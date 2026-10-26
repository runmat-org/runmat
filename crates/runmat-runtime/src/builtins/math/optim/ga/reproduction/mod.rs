//! A generation is published only after every island's offspring are admitted.
mod destination;
pub(super) mod mutation;
mod parents;
mod variation;
#[cfg(test)]
mod tests;
use super::callbacks::Invocation;
use super::{
    common, invalid, linear,
    operators::{custom, ranking::Score},
    options,
    population::Generation,
    ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use common::{population::projection::Arguments, solver_timing::SolverTiming};
use destination::Destination;
pub(super) use mutation::Mutation;
use runmat_value::{Tensor, Value};
use std::ops::ControlFlow;

pub(super) struct Current<'a> {
    pub generation: &'a Generation,
    pub comparison: &'a [Score],
    pub public_scores: &'a Value,
    pub controls: Option<&'a Value>,
    pub state: Option<&'a Value>,
    pub problem: &'a Value,
    pub options: &'a options::Options,
    pub timing: &'a SolverTiming,
    pub record: bool,
}
pub(super) struct Produced {
    pub generation: Generation,
    /// Exact unchanged copies, not equality guesses about nearby chromosomes.
    pub copied_from: Vec<Option<usize>>,
    pub expectation: Option<Tensor>,
    pub selection: Vec<usize>,
    pub lineage: Vec<super::lineage::Lineage>,
    pub ancestors: Vec<Option<super::lineage::Parents>>,
    pub unverified_proposals: bool,
}
/// None means the original MaxTime deadline was reached before publication.
/// Completed generations move their existing buffers without heap indirection.
pub(super) async fn next_with<G: linear::Geometry, I: Invocation>(
    current: Current<'_>,
    mut domain: Option<&mut G>,
    mutation: &mut Mutation,
    invocation: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Option<Produced>>> {
    cooperation.completed(0).await?;
    let total = current.generation.members();
    if current.options.layout.total() != total || current.comparison.len() != total {
        return Err(invalid("reproduction population and score geometry differ"));
    }
    if current.options.maximum.reached(current.timing.elapsed()) {
        return Ok(ControlFlow::Continue(None));
    }
    let mut next =
        Destination::new(current.generation, current.options.population, cooperation).await?;
    let mut copied_from = POLICY.capacity(total, cooperation.runtime())?;
    let mut expectation = POLICY.capacity(
        if current.record { total } else { 0 },
        cooperation.runtime(),
    )?;
    let max_parents = if current.record {
        total
            .checked_mul(2)
            .ok_or_else(|| invalid("selection history extent overflow"))?
    } else {
        0
    };
    let mut selection = POLICY.capacity(max_parents, cooperation.runtime())?;
    let mut lineage = POLICY.capacity(
        if current.record { total } else { 0 },
        cooperation.runtime(),
    )?;
    let mut ancestors = POLICY.capacity(
        if current.record { total } else { 0 },
        cooperation.runtime(),
    )?;
    for _ in 0..total {
        copied_from.push(None);
        if current.record {
            expectation.push(0.);
        }
        cooperation.completed(1).await?;
    }
    let mut unverified_proposals = false;
    for group in 0..current.options.layout.groups() {
        if current.options.maximum.reached(current.timing.elapsed()) {
            return Ok(ControlFlow::Continue(None));
        }
        let range = current.options.layout.range(group);
        let elites = current.options.elites[group];
        let children = range.len() - elites;
        let crossover =
            ((children as f64 * current.options.crossover_fraction).round() as usize).min(children);
        let parent_count = crossover
            .checked_mul(2)
            .and_then(|n| n.checked_add(children - crossover))
            .ok_or_else(|| invalid("selected parent extent overflow"))?;
        let chosen = parents::choose(
            parents::Request {
                comparison: &current.comparison[range.clone()],
                scores: current.public_scores,
                offset: range.start,
                count: parent_count,
                options: current.options,
                controls: current.controls,
            },
            invocation,
            cooperation,
        )
        .await?;
        let chosen = match chosen {
            ControlFlow::Continue(chosen) => chosen,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        if current.record {
            for (row, &weight) in chosen
                .expectation
                .as_f64_slice()
                .expect("double expectations")
                .iter()
                .enumerate()
            {
                expectation[range.start + row] = weight;
                cooperation.completed(1).await?;
            }
        }
        let mut destinations = POLICY.capacity(range.len(), cooperation.runtime())?;
        let mut sources = POLICY.capacity(elites, cooperation.runtime())?;
        for (offset, &source) in chosen.order.as_slice()[..elites].iter().enumerate() {
            let destination = range.start + offset;
            let source = range.start + source;
            destinations.push(destination);
            sources.push(source);
            copied_from[destination] = Some(source);
            if current.record {
                selection.push(source);
                lineage.push(super::lineage::Lineage::Elite);
                ancestors.push(Some(super::lineage::Parents {
                    first: source,
                    second: None,
                }));
            }
            cooperation.completed(1).await?;
        }
        if let ControlFlow::Break(stop) = next
            .copy(
                &destinations,
                &sources,
                current.generation,
                invocation,
                cooperation,
            )
            .await?
        {
            return Ok(ControlFlow::Break(stop));
        }
        let mut selected = chosen.selected;
        for parent in &mut selected {
            *parent = parent
                .checked_add(range.start)
                .ok_or_else(|| invalid("parent row overflow"))?;
            if current.record {
                selection.push(*parent);
            }
            cooperation.completed(1).await?;
        }
        let context = current.controls.map(|controls| custom::Current {
            controls,
            problem: current.problem,
            scores: current.public_scores,
            population: current.generation,
            kind: current.options.population,
        });
        let split = crossover * 2; // parent_count checked this product above.
        for (is_crossover, parents, first, count) in [
            (true, &selected[..split], range.start + elites, crossover),
            (
                false,
                &selected[split..],
                range.start + elites + crossover,
                children - crossover,
            ),
        ] {
            if count == 0 {
                continue;
            }
            destinations.clear();
            for row in first..first + count {
                destinations.push(row);
                cooperation.completed(1).await?;
            }
            let custom = if is_crossover {
                current.options.crossover.custom().is_some()
            } else {
                current.options.mutation.custom().is_some()
            };
            if custom {
                let context = context
                    .as_ref()
                    .ok_or_else(|| invalid("custom variation requires its effective controls"))?;
                let children = if is_crossover {
                    custom::crossover(
                        &current.options.crossover,
                        parents,
                        context,
                        invocation,
                        cooperation,
                    )
                    .await?
                } else {
                    custom::mutation(
                        &current.options.mutation,
                        parents,
                        context,
                        current.state.ok_or_else(|| {
                            invalid("custom mutation requires its completed-generation state")
                        })?,
                        invocation,
                        cooperation,
                    )
                    .await?
                };
                let children = match children {
                    ControlFlow::Continue(children) => children,
                    ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
                };
                let status = variation::install_custom(
                    variation::Custom {
                        rows: &destinations,
                        parents,
                        crossover: is_crossover,
                        children,
                        current: &current,
                    },
                    &mut next,
                    domain.as_deref_mut(),
                    &mut copied_from,
                    invocation,
                    cooperation,
                )
                .await?;
                let status = match status {
                    ControlFlow::Continue(status) => status,
                    ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
                };
                if status.timed_out {
                    return Ok(ControlFlow::Continue(None));
                }
                unverified_proposals |= status.unverified;
            } else {
                for (child, &destination) in destinations.iter().enumerate() {
                    if current.options.maximum.reached(current.timing.elapsed()) {
                        return Ok(ControlFlow::Continue(None));
                    }
                    let first_parent = parents[if is_crossover { child * 2 } else { child }];
                    let status = variation::builtin(
                        variation::Request {
                            crossover: is_crossover,
                            first: first_parent,
                            second: is_crossover.then(|| parents[child * 2 + 1]),
                            destination,
                            current: &current,
                        },
                        &mut next,
                        domain.as_deref_mut(),
                        mutation,
                        cooperation,
                    )
                    .await?;
                    if status.timed_out {
                        return Ok(ControlFlow::Continue(None));
                    }
                    unverified_proposals |= status.unverified;
                    if status.retained {
                        copied_from[destination] = Some(first_parent);
                    }
                }
            }
            if current.record {
                for (child, &destination) in destinations.iter().enumerate() {
                    lineage.push(if copied_from[destination].is_some() {
                        super::lineage::Lineage::Retained
                    } else if is_crossover {
                        super::lineage::Lineage::Crossover
                    } else {
                        super::lineage::Lineage::Mutation
                    });
                    ancestors.push(Some(super::lineage::Parents {
                        first: parents[if is_crossover { child * 2 } else { child }],
                        second: (is_crossover && copied_from[destination].is_none())
                            .then(|| parents[child * 2 + 1]),
                    }));
                    cooperation.completed(1).await?;
                }
            }
        }
    }
    let expectation = if current.record {
        Some(Tensor::new(expectation, vec![total, 1]).map_err(invalid)?)
    } else {
        None
    };
    Ok(ControlFlow::Continue(Some(Produced {
        generation: next.finish(),
        copied_from,
        expectation,
        selection,
        lineage,
        ancestors,
        unverified_proposals,
    })))
}
