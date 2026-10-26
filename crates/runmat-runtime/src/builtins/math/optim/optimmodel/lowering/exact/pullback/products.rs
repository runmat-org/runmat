//! Exact affine product slopes use only structurally constant factors.
use super::super::super::{factors::Factors, operations::Operation};
use super::*;
mod retained;
#[derive(Clone)]
pub(super) struct Entry {
    source: Option<usize>,
    constant: Rational,
}
impl Entry {
    fn one() -> Self {
        Self {
            source: None,
            constant: Rational::from_usize(1),
        }
    }
    async fn consume(
        &mut self,
        state: &State<'_>,
        constants: &Evaluated,
        root: GcHandle,
        row: usize,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if state.active_at(root, row)? {
            if self.source.is_some() {
                return Err(state
                    .policy
                    .invalid("non-affine factor reached exact reverse"));
            }
            self.source = Some(row);
        } else {
            let value = constants
                .value(root, row, state.policy)?
                .ok_or_else(|| state.policy.invalid("unassigned exact constant factor"))?;
            self.constant = self.constant.mul(&value, state.policy, c).await?;
        }
        Ok(())
    }
}
pub(super) async fn propagate(
    state: &mut State<'_>,
    constants: &Evaluated,
    handle: GcHandle,
    operation: Operation,
    source: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if constants.retains_operands() {
        return retained::propagate(
            state, constants, operation, source, arguments, shape, weights, c,
        )
        .await;
    }
    if !state.active(source)? {
        return Ok(());
    }
    let policy = state.policy;
    let cached = state
        .plan
        .as_ref()
        .and_then(|plan| plan.products.get(&handle))
        .cloned();
    let factors;
    let shared;
    let prepared = if let Some(cached) = cached {
        shared = cached;
        &shared[..]
    } else {
        let built = prepare(state, constants, operation, source, arguments, shape, c).await?;
        if let Some(plan) = &mut state.plan {
            let cached = std::rc::Rc::new(built);
            plan.products
                .try_reserve(1)
                .map_err(|e| policy.reserve(e))?;
            plan.products.insert(handle, std::rc::Rc::clone(&cached));
            shared = cached;
            &shared[..]
        } else {
            factors = built;
            &factors[..]
        }
    };
    for (row, value) in weights {
        if let Some(source_row) = prepared[row].source {
            state
                .add(
                    source,
                    source_row,
                    value.mul(&prepared[row].constant, policy, c).await?,
                    c,
                )
                .await?;
        }
        c.completed(1).await?;
    }
    Ok(())
}

async fn prepare(
    state: &State<'_>,
    constants: &Evaluated,
    operation: Operation,
    source: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Entry>> {
    let policy = state.policy;
    let input = state
        .program
        .source
        .nodes
        .get(&source)
        .ok_or_else(|| policy.invalid("factor source is unavailable"))?
        .node()
        .shape
        .clone();
    let mut plan = Factors::prepare(operation, &input, arguments, shape, policy, c).await?;
    let count = evaluation::element_count(shape, policy)?;
    let mut factors = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for _ in 0..count {
        factors.push(Entry::one());
        c.completed(1).await?;
    }
    match &mut plan {
        Factors::Reduction(reduction) => {
            for row in 0..evaluation::element_count(&input, policy)? {
                let target = reduction
                    .project(row)
                    .ok_or_else(|| policy.invalid("reverse factor exceeds reduction"))?;
                factors[target]
                    .consume(state, constants, source, row, c)
                    .await?;
                c.completed(1).await?;
            }
        }
        Factors::Scan(scan) => {
            let mut running = Entry::one();
            for step in scan.steps(false) {
                if step.first {
                    running = Entry::one();
                }
                running
                    .consume(state, constants, source, step.index, c)
                    .await?;
                factors[step.index] = running.clone();
                c.completed(1).await?;
            }
        }
    }
    Ok(factors)
}
