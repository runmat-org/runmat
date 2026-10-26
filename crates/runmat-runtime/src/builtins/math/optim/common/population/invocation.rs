//! One original selected callback wave, shared by population objectives and
//! multi-output constraints. Numerical decoding/counter interpretation stays
//! with the caller, and no accepted worker result can replay serially.
use super::super::callback_source::Callable;
use super::{projection::Arguments, Policy, ScalarCooperation};
use crate::{
    execution::{CallbackBatch, RootedValueSequence},
    BuiltinResult,
};
#[derive(Default, Clone, Copy)]
pub(crate) struct Counts {
    pub points: usize,
    pub callbacks: usize,
}
pub(crate) enum Wave {
    Vectorized {
        start: usize,
        end: usize,
        returned: RootedValueSequence,
    },
    Rows {
        start: usize,
        end: usize,
        returned: Vec<RootedValueSequence>,
    },
}
impl Wave {
    pub(crate) fn end(&self) -> usize {
        match self {
            Self::Vectorized { end, .. } | Self::Rows { end, .. } => *end,
        }
    }
}
pub(crate) struct Invocation<'a> {
    pub policy: Policy,
    pub function: Callable<'a>,
    pub vectorized: bool,
    pub parallel: Option<&'a CallbackBatch>,
    pub counts: &'a mut Counts,
    pub cooperation: &'a mut ScalarCooperation,
}
impl Invocation<'_> {
    pub(crate) async fn next<P: Arguments>(
        self,
        population: &P,
        selected: &[usize],
        start: usize,
        outputs: usize,
    ) -> BuiltinResult<Wave> {
        let policy = self.policy;
        if outputs == 0 || start >= selected.len() {
            return Err(policy.invalid("callback wave requires actual selected rows and outputs"));
        }
        let width = if self.vectorized {
            selected.len() - start
        } else {
            self.parallel.map_or(1, CallbackBatch::width)
        };
        if width == 0 {
            return Err(policy.invalid("selected callback wave has zero width"));
        }
        let end = start.saturating_add(width).min(selected.len());
        let rows = &selected[start..end];
        for &row in rows {
            if row >= population.members() {
                return Err(policy.invalid("callback row is outside the population"));
            }
            self.cooperation.completed(1).await?;
        }
        let returned = if self.vectorized {
            let (argument, roots) = population
                .project(rows, self.cooperation)
                .await?
                .into_parts();
            let returned = self
                .function
                .invoke(self.cooperation.runtime(), [argument], outputs)
                .await?;
            drop(roots);
            charge(policy, self.counts, rows.len(), 1)?;
            return Ok(Wave::Vectorized {
                start,
                end,
                returned,
            });
        } else if let Some(batch) = self.parallel {
            let mut arguments = policy.capacity(rows.len(), self.cooperation.runtime())?;
            let mut roots = Vec::new();
            for row in rows {
                let (argument, lease) = population
                    .project(std::slice::from_ref(row), self.cooperation)
                    .await?
                    .into_parts();
                arguments.push(vec![argument]);
                if let Some(lease) = lease {
                    if roots.capacity() == 0 {
                        roots = policy.capacity(rows.len(), self.cooperation.runtime())?;
                    }
                    roots.push(lease);
                }
            }
            let returned = {
                let (_, budget) = self.cooperation.runtime_budget();
                batch
                    .execute(
                        arguments,
                        outputs,
                        budget.ok_or_else(|| {
                            policy.invalid("parallel route lacks its original CPU budget")
                        })?,
                    )?
                    .await?
            };
            drop(roots);
            charge(policy, self.counts, rows.len(), rows.len())?;
            if returned.len() != rows.len() {
                return Err(policy.invalid("worker callback result cardinality changed"));
            }
            returned
        } else {
            let (argument, roots) = population
                .project(&rows[..1], self.cooperation)
                .await?
                .into_parts();
            let returned = self
                .function
                .invoke(self.cooperation.runtime(), [argument], outputs)
                .await?;
            drop(roots);
            charge(policy, self.counts, 1, 1)?;
            let mut result = policy.capacity(1, self.cooperation.runtime())?;
            result.push(returned);
            result
        };
        Ok(Wave::Rows {
            start,
            end,
            returned,
        })
    }
}
fn charge(policy: Policy, counts: &mut Counts, points: usize, calls: usize) -> BuiltinResult<()> {
    counts.points = counts
        .points
        .checked_add(points)
        .ok_or_else(|| policy.invalid("function point count overflow"))?;
    counts.callbacks = counts
        .callbacks
        .checked_add(calls)
        .ok_or_else(|| policy.invalid("callback call count overflow"))?;
    Ok(())
}
