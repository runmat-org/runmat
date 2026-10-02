//! Stable numeric ordering with bounded synchronous runs and cooperative merge.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::execution::{CpuCooperationBudget, ExecutionServiceError};

// This bounds one uninterrupted library sort, not the input size or a
// scheduling threshold. The runtime service still owns when a turn is due.
const RUN_LEN: usize = 256;

#[derive(Debug)]
pub(crate) enum CooperativeSortError {
    Resource,
    Service(ExecutionServiceError),
}

#[derive(Clone, Copy, Debug)]
struct RunHead {
    value: f64,
    run: usize,
    offset: usize,
}

impl PartialEq for RunHead {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RunHead {}

impl PartialOrd for RunHead {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RunHead {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.run.cmp(&other.run))
            .then_with(|| self.offset.cmp(&other.offset))
    }
}

async fn checkpoint(
    budget: &mut Option<&mut CpuCooperationBudget>,
    units: usize,
) -> Result<(), CooperativeSortError> {
    if let Some(budget) = budget.as_deref_mut() {
        budget
            .checkpoint(u64::try_from(units).unwrap_or(u64::MAX))
            .await
            .map_err(CooperativeSortError::Service)?;
    }
    Ok(())
}

/// Sort a vector without NaNs using the same stable partial ordering as
/// `slice::sort_by(|a,b| a.partial_cmp(b).unwrap_or(Equal))`. The caller owns
/// NaN admission and any public error policy. Signed zeros keep input order.
pub(crate) async fn stable_sort_non_nan_f64(
    values: &mut Vec<f64>,
    mut budget: Option<&mut CpuCooperationBudget>,
) -> Result<(), CooperativeSortError> {
    checkpoint(&mut budget, 0).await?;
    for run in values.chunks_mut(RUN_LEN) {
        run.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        checkpoint(&mut budget, run.len()).await?;
    }
    if values.len() <= RUN_LEN {
        return Ok(());
    }

    let run_count = values.len().div_ceil(RUN_LEN);
    let mut heap = BinaryHeap::<Reverse<RunHead>>::new();
    heap.try_reserve(run_count)
        .map_err(|_| CooperativeSortError::Resource)?;
    for run in 0..run_count {
        heap.push(Reverse(RunHead {
            value: values[run * RUN_LEN],
            run,
            offset: 0,
        }));
        checkpoint(&mut budget, 1).await?;
    }

    let mut merged = Vec::new();
    merged
        .try_reserve_exact(values.len())
        .map_err(|_| CooperativeSortError::Resource)?;
    while let Some(Reverse(head)) = heap.pop() {
        merged.push(head.value);
        let offset = head.offset + 1;
        let index = head.run * RUN_LEN + offset;
        if offset < RUN_LEN && index < values.len() {
            heap.push(Reverse(RunHead {
                value: values[index],
                run: head.run,
                offset,
            }));
        }
        checkpoint(&mut budget, 1).await?;
    }
    *values = merged;
    Ok(())
}

#[cfg(test)]
mod tests;
