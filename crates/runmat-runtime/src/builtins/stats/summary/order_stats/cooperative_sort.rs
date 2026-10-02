//! Stable numeric and associated-value ordering with bounded work.

use std::cmp::Ordering;

use crate::execution::{CpuCooperationBudget, ExecutionServiceError};

mod generic;
#[cfg(test)]
use generic::RUN_LEN;
pub(crate) use generic::{stable_sort_by_sync, SortProgress, StableSortWork};

#[derive(Debug)]
pub(crate) enum CooperativeSortError {
    Resource(std::collections::TryReserveError),
    Service(ExecutionServiceError),
}

/// Sort non-NaN values using the same stable partial order as before. NaN
/// admission and signed-zero policy remain with the caller.
pub(crate) async fn stable_sort_non_nan_f64(
    values: &mut Vec<f64>,
    mut budget: Option<&mut CpuCooperationBudget>,
) -> Result<(), CooperativeSortError> {
    if let Some(budget) = budget.as_deref_mut() {
        budget
            .checkpoint(0)
            .await
            .map_err(CooperativeSortError::Service)?;
    }
    if budget.is_none() {
        return stable_sort_by_sync(values, |left: &f64, right: &f64| {
            left.partial_cmp(right).unwrap_or(Ordering::Equal)
        });
    }
    let mut work = StableSortWork::new(values, |left: &f64, right: &f64| {
        left.partial_cmp(right).unwrap_or(Ordering::Equal)
    });
    loop {
        let progress = work.advance()?;
        let units = match progress {
            SortProgress::Pending(units) | SortProgress::Complete(units) => units,
        };
        if let Some(budget) = budget.as_deref_mut() {
            budget
                .checkpoint(units)
                .await
                .map_err(CooperativeSortError::Service)?;
        }
        if matches!(progress, SortProgress::Complete(_)) {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests;
