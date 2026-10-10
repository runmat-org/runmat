//! Stable numeric and associated-value ordering with bounded work.

#[cfg(any(feature = "plot-core", test))]
use std::cmp::Ordering;

#[cfg(any(feature = "plot-core", test))]
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};

#[cfg(test)]
use crate::builtins::common::cooperative_sort::RUN_LEN;
#[cfg(any(feature = "plot-core", test))]
use crate::builtins::common::cooperative_sort::{
    stable_sort_by_sync, SortProgress, SortResourceError, StableSortWork,
};

#[cfg(any(feature = "plot-core", test))]
#[derive(Debug, thiserror::Error)]
pub(crate) enum CooperativeSortError {
    #[error("sort allocation failed: {0}")]
    Resource(#[source] std::collections::TryReserveError),
    #[cfg(any(feature = "plot-core", test))]
    #[error("sort execution failed: {0}")]
    Service(#[source] ExecutionServiceError),
}

/// Sort non-NaN values using the same stable partial order as before. NaN
/// admission and signed-zero policy remain with the caller.
#[cfg(any(feature = "plot-core", test))]
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
        })
        .map_err(|SortResourceError(source)| CooperativeSortError::Resource(source));
    }
    let mut work = StableSortWork::new(values, |left: &f64, right: &f64| {
        left.partial_cmp(right).unwrap_or(Ordering::Equal)
    });
    loop {
        let progress = work
            .advance()
            .map_err(|SortResourceError(source)| CooperativeSortError::Resource(source))?;
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
