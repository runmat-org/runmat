use std::sync::atomic::{AtomicU64, Ordering};

use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};

use crate::{
    exact::ExactCadImportOptions,
    import::{GeometryImportContext, GeometryImportError},
};

pub(super) struct ImportEvaluationControl<'a> {
    context: &'a GeometryImportContext,
    iterations: AtomicU64,
    search_work: AtomicU64,
    allocation_bytes: AtomicU64,
}

impl<'a> ImportEvaluationControl<'a> {
    pub fn new(context: &'a GeometryImportContext, options: &ExactCadImportOptions) -> Self {
        Self {
            context,
            iterations: AtomicU64::new(options.max_validation_iterations),
            search_work: AtomicU64::new(options.max_validation_search_work),
            allocation_bytes: AtomicU64::new(options.max_validation_allocation_bytes),
        }
    }
}

impl GeometryEvaluationControl for ImportEvaluationControl<'_> {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.context.is_cancelled() {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "exact CAD import validation was cancelled",
            ))
        } else {
            Ok(())
        }
    }

    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        consume(&self.iterations, count, "iteration")
    }

    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        consume(&self.search_work, count, "search-work")
    }

    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        consume(&self.allocation_bytes, count, "allocation-byte")
    }
}

pub(super) fn map_validation_error(error: GeometryEvaluationError) -> GeometryImportError {
    match error.kind {
        GeometryEvaluationErrorKind::Cancelled => GeometryImportError::Cancelled,
        GeometryEvaluationErrorKind::BudgetExceeded
        | GeometryEvaluationErrorKind::TimeBudgetExceeded
        | GeometryEvaluationErrorKind::AllocationBudgetExceeded
        | GeometryEvaluationErrorKind::SearchWorkBudgetExceeded
        | GeometryEvaluationErrorKind::IterationBudgetExceeded => {
            GeometryImportError::ExactValidationBudgetExceeded(error.reason)
        }
        _ => GeometryImportError::InvalidGeometry(error.to_string()),
    }
}

fn consume(
    remaining: &AtomicU64,
    count: u64,
    resource: &str,
) -> Result<(), GeometryEvaluationError> {
    remaining
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |available| {
            available.checked_sub(count)
        })
        .map(|_| ())
        .map_err(|_| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::BudgetExceeded,
                format!("exact CAD import exhausted its {resource} validation budget"),
            )
        })
}
