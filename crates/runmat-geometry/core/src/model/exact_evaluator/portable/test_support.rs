use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::super::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};

pub(super) struct BudgetControl {
    cancelled: AtomicBool,
    iterations: AtomicU64,
    search_work: AtomicU64,
    allocation_bytes: AtomicU64,
}

impl BudgetControl {
    pub(super) fn new(iterations: u64, search_work: u64) -> Self {
        Self::with_limits(iterations, search_work, u64::MAX)
    }

    pub(super) fn with_limits(iterations: u64, search_work: u64, allocation_bytes: u64) -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            iterations: AtomicU64::new(iterations),
            search_work: AtomicU64::new(search_work),
            allocation_bytes: AtomicU64::new(allocation_bytes),
        }
    }

    pub(super) fn generous() -> Self {
        Self::with_limits(10_000_000, 10_000_000, 10_000_000)
    }

    pub(super) fn cancelled() -> Self {
        let control = Self::with_limits(u64::MAX, u64::MAX, u64::MAX);
        control.cancelled.store(true, Ordering::Relaxed);
        control
    }

    pub(super) fn allocation_limited(allocation_bytes: u64) -> Self {
        Self::with_limits(u64::MAX, u64::MAX, allocation_bytes)
    }

    fn consume(remaining: &AtomicU64, count: u64) -> Result<(), GeometryEvaluationError> {
        remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_sub(count)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::BudgetExceeded,
                    "test geometry evaluation budget exceeded",
                )
            })
    }
}

impl GeometryEvaluationControl for BudgetControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "test geometry evaluation cancelled",
            ));
        }
        Ok(())
    }

    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.iterations, count)
    }

    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.search_work, count)
    }

    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.allocation_bytes, count)
    }
}
