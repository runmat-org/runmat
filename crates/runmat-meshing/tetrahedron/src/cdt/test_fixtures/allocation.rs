//! Finite requested-storage authority; neither allocator bookkeeping nor RSS.
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};
use std::sync::atomic::{AtomicU64, Ordering};
pub(in crate::cdt) struct AllocationBudget {
    limit: u64,
    live: AtomicU64,
    peak: AtomicU64,
}
impl AllocationBudget {
    pub fn live_bytes(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }
    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }
    }
}
impl GeometryEvaluationControl for AllocationBudget {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        let previous = self
            .live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "test allocation admission",
                )
            })?;
        self.peak.fetch_max(previous + bytes, Ordering::Relaxed);
        Ok(())
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        let previous = self.live.fetch_sub(bytes, Ordering::Relaxed);
        assert!(previous >= bytes);
    }
}
