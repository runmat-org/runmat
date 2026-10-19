use crate::{GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind};
use std::sync::atomic::{AtomicU64, Ordering};
pub(crate) struct Budget {
    limit: u64,
    live: AtomicU64,
    peak: AtomicU64,
    probes: AtomicU64,
    stop_at: u64,
}
impl Budget {
    pub fn new(limit: u64) -> Self {
        Self::with_stop(limit, u64::MAX)
    }
    pub fn with_stop(limit: u64, stop_at: u64) -> Self {
        Self {
            limit,
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            probes: AtomicU64::new(0),
            stop_at,
        }
    }
    pub fn live(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
    pub fn probes(&self) -> u64 {
        self.probes.load(Ordering::Relaxed)
    }
}
impl GeometryEvaluationControl for Budget {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.probes.fetch_add(1, Ordering::Relaxed) >= self.stop_at {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::TimeBudgetExceeded,
                "original test deadline",
            ));
        }
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        let before = self
            .live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "original test storage ceiling",
                )
            })?;
        self.peak.fetch_max(before + bytes, Ordering::Relaxed);
        Ok(())
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        assert!(self.live.fetch_sub(bytes, Ordering::Relaxed) >= bytes);
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
}
