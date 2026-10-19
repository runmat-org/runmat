//! Allocation ownership for a nested calculation; work and cancellation remain global.
use super::{
    GeometryAllocationReservation, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind,
};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct GeometryAllocationScope<'a> {
    parent: &'a dyn GeometryEvaluationControl,
    held: AtomicU64,
}

impl<'a> GeometryAllocationScope<'a> {
    pub fn new(parent: &'a dyn GeometryEvaluationControl) -> Self {
        Self {
            parent,
            held: AtomicU64::new(0),
        }
    }

    /// Transfer an already admitted retained result to the parent calculation.
    /// All other successful charges are released when this scope ends.
    pub fn retain_bytes(&self, bytes: u64) {
        self.held
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                held.checked_sub(bytes)
            })
            .expect("retained more geometry storage than this scope owns");
    }

    /// Move already admitted storage into a guard with the parent's lifetime.
    /// Failure leaves this scope's charge intact, so its owned values can be
    /// dropped before the scope releases storage. Drop unused values before
    /// ending the scope, and keep the returned guard until the retained value dies.
    pub fn transfer_reservation(
        &self,
        bytes: u64,
    ) -> Result<GeometryAllocationReservation<'a>, GeometryEvaluationError> {
        self.parent.checkpoint()?;
        self.held
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                held.checked_sub(bytes)
            })
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "retained geometry storage exceeds its admitted scope",
                )
            })?;
        Ok(GeometryAllocationReservation::from_admitted(
            self.parent,
            bytes,
        ))
    }
}

impl GeometryEvaluationControl for GeometryAllocationScope<'_> {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        self.parent.checkpoint()
    }
    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.parent.consume_iterations(count)
    }
    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.parent.consume_search_work(count)
    }
    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.parent.consume_allocation_bytes(count)?;
        self.held
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                held.checked_add(count)
            })
            .expect("scope storage exceeds its admitted parent inventory");
        Ok(())
    }
    fn release_allocation_bytes(&self, count: u64) {
        self.retain_bytes(count);
        self.parent.release_allocation_bytes(count);
    }
}

impl Drop for GeometryAllocationScope<'_> {
    fn drop(&mut self) {
        self.parent
            .release_allocation_bytes(self.held.load(Ordering::Relaxed));
    }
}
