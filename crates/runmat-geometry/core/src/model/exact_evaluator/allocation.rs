use super::{GeometryEvaluationControl, GeometryEvaluationError};

/// A non-cloneable reservation for temporary evaluation storage. Declare it
/// before the storage so normal return, errors and unwinding drop storage first.
/// Retained results must keep their charge for their full lifetime instead.
pub struct GeometryAllocationReservation<'a> {
    control: &'a dyn GeometryEvaluationControl,
    bytes: u64,
}

impl<'a> GeometryAllocationReservation<'a> {
    pub(super) fn from_admitted(control: &'a dyn GeometryEvaluationControl, bytes: u64) -> Self {
        Self { control, bytes }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn new(
        control: &'a dyn GeometryEvaluationControl,
        bytes: u64,
    ) -> Result<Self, GeometryEvaluationError> {
        control.checkpoint()?;
        control.consume_allocation_bytes(bytes)?;
        Ok(Self { control, bytes })
    }

    /// Extend this charge by `bytes` before the storage it covers grows. A refused extension
    /// leaves the charge unchanged.
    pub fn grow(&mut self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.control.checkpoint()?;
        self.control.consume_allocation_bytes(bytes)?;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .expect("an admitted charge fits the allocation budget");
        Ok(())
    }

    /// Release `bytes` of this charge once the storage they covered is gone.
    pub fn release(&mut self, bytes: u64) {
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .expect("released more storage than reserved");
        self.control.release_allocation_bytes(bytes);
    }

    /// Keep part of this charge after returning a retained result. That part is
    /// conservatively charged for the remaining evaluation lifetime.
    pub fn retain_bytes(&mut self, bytes: u64) {
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .expect("retained more storage than reserved");
    }
}

impl std::fmt::Debug for GeometryAllocationReservation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeometryAllocationReservation")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for GeometryAllocationReservation<'_> {
    fn drop(&mut self) {
        self.control.release_allocation_bytes(self.bytes);
    }
}
