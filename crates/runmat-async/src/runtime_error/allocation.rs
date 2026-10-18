//! Retained diagnostic allocation ownership, independent of error formatting.

use std::mem::ManuallyDrop;
use std::sync::Arc;

/// An allocation owner that survives with a diagnostic. This is ownership,
/// not a certificate for the publicly mutable message, context, or later
/// executor presentation allocations.
pub trait DiagnosticAllocationLease: std::fmt::Debug + Send + Sync {}

impl<T: std::fmt::Debug + Send + Sync + ?Sized> DiagnosticAllocationLease for T {}

/// A managed diagnostic allocation whose uniquely owned backing may carry
/// the reservation for that same backing. Installed finite owners override
/// retirement so physical metadata is released before its charge is refunded.
pub trait ManagedDiagnosticAllocationLease: DiagnosticAllocationLease {
    fn retire_owned(self: Box<Self>) {
        drop(self);
    }
}

#[derive(Debug)]
pub struct ManagedDiagnosticAllocation {
    lease: ManuallyDrop<Box<dyn ManagedDiagnosticAllocationLease>>,
}

impl ManagedDiagnosticAllocation {
    pub fn new(lease: Box<dyn ManagedDiagnosticAllocationLease>) -> Self {
        Self {
            lease: ManuallyDrop::new(lease),
        }
    }
}

impl Drop for ManagedDiagnosticAllocation {
    fn drop(&mut self) {
        // SAFETY: this is the sole Drop path for the move-only field.
        let lease = unsafe { ManuallyDrop::take(&mut self.lease) };
        lease.retire_owned();
    }
}

/// Move an already-allocated, domain-charged lease into an error without
/// allocating a wrapper or changing its typed `source()` chain.
#[derive(Debug)]
pub enum RetainedDiagnosticAllocation {
    Shared(Arc<dyn DiagnosticAllocationLease>),
    Owned(Box<dyn DiagnosticAllocationLease>),
    ManagedOwned(ManagedDiagnosticAllocation),
}
