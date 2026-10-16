//! Owning allocation leases and validated tensor views.
//!
//! The provider's existing allocation map remains the registration and physical
//! retirement authority. This module owns no map, ambient provider lookup,
//! thread, timer, or retry policy.

mod cleanup;
mod descriptor;
mod lease;
mod owner;
mod release;
mod shared;
mod tensor;

pub use cleanup::{catch_allocation_cleanup_unwind, AllocationCleanupPanicked};
pub use descriptor::{
    AllocationAcquisitionError, GpuPhysicalDescriptor, GpuTensorLayout, GpuTensorView,
    GpuTensorWireDescriptor, GpuViewError,
};
pub use lease::{AllocationLease, RetainedLeaseHostMetadata, WeakAllocationLease};
pub use owner::{
    AllocationOwner, AllocationOwnerHandle, AllocationRegistrationError,
    AllocationRetirementOutcome,
};
pub use release::{AllocationReleaseError, AllocationReleaseOutcome, GpuHandleReleaseError};
pub use tensor::{GpuTensorHandle, TensorViewIdentity, WeakGpuTensorHandle};

/// Immutable process-local allocation identity, independent of tensor views.
/// Constructing this descriptor alone grants no ownership or access.
/// Providers must never reuse a live `(device, allocation, generation)` tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AllocationIdentity {
    device_id: u32,
    allocation_id: u64,
    generation: u64,
}

impl AllocationIdentity {
    pub const fn new(device_id: u32, allocation_id: u64, generation: u64) -> Self {
        Self {
            device_id,
            allocation_id,
            generation,
        }
    }

    pub const fn device_id(self) -> u32 {
        self.device_id
    }
    pub const fn allocation_id(self) -> u64 {
        self.allocation_id
    }
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests;
