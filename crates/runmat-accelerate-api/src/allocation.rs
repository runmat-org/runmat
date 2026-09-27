//! Additive allocation lifetime primitive; not yet wired into tensor handles.
//!
//! The provider's existing allocation map remains the registration and physical
//! retirement authority. This module owns no map, ambient provider lookup,
//! thread, timer, or retry policy.

mod lease;
mod owner;

pub use lease::{AllocationLease, WeakAllocationLease};
pub use owner::{
    AllocationOwner, AllocationOwnerThreadBound, AllocationRegistrationError,
    AllocationRetirementOutcome,
};

/// Immutable process-local allocation identity, independent of tensor views.
/// Constructing this descriptor alone grants no ownership or access.
/// Providers must never reuse a live `(device, allocation, generation)` tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
