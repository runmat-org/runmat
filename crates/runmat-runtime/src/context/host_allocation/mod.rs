//! Managed physical host allocations for ordinary numeric calls.
//!
//! The installed parallel service remains the only finite-capacity authority.
//! This module supplies checked workload bounds and an allocation receipt; it
//! neither estimates process RSS nor certifies host I/O effects or placement.

mod collection_allocator;
mod control_error;
mod frame_collection;
mod plan;
mod provider_registration;
mod receipt;
mod session;
mod temporary;
mod trace_scratch;

pub use collection_allocator::{
    hashbrown_table_layout, AdmittedCollectionAllocator, OperationCollectionAllocator,
};
pub(crate) use collection_allocator::{OperationPermitGate, OperationPermitGateError};
pub use control_error::host_allocation_runtime_error;
pub(crate) use control_error::HostAllocationDisposition;
pub use frame_collection::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, FrameCollectionGate,
    FrameCollectionPermit,
};
pub use plan::HostAllocationPlan;
pub use provider_registration::{provider_registration_host_quote, ProviderRegistrationHostQuote};
pub(crate) use receipt::HostOutputPermit;
pub use receipt::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt, HostRetainedViewPermit,
};
pub(crate) use session::HostRetainedSessionAllocation;
pub use temporary::{HostComplexWorkspace, HostNumericWorkspace};
pub(crate) use trace_scratch::HostTraceScratchPermit;

#[cfg(test)]
mod tests;
