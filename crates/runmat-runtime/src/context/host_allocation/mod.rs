//! Managed physical host allocations for ordinary numeric calls.
//!
//! The installed parallel service remains the only finite-capacity authority.
//! This module supplies checked workload bounds and an allocation receipt; it
//! neither estimates process RSS nor certifies host I/O effects or placement.

mod aligned;
mod checkpoint;
mod collection_allocator;
mod control_error;
mod frame_collection;
mod owner;
mod plan;
mod provider_registration;
mod receipt;
mod session;
mod temporary;
mod text;
mod trace_scratch;
mod unbounded_workspace;
pub use unbounded_workspace::require_unbounded_workspace;

pub(crate) use aligned::AlignedHostWorkspace;
pub(crate) use checkpoint::HostAllocationCheckpoint;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use checkpoint::HostAllocationWorkerCheckpoint;
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
pub(in crate::context) use owner::HostAllocationContext;
pub use plan::HostAllocationPlan;
pub use provider_registration::{provider_registration_host_quote, ProviderRegistrationHostQuote};
pub(crate) use receipt::HostOutputPermit;
pub use receipt::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt, HostRetainedViewPermit,
};
pub(crate) use session::HostRetainedSessionAllocation;
pub use temporary::{
    HostComplexWorkspace, HostNumericWorkspace, HostTypedElement, HostTypedWorkspace,
};
pub use text::HostTextError;
pub(crate) use text::{HostText, HostTextAllocation};
pub(crate) use trace_scratch::HostTraceScratchPermit;

mod array;
#[cfg(test)]
mod tests;
pub use array::{
    initialize_host_array, prepare_host_array, prepare_host_array_optional, PreparedHostArray,
};
