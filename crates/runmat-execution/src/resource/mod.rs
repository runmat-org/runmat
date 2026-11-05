//! Portable resource requests, inventories, assignments, and metering records.
//!
//! Cluster scheduling owns worker-local resource assignment. Accelerator
//! providers continue to own kernel placement within an assigned device.

mod accelerator;
mod accounting;
mod capability;
mod exact_shared;
mod interval;
mod matching;
mod request;
mod requested_usage;
mod registration_slots;
mod reservation;
mod usage;
pub use registration_slots::{ProviderRegistrationSlot, ProviderRegistrationSlotPool};

/// Opaque lifetime claim for host metadata retained by an executable value.
/// The allocating resource owner, not this neutral contract, decides its size
/// and when the physical reservation is refunded.
pub trait RetainedHostAllocationClaim: std::fmt::Debug + Send + Sync {}
impl<T: std::fmt::Debug + Send + Sync + ?Sized> RetainedHostAllocationClaim for T {}

pub use exact_shared::ExactShared;

/// Move-only funding for one owned view record. A handle clone shares that
/// record; constructing another view consumes another grant from the issuer.
/// The implementing resource owner retains the actual reservation inside it.
pub trait RetainedHostViewGrant: std::fmt::Debug + Send + Sync {
    fn view_bytes(&self) -> u64;
    fn issuer(&self) -> std::sync::Arc<dyn RetainedHostViewIssuer>;
    /// Keep the existing reservation alive while a weak view still retains
    /// the ref-counted backing. This does not authorize another view.
    fn backing_claim(&self) -> std::sync::Arc<dyn RetainedHostAllocationClaim>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedHostViewError {
    Overflow,
    Revoked,
    ExceedsCapacity,
}

/// A live host resource owner can admit a new metadata record derived from
/// an already-published view. It does not copy or authorize device payloads.
pub trait RetainedHostViewIssuer: std::fmt::Debug + Send + Sync {
    fn reserve_view(
        self: std::sync::Arc<Self>,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, RetainedHostViewError>;
}

/// Request-bound, precharged host metadata admission for a provider-produced
/// physical allocation. This authority is passed through the selected
/// execution recipe; provider registration cannot obtain it from ambient
/// state or infer it from a device buffer's size.
pub trait ProviderHostMetadataAdmission: std::fmt::Debug {
    /// Legacy admissions consume a one-shot slot. Reusable admissions return
    /// a move-only token which follows the physical record until removal.
    fn claim_registration_slot(
        &self,
        provider_owner: usize,
    ) -> Result<Option<ProviderRegistrationSlot>, ProviderHostMetadataError> {
        self.commit_registration_slot(provider_owner)?;
        Ok(None)
    }

    fn claim_view(
        &self,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError>;
    fn claim_record(
        &self,
        record_bytes: u64,
    ) -> Result<std::sync::Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError>;

    /// Consume one slot from a provider-owned table reservation made before
    /// the selected route's first physical effect. The caller holds the
    /// provider registry lock through the ensuing insertion, so no other
    /// route can take that slot. A generic host receipt cannot mint one.
    fn commit_registration_slot(
        &self,
        _provider_owner: usize,
    ) -> Result<(), ProviderHostMetadataError> {
        Err(ProviderHostMetadataError::OwnerFailure)
    }
}

/// Physical provider-owned host metadata for one newly registered output.
/// These are payload bytes; the managed allocation owner adds the exact
/// claim/issuer bookkeeping before reserving finite headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderHostRegistrationMetadata {
    pub record_payload_bytes: u64,
    pub view_payload_bytes: u64,
    pub lease_payload_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderHostMetadataError {
    Overflow,
    Cancelled,
    StaleAuthority,
    ExceedsCapacity,
    OwnerFailure,
}

impl std::fmt::Display for ProviderHostMetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "provider host metadata admission: {self:?}")
    }
}
impl std::error::Error for ProviderHostMetadataError {}

/// Encoded Graph V10 Program transport budget. The selected archive aggregate
/// is separately limited to 4 GiB decoded; JSON byte arrays, duplicated
/// selected-unit bytes, and graph metadata require a larger wire allowance.
/// An actual host's scratch/memory policy may admit less.
pub const MAX_GRAPH_PROGRAM_WIRE_BYTES: u64 = 24 * 1024 * 1024 * 1024;
pub const GRAPH_PROGRAM_CHUNK_PLAINTEXT_BYTES: u64 = 4 * 1024 * 1024;

/// Parent-driver capacity reserved for one retained terminal control envelope.
/// This portable component includes allocation headers and claim metadata.
/// Native drivers must refuse a backing layout that exceeds this capacity.
/// Unused capacity stays reserved; it is not numerical work or memory usage.
pub const REMOTE_DRIVER_COMPLETION_CONTROL_BYTES: u64 = 512;

/// Request the child limit plus its separate parent control component before
/// allocation authorization. The child's signed numerical limit stays intact.
pub fn remote_driver_memory_bytes(child_memory_bytes: u64) -> Result<u64, crate::ContractError> {
    child_memory_bytes
        .checked_add(REMOTE_DRIVER_COMPLETION_CONTROL_BYTES)
        .ok_or_else(|| crate::ContractError::invalid("driver memory", "parent capacity overflows"))
}

pub const MAX_GRAPH_PROGRAM_CHUNKS: usize =
    (MAX_GRAPH_PROGRAM_WIRE_BYTES / GRAPH_PROGRAM_CHUNK_PLAINTEXT_BYTES) as usize;

pub use accelerator::{
    accelerator_requirements_for_capabilities, merge_accelerator_requirements,
    AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
    AcceleratorDeviceLease, AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
    AcceleratorProviderVersion, AcceleratorRequest,
};
pub use accounting::{ResourceAccounting, ResourceAccountingSummary};
pub use capability::Capability;
pub use matching::{
    accelerator_devices_exactly_satisfy, accelerator_request_is_within,
    accelerator_requests_satisfy_requirements, select_accelerator_devices,
};
pub use request::{ResourceAssignment, ResourceInventory, ResourceRequest};
pub use requested_usage::{RequestedAcceleratorUsage, RequestedResourceUsage};
pub use usage::{AcceleratorUsage, ResourceUsage};
