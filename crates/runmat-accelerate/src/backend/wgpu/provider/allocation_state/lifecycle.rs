use super::BufferEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    Registration,
    Cache,
    Provenance,
    Resource,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Disposition {
    Retained,
    Released,
    Uncertain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Failure {
    MapPoisoned,
    Cache,
    Provenance,
    /// Cleanup admission refused before the destroy closure was entered.
    DestroyAdmission,
    Unwind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Registering(Stage),
    Live,
    Retiring(Stage),
    Failed {
        stage: Stage,
        failure: Failure,
        disposition: Disposition,
    },
}

impl Phase {
    pub(super) fn retirement_start(self, explicit_reset: bool) -> Option<Stage> {
        match self {
            Self::Live => Some(Stage::Cache),
            Self::Failed {
                stage: Stage::Resource,
                failure: Failure::DestroyAdmission,
                disposition: Disposition::Retained,
            } if explicit_reset => Some(Stage::Resource),
            _ => None,
        }
    }
}

pub(super) struct Record {
    /// Immutable even after the physical payload is detached for retirement.
    pub(super) scope: Option<runmat_accelerate_api::DeviceAllocationScope>,
    pub(super) registration_slot: Option<runmat_execution::resource::ProviderRegistrationSlot>,
    pub(super) phase: Phase,
    pub(super) entry: Option<BufferEntry>,
    /// Includes a temporarily detached retirement ticket, until released.
    pub(super) retained_bytes: u64,
    pub(super) physical: runmat_accelerate_api::allocation::GpuPhysicalDescriptor,
    pub(super) lease: Option<runmat_accelerate_api::allocation::WeakAllocationLease>,
    /// Issued operation identity follows this canonical physical record until
    /// its actual retirement, even when the current task context changes.
    pub(super) resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
    /// Set only by a logical-producing upload/kernel at physical registration.
    pub(super) logical_producer: bool,
    /// Request-derived record and shape metadata is charged until actual
    /// physical retirement (or a proven released registration failure).
    pub(super) host_metadata_claim:
        Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AllocationStateError {
    Poisoned,
    IdentifierExhausted,
    Capacity,
    NotLive,
    Missing,
    DeviceLost,
    Cleanup { stage: Stage, failure: Failure },
}

impl std::fmt::Display for AllocationStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WGPU allocation state: {self:?}")
    }
}

impl std::error::Error for AllocationStateError {}
