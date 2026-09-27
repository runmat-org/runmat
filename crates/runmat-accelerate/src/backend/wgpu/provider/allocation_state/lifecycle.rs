use super::BufferEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    RegistrationResidency,
    RegistrationAnnotations,
    Cache,
    Provenance,
    Residency,
    Annotations,
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
    Callback,
    AnnotationLock,
    Unwind,
    DeviceLost,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidencyCleanup {
    Pending,
    Complete,
    Incomplete,
}

impl Phase {
    pub(super) fn residency_cleanup(self) -> ResidencyCleanup {
        match self {
            Self::Live
            | Self::Retiring(Stage::Cache | Stage::Provenance)
            | Self::Failed {
                stage: Stage::Cache | Stage::Provenance,
                ..
            } => ResidencyCleanup::Pending,
            Self::Registering(Stage::RegistrationAnnotations)
            | Self::Retiring(Stage::Annotations | Stage::Resource | Stage::Complete)
            | Self::Failed {
                stage:
                    Stage::RegistrationAnnotations
                    | Stage::Annotations
                    | Stage::Resource
                    | Stage::Complete,
                ..
            }
            | Self::Failed {
                stage: Stage::Residency,
                failure: Failure::MapPoisoned,
                ..
            } => ResidencyCleanup::Complete,
            // Invocation was recorded before entering an external callback.
            // Missing a completion receipt is not success or retry authority.
            _ => ResidencyCleanup::Incomplete,
        }
    }
}

pub(super) struct Record {
    pub(super) phase: Phase,
    pub(super) entry: Option<BufferEntry>,
    /// Includes a temporarily detached retirement ticket, until released.
    pub(super) retained_bytes: u64,
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
