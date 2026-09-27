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
