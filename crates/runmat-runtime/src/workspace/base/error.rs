use runmat_value::TransientValueError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BaseWorkspaceError {
    #[error("the base workspace is closed")]
    Closed,
    #[error("the base workspace is already borrowed by another operation")]
    BorrowConflict,
    #[error("base workspace revision {requested} is newer than current revision {current}")]
    FutureRevision { requested: u64, current: u64 },
    #[error("the workspace changed after transactional completion capture")]
    CompletionConflict,
    #[error("the base workspace revision domain is exhausted")]
    RevisionExhausted,
    #[error("a workspace value is not storable: {0}")]
    TransientValue(#[from] TransientValueError),
    #[error("could not retain a base workspace value: {0}")]
    Rooting(#[source] runmat_gc::GcError),
    #[error("could not admit base-workspace storage: {0}")]
    HostAllocation(#[source] crate::context::host_allocation::HostAllocationError),
    #[error("publish pending workspace changes before reconciling the projection")]
    PendingChanges,
    #[error("the private workspace namespace is available only through its issuing projection")]
    PrivateProjectionRequired,
}

impl From<runmat_gc::GcError> for BaseWorkspaceError {
    fn from(error: runmat_gc::GcError) -> Self {
        Self::Rooting(error)
    }
}
