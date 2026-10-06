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
    #[error("the base workspace revision domain is exhausted")]
    RevisionExhausted,
    #[error("a workspace delta contains {actual} changes, but the maximum is {maximum}")]
    DeltaTooLarge { actual: usize, maximum: usize },
    #[error("the base workspace contains {actual} entries, but the maximum is {maximum}")]
    TooManyEntries { actual: usize, maximum: usize },
    #[error("a workspace value is not storable: {0}")]
    TransientValue(#[from] TransientValueError),
    #[error("could not retain a base workspace value: {0}")]
    Rooting(String),
    #[error("publish pending workspace changes before reconciling the projection")]
    PendingChanges,
}

impl From<runmat_gc::GcError> for BaseWorkspaceError {
    fn from(error: runmat_gc::GcError) -> Self {
        Self::Rooting(error.to_string())
    }
}
