use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExecutionServiceError {
    #[error("execution handle belongs to a different scope")]
    ForeignScope,
    #[error("execution handle is unknown or stale")]
    UnknownHandle,
    #[error("execution was cancelled")]
    Cancelled,
    #[error("execution failed: {0}")]
    Failed(String),
    #[error("requested output count exceeds the execution contract")]
    InvalidOutputContract,
}
