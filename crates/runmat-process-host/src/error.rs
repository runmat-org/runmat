use std::io;

#[derive(Debug, thiserror::Error)]
pub enum ProcessHostError {
    #[error("invalid process host configuration: {0}")]
    Configuration(String),
    #[error("local IPC protocol error: {0}")]
    Protocol(String),
    #[error("child process I/O failed: {0}")]
    Io(#[from] io::Error),
}

pub type ProcessHostResult<T> = Result<T, ProcessHostError>;
