#[derive(Debug, thiserror::Error)]
pub enum NativeRunnerError {
    #[error("native test-runner I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("native test-runner protocol failed: {0}")]
    Protocol(String),
    #[error("native test-runner configuration is invalid: {0}")]
    Configuration(String),
}

pub type NativeRunnerResult<T> = Result<T, NativeRunnerError>;
