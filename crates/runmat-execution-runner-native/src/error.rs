#[derive(Debug, thiserror::Error)]
pub enum NativeExecutionError {
    #[error("invalid native execution configuration: {0}")]
    Configuration(String),
    #[error("native process host failed: {0}")]
    Host(#[from] runmat_process_host::ProcessHostError),
    #[error("native execution protocol failed: {0}")]
    Protocol(String),
    #[error("portable execution driver failed: {0}")]
    Driver(#[from] runmat_execution_runner::RunnerError),
}

pub type NativeExecutionResult<T> = Result<T, NativeExecutionError>;
