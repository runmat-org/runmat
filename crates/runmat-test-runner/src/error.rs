use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RunnerError {
    #[error("invalid runner configuration: {0}")]
    InvalidConfiguration(String),
    #[error("requested isolation '{requested}' is unavailable; host supports {available}")]
    IsolationUnavailable {
        requested: String,
        available: String,
    },
    #[error("worker protocol error: {0}")]
    Protocol(String),
    #[error("worker backend error: {0}")]
    Backend(String),
    #[error("reporter error: {0}")]
    Reporter(String),
    #[error("artifact store error: {0}")]
    Artifact(String),
}

pub type RunnerResult<T> = Result<T, RunnerError>;
