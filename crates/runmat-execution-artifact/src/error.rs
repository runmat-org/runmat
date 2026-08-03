#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("invalid execution artifact: {0}")]
    Invalid(String),
    #[error("execution artifact limit exceeded: {0}")]
    Limit(String),
    #[error("execution artifact identity mismatch: {0}")]
    Identity(String),
    #[error("execution artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("execution artifact encoding failed: {0}")]
    Encoding(String),
    #[error("execution artifact encryption failed: {0}")]
    Encryption(String),
}

pub type ArtifactResult<T> = Result<T, ArtifactError>;
