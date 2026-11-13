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

pub(crate) fn contract_error(error: runmat_execution::ContractError) -> ArtifactError {
    contract_error_in("execution contract", error)
}

pub(crate) fn contract_error_in(
    context: &str,
    error: runmat_execution::ContractError,
) -> ArtifactError {
    let message = format!("{context}: {error}");
    match error {
        runmat_execution::ContractError::Limit { .. } => ArtifactError::Limit(message),
        runmat_execution::ContractError::Invalid { .. }
        | runmat_execution::ContractError::UnsupportedSchema { .. }
        | runmat_execution::ContractError::MalformedProtocol(_) => ArtifactError::Invalid(message),
    }
}
