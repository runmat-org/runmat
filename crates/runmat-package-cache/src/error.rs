use runmat_package::ContentDigest;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BackendError {
    #[error("cache backend failed: {0}")]
    Failure(String),
    #[error("cache backend schema is incompatible: {0}")]
    IncompatibleSchema(String),
    #[error(
        "cache quota exceeded: requested {requested_bytes} bytes with {available_bytes} available"
    )]
    QuotaExceeded {
        requested_bytes: u64,
        available_bytes: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CacheError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("invalid cache state: {0}")]
    InvalidState(String),
    #[error("invalid cache object: {0}")]
    InvalidObject(String),
    #[error("cache object {0} failed digest verification")]
    DigestMismatch(ContentDigest),
    #[error("cache transaction conflicted after {attempts} attempts")]
    ConflictExhausted { attempts: usize },
    #[error("cache object {0} is missing")]
    Miss(ContentDigest),
    #[error("cache object {digest} is recorded as corrupt: {reason}")]
    Corrupt {
        digest: ContentDigest,
        reason: String,
    },
    #[error("lease operation failed: {0}")]
    Lease(String),
    #[error("materialization transition failed: {0}")]
    Materialization(String),
}
