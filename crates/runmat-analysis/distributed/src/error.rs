//! Typed failures of partitioning, assembly, factorization, communication and iteration.

#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("invalid distributed problem: {0}")]
    InvalidProblem(String),
    #[error("element kernel failed: {0}")]
    Kernel(String),
    #[error("subdomain factorization failed: {0}")]
    Factorization(String),
    #[error("rank communication failed: {0}")]
    Communication(String),
    #[error("distributed iteration did not converge: {0}")]
    NotConverged(String),
}

pub type DistributedResult<T> = Result<T, DistributedError>;

impl From<std::io::Error> for DistributedError {
    fn from(error: std::io::Error) -> Self {
        Self::Communication(error.to_string())
    }
}
