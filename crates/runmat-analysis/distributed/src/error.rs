//! Typed failures of partitioning, assembly, factorization, communication and iteration.

#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("invalid distributed problem: {0}")]
    InvalidProblem(String),
    #[error("element kernel failed: {0}")]
    Kernel(String),
    #[error("element kernel operation accounting failed: {0}")]
    Usage(#[from] runmat_simulation_usage::LedgerError),
    #[error("subdomain factorization failed: {0}")]
    Factorization(String),
    #[error("rank communication failed: {0}")]
    Communication(String),
    #[error("distributed iteration did not converge: {0}")]
    NotConverged(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ElementKernelError {
    #[error("{0}")]
    Detail(String),
    #[error("operation accounting failed: {0}")]
    Usage(#[from] runmat_simulation_usage::LedgerError),
}

impl From<String> for ElementKernelError {
    fn from(detail: String) -> Self {
        Self::Detail(detail)
    }
}

impl From<&str> for ElementKernelError {
    fn from(detail: &str) -> Self {
        Self::Detail(detail.to_owned())
    }
}

impl From<ElementKernelError> for DistributedError {
    fn from(error: ElementKernelError) -> Self {
        match error {
            ElementKernelError::Detail(detail) => Self::Kernel(detail),
            ElementKernelError::Usage(cause) => Self::Usage(cause),
        }
    }
}

pub type DistributedResult<T> = Result<T, DistributedError>;

impl From<std::io::Error> for DistributedError {
    fn from(error: std::io::Error) -> Self {
        Self::Communication(error.to_string())
    }
}
