//! Typed failures shared by every solver family.

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SolverError {
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    #[error("invalid sparse structure: {0}")]
    InvalidStructure(String),
    #[error("non-finite value at entry {entry}")]
    NonFinite { entry: usize },
    #[error("matrix is not positive definite (pivot {pivot})")]
    NotPositiveDefinite { pivot: usize },
    #[error("matrix is numerically singular (pivot {pivot})")]
    Singular { pivot: usize },
    #[error("row {row} has no stored diagonal entry")]
    MissingDiagonal { row: usize },
    #[error("sparsity pattern differs from the analyzed pattern")]
    PatternMismatch,
    #[error("out of memory: requested {bytes} bytes")]
    OutOfMemory { bytes: u64 },
    #[error("memory estimate {required} bytes exceeds the admitted budget {budget} bytes")]
    OverBudget { required: u64, budget: u64 },
    #[error("cancelled")]
    Cancelled,
    #[error(transparent)]
    Usage(#[from] runmat_simulation_usage::LedgerError),
    #[error("index overflow")]
    IndexOverflow,
    #[error("operator application failed: {0}")]
    Operator(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("numerical breakdown: {0}")]
    Breakdown(String),
}

pub type SolverResult<T> = Result<T, SolverError>;

impl From<faer::sparse::FaerError> for SolverError {
    fn from(value: faer::sparse::FaerError) -> Self {
        match value {
            faer::sparse::FaerError::OutOfMemory => SolverError::OutOfMemory { bytes: 0 },
            faer::sparse::FaerError::IndexOverflow => SolverError::IndexOverflow,
            other => SolverError::Unsupported(format!("{other:?}")),
        }
    }
}
