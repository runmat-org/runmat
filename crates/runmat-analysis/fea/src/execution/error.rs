use runmat_canonical_codec::CanonicalCodecError;

pub type AnalysisExecutionResult<T> = Result<T, AnalysisExecutionError>;

#[derive(Debug, thiserror::Error)]
pub enum AnalysisExecutionError {
    #[error("invalid analysis execution contract: {0}")]
    Invalid(String),
    #[error("analysis execution identity mismatch: {0}")]
    Identity(&'static str),
    #[error("analysis execution object limit exceeded: {0}")]
    Limit(&'static str),
    #[error("analysis execution codec failure: {0}")]
    Codec(#[from] CanonicalCodecError),
    #[error("analysis execution numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
    #[error("analysis execution cancelled")]
    Cancelled,
}
