//! Typed data-domain rejection from covariance's existing weight preflight.
//! This is not an unavailable operation or a transport/allocation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceWeightDomainError {
    NonFinite,
    Negative,
}

impl std::fmt::Display for CovarianceWeightDomainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let detail = match self {
            Self::NonFinite => "nonfinite weight",
            Self::Negative => "negative weight",
        };
        write!(
            formatter,
            "covariance: weights must be non-negative finite values ({detail})"
        )
    }
}
impl std::error::Error for CovarianceWeightDomainError {}
