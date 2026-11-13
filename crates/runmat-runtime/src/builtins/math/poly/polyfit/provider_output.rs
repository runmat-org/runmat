//! Validation failures from a provider that reported a successful fit.

use std::fmt;

use super::{polyfit_error_with, POLYFIT_ERROR_PROVIDER_FAILED};

#[derive(Debug)]
pub(super) enum ProviderOutputFault {
    CoefficientCount { expected: usize, observed: usize },
    NonFiniteCoefficient { index: usize },
    StatisticsPresence,
    ResidualPresence,
    RLength { expected: usize, observed: usize },
    NonFiniteR { index: usize },
    NonFiniteResidual,
    DegreesOfFreedom { expected: f64, observed: f64 },
    InvalidNormalization,
}

impl fmt::Display for ProviderOutputFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoefficientCount { expected, observed } => write!(
                f,
                "coefficient count {observed} does not match requested degree ({expected})"
            ),
            Self::NonFiniteCoefficient { index } => {
                write!(f, "coefficient {} is nonfinite", index + 1)
            }
            Self::StatisticsPresence => f.write_str("R presence disagrees with requested outputs"),
            Self::ResidualPresence => {
                f.write_str("residual norm presence disagrees with requested outputs")
            }
            Self::RLength { expected, observed } => {
                write!(f, "R length {observed} does not match {expected}")
            }
            Self::NonFiniteR { index } => write!(f, "R element {} is nonfinite", index + 1),
            Self::NonFiniteResidual => f.write_str("residual norm is nonfinite or negative"),
            Self::DegreesOfFreedom { expected, observed } => write!(
                f,
                "degrees of freedom {observed} do not match sample geometry ({expected})"
            ),
            Self::InvalidNormalization => {
                f.write_str("normalization parameters are invalid for the requested basis")
            }
        }
    }
}

impl std::error::Error for ProviderOutputFault {}

pub(super) fn failure(fault: ProviderOutputFault) -> crate::RuntimeError {
    let mut error = polyfit_error_with(
        format!("polyfit: provider returned {fault}"),
        &POLYFIT_ERROR_PROVIDER_FAILED,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error.source = Some(Box::new(fault));
    error
}
