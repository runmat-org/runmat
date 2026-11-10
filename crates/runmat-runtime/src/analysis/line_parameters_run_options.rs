//! Publication controls of a cross-section line-parameter solve.
use super::PrecisionMode;
use serde::{Deserialize, Serialize};

/// The section's two potential solves are direct Cholesky solves in f64; the residual target
/// bounds the largest relative residual the publication accepts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisLineParametersRunOptions {
    pub precision_mode: PrecisionMode,
    pub residual_target: f64,
}

impl Default for AnalysisLineParametersRunOptions {
    fn default() -> Self {
        Self {
            precision_mode: PrecisionMode::Fp64,
            residual_target: 1e-8,
        }
    }
}
