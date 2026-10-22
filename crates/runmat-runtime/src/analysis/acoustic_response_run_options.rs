//! Acoustic cavity-mode and pressure-transient publication controls.
use super::PrecisionMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisAcousticResponseRunOptions {
    pub precision_mode: PrecisionMode,
    /// The accepted eigenpair or per-step linear residual bound of the original solve.
    pub residual_target: f64,
}

impl Default for AnalysisAcousticResponseRunOptions {
    fn default() -> Self {
        Self {
            precision_mode: PrecisionMode::Fp64,
            residual_target: 1e-8,
        }
    }
}
