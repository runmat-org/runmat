//! Harmonic Maxwell publication controls; frequency remains in the original workload.
use super::PrecisionMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMaxwellRunOptions {
    pub precision_mode: PrecisionMode,
    pub residual_target: f64,
}

impl Default for AnalysisMaxwellRunOptions {
    fn default() -> Self {
        Self {
            precision_mode: PrecisionMode::Fp64,
            residual_target: 1e-8,
        }
    }
}
