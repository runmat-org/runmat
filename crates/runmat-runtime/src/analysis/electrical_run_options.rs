//! Static electrical publication controls; no harmonic frequency or sweep placeholders.
use super::{PrecisionMode, QualityPolicy};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisElectricalRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub residual_target: f64,
}

impl Default for AnalysisElectricalRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            residual_target: 1e-8,
        }
    }
}
