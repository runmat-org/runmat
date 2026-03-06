use runmat_analysis_fea::{ComputeBackend, FeaRunResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisValidateResult {
    pub valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    Fp32,
    Fp64,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
}

impl Default for AnalysisRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityGate {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Publishable,
    Degraded,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunProvenance {
    pub backend: ComputeBackend,
    pub precision_mode: String,
    pub deterministic_mode: bool,
    pub fallback_events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRunResult {
    pub run: FeaRunResult,
    pub model_validity: QualityGate,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub provenance: RunProvenance,
}

pub(crate) fn format_precision_mode(mode: PrecisionMode) -> String {
    match mode {
        PrecisionMode::Fp32 => "fp32".to_string(),
        PrecisionMode::Fp64 => "fp64".to_string(),
        PrecisionMode::Mixed => "mixed".to_string(),
    }
}
