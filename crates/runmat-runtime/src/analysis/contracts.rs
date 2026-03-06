use runmat_analysis_core::AnalysisField;
use runmat_analysis_fea::diagnostics::FeaDiagnostic;
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
#[serde(rename_all = "snake_case")]
pub enum PreconditionerMode {
    Auto,
    Jacobi,
    Amg,
    Ilu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub preconditioner_mode: PreconditionerMode,
}

impl Default for AnalysisRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
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
    pub solver_backend: String,
    pub precision_mode: String,
    pub deterministic_mode: bool,
    pub solver_method: String,
    pub preconditioner: String,
    pub fallback_events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRunResult {
    pub run_id: String,
    pub run: FeaRunResult,
    pub model_validity: QualityGate,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub provenance: RunProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisArtifactRecord {
    pub run_id: String,
    pub created_at: String,
    pub op_version: String,
    pub field_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResultsQuery {
    pub include_fields: Vec<String>,
    pub include_diagnostics: bool,
}

impl Default for AnalysisResultsQuery {
    fn default() -> Self {
        Self {
            include_fields: Vec::new(),
            include_diagnostics: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResultsSummary {
    pub field_count: usize,
    pub total_elements: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsData {
    pub fields: Vec<AnalysisField>,
    pub diagnostics: Option<Vec<FeaDiagnostic>>,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub provenance: RunProvenance,
    pub summary: AnalysisResultsSummary,
}

pub(crate) fn format_precision_mode(mode: PrecisionMode) -> String {
    match mode {
        PrecisionMode::Fp32 => "fp32".to_string(),
        PrecisionMode::Fp64 => "fp64".to_string(),
        PrecisionMode::Mixed => "mixed".to_string(),
    }
}
