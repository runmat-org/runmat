use runmat_analysis_core::AnalysisField;
use runmat_analysis_fea::diagnostics::FeaDiagnostic;
use runmat_analysis_fea::{ComputeBackend, FeaRunResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisValidateResult {
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisCreateModelIntentSpec {
    pub model_id: String,
    pub profile: AnalysisCreateModelProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisCreateModelProfile {
    LinearStaticStructural,
    ModalStructural,
    TransientStructural,
    NonlinearStructural,
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
#[serde(rename_all = "snake_case")]
pub enum QualityPolicy {
    Strict,
    Balanced,
    Exploratory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityReasonCode {
    MaterialAssignmentConflict,
    SolverNotConverged,
    SolverBackendFallback,
    FieldPromotionFallback,
    ModalPlaceholder,
    ModalResidualExceeded,
    TransientPlaceholder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityReason {
    pub code: QualityReasonCode,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub preconditioner_mode: PreconditionerMode,
    pub quality_policy: QualityPolicy,
}

impl Default for AnalysisRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    pub backend: ComputeBackend,
    pub solver_backend: String,
    pub solver_device_apply_k_ratio: f64,
    pub solver_host_sync_count: u32,
    pub precision_mode: String,
    pub deterministic_mode: bool,
    pub solver_method: String,
    pub preconditioner: String,
    pub quality_policy: String,
    pub fallback_events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRunResult {
    pub run_id: String,
    pub run: FeaRunResult,
    pub modal_results: Option<ModalResultsData>,
    pub transient_results: Option<TransientResultsData>,
    pub model_validity: QualityGate,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub quality_reasons: Vec<QualityReason>,
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
    pub include_modal_results: bool,
    pub mode_indices: Vec<usize>,
}

impl Default for AnalysisResultsQuery {
    fn default() -> Self {
        Self {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsSummary {
    pub field_count: usize,
    pub total_elements: usize,
    pub mode_count: usize,
    pub available_mode_indices: Vec<usize>,
    pub min_frequency_hz: Option<f64>,
    pub max_frequency_hz: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsData {
    pub fields: Vec<AnalysisField>,
    pub modal_results: Option<ModalResultsData>,
    pub transient_results: Option<TransientResultsData>,
    pub diagnostics: Option<Vec<FeaDiagnostic>>,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub summary: AnalysisResultsSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalResultsData {
    pub modal_payload_version: String,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub mode_units: ModalFrequencyUnits,
    pub frequency_basis: ModalFrequencyBasis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientResultsData {
    pub transient_payload_version: String,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub integration_method: TransientIntegrationMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransientIntegrationMethod {
    ImplicitEuler,
    PlaceholderLinearStatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalFrequencyUnits {
    Hz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalFrequencyBasis {
    PlaceholderLinearStatic,
    NativeEigenSolve,
}

pub(crate) fn format_precision_mode(mode: PrecisionMode) -> String {
    match mode {
        PrecisionMode::Fp32 => "fp32".to_string(),
        PrecisionMode::Fp64 => "fp64".to_string(),
        PrecisionMode::Mixed => "mixed".to_string(),
    }
}

pub(crate) fn format_quality_policy(mode: QualityPolicy) -> String {
    match mode {
        QualityPolicy::Strict => "strict".to_string(),
        QualityPolicy::Balanced => "balanced".to_string(),
        QualityPolicy::Exploratory => "exploratory".to_string(),
    }
}
