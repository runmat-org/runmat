use runmat_analysis_core::AnalysisField;
use runmat_analysis_fea::diagnostics::FeaDiagnostic;
use runmat_analysis_fea::{ComputeBackend, FeaRunResult};
use runmat_meshing_core::RegionMeshMapping;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisValidateResult {
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisCreateModelIntentSpec {
    pub model_id: String,
    pub profile: AnalysisCreateModelProfile,
    #[serde(default)]
    pub prep_context: Option<AnalysisCreateModelPrepContext>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisCreateModelPrepContext {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub region_mappings: Vec<RegionMeshMapping>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRunPrepContext {
    pub prepared_mesh_count: usize,
    pub prepared_node_count: usize,
    pub prepared_element_count: usize,
    pub mapped_region_count: usize,
    pub min_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub inverted_element_count: usize,
    pub mapped_load_count: usize,
    pub mapped_bc_count: usize,
    pub layout_seed: u64,
    pub topology_dof_multiplier: f64,
    pub topology_bandwidth_proxy: u32,
    pub mapped_region_participation_ratio: f64,
    pub topology_surface_patch_ratio: f64,
    pub topology_volume_core_ratio: f64,
    pub topology_mixed_family_ratio: f64,
    pub topology_region_span_mean: f64,
    pub topology_region_block_count: usize,
    pub topology_region_mesh_mean: f64,
    pub topology_region_mesh_variance: f64,
    pub topology_triangle_family_ratio: f64,
    pub topology_quad_family_ratio: f64,
    pub topology_tet_family_ratio: f64,
    pub topology_hex_family_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisCreateModelProfile {
    LinearStaticStructural,
    ThermoMechanicalCoupled,
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
pub enum PrepCalibrationProfile {
    Auto,
    Fast,
    Balanced,
    Conservative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoRegionTemperatureDelta {
    pub region_id: String,
    pub temperature_delta_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoTimeProfilePoint {
    pub normalized_time: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalCouplingOptions {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    #[serde(default)]
    pub region_temperature_deltas: Vec<ThermoRegionTemperatureDelta>,
    #[serde(default)]
    pub time_profile: Vec<ThermoTimeProfilePoint>,
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
    ModalOrthogonalityExceeded,
    ModalSeparationLow,
    TransientPlaceholder,
    TransientResidualExceeded,
    TransientStabilityExceeded,
    TransientStepFailure,
    ThermoMechanicalTransientStress,
    ThermoMechanicalConstitutiveSpreadHigh,
    ThermoMechanicalAssignmentHeterogeneityHigh,
    ThermoMechanicalGradientInstability,
    NonlinearResidualExceeded,
    NonlinearIncrementFailure,
    ThermoMechanicalNonlinearStress,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityReason {
    pub code: QualityReasonCode,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub preconditioner_mode: PreconditionerMode,
    pub quality_policy: QualityPolicy,
    #[serde(default)]
    pub prep_context: Option<AnalysisRunPrepContext>,
    #[serde(default)]
    pub prep_artifact_id: Option<String>,
    #[serde(default)]
    pub prep_calibration_profile: Option<PrepCalibrationProfile>,
    #[serde(default)]
    pub thermo_mechanical_coupling: Option<ThermoMechanicalCouplingOptions>,
}

impl Default for AnalysisRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisTransientRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub time_step_s: f64,
    pub min_time_step_s: f64,
    pub max_time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_target: f64,
    pub adaptive_time_step: bool,
    pub max_step_retries: usize,
    pub adapt_min_scale: f64,
    pub adapt_max_scale: f64,
    pub adapt_growth_exponent: f64,
    pub adapt_retry_growth_cap: f64,
    pub adapt_nonconverged_shrink: f64,
    pub dt_bucket_rel_tolerance: f64,
    #[serde(default)]
    pub prep_context: Option<AnalysisRunPrepContext>,
    #[serde(default)]
    pub prep_artifact_id: Option<String>,
    #[serde(default)]
    pub prep_calibration_profile: Option<PrepCalibrationProfile>,
    #[serde(default)]
    pub thermo_mechanical_coupling: Option<ThermoMechanicalCouplingOptions>,
}

impl Default for AnalysisTransientRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            min_time_step_s: 1.0e-6,
            max_time_step_s: 2.0e-2,
            step_count: 10,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: true,
            max_step_retries: 4,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }
}

impl AnalysisTransientRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            time_step_s: 5.0e-3,
            min_time_step_s: 5.0e-4,
            max_time_step_s: 2.0e-2,
            step_count: 6,
            max_linear_iters: 64,
            tolerance: 1.0e-6,
            residual_target: 1.0e-4,
            adaptive_time_step: true,
            max_step_retries: 2,
            adapt_min_scale: 0.75,
            adapt_max_scale: 1.3,
            adapt_growth_exponent: 0.3,
            adapt_retry_growth_cap: 1.02,
            adapt_nonconverged_shrink: 0.7,
            dt_bucket_rel_tolerance: 0.02,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn production_recommended() -> Self {
        Self {
            quality_policy: QualityPolicy::Balanced,
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            dt_bucket_rel_tolerance: 0.01,
            ..Self::balanced()
        }
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            time_step_s: 5.0e-4,
            min_time_step_s: 5.0e-6,
            max_time_step_s: 2.0e-3,
            step_count: 24,
            max_linear_iters: 256,
            tolerance: 1.0e-10,
            residual_target: 1.0e-7,
            adaptive_time_step: true,
            max_step_retries: 8,
            adapt_min_scale: 0.85,
            adapt_max_scale: 1.2,
            adapt_growth_exponent: 0.45,
            adapt_retry_growth_cap: 1.03,
            adapt_nonconverged_shrink: 0.8,
            dt_bucket_rel_tolerance: 0.005,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModalRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub mode_count: usize,
    pub residual_warn_threshold: f64,
    #[serde(default)]
    pub prep_context: Option<AnalysisRunPrepContext>,
    #[serde(default)]
    pub prep_artifact_id: Option<String>,
    #[serde(default)]
    pub prep_calibration_profile: Option<PrepCalibrationProfile>,
    #[serde(default)]
    pub thermo_mechanical_coupling: Option<ThermoMechanicalCouplingOptions>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisNonlinearRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub increment_count: usize,
    pub max_newton_iters: usize,
    pub tolerance: f64,
    pub residual_convergence_factor: f64,
    pub increment_norm_tolerance: f64,
    pub line_search: bool,
    pub max_line_search_backtracks: usize,
    pub line_search_reduction: f64,
    pub tangent_refresh_interval: usize,
    #[serde(default)]
    pub prep_context: Option<AnalysisRunPrepContext>,
    #[serde(default)]
    pub prep_artifact_id: Option<String>,
    #[serde(default)]
    pub prep_calibration_profile: Option<PrepCalibrationProfile>,
    #[serde(default)]
    pub thermo_mechanical_coupling: Option<ThermoMechanicalCouplingOptions>,
}

impl Default for AnalysisNonlinearRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            increment_count: 12,
            max_newton_iters: 24,
            tolerance: 1.0e-6,
            residual_convergence_factor: 5.0,
            increment_norm_tolerance: 1.0e-7,
            line_search: true,
            max_line_search_backtracks: 6,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 2,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }
}

impl AnalysisNonlinearRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            increment_count: 8,
            max_newton_iters: 16,
            tolerance: 5.0e-6,
            residual_convergence_factor: 8.0,
            increment_norm_tolerance: 5.0e-7,
            line_search: false,
            max_line_search_backtracks: 0,
            line_search_reduction: 0.6,
            tangent_refresh_interval: 4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            increment_count: 24,
            max_newton_iters: 40,
            tolerance: 1.0e-7,
            residual_convergence_factor: 3.0,
            increment_norm_tolerance: 5.0e-8,
            line_search: true,
            max_line_search_backtracks: 10,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 1,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }

    pub fn production_recommended() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            increment_count: 24,
            max_newton_iters: 28,
            tolerance: 1.0e-6,
            residual_convergence_factor: 4.0,
            increment_norm_tolerance: 8.0e-8,
            line_search: true,
            max_line_search_backtracks: 8,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 2,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }
}

impl Default for AnalysisModalRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 3,
            residual_warn_threshold: 1.0e-3,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }
}

impl AnalysisModalRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            mode_count: 2,
            residual_warn_threshold: 5.0e-3,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            mode_count: 8,
            residual_warn_threshold: 5.0e-4,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
            thermo_mechanical_coupling: None,
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
    pub nonlinear_results: Option<NonlinearResultsData>,
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
    pub diagnostic_codes: Vec<String>,
    pub include_modal_results: bool,
    pub mode_indices: Vec<usize>,
    pub include_transient_results: bool,
    pub transient_snapshot_indices: Vec<usize>,
    pub include_nonlinear_results: bool,
}

impl Default for AnalysisResultsQuery {
    fn default() -> Self {
        Self {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
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
    pub max_modal_residual_norm: Option<f64>,
    pub first_mode_converged: Option<bool>,
    pub snapshot_count: usize,
    pub time_start_s: Option<f64>,
    pub time_end_s: Option<f64>,
    pub max_transient_residual_norm: Option<f64>,
    pub final_step_converged: Option<bool>,
    pub increment_count: usize,
    pub failed_increment_count: Option<usize>,
    pub max_nonlinear_residual_norm: Option<f64>,
    pub max_nonlinear_increment_norm: Option<f64>,
    pub max_nonlinear_iteration_count: Option<usize>,
    pub final_increment_converged: Option<bool>,
    pub nonlinear_line_search_backtracks: Option<usize>,
    pub nonlinear_max_backtracks_per_increment: Option<usize>,
    pub nonlinear_tangent_rebuild_count: Option<usize>,
    pub nonlinear_iteration_spike_count: Option<usize>,
    pub nonlinear_convergence_stall_count: Option<usize>,
    pub nonlinear_backtrack_burst_count: Option<usize>,
    pub prep_calibration_profile: Option<String>,
    pub prep_calibration_fingerprint: Option<u64>,
    pub prep_acceptance_score: Option<f64>,
    pub prep_acceptance_passed: Option<bool>,
    pub prep_acceptance_fingerprint: Option<u64>,
    pub thermo_coupling_enabled: Option<bool>,
    pub thermo_coupling_fingerprint: Option<u64>,
    pub thermo_constitutive_temperature_factor: Option<f64>,
    pub thermo_effective_modulus_scale: Option<f64>,
    pub thermo_constitutive_material_spread_ratio: Option<f64>,
    pub thermo_assignment_heterogeneity_index: Option<f64>,
    pub thermo_transient_severity: Option<f64>,
    pub thermo_nonlinear_severity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsData {
    pub fields: Vec<AnalysisField>,
    pub modal_results: Option<ModalResultsData>,
    pub transient_results: Option<TransientResultsData>,
    pub nonlinear_results: Option<NonlinearResultsData>,
    pub diagnostics: Option<Vec<FeaDiagnostic>>,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub summary: AnalysisResultsSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResultsCompareQuery {
    pub baseline_run_id: String,
    pub candidate_run_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsCompareData {
    pub baseline_run_id: String,
    pub candidate_run_id: String,
    pub publishable_changed: bool,
    pub run_status_changed: bool,
    pub quality_reason_count_delta: i64,
    pub failed_increment_delta: Option<i64>,
    pub max_iteration_delta: Option<i64>,
    pub nonlinear_spike_count_delta: Option<i64>,
    pub nonlinear_stall_count_delta: Option<i64>,
    pub solve_ms_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisTrendsQuery {
    pub window_size: usize,
}

impl Default for AnalysisTrendsQuery {
    fn default() -> Self {
        Self { window_size: 16 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisRunKind {
    LinearStatic,
    Modal,
    Transient,
    Nonlinear,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisTrendKindSummary {
    pub run_kind: AnalysisRunKind,
    pub sample_count: usize,
    pub median_solve_ms: Option<f64>,
    pub p95_solve_ms: Option<f64>,
    pub publishable_rate: f64,
    pub failed_increment_rate: Option<f64>,
    pub mean_spike_count: Option<f64>,
    pub mean_stall_count: Option<f64>,
    pub prep_acceptance_rate: Option<f64>,
    pub prep_calibration_fast_rate: Option<f64>,
    pub prep_calibration_balanced_rate: Option<f64>,
    pub prep_calibration_conservative_rate: Option<f64>,
    pub thermo_coupling_enabled_rate: Option<f64>,
    pub thermo_transient_warn_rate: Option<f64>,
    pub thermo_nonlinear_warn_rate: Option<f64>,
    pub thermo_spread_breach_rate: Option<f64>,
    pub thermo_heterogeneity_breach_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisTrendsData {
    pub window_size: usize,
    pub summaries: Vec<AnalysisTrendKindSummary>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearResultsData {
    pub nonlinear_payload_version: String,
    pub load_factors: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    #[serde(default)]
    pub increment_norms: Vec<f64>,
    #[serde(default)]
    pub iteration_counts: Vec<usize>,
    #[serde(default)]
    pub failed_increments: usize,
    #[serde(default)]
    pub line_search_backtracks: usize,
    #[serde(default)]
    pub max_line_search_backtracks_per_increment: usize,
    #[serde(default)]
    pub tangent_rebuild_count: usize,
    #[serde(default)]
    pub iteration_spike_count: usize,
    #[serde(default)]
    pub convergence_stall_count: usize,
    #[serde(default)]
    pub backtrack_burst_count: usize,
    pub method: NonlinearMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransientIntegrationMethod {
    ImplicitEuler,
    PlaceholderLinearStatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonlinearMethod {
    IncrementalNewtonRaphson,
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
