use runmat_analysis_core::AnalysisField;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::solve::{
    backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaRunResult {
    pub backend: ComputeBackend,
    pub solver_backend: String,
    pub solver_device_apply_k_ratio: f64,
    pub solver_method: String,
    pub preconditioner: String,
    pub solver_host_sync_count: u32,
    pub diagnostics: Vec<crate::diagnostics::FeaDiagnostic>,
    pub displacement_field: AnalysisField,
    pub von_mises_field: AnalysisField,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeaPrepContext {
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
    pub calibration_profile_override: Option<FeaPrepCalibrationProfile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoRegionTemperatureDelta {
    pub region_id: String,
    pub temperature_delta_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoTimeProfilePoint {
    pub normalized_time: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaThermoFieldInterpolationMode {
    Linear,
    Step,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoFieldSource {
    pub source_id: String,
    pub revision: u32,
    pub interpolation_mode: Option<FeaThermoFieldInterpolationMode>,
    pub expected_region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoMechanicalContext {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    pub field_source: Option<FeaThermoFieldSource>,
    pub region_temperature_deltas: Vec<FeaThermoRegionTemperatureDelta>,
    pub time_profile: Vec<FeaThermoTimeProfilePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectroRegionConductivityScale {
    pub region_id: String,
    pub conductivity_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectroTimeProfilePoint {
    pub normalized_time: f64,
    pub current_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectroThermalContext {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    pub resistive_heating_coefficient: f64,
    pub region_conductivity_scales: Vec<FeaElectroRegionConductivityScale>,
    pub time_profile: Vec<FeaElectroTimeProfilePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaPlasticityConstitutiveContext {
    pub enabled: bool,
    pub yield_strain: f64,
    pub hardening_modulus_ratio: f64,
    pub saturation_exponent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaContactInterfaceContext {
    pub enabled: bool,
    pub penalty_stiffness_scale: f64,
    pub max_penetration_ratio: f64,
    pub friction_coefficient: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaPrepCalibrationProfile {
    Fast,
    Balanced,
    Conservative,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearStaticSolveOptions {
    pub preconditioner_kind: SpdPreconditionerKind,
    pub algebra_backend_kind: LinearAlgebraBackendKind,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    pub electro_thermal_context: Option<FeaElectroThermalContext>,
}

impl Default for LinearStaticSolveOptions {
    fn default() -> Self {
        Self {
            preconditioner_kind: SpdPreconditionerKind::Jacobi,
            algebra_backend_kind: LinearAlgebraBackendKind::CpuReference,
            prep_context: None,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalSolveOptions {
    pub mode_count: usize,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    pub electro_thermal_context: Option<FeaElectroThermalContext>,
}

impl Default for ModalSolveOptions {
    fn default() -> Self {
        Self {
            mode_count: 3,
            prep_context: None,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaModalRunResult {
    pub run: FeaRunResult,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaTransientRunResult {
    pub run: FeaRunResult,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermalRunResult {
    pub run: FeaRunResult,
    pub time_points_s: Vec<f64>,
    pub temperature_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub reference_temperature_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectromagneticRunResult {
    pub run: FeaRunResult,
    pub reference_frequency_hz: f64,
    pub applied_current_a: f64,
    pub vector_potential_field: AnalysisField,
    pub flux_density_field: AnalysisField,
    pub max_residual_norm: f64,
    pub solve_quality: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectromagneticSolveOptions {
    pub prep_context: Option<FeaPrepContext>,
    pub residual_target: f64,
}

impl Default for ElectromagneticSolveOptions {
    fn default() -> Self {
        Self {
            prep_context: None,
            residual_target: 1.0e-6,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalSolveOptions {
    pub step_count: usize,
    pub time_step_s: f64,
    pub residual_target: f64,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
}

impl Default for ThermalSolveOptions {
    fn default() -> Self {
        Self {
            step_count: 10,
            time_step_s: 1.0e-2,
            residual_target: 1.0e-4,
            prep_context: None,
            thermo_mechanical_context: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaNonlinearRunResult {
    pub run: FeaRunResult,
    pub load_factors: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub increment_norms: Vec<f64>,
    pub iteration_counts: Vec<usize>,
    pub failed_increments: usize,
    pub line_search_backtracks: usize,
    pub max_line_search_backtracks_per_increment: usize,
    pub tangent_rebuild_count: usize,
    pub iteration_spike_count: usize,
    pub convergence_stall_count: usize,
    pub backtrack_burst_count: usize,
}

#[derive(Debug, Error)]
pub enum FeaRunError {
    #[error("FEA_MODEL_INVALID: {0}")]
    InvalidModel(String),
}
