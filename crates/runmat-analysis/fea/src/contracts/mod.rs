use runmat_analysis_core::AnalysisField;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::solve::{
    backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind,
};

pub const FEA_FIELD_STRUCTURAL_DISPLACEMENT: &str = "structural.displacement";
pub const FEA_FIELD_STRUCTURAL_VON_MISES: &str = "structural.von_mises";
pub const FEA_FIELD_STRUCTURAL_STRAIN: &str = "structural.strain";
pub const FEA_FIELD_STRUCTURAL_STRESS: &str = "structural.stress";
pub const FEA_FIELD_STRUCTURAL_REACTION_FORCE: &str = "structural.reaction_force";
pub const FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY: &str = "structural.total_strain_energy";
pub const FEA_FIELD_STRUCTURAL_RESIDUAL_NORM: &str = "structural.residual_norm";
pub const FEA_FIELD_STRUCTURAL_EQUATION_SCALE: &str = "structural.equation_scale";
pub const FEA_FIELD_EM_VECTOR_POTENTIAL_REAL: &str = "em.vector_potential_real";
pub const FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE: &str = "em.magnetic_flux_density_magnitude";

pub fn fea_modal_mode_shape_field_id(mode_number: usize) -> String {
    format!("modal.mode_shape.{mode_number}")
}

pub const FEA_FIELD_MODAL_FREQUENCY_HZ: &str = "modal.frequency_hz";
pub const FEA_FIELD_MODAL_EIGENVALUE: &str = "modal.eigenvalue";
pub const FEA_FIELD_MODAL_MODAL_MASS: &str = "modal.modal_mass";
pub const FEA_FIELD_MODAL_MODAL_STIFFNESS: &str = "modal.modal_stiffness";
pub const FEA_FIELD_MODAL_PARTICIPATION_FACTOR: &str = "modal.participation_factor";
pub const FEA_FIELD_MODAL_RESIDUAL_NORM: &str = "modal.residual_norm";
pub const FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION: &str =
    "modal.relative_frequency_separation";
pub const FEA_FIELD_MODAL_M_ORTHOGONALITY: &str = "modal.m_orthogonality";

pub const FEA_FIELD_ACOUSTIC_PRESSURE_REAL: &str = "acoustic.pressure_real";
pub const FEA_FIELD_ACOUSTIC_PRESSURE_IMAG: &str = "acoustic.pressure_imag";
pub const FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE: &str = "acoustic.pressure_magnitude";
pub const FEA_FIELD_ACOUSTIC_PHASE: &str = "acoustic.phase";
pub const FEA_FIELD_ACOUSTIC_SOUND_PRESSURE_LEVEL_DB: &str = "acoustic.sound_pressure_level_db";
pub const FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY: &str = "acoustic.particle_velocity";

pub const FEA_FIELD_CFD_VELOCITY: &str = "cfd.velocity";
pub const FEA_FIELD_CFD_PRESSURE: &str = "cfd.pressure";
pub const FEA_FIELD_CFD_VORTICITY: &str = "cfd.vorticity";
pub const FEA_FIELD_CFD_WALL_SHEAR_STRESS: &str = "cfd.wall_shear_stress";
pub const FEA_FIELD_CFD_RESIDUAL_MOMENTUM: &str = "cfd.residual_momentum";
pub const FEA_FIELD_CFD_RESIDUAL_CONTINUITY: &str = "cfd.residual_continuity";
pub const FEA_FIELD_CFD_REYNOLDS_NUMBER: &str = "cfd.reynolds_number";

pub const FEA_FIELD_CHT_FLUID_VELOCITY: &str = "cht.fluid_velocity";
pub const FEA_FIELD_CHT_FLUID_PRESSURE: &str = "cht.fluid_pressure";

pub fn fea_cht_fluid_temperature_field_id(step_index: usize) -> String {
    format!("cht.fluid_temperature.{step_index}")
}

pub fn fea_cht_solid_temperature_field_id(step_index: usize) -> String {
    format!("cht.solid_temperature.{step_index}")
}

pub fn fea_cht_interface_heat_flux_field_id(step_index: usize) -> String {
    format!("cht.interface_heat_flux.{step_index}")
}

pub fn fea_cht_interface_temperature_jump_field_id(step_index: usize) -> String {
    format!("cht.interface_temperature_jump.{step_index}")
}

pub fn fea_cht_energy_residual_field_id(step_index: usize) -> String {
    format!("cht.energy_residual.{step_index}")
}

pub fn fea_fsi_fluid_velocity_field_id(step_index: usize) -> String {
    format!("fsi.fluid_velocity.{step_index}")
}

pub fn fea_fsi_fluid_pressure_field_id(step_index: usize) -> String {
    format!("fsi.fluid_pressure.{step_index}")
}

pub fn fea_fsi_structural_displacement_field_id(step_index: usize) -> String {
    format!("fsi.structural_displacement.{step_index}")
}

pub fn fea_fsi_interface_pressure_field_id(step_index: usize) -> String {
    format!("fsi.interface_pressure.{step_index}")
}

pub fn fea_fsi_interface_traction_field_id(step_index: usize) -> String {
    format!("fsi.interface_traction.{step_index}")
}

pub fn fea_fsi_interface_displacement_field_id(step_index: usize) -> String {
    format!("fsi.interface_displacement.{step_index}")
}

pub fn fea_fsi_interface_residual_field_id(step_index: usize) -> String {
    format!("fsi.interface_residual.{step_index}")
}

pub fn fea_fsi_coupling_iteration_count_field_id(step_index: usize) -> String {
    format!("fsi.coupling_iteration_count.{step_index}")
}

pub fn fea_thermal_temperature_field_id(snapshot_index: usize) -> String {
    format!("thermal.temperature.{snapshot_index}")
}

pub fn fea_thermal_temperature_gradient_field_id(snapshot_index: usize) -> String {
    format!("thermal.temperature_gradient.{snapshot_index}")
}

pub fn fea_thermal_heat_flux_field_id(snapshot_index: usize) -> String {
    format!("thermal.heat_flux.{snapshot_index}")
}

pub fn fea_thermal_heat_source_field_id(snapshot_index: usize) -> String {
    format!("thermal.heat_source.{snapshot_index}")
}

pub fn fea_thermal_boundary_heat_flux_field_id(snapshot_index: usize) -> String {
    format!("thermal.boundary_heat_flux.{snapshot_index}")
}

pub fn fea_transient_displacement_field_id(snapshot_index: usize) -> String {
    format!("transient.displacement.{snapshot_index}")
}

pub fn fea_transient_velocity_field_id(snapshot_index: usize) -> String {
    format!("transient.velocity.{snapshot_index}")
}

pub fn fea_transient_acceleration_field_id(snapshot_index: usize) -> String {
    format!("transient.acceleration.{snapshot_index}")
}

pub fn fea_transient_von_mises_field_id(snapshot_index: usize) -> String {
    format!("transient.von_mises.{snapshot_index}")
}

pub fn fea_transient_kinetic_energy_field_id(snapshot_index: usize) -> String {
    format!("transient.kinetic_energy.{snapshot_index}")
}

pub fn fea_transient_strain_energy_field_id(snapshot_index: usize) -> String {
    format!("transient.strain_energy.{snapshot_index}")
}

pub fn fea_transient_residual_norm_field_id(snapshot_index: usize) -> String {
    format!("transient.residual_norm.{snapshot_index}")
}

pub fn fea_nonlinear_displacement_field_id(increment_index: usize) -> String {
    format!("nonlinear.displacement.{increment_index}")
}

pub fn fea_nonlinear_von_mises_field_id(increment_index: usize) -> String {
    format!("nonlinear.von_mises.{increment_index}")
}

pub fn fea_nonlinear_plastic_strain_field_id(increment_index: usize) -> String {
    format!("nonlinear.plastic_strain.{increment_index}")
}

pub fn fea_nonlinear_equivalent_plastic_strain_field_id(increment_index: usize) -> String {
    format!("nonlinear.equivalent_plastic_strain.{increment_index}")
}

pub fn fea_nonlinear_contact_pressure_field_id(increment_index: usize) -> String {
    format!("nonlinear.contact_pressure.{increment_index}")
}

pub fn fea_nonlinear_contact_gap_field_id(increment_index: usize) -> String {
    format!("nonlinear.contact_gap.{increment_index}")
}

pub fn fea_nonlinear_load_factor_field_id(increment_index: usize) -> String {
    format!("nonlinear.load_factor.{increment_index}")
}

pub fn fea_nonlinear_residual_norm_field_id(increment_index: usize) -> String {
    format!("nonlinear.residual_norm.{increment_index}")
}

pub fn fea_thermo_mechanical_temperature_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.temperature.{step_index}")
}

pub fn fea_thermo_mechanical_thermal_strain_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.thermal_strain.{step_index}")
}

pub fn fea_thermo_mechanical_thermal_stress_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.thermal_stress.{step_index}")
}

pub fn fea_thermo_mechanical_displacement_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.displacement.{step_index}")
}

pub fn fea_thermo_mechanical_von_mises_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.von_mises.{step_index}")
}

pub fn fea_thermo_mechanical_coupling_residual_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.coupling_residual.{step_index}")
}

pub const FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL: &str = "electro_thermal.electric_potential";
pub const FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD: &str = "electro_thermal.electric_field";
pub const FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY: &str = "electro_thermal.current_density";
pub const FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT: &str = "electro_thermal.joule_heat";

pub fn fea_electro_thermal_temperature_field_id(step_index: usize) -> String {
    format!("electro_thermal.temperature.{step_index}")
}

pub fn fea_electro_thermal_thermal_residual_field_id(step_index: usize) -> String {
    format!("electro_thermal.thermal_residual.{step_index}")
}

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
    pub fields: Vec<AnalysisField>,
}

impl FeaRunResult {
    pub fn field(&self, field_id: &str) -> Option<&AnalysisField> {
        self.fields.iter().find(|field| field.field_id == field_id)
    }

    pub fn fields_are_empty(&self) -> bool {
        self.fields.is_empty() || self.fields.iter().any(AnalysisField::is_empty)
    }

    pub fn field_ids(&self) -> Vec<String> {
        self.fields
            .iter()
            .map(|field| field.field_id.clone())
            .collect()
    }
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
    pub velocity_snapshots: Vec<AnalysisField>,
    pub acceleration_snapshots: Vec<AnalysisField>,
    pub von_mises_snapshots: Vec<AnalysisField>,
    pub kinetic_energy_snapshots: Vec<AnalysisField>,
    pub strain_energy_snapshots: Vec<AnalysisField>,
    pub residual_norm_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_temperature_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_thermal_strain_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_thermal_stress_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_displacement_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_von_mises_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_coupling_residual_snapshots: Vec<AnalysisField>,
    pub electro_thermal_temperature_snapshots: Vec<AnalysisField>,
    pub electro_thermal_thermal_residual_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermalRunResult {
    pub run: FeaRunResult,
    pub time_points_s: Vec<f64>,
    pub temperature_snapshots: Vec<AnalysisField>,
    pub temperature_gradient_snapshots: Vec<AnalysisField>,
    pub heat_flux_snapshots: Vec<AnalysisField>,
    pub heat_source_snapshots: Vec<AnalysisField>,
    pub boundary_heat_flux_snapshots: Vec<AnalysisField>,
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
    pub harmonic_tolerance: f64,
    pub harmonic_max_iterations: usize,
}

impl Default for ElectromagneticSolveOptions {
    fn default() -> Self {
        Self {
            prep_context: None,
            residual_target: 1.0e-6,
            harmonic_tolerance: 1.0e-7,
            harmonic_max_iterations: 96,
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
    pub von_mises_snapshots: Vec<AnalysisField>,
    pub plastic_strain_snapshots: Vec<AnalysisField>,
    pub equivalent_plastic_strain_snapshots: Vec<AnalysisField>,
    pub contact_pressure_snapshots: Vec<AnalysisField>,
    pub contact_gap_snapshots: Vec<AnalysisField>,
    pub load_factor_snapshots: Vec<AnalysisField>,
    pub residual_norm_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_temperature_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_thermal_strain_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_thermal_stress_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_displacement_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_von_mises_snapshots: Vec<AnalysisField>,
    pub thermo_mechanical_coupling_residual_snapshots: Vec<AnalysisField>,
    pub electro_thermal_temperature_snapshots: Vec<AnalysisField>,
    pub electro_thermal_thermal_residual_snapshots: Vec<AnalysisField>,
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
    #[error("FEA_CANCELLED: execution cancelled by user")]
    Cancelled,
}
