mod field_ids;
mod solver_compatibility;

pub use field_ids::*;

pub use solver_compatibility::{
    validate_solver_formulation_compatibility, SolverFormulationCompatibilityError,
};

use runmat_analysis_core::{
    AnalysisField, AnalysisProcedure, FormulationRequirement, InitialConditionTarget, SignalId,
    SignalSampleApproach,
};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::planning::resource::LocalExecutionPlan;
use crate::solve::{
    backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackend {
    Cpu,
    Accelerator,
}

impl ExecutionBackend {
    pub(crate) fn from_solver_backend(solver_backend: LinearAlgebraBackendKind) -> Self {
        match solver_backend {
            LinearAlgebraBackendKind::CpuReference => Self::Cpu,
            LinearAlgebraBackendKind::Accelerator => Self::Accelerator,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverMethod {
    DenseDirect,
    AssembledPcg,
    MatrixFreePcg,
    MatrixFreeSaddlePointGmres,
    MatrixFreeSubspaceIteration,
    MatrixFreeBucklingSubspaceIteration,
    NewmarkAverageAccelerationPcg,
    StructuralModalTransientNewmarkDenseFactorization,
    StructuralDirectHarmonicGmres,
    StructuralModalHarmonicDenseDirect,
    StructuralModalResponseSpectrumCombination,
    StructuralRandomResponseIntegration,
    IncrementalNewtonRaphson,
    ElectromagneticEdgeCurlCurlHarmonicBlockBicgstab,
    AcousticDomainGraphHelmholtzHarmonic,
    CfdVelocityPressureFiniteVolume,
    ChtConjugateProjection,
    FsiPartitionedProjection,
}

impl SolverMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseDirect => "dense_direct",
            Self::AssembledPcg => "assembled_pcg",
            Self::MatrixFreePcg => "matrix_free_pcg",
            Self::MatrixFreeSaddlePointGmres => "matrix_free_saddle_point_gmres",
            Self::MatrixFreeSubspaceIteration => "matrix_free_subspace_iteration",
            Self::MatrixFreeBucklingSubspaceIteration => "matrix_free_buckling_subspace_iteration",
            Self::NewmarkAverageAccelerationPcg => "newmark_average_acceleration_pcg",
            Self::StructuralModalTransientNewmarkDenseFactorization => {
                "structural_modal_transient_newmark_dense_factorization"
            }
            Self::StructuralDirectHarmonicGmres => "structural_direct_harmonic_gmres",
            Self::StructuralModalHarmonicDenseDirect => "structural_modal_harmonic_dense_direct",
            Self::StructuralModalResponseSpectrumCombination => {
                "structural_modal_response_spectrum_combination"
            }
            Self::StructuralRandomResponseIntegration => "structural_random_response_integration",
            Self::IncrementalNewtonRaphson => "incremental_newton_raphson",
            Self::ElectromagneticEdgeCurlCurlHarmonicBlockBicgstab => {
                "electromagnetic_edge_curl_curl_harmonic_block_bicgstab"
            }
            Self::AcousticDomainGraphHelmholtzHarmonic => {
                "acoustic_domain_graph_helmholtz_harmonic"
            }
            Self::CfdVelocityPressureFiniteVolume => "cfd_velocity_pressure_finite_volume",
            Self::ChtConjugateProjection => "cht_conjugate_projection",
            Self::FsiPartitionedProjection => "fsi_partitioned_projection",
        }
    }
}

impl std::fmt::Display for SolverMethod {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverPreconditioner {
    None,
    Jacobi,
    Ilu0,
    BlockJacobi,
    FiniteVolumePressureBalance,
    ThermalCfdProjection,
    InterfaceRelaxation,
}

impl SolverPreconditioner {
    pub const fn from_spd(kind: SpdPreconditionerKind) -> Self {
        match kind {
            SpdPreconditionerKind::Jacobi => Self::Jacobi,
            SpdPreconditionerKind::Ilu0 => Self::Ilu0,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Jacobi => "jacobi",
            Self::Ilu0 => "ilu0",
            Self::BlockJacobi => "block_jacobi",
            Self::FiniteVolumePressureBalance => "finite_volume_pressure_balance",
            Self::ThermalCfdProjection => "thermal_cfd_projection",
            Self::InterfaceRelaxation => "interface_relaxation",
        }
    }
}

impl std::fmt::Display for SolverPreconditioner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "solver_compatibility::FeaRunResultWire")]
pub struct FeaRunResult {
    pub identity: crate::admission::AnalysisRunIdentity,
    pub requested_backend: ComputeBackend,
    pub execution_backend: ExecutionBackend,
    pub solver_backend: LinearAlgebraBackendKind,
    pub solver_device_apply_k_ratio: f64,
    pub solver_method: SolverMethod,
    pub preconditioner: SolverPreconditioner,
    pub solver_host_sync_count: u32,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
    pub diagnostics: Vec<crate::diagnostics::FeaDiagnostic>,
    #[serde(default)]
    pub spatial_projection_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    #[serde(default)]
    pub structural_static_balance: Option<Box<crate::solve::StructuralStaticBalanceEvidence>>,
    #[serde(default)]
    pub structural_constraint_result:
        Option<Box<crate::constraints::CanonicalStructuralConstraintResult>>,
    #[serde(default)]
    pub structural_connector_result:
        Option<Box<crate::connectors::CanonicalStructuralConnectorResult>>,
    #[serde(default)]
    pub observations: Vec<crate::results::observation::ObservationResult>,
    pub observation_histories: Vec<crate::results::observation::ObservationHistory>,
    pub series_reductions: Vec<crate::results::observation::SeriesReductionResult>,
    pub fields: Vec<AnalysisField>,
}

impl FeaRunResult {
    pub fn field(&self, field_id: &str) -> Option<&AnalysisField> {
        self.fields
            .iter()
            .find(|field| field.field_id == field_id)
            .or_else(|| {
                self.series_reductions
                    .iter()
                    .find_map(|reduction| reduction.payload.field(field_id))
            })
    }

    pub fn fields_are_empty(&self) -> bool {
        self.fields.iter().all(AnalysisField::is_empty)
            && self
                .series_reductions
                .iter()
                .flat_map(|reduction| reduction.payload.fields())
                .flatten()
                .all(AnalysisField::is_empty)
    }

    pub fn field_ids(&self) -> Vec<String> {
        self.fields
            .iter()
            .chain(
                self.series_reductions
                    .iter()
                    .flat_map(|reduction| reduction.payload.fields())
                    .flatten(),
            )
            .map(|field| field.field_id.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoRegionTemperatureDelta {
    pub region_id: String,
    pub temperature_delta_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoMechanicalContext {
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    pub region_temperature_deltas: Vec<FeaThermoRegionTemperatureDelta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectroThermalContext {
    pub reference_temperature_k: f64,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    pub resistive_heating_coefficient: f64,
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

#[derive(Debug, Clone, PartialEq)]
pub struct LinearStaticSolveOptions {
    pub formulation_requirement: FormulationRequirement,
    pub preconditioner_kind: SpdPreconditionerKind,
    pub algebra_backend_kind: LinearAlgebraBackendKind,
    pub solver_mesh_artifact_path: Option<String>,
    pub solver_mesh: Option<SolverMeshArtifact>,
    pub require_solver_mesh_for_solid: bool,
    /// Complete displacement state used only to initialize an iterative linear solve.
    pub initial_displacement: Option<Vec<f64>>,
    /// Admitted CPU concurrency for deterministic local numerical execution.
    pub local_execution: LocalExecutionPlan,
}

impl Default for LinearStaticSolveOptions {
    fn default() -> Self {
        Self {
            formulation_requirement: FormulationRequirement::Production,
            preconditioner_kind: SpdPreconditionerKind::Jacobi,
            algebra_backend_kind: LinearAlgebraBackendKind::CpuReference,
            solver_mesh_artifact_path: None,
            solver_mesh: None,
            require_solver_mesh_for_solid: true,
            initial_displacement: None,
            local_execution: LocalExecutionPlan::serial(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalSolveOptions {
    pub formulation_requirement: FormulationRequirement,
    pub truncation: crate::pipeline::modal_governance::ModalTruncationPolicy,
    pub convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    /// Global base-acceleration directions whose load-dependent residual vectors are retained in
    /// the modal basis for later response-spectrum analysis.
    pub residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
    pub solver_mesh: Option<SolverMeshArtifact>,
}

impl Default for ModalSolveOptions {
    fn default() -> Self {
        Self {
            formulation_requirement: FormulationRequirement::Production,
            truncation: crate::pipeline::modal_governance::ModalTruncationPolicy::default(),
            convergence: crate::pipeline::modal_governance::ModalConvergencePolicy::default(),
            residual_vector_directions: Vec::new(),
            solver_mesh: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaModalRunResult {
    pub run: FeaRunResult,
    pub basis: crate::pipeline::modal_basis::ModalBasisArtifact,
    pub mode_shapes: Vec<AnalysisField>,
    pub rotation_mode_shapes: Vec<AnalysisField>,
    pub recovered_modes: Vec<ModalRecoveredMode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaBucklingRunResult {
    pub run: FeaRunResult,
    pub load_factors: Vec<f64>,
    pub residual_norms: Vec<f64>,
    pub mode_shapes: Vec<AnalysisField>,
    pub rotation_mode_shapes: Vec<AnalysisField>,
    pub recovered_modes: Vec<ModalRecoveredMode>,
}

impl FeaModalRunResult {
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.basis
            .modes()
            .iter()
            .map(|mode| mode.frequency_hz())
            .collect()
    }

    pub fn residual_norms(&self) -> Vec<f64> {
        self.basis
            .modes()
            .iter()
            .map(|mode| mode.residual_norm())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalRecoveredMode {
    pub mode_index: usize,
    pub fields: Vec<AnalysisField>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaTransientRunResult {
    pub run: FeaRunResult,
    pub initial_state: StructuralInitialStateEvidence,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub rotation_snapshots: Vec<AnalysisField>,
    pub velocity_snapshots: Vec<AnalysisField>,
    pub angular_velocity_snapshots: Vec<AnalysisField>,
    pub acceleration_snapshots: Vec<AnalysisField>,
    pub angular_acceleration_snapshots: Vec<AnalysisField>,
    pub strain_snapshots: Vec<AnalysisField>,
    pub stress_snapshots: Vec<AnalysisField>,
    pub von_mises_snapshots: Vec<AnalysisField>,
    pub strain_energy_density_snapshots: Vec<AnalysisField>,
    pub kinetic_energy_snapshots: Vec<AnalysisField>,
    pub strain_energy_snapshots: Vec<AnalysisField>,
    pub residual_norm_snapshots: Vec<AnalysisField>,
    pub reaction_force_snapshots: Vec<AnalysisField>,
    pub reaction_moment_snapshots: Vec<AnalysisField>,
    pub connector_results: Vec<crate::connectors::CanonicalStructuralConnectorResult>,
    pub connector_force_snapshots: Vec<AnalysisField>,
    pub connector_stored_energy_snapshots: Vec<AnalysisField>,
    pub external_work_snapshots: Vec<AnalysisField>,
    pub support_work_snapshots: Vec<AnalysisField>,
    pub damping_dissipation_snapshots: Vec<AnalysisField>,
    pub energy_balance_residual_snapshots: Vec<AnalysisField>,
    pub relative_energy_balance_error_snapshots: Vec<AnalysisField>,
    pub dynamic_balance_snapshots: Vec<StructuralDynamicBalanceSnapshot>,
    pub electro_thermal_temperature_snapshots: Vec<AnalysisField>,
    pub electro_thermal_thermal_residual_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralDynamicBalanceSnapshot {
    pub applied_load: Vec<f64>,
    pub inertia_force: Vec<f64>,
    pub damping_force: Vec<f64>,
    pub elastic_force: Vec<f64>,
    pub equilibrium_residual: Vec<f64>,
}

impl StructuralDynamicBalanceSnapshot {
    pub fn validate(&self) -> Result<(), FeaRunError> {
        let dof_count = self.applied_load.len();
        if self.inertia_force.len() != dof_count
            || self.damping_force.len() != dof_count
            || self.elastic_force.len() != dof_count
            || self.equilibrium_residual.len() != dof_count
            || self
                .applied_load
                .iter()
                .chain(&self.inertia_force)
                .chain(&self.damping_force)
                .chain(&self.elastic_force)
                .chain(&self.equilibrium_residual)
                .any(|value| !value.is_finite())
        {
            return Err(FeaRunError::Solve(
                "structural dynamic-balance snapshot is malformed".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralInitialConditionQuantity {
    Displacement,
    Velocity,
    AngularVelocity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralInitialValueSourceEvidence {
    UniformVector3,
    SignalSample {
        signal_id: SignalId,
        abscissa_bits: u64,
        approach: SignalSampleApproach,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppliedStructuralInitialConditionEvidence {
    pub initial_condition_id: String,
    pub quantity: StructuralInitialConditionQuantity,
    pub target: InitialConditionTarget,
    pub resolved_node_identities: Vec<StableDigest>,
    pub source: StructuralInitialValueSourceEvidence,
    pub realized_component_bits: [u64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralInitialStateEvidenceBasis {
    CanonicalTopology {
        discretization_identity: Digest,
        mesh_digest: Option<StableDigest>,
    },
    OperatorBoundZeroDefault {
        dof_count: usize,
        operator_identity: Digest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "StructuralInitialStateEvidenceData")]
pub struct StructuralInitialStateEvidence {
    pub procedure_step_id: String,
    pub basis: StructuralInitialStateEvidenceBasis,
    pub total_dof_count: usize,
    pub authored_displacement_dof_count: usize,
    pub constraint_derived_displacement_dof_count: usize,
    pub zero_default_displacement_dof_count: usize,
    pub authored_velocity_dof_count: usize,
    pub constraint_derived_velocity_dof_count: usize,
    pub zero_default_velocity_dof_count: usize,
    pub direct_constrained_dof_count: usize,
    pub constraint_equation_count: usize,
    pub maximum_displacement_constraint_residual_bits: u64,
    pub maximum_velocity_constraint_residual_bits: u64,
    pub applied_conditions: Vec<AppliedStructuralInitialConditionEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralInitialStateEvidenceData {
    procedure_step_id: String,
    basis: StructuralInitialStateEvidenceBasis,
    total_dof_count: usize,
    authored_displacement_dof_count: usize,
    constraint_derived_displacement_dof_count: usize,
    zero_default_displacement_dof_count: usize,
    authored_velocity_dof_count: usize,
    constraint_derived_velocity_dof_count: usize,
    zero_default_velocity_dof_count: usize,
    direct_constrained_dof_count: usize,
    constraint_equation_count: usize,
    maximum_displacement_constraint_residual_bits: u64,
    maximum_velocity_constraint_residual_bits: u64,
    applied_conditions: Vec<AppliedStructuralInitialConditionEvidence>,
}

impl TryFrom<StructuralInitialStateEvidenceData> for StructuralInitialStateEvidence {
    type Error = String;

    fn try_from(data: StructuralInitialStateEvidenceData) -> Result<Self, Self::Error> {
        if data.procedure_step_id.is_empty() || data.total_dof_count == 0 {
            return Err("structural initial-state evidence has an empty identity domain".into());
        }
        let accounted_displacement_dofs = data
            .authored_displacement_dof_count
            .checked_add(data.constraint_derived_displacement_dof_count)
            .and_then(|count| count.checked_add(data.zero_default_displacement_dof_count));
        let accounted_velocity_dofs = data
            .authored_velocity_dof_count
            .checked_add(data.constraint_derived_velocity_dof_count)
            .and_then(|count| count.checked_add(data.zero_default_velocity_dof_count));
        if accounted_displacement_dofs != Some(data.total_dof_count)
            || accounted_velocity_dofs != Some(data.total_dof_count)
            || data.direct_constrained_dof_count > data.total_dof_count
        {
            return Err("structural initial-state evidence DOF accounting is inconsistent".into());
        }
        let displacement_residual =
            f64::from_bits(data.maximum_displacement_constraint_residual_bits);
        let velocity_residual = f64::from_bits(data.maximum_velocity_constraint_residual_bits);
        if !displacement_residual.is_finite()
            || displacement_residual < 0.0
            || displacement_residual.to_bits() == (-0.0_f64).to_bits()
            || !velocity_residual.is_finite()
            || velocity_residual < 0.0
            || velocity_residual.to_bits() == (-0.0_f64).to_bits()
            || data.applied_conditions.iter().any(|condition| {
                condition.initial_condition_id.is_empty()
                    || condition.resolved_node_identities.is_empty()
                    || condition.resolved_node_identities.iter().enumerate().any(
                        |(index, identity)| {
                            condition.resolved_node_identities[index + 1..].contains(identity)
                        },
                    )
                    || condition
                        .realized_component_bits
                        .iter()
                        .any(|bits| !f64::from_bits(*bits).is_finite())
                    || matches!(
                        &condition.source,
                        StructuralInitialValueSourceEvidence::SignalSample {
                            signal_id,
                            abscissa_bits,
                            ..
                        } if signal_id.0.is_empty()
                            || !f64::from_bits(*abscissa_bits).is_finite()
                            || *abscissa_bits == (-0.0_f64).to_bits()
                    )
            })
            || data
                .applied_conditions
                .windows(2)
                .any(|pair| pair[0].initial_condition_id >= pair[1].initial_condition_id)
        {
            return Err(
                "structural initial-state evidence contains invalid realized values".into(),
            );
        }
        match data.basis {
            StructuralInitialStateEvidenceBasis::CanonicalTopology {
                discretization_identity,
                mesh_digest,
            } if discretization_identity.bytes() == &[0; 32]
                || discretization_identity == Digest::sha256([])
                || mesh_digest == Some(StableDigest::ZERO) =>
            {
                return Err(
                    "structural initial-state evidence has an invalid topology identity".into(),
                );
            }
            StructuralInitialStateEvidenceBasis::OperatorBoundZeroDefault {
                dof_count,
                operator_identity,
            } => {
                if dof_count != data.total_dof_count {
                    return Err(
                        "operator-bound structural initial-state evidence has inconsistent DOF identity"
                            .into(),
                    );
                }
                if operator_identity.bytes() == &[0; 32] || operator_identity == Digest::sha256([])
                {
                    return Err("operator-bound structural initial-state evidence has an invalid operator identity".into());
                }
                if data.authored_displacement_dof_count != 0
                    || data.authored_velocity_dof_count != 0
                    || !data.applied_conditions.is_empty()
                {
                    return Err(
                        "operator-bound structural initial-state evidence is not an implicit zero state"
                            .into(),
                    );
                }
            }
            _ => {}
        }
        Ok(Self {
            procedure_step_id: data.procedure_step_id,
            basis: data.basis,
            total_dof_count: data.total_dof_count,
            authored_displacement_dof_count: data.authored_displacement_dof_count,
            constraint_derived_displacement_dof_count: data
                .constraint_derived_displacement_dof_count,
            zero_default_displacement_dof_count: data.zero_default_displacement_dof_count,
            authored_velocity_dof_count: data.authored_velocity_dof_count,
            constraint_derived_velocity_dof_count: data.constraint_derived_velocity_dof_count,
            zero_default_velocity_dof_count: data.zero_default_velocity_dof_count,
            direct_constrained_dof_count: data.direct_constrained_dof_count,
            constraint_equation_count: data.constraint_equation_count,
            maximum_displacement_constraint_residual_bits: data
                .maximum_displacement_constraint_residual_bits,
            maximum_velocity_constraint_residual_bits: data
                .maximum_velocity_constraint_residual_bits,
            applied_conditions: data.applied_conditions,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermalRunResult {
    pub run: FeaRunResult,
    pub time_points_s: Vec<f64>,
    pub temperature_snapshots: Vec<AnalysisField>,
    pub temperature_gradient_snapshots: Vec<AnalysisField>,
    pub heat_flux_snapshots: Vec<AnalysisField>,
    pub energy_storage_rate_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub reference_temperature_k: f64,
    pub initial_state: ThermalInitialStateEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalInitialStateEvidence {
    pub procedure_step_id: String,
    pub reference_temperature_k: f64,
    pub applied_initial_condition_ids: Vec<String>,
    pub authored_node_count: usize,
    pub prescribed_node_count: usize,
    pub authored_and_prescribed_node_count: usize,
    pub reference_node_count: usize,
    pub minimum_temperature_k: f64,
    pub maximum_temperature_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectromagneticRunResult {
    pub run: FeaRunResult,
    pub reference_frequency_hz: f64,
    pub applied_current_a: f64,
    pub vector_potential_real_field: AnalysisField,
    pub vector_potential_imag_field: AnalysisField,
    pub magnetic_flux_density_real_field: AnalysisField,
    pub magnetic_flux_density_imag_field: AnalysisField,
    pub magnetic_flux_density_magnitude_field: AnalysisField,
    pub magnetic_field_real_field: AnalysisField,
    pub magnetic_field_imag_field: AnalysisField,
    pub current_density_real_field: AnalysisField,
    pub current_density_imag_field: AnalysisField,
    pub electric_field_real_field: AnalysisField,
    pub electric_field_imag_field: AnalysisField,
    pub power_loss_density_field: AnalysisField,
    pub energy_density_field: AnalysisField,
    pub residual_real_field: AnalysisField,
    pub residual_imag_field: AnalysisField,
    pub electric_flux_density_real_field: AnalysisField,
    pub electric_flux_density_imag_field: AnalysisField,
    pub poynting_vector_real_field: AnalysisField,
    pub poynting_vector_imag_field: AnalysisField,
    pub max_residual_norm: f64,
    pub solve_quality: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectromagneticSolveOptions {
    pub formulation_requirement: FormulationRequirement,
    pub procedure: AnalysisProcedure,
    pub residual_target: f64,
    pub harmonic_tolerance: f64,
    pub harmonic_max_iterations: usize,
    pub solver_mesh: Option<SolverMeshArtifact>,
}

impl Default for ElectromagneticSolveOptions {
    fn default() -> Self {
        Self {
            formulation_requirement: FormulationRequirement::Production,
            procedure: AnalysisProcedure::HarmonicResponse,
            residual_target: 1.0e-6,
            harmonic_tolerance: 1.0e-7,
            harmonic_max_iterations: 96,
            solver_mesh: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalProcedureControl {
    Steady,
    BackwardEuler { snapshot_count: usize },
}

impl ThermalProcedureControl {
    pub const fn procedure(&self) -> AnalysisProcedure {
        match self {
            Self::Steady => AnalysisProcedure::Static,
            Self::BackwardEuler { .. } => AnalysisProcedure::TransientResponse,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalSolveOptions {
    pub formulation_requirement: FormulationRequirement,
    pub procedure: ThermalProcedureControl,
    pub residual_target: f64,
    pub solver_mesh: Option<SolverMeshArtifact>,
}

impl Default for ThermalSolveOptions {
    fn default() -> Self {
        Self {
            formulation_requirement: FormulationRequirement::Production,
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 10 },
            residual_target: 1.0e-4,
            solver_mesh: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaNonlinearRunResult {
    pub run: FeaRunResult,
    pub load_factors: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub rotation_snapshots: Vec<AnalysisField>,
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
    #[error("FEA_ASSEMBLY_FAILED: {0}")]
    Assembly(String),
    #[error("FEA_SOLVE_FAILED: {0}")]
    Solve(String),
    #[error("FEA_OBSERVATION_FAILED: {0}")]
    Observation(String),
    #[error(transparent)]
    FrequencyPlan(#[from] crate::frequency::FrequencyPlanError),
    #[error("FEA_ELECTROMAGNETIC_SOURCE_MISSING: no nonzero source at {frequency_hz} Hz")]
    MissingElectromagneticSource { frequency_hz: f64 },
    #[error(transparent)]
    FormulationAdmission(#[from] crate::admission::FormulationAdmissionRejection),
    #[error(transparent)]
    FormulationAvailability(#[from] crate::admission::FormulationAvailabilityRejection),
    #[error(transparent)]
    SolverFormulationCompatibility(#[from] SolverFormulationCompatibilityError),
    #[error("FEA_CANCELLED: execution cancelled by user")]
    Cancelled,
}
