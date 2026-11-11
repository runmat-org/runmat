mod static_convergence;
pub use static_convergence::{
    SpectralErrorEstimate, StaticOperatorProof, StructuralStaticConvergence,
};
pub mod accelerator_admission;
pub mod canonical_digest;
mod field_ids;
mod run_error;
mod solver_compatibility;
mod thermal_state;

pub use field_ids::*;
pub use run_error::FeaRunError;
pub use thermal_state::*;

pub use solver_compatibility::{
    validate_solver_formulation_compatibility, SolverFormulationCompatibilityError,
};

use runmat_analysis_core::AnalysisField;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

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

mod solver_method;
pub use solver_method::SolverMethod;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverPreconditioner {
    None,
    Jacobi,
    Ilu0,
    /// Smoothed-aggregation algebraic multigrid V-cycle.
    Amg,
    /// Exact sparse Cholesky factor (the inner solve of a composite method is direct).
    SparseCholesky,
    BlockJacobi,
    FiniteVolumePressureBalance,
}

impl SolverPreconditioner {
    pub const fn from_spd(kind: SpdPreconditionerKind) -> Self {
        match kind {
            SpdPreconditionerKind::Jacobi => Self::Jacobi,
            SpdPreconditionerKind::Ilu0 => Self::Ilu0,
            SpdPreconditionerKind::Amg => Self::Amg,
            SpdPreconditionerKind::Auto | SpdPreconditionerKind::Direct => Self::SparseCholesky,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Jacobi => "jacobi",
            Self::Ilu0 => "ilu0",
            Self::Amg => "amg",
            Self::SparseCholesky => "sparse_cholesky",
            Self::BlockJacobi => "block_jacobi",
            Self::FiniteVolumePressureBalance => "finite_volume_pressure_balance",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_static_convergence: Option<StructuralStaticConvergence>,
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
    /// The authored volume target with its resolved entities; solver elements match it through
    /// their region or CAD provenance, never through the authored alias alone.
    pub target: runmat_analysis_core::RegionTarget,
    pub temperature_delta_k: f64,
}

impl FeaThermoRegionTemperatureDelta {
    pub fn region_id(&self) -> &str {
        &self.target.region_id
    }

    /// Whether a solver element with this region and provenance lies in the target.
    pub fn contains_element(
        &self,
        region: &runmat_meshing_core::PersistentEntityId,
        provenance: &[runmat_meshing_core::PersistentEntityId],
    ) -> bool {
        self.target.matches_persistent_entity(region)
            || provenance
                .iter()
                .any(|entity| self.target.matches_persistent_entity(entity))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoMechanicalContext {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub applied_temperature_delta_k: runmat_types::physical::TemperatureDifference,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion,
    pub region_temperature_deltas: Vec<FeaThermoRegionTemperatureDelta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaElectroThermalContext {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub resistive_heating_coefficient: runmat_types::physical::InverseTemperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Run controls of penalty contact enforcement. The interfaces, their surfaces and their
/// friction law come from the analysis model only.
#[serde(deny_unknown_fields)]
pub struct FeaContactEnforcement {
    /// Multiplier on the default normal penalty `10 E_min / h` (h: mean secondary face size).
    pub penalty_stiffness_scale: f64,
    /// Admitted initial penetration as a fraction of h.
    pub max_penetration_ratio: f64,
}

impl Default for FeaContactEnforcement {
    fn default() -> Self {
        Self {
            penalty_stiffness_scale: 1.0,
            max_penetration_ratio: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearStaticSolveOptions {
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
            preconditioner_kind: SpdPreconditionerKind::Auto,
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

    pub fn backward_errors(&self) -> Vec<f64> {
        self.basis
            .modes()
            .iter()
            .map(|mode| mode.backward_error())
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

mod structural_initial_state;
pub use structural_initial_state::{
    AppliedStructuralInitialConditionEvidence, StructuralInitialConditionQuantity,
    StructuralInitialStateEvidence, StructuralInitialStateEvidenceBasis,
    StructuralInitialValueSourceEvidence,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaNonlinearRunResult {
    pub run: FeaRunResult,
    /// Complete original converged point history. Element fields are projections,
    /// not restart authority; deserialization alone does not admit continuation.
    pub accepted_state: crate::solve::nonlinear::AcceptedSolidState,
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
