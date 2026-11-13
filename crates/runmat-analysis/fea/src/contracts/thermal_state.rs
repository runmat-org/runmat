//! Thermal solve controls, solved output and initial/reference-state evidence.

use runmat_analysis_core::{AnalysisField, AnalysisProcedure};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use super::FeaRunResult;

mod energy;
mod initial_evidence;
mod nonlinear_evidence;
pub use energy::{
    ThermalEnergyBalance, ThermalEnergyBalanceError, ThermalPowerScale, ThermalPrescribedHeatFlow,
};
pub use nonlinear_evidence::{
    ThermalNonlinearSampleEvidence, THERMAL_MAXIMUM_BACKTRACKING_TRIALS,
    THERMAL_MAXIMUM_NEWTON_CORRECTIONS,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
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
    pub initial_fields: Vec<crate::execution::InitialFieldBinding>,
    pub procedure: ThermalProcedureControl,
    pub residual_target: f64,
    pub solver_mesh: Option<SolverMeshArtifact>,
}

impl Default for ThermalSolveOptions {
    fn default() -> Self {
        Self {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 10 },
            residual_target: 1.0e-4,
            solver_mesh: None,
            initial_fields: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaThermalRunResult {
    /// Exact authored numerical intent captured by the executing thermal pipeline.
    pub source_workload: crate::execution::AnalysisWorkloadIdentity,
    pub numerical_intent: crate::execution::ThermalNumericalIntentIdentity,
    pub run: FeaRunResult,
    pub producer: crate::execution::StepIdentity,
    pub topology_digest: Digest,
    /// Persistent node identities in exactly the temperature field's storage order.
    pub ordered_node_ids: Vec<u64>,
    /// Persistent volume-element IDs in recovered field order.
    pub ordered_element_ids: Vec<u64>,
    pub time_points_s: Vec<f64>,
    pub temperature_snapshots: Vec<AnalysisField>,
    /// Mesh nodes of finite-conductance thermal interfaces, in the interface field's order.
    /// `temperature_snapshots` holds the first side's temperature there.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interface_node_ids: Vec<u64>,
    /// The second side's temperature at `interface_node_ids`, one field per time point (empty
    /// when the study has no finite-conductance interface).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interface_temperature_snapshots: Vec<AnalysisField>,
    pub temperature_gradient_snapshots: Vec<AnalysisField>,
    pub heat_flux_snapshots: Vec<AnalysisField>,
    pub energy_storage_rate_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    /// Final original nonlinear equilibrium evidence, never a Newton correction residual.
    /// Empty only for a linear thermal workload; the receipt checks the original source law.
    pub nonlinear_evidence: Vec<ThermalNonlinearSampleEvidence>,
    pub residual_target: f64,
    /// One global power balance per solve (no initial-only transient sample).
    pub energy_balances: Vec<ThermalEnergyBalance>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub initial_state: ThermalInitialStateEvidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalInitialStateEvidence {
    pub procedure_step_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub applied_initial_condition_ids: Vec<String>,
    pub authored_node_count: usize,
    pub prescribed_node_count: usize,
    pub authored_and_prescribed_node_count: usize,
    pub reference_node_count: usize,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub minimum_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub maximum_temperature_k: runmat_types::physical::Temperature,
}
