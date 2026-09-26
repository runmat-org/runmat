//! Solved thermal output and its initial/reference-state evidence.

use runmat_analysis_core::AnalysisField;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::FeaRunResult;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermalRunResult {
    pub run: FeaRunResult,
    pub producer: crate::execution::StepIdentity,
    pub topology_digest: Digest,
    /// Persistent node identities in exactly the temperature field's storage order.
    pub ordered_node_ids: Vec<u64>,
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
