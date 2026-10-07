use super::super::{SolverFieldTransferErrorEvidence, StructuralRecoveryEstimate};
use super::constraints::StructuralAdaptationConstraintLineage;
use crate::{execution::SolverResultIdentity, results::accuracy::AccuracyEvaluation};
use runmat_analysis_core::{AccuracyPlan, AnalysisModel, ObservationQuantity};
use runmat_meshing_core::{
    SolverMeshAdaptationLineage, SolverMeshArtifact, SolverMeshTransferMap, StableDigest,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationSolverResult {
    pub result_identity: SolverResultIdentity,
    pub converged: bool,
    pub iteration_count: u64,
    pub normalized_residual: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTargetQuantity {
    pub quantity_id: String,
    pub quantity: ObservationQuantity,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralAdaptationDecisionStatus {
    Continue,
    Converged,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationConvergenceDecision {
    pub status: StructuralAdaptationDecisionStatus,
    pub estimator_reduction: Option<f64>,
    pub target_absolute_change: f64,
    pub target_relative_change: f64,
    pub refinement_ceiling_exhausted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationIteration {
    pub schema_version: u16,
    pub iteration_index: u64,
    pub previous_iteration_digest: Option<StableDigest>,
    pub source_solver_artifact_digest: StableDigest,
    pub target_solver_artifact_digest: StableDigest,
    pub adaptation_lineage_digest: StableDigest,
    pub transfer_map_digest: StableDigest,
    pub source_estimator: StructuralRecoveryEstimate,
    pub estimator: StructuralRecoveryEstimate,
    pub transfer_errors: Vec<SolverFieldTransferErrorEvidence>,
    pub source_solver_result: StructuralAdaptationSolverResult,
    pub solver_result: StructuralAdaptationSolverResult,
    pub source_target_quantity: StructuralTargetQuantity,
    pub target_quantity: StructuralTargetQuantity,
    pub constraint_lineage: StructuralAdaptationConstraintLineage,
    pub accuracy_plan: AccuracyPlan,
    pub accuracy_evaluation: AccuracyEvaluation,
    pub decision: StructuralAdaptationConvergenceDecision,
}

#[derive(Clone, Copy)]
pub struct StructuralAdaptationIterationInput<'a> {
    pub model: &'a AnalysisModel,
    pub source_artifact: &'a SolverMeshArtifact,
    pub target_artifact: &'a SolverMeshArtifact,
    pub transfer_map: &'a SolverMeshTransferMap,
    pub lineage: &'a SolverMeshAdaptationLineage,
    pub source_estimator: &'a StructuralRecoveryEstimate,
    pub estimator: &'a StructuralRecoveryEstimate,
    pub transfer_errors: &'a [SolverFieldTransferErrorEvidence],
    pub source_solver_result: &'a StructuralAdaptationSolverResult,
    pub solver_result: &'a StructuralAdaptationSolverResult,
    pub source_target_quantity: &'a StructuralTargetQuantity,
    pub target_quantity: &'a StructuralTargetQuantity,
    pub previous: Option<&'a StructuralAdaptationIteration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralAdaptationIterationError {
    InvalidAccuracyPlan,
    InvalidAccuracyEvidence(String),
    InvalidConstraints(String),
    InvalidArtifact(String),
    InvalidLineage(String),
    InvalidEstimator,
    InvalidTransferEvidence,
    InvalidSolverResult,
    InvalidTargetQuantity,
    InvalidIterationChain,
    InvalidDecision,
    Codec(String),
}
