use serde::{Deserialize, Serialize};

/// Placement choices for a checked, prepared source revision. A missing time
/// estimate stays absent; a deadline is never presented as predicted duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisExecutionMode {
    SingleCore,
    MultiCore,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisExecutionReadiness {
    Ready,
    DesktopRequired,
    Unavailable,
}
/// Recent successful timings of this exact study and allocation on this runtime.
/// Bounds describe observed variability; they are not a cross-machine promise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalysisExecutionDuration {
    #[serde(with = "runmat_types::wire_integer")]
    pub lower_millis: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub upper_millis: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub sample_count: u64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalysisExecutionCandidate {
    pub mode: AnalysisExecutionMode,
    pub cores: Option<u32>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub memory_bytes: Option<u64>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub estimated_millis: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_evidence: Option<AnalysisExecutionDuration>,
    pub readiness: AnalysisExecutionReadiness,
    pub reason: Option<String>,
    /// Tetrahedra the check estimates the study's mesh settings make of its geometry, before
    /// meshing. Absent when the geometry has no measured preview.
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub estimated_element_count: Option<u64>,
    /// Mesh nodes of the same estimate (corner and midside nodes).
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub estimated_node_count: Option<u64>,
    /// Present when the estimate exceeds this target's admitted memory: the candidate is not
    /// ready, and the limit says by how much.
    #[serde(default)]
    pub resource_limit: Option<AnalysisResourceLimit>,
    /// Operations this run is estimated to charge against the simulation allowance, by usage
    /// category. A host refuses the run before it starts when the total exceeds the allowance
    /// left; admission refuses it too. Absent when the study type or geometry has no estimate.
    #[serde(default)]
    pub estimated_operations: Option<AnalysisOperationEstimate>,
}
/// Operations a run is estimated to charge, by the usage ledger's categories, as decimal
/// counts. `total` is the figure compared with the remaining allowance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalysisOperationEstimate {
    pub meshing: runmat_simulation_usage::OperationCount,
    pub assembly: runmat_simulation_usage::OperationCount,
    pub factorization: runmat_simulation_usage::OperationCount,
    pub solve: runmat_simulation_usage::OperationCount,
    pub other: runmat_simulation_usage::OperationCount,
    pub total: runmat_simulation_usage::OperationCount,
    /// Conjugate-gradient iterations the solve figure covers; absent for a direct solve. A solve
    /// that needs more iterations charges more than the estimate.
    pub solve_iterations: Option<u32>,
}
/// A check-time refusal: the estimated requirement of one resource term against the maximum
/// the target admits, with the action that resolves it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalysisResourceLimit {
    /// The diagnostic code, e.g. `RM.FEA.RUN_STUDY.BROWSER_RESOURCE_LIMIT`.
    pub code: String,
    /// The term that does not fit: `estimated_memory` or `estimated_scratch`.
    pub term: String,
    /// Absent only when another authored limit bound before the requirement was sized.
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub required_bytes: Option<u64>,
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_bytes: u64,
    pub next_action: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalysisExecutionChoices {
    pub schema_version: u16,
    pub plan_id: String,
    pub revision: [u8; 32],
    pub candidates: Vec<AnalysisExecutionCandidate>,
}
impl AnalysisOperationEstimate {
    /// The contract form of a plan's estimate; `None` when its total exceeds the count range.
    pub fn from_estimate(
        estimate: &runmat_analysis_fea::planning::operations::FeaOperationEstimate,
    ) -> Option<Self> {
        Some(Self {
            meshing: estimate.meshing,
            assembly: estimate.assembly,
            factorization: estimate.factorization,
            solve: estimate.solve,
            other: estimate.other,
            total: estimate.total().ok()?,
            solve_iterations: estimate.solve_iterations,
        })
    }
}
