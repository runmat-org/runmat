use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingStage {
    CadTopology,
    Sizing,
    CurveMesh,
    SurfaceMesh,
    ProtectedBoundaryComplex,
    TetMesh,
    ConstraintRecovery,
    Optimization,
    SolveReadiness,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TopologyEntityId {
    pub stage: MeshingStage,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageEvidence {
    pub stage: MeshingStage,
    pub status: StageEvidenceStatus,
    #[serde(default)]
    pub entity_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub rejection_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub max_projection_error_m: Option<f64>,
    #[serde(default)]
    pub min_scaled_jacobian: Option<f64>,
}

impl StageEvidence {
    pub fn complete(stage: MeshingStage) -> Self {
        Self {
            stage,
            status: StageEvidenceStatus::Complete,
            entity_counts: BTreeMap::new(),
            rejection_counts: BTreeMap::new(),
            max_projection_error_m: None,
            min_scaled_jacobian: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageEvidenceStatus {
    Complete,
    Failed,
}
