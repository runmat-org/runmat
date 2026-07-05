use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT: &str =
    "optimization_local_reconnection_attempts";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT: &str =
    "optimization_local_reconnection_accepted";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT: &str =
    "optimization_local_reconnection_rejected";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT: &str =
    "optimization_local_reconnection_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX: &str =
    "optimization_local_reconnection_rejected_";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT: &str =
    "optimization_interior_smoothing_attempts";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT: &str =
    "optimization_interior_smoothing_accepted";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT: &str =
    "optimization_interior_smoothing_rejected";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT: &str =
    "optimization_interior_smoothing_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX: &str =
    "optimization_interior_smoothing_rejected_";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT: &str =
    "optimization_boundary_smoothing_attempts";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT: &str =
    "optimization_boundary_smoothing_accepted";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT: &str =
    "optimization_boundary_smoothing_rejected";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT: &str =
    "optimization_boundary_smoothing_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX: &str =
    "optimization_boundary_smoothing_rejected_";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronMesh {
    pub mesh_id: String,
    #[serde(default = "unknown_tetrahedron_generation_family")]
    pub tetrahedron_generation_family: String,
    #[serde(default)]
    pub nodes: Vec<TetrahedronMeshNode>,
    #[serde(default)]
    pub elements: Vec<Tetrahedron4Element>,
    #[serde(default)]
    pub boundary_faces: Vec<TetrahedronBoundaryFace>,
    pub recovery_complete: bool,
    pub quality_optimized: bool,
    pub evidence: StageEvidence,
}

fn unknown_tetrahedron_generation_family() -> String {
    "unknown".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronMeshNode {
    pub node_id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tetrahedron4Element {
    pub element_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 4],
    pub material_region_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronBoundaryFace {
    pub face_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 3],
    pub source_face_id: TopologyEntityId,
    #[serde(default)]
    pub source_edge_ids: [Option<TopologyEntityId>; 3],
}
