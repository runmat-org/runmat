use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetMesh {
    pub mesh_id: String,
    #[serde(default)]
    pub nodes: Vec<TetMeshNode>,
    #[serde(default)]
    pub elements: Vec<Tet4Element>,
    #[serde(default)]
    pub boundary_faces: Vec<TetBoundaryFace>,
    pub recovery_complete: bool,
    pub quality_optimized: bool,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetMeshNode {
    pub node_id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tet4Element {
    pub element_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 4],
    pub material_region_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetBoundaryFace {
    pub face_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 3],
    pub source_face_id: TopologyEntityId,
}
