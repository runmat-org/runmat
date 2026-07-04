use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronMesh {
    pub mesh_id: String,
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
