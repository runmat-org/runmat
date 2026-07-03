use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub mesh_id: String,
    #[serde(default)]
    pub nodes: Vec<SurfaceMeshNode>,
    #[serde(default)]
    pub triangles: Vec<SurfaceMeshTriangle>,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMeshNode {
    pub node_id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
    #[serde(default)]
    pub source_edge_id: Option<TopologyEntityId>,
    pub source_face_id: TopologyEntityId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMeshTriangle {
    pub triangle_id: TopologyEntityId,
    pub source_face_id: TopologyEntityId,
    #[serde(default)]
    pub source_edge_ids: Vec<TopologyEntityId>,
    pub node_ids: [TopologyEntityId; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
    pub area_m2: f64,
}
