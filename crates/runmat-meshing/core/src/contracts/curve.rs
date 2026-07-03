use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveMesh {
    pub mesh_id: String,
    #[serde(default)]
    pub nodes: Vec<CurveMeshNode>,
    #[serde(default)]
    pub elements: Vec<CurveMeshElement>,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveMeshNode {
    pub node_id: TopologyEntityId,
    pub source_edge_id: TopologyEntityId,
    pub parameter: f64,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveMeshElement {
    pub element_id: TopologyEntityId,
    pub source_edge_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 2],
    pub length_m: f64,
}
