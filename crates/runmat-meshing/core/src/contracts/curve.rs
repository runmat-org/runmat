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
    #[serde(default)]
    pub source_vertex_id: Option<TopologyEntityId>,
    pub parameter: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveMeshElement {
    pub element_id: TopologyEntityId,
    pub source_edge_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 2],
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub length_m: runmat_types::physical::Length,
}
