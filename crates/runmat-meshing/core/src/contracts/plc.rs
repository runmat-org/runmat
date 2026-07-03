use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtectedBoundaryComplex {
    pub complex_id: String,
    #[serde(default)]
    pub nodes: Vec<PlcNode>,
    #[serde(default)]
    pub facets: Vec<PlcFacet>,
    #[serde(default)]
    pub protected_edges: Vec<PlcProtectedEdge>,
    pub validation: PlcValidationSummary,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlcNode {
    pub node_id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlcFacet {
    pub facet_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 3],
    pub source_face_id: TopologyEntityId,
    #[serde(default)]
    pub material_interface_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlcProtectedEdge {
    pub edge_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 2],
    pub source_edge_id: TopologyEntityId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlcValidationSummary {
    pub watertight: bool,
    pub manifold: bool,
    pub shell_nesting_classified: bool,
    pub material_interfaces_classified: bool,
}

impl PlcValidationSummary {
    pub fn valid_for_volume_meshing(&self) -> bool {
        self.watertight
            && self.manifold
            && self.shell_nesting_classified
            && self.material_interfaces_classified
    }
}
