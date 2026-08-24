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
    pub conformal_interface_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlcProtectedEdge {
    pub edge_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 2],
    pub source_edge_id: TopologyEntityId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cad_curve_boundary: Option<PlcProtectedEdgeCadCurveBoundary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlcProtectedEdgeCadCurveBoundary {
    pub cad_edge_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_curve_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_id: Option<String>,
    #[serde(default)]
    pub evaluator_supports_point_evaluation: bool,
    #[serde(default)]
    pub evaluator_supports_projection: bool,
    #[serde(default)]
    pub evaluator_supports_tangent: bool,
    #[serde(default)]
    pub evaluator_supports_curvature: bool,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub live_query_backed: bool,
    #[serde(default)]
    pub live_query_sample_count: usize,
    #[serde(default)]
    pub rejected_evaluator_sample_count: usize,
    #[serde(default)]
    pub curvature_sample_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curvature_limited_target_size_m: Option<f64>,
    pub boundary_segment_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlcValidationSummary {
    pub watertight: bool,
    pub manifold: bool,
    pub shell_nesting_classified: bool,
    pub conformal_interfaces_classified: bool,
}

impl PlcValidationSummary {
    pub fn valid_for_volume_meshing(&self) -> bool {
        self.watertight
            && self.manifold
            && self.shell_nesting_classified
            && self.conformal_interfaces_classified
    }
}
