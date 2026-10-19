use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadModel {
    pub model_id: String,
    pub unit_scale_to_m: f64,
    #[serde(default)]
    pub vertices: Vec<CadVertexContract>,
    #[serde(default)]
    pub edges: Vec<CadEdgeContract>,
    #[serde(default)]
    pub faces: Vec<CadFaceContract>,
    #[serde(default)]
    pub shells: Vec<CadShellContract>,
    #[serde(default)]
    pub volumes: Vec<CadVolumeContract>,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadVertexContract {
    pub id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEdgeContract {
    pub id: TopologyEntityId,
    pub vertex_ids: [TopologyEntityId; 2],
    #[serde(default)]
    pub adjacent_face_ids: Vec<TopologyEntityId>,
    #[serde(default)]
    pub imported_handle: Option<String>,
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadFaceContract {
    pub id: TopologyEntityId,
    #[serde(default)]
    pub outer_loop_edge_ids: Vec<TopologyEntityId>,
    #[serde(default)]
    pub hole_loop_edge_ids: Vec<Vec<TopologyEntityId>>,
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub imported_handle: Option<String>,
    pub evaluator: CadEvaluatorCapabilities,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadShellContract {
    pub id: TopologyEntityId,
    #[serde(default)]
    pub face_ids: Vec<TopologyEntityId>,
    pub closed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadVolumeContract {
    pub id: TopologyEntityId,
    #[serde(default)]
    pub shell_ids: Vec<TopologyEntityId>,
    #[serde(default)]
    pub physical_region_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CadEvaluatorCapabilities {
    pub point_evaluation: bool,
    pub projection: bool,
    pub normal: bool,
    pub first_derivatives: bool,
    pub second_derivatives: bool,
    pub curvature: bool,
}
