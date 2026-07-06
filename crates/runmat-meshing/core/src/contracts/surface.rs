use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub mesh_id: String,
    #[serde(default)]
    pub nodes: Vec<SurfaceMeshNode>,
    #[serde(default)]
    pub triangles: Vec<SurfaceMeshTriangle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve_boundary_validation: Option<SurfaceCurveBoundaryValidation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loop_coverage: Option<SurfaceLoopCoverage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cad_curve_boundary_provenance: Option<SurfaceCadCurveBoundaryProvenance>,
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
    pub source_edge_ids: [Option<TopologyEntityId>; 3],
    pub node_ids: [TopologyEntityId; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub material_region_ids: Vec<String>,
    #[serde(default)]
    pub max_projection_error_m: f64,
    pub area_m2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCurveBoundaryValidation {
    pub source_edge_count: usize,
    pub curve_node_count: usize,
    pub curve_element_count: usize,
    pub max_endpoint_error_m: f64,
    pub max_projection_error_m: f64,
    pub max_length_error_m: f64,
    pub max_segment_length_m: f64,
    pub max_parameter_gap: f64,
    pub max_adjacent_length_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceLoopCoverage {
    pub source_face_count: usize,
    pub recovered_face_count: usize,
    pub boundary_loop_count: usize,
    #[serde(default)]
    pub hole_loop_count: usize,
    #[serde(default)]
    pub boundary_node_count: usize,
    pub recovered_source_edge_count: usize,
    pub boundary_segment_count: usize,
    pub max_loops_per_face: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCadCurveBoundaryProvenance {
    pub recovered_source_edge_count: usize,
    pub boundary_segment_count: usize,
    pub imported_curve_edge_count: usize,
    pub evaluator_curve_edge_count: usize,
    pub evaluator_sample_count: usize,
    pub live_query_edge_count: usize,
    pub live_query_sample_count: usize,
    pub rejected_evaluator_sample_count: usize,
    pub curvature_sized_edge_count: usize,
    pub curvature_sample_count: usize,
    #[serde(default)]
    pub edges: Vec<SurfaceCadCurveBoundaryEdgeProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCadCurveBoundaryEdgeProvenance {
    pub source_edge_id: TopologyEntityId,
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
