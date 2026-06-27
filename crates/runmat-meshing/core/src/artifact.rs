use serde::{Deserialize, Serialize};

use crate::{
    adaptive::AdaptiveIterationSummary,
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance},
    quality::AnalysisMeshQualityReport,
    sizing::MeshSizingField,
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub const ANALYSIS_MESH_SCHEMA_VERSION: &str = "analysis-mesh/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMeshNode {
    pub node_id: u32,
    pub coordinates_m: [f64; 3],
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisVolumeElement {
    pub element_id: String,
    pub kind: VolumeElementKind,
    pub node_ids: Vec<u32>,
    pub material_region_id: String,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisBoundaryFace {
    pub face_id: String,
    pub kind: BoundaryElementKind,
    pub node_ids: Vec<u32>,
    #[serde(default)]
    pub adjacent_volume_element_ids: Vec<String>,
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisBoundaryEdge {
    pub edge_id: String,
    pub node_ids: [u32; 2],
    #[serde(default)]
    pub adjacent_boundary_face_ids: Vec<String>,
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBackendSummary {
    pub backend: String,
    pub algorithm: String,
    #[serde(default)]
    pub source_topology_vertex_count: usize,
    #[serde(default)]
    pub source_topology_edge_count: usize,
    #[serde(default)]
    pub source_topology_face_count: usize,
    #[serde(default)]
    pub cad_topology_source: String,
    #[serde(default)]
    pub cad_vertex_count: usize,
    #[serde(default)]
    pub cad_edge_count: usize,
    #[serde(default)]
    pub cad_face_count: usize,
    #[serde(default)]
    pub cad_shell_count: usize,
    #[serde(default)]
    pub cad_volume_count: usize,
    #[serde(default)]
    pub cad_semantic_face_count: usize,
    #[serde(default)]
    pub cad_imported_face_count: usize,
    #[serde(default)]
    pub cad_evaluator_face_count: usize,
    #[serde(default)]
    pub cad_generic_face_count: usize,
    #[serde(default)]
    pub cad_closed_shell_count: usize,
    #[serde(default)]
    pub cad_evaluation_source: String,
    #[serde(default)]
    pub cad_face_frame_count: usize,
    #[serde(default)]
    pub cad_evaluation_evaluator_face_count: usize,
    #[serde(default)]
    pub cad_projection_query_count: usize,
    #[serde(default)]
    pub cad_max_projection_error_m: f64,
    #[serde(default)]
    pub cad_max_normal_deviation: f64,
    #[serde(default)]
    pub curve_element_count: usize,
    #[serde(default)]
    pub surface_element_count: usize,
    #[serde(default)]
    pub surface_source_edge_loop_count: usize,
    #[serde(default)]
    pub surface_closed_edge_loop_count: usize,
    #[serde(default)]
    pub surface_projection_error_m: f64,
    #[serde(default)]
    pub surface_face_coverage_ratio: f64,
    #[serde(default)]
    pub surface_cad_face_count: usize,
    #[serde(default)]
    pub surface_max_cad_projection_error_m: f64,
    #[serde(default)]
    pub volume_candidate_count: usize,
    #[serde(default)]
    pub interior_seed_point_count: usize,
    #[serde(default)]
    pub tet_candidate_count: usize,
    #[serde(default)]
    pub tet_recovered_component_ratio: f64,
    #[serde(default)]
    pub tet_fan_fallback_component_count: usize,
    #[serde(default)]
    pub tet_candidate_volume_ratio: f64,
    #[serde(default)]
    pub tet_refinement_pass_count: usize,
    #[serde(default)]
    pub tet_refinement_point_count: usize,
    #[serde(default)]
    pub tet_max_radius_edge_ratio: f64,
    #[serde(default)]
    pub tet_sizing_violation_count: usize,
    #[serde(default)]
    pub tet_optimization_pass_count: usize,
    #[serde(default)]
    pub tet_smoothed_point_count: usize,
    #[serde(default)]
    pub tet_sliver_candidate_count: usize,
    #[serde(default)]
    pub boundary_face_recovery_ratio: f64,
    #[serde(default)]
    pub boundary_edge_recovery_ratio: f64,
}

impl Default for MeshBackendSummary {
    fn default() -> Self {
        Self {
            backend: "unknown".to_string(),
            algorithm: "unknown".to_string(),
            source_topology_vertex_count: 0,
            source_topology_edge_count: 0,
            source_topology_face_count: 0,
            cad_topology_source: "unknown".to_string(),
            cad_vertex_count: 0,
            cad_edge_count: 0,
            cad_face_count: 0,
            cad_shell_count: 0,
            cad_volume_count: 0,
            cad_semantic_face_count: 0,
            cad_imported_face_count: 0,
            cad_evaluator_face_count: 0,
            cad_generic_face_count: 0,
            cad_closed_shell_count: 0,
            cad_evaluation_source: "unknown".to_string(),
            cad_face_frame_count: 0,
            cad_evaluation_evaluator_face_count: 0,
            cad_projection_query_count: 0,
            cad_max_projection_error_m: 0.0,
            cad_max_normal_deviation: 0.0,
            curve_element_count: 0,
            surface_element_count: 0,
            surface_source_edge_loop_count: 0,
            surface_closed_edge_loop_count: 0,
            surface_projection_error_m: 0.0,
            surface_face_coverage_ratio: 0.0,
            surface_cad_face_count: 0,
            surface_max_cad_projection_error_m: 0.0,
            volume_candidate_count: 0,
            interior_seed_point_count: 0,
            tet_candidate_count: 0,
            tet_recovered_component_ratio: 0.0,
            tet_fan_fallback_component_count: 0,
            tet_candidate_volume_ratio: 0.0,
            tet_refinement_pass_count: 0,
            tet_refinement_point_count: 0,
            tet_max_radius_edge_ratio: 0.0,
            tet_sizing_violation_count: 0,
            tet_optimization_pass_count: 0,
            tet_smoothed_point_count: 0,
            tet_sliver_candidate_count: 0,
            boundary_face_recovery_ratio: 0.0,
            boundary_edge_recovery_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMeshArtifact {
    pub schema_version: String,
    pub mesh_id: String,
    pub nodes: Vec<AnalysisMeshNode>,
    pub volume_elements: Vec<AnalysisVolumeElement>,
    #[serde(default)]
    pub boundary_faces: Vec<AnalysisBoundaryFace>,
    #[serde(default)]
    pub boundary_edges: Vec<AnalysisBoundaryEdge>,
    pub quality: AnalysisMeshQualityReport,
    pub sizing: MeshSizingField,
    #[serde(default)]
    pub backend: MeshBackendSummary,
    #[serde(default)]
    pub adaptive_iterations: Vec<AdaptiveIterationSummary>,
    pub provenance: AnalysisMeshProvenance,
}
