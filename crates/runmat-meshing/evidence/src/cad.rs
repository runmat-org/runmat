use serde::{Deserialize, Serialize};

use runmat_meshing_core::contracts::AnalysisMeshArtifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshCadEvidence {
    pub topology_source: String,
    pub evaluation_source: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub shell_count: usize,
    pub volume_count: usize,
    pub imported_face_count: usize,
    pub evaluator_face_count: usize,
    #[serde(default)]
    pub live_query_face_count: usize,
    pub exact_query_face_count: usize,
    #[serde(default)]
    pub missing_exact_query_face_count: usize,
    #[serde(default)]
    pub missing_derivative_query_face_count: usize,
    #[serde(default)]
    pub missing_curvature_query_face_count: usize,
    #[serde(default)]
    pub point_evaluation_supported_face_count: usize,
    #[serde(default)]
    pub projection_supported_face_count: usize,
    #[serde(default)]
    pub normal_supported_face_count: usize,
    #[serde(default)]
    pub derivative_supported_face_count: usize,
    #[serde(default)]
    pub curvature_supported_face_count: usize,
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub evaluator_rejected_sample_count: usize,
    pub projection_query_count: usize,
    #[serde(default)]
    pub derivative_query_count: usize,
    #[serde(default)]
    pub curvature_query_count: usize,
    #[serde(default)]
    pub uv_domain_face_count: usize,
    #[serde(default)]
    pub uv_projection_out_of_bounds_count: usize,
    pub max_projection_error_m: f64,
    pub max_normal_deviation: f64,
    #[serde(default)]
    pub max_curvature_estimate_1_per_m: f64,
    pub surface_cad_face_count: usize,
    #[serde(default)]
    pub surface_source_edge_loop_count: usize,
    #[serde(default)]
    pub surface_closed_edge_loop_count: usize,
    #[serde(default)]
    pub surface_conforming_source_edge_count: usize,
    #[serde(default)]
    pub surface_missing_source_edge_count: usize,
    #[serde(default)]
    pub surface_exact_cad_sample_node_count: usize,
    #[serde(default)]
    pub surface_rejected_exact_cad_sample_count: usize,
    pub surface_max_projection_error_m: f64,
}

pub(super) fn cad_evidence(mesh: &AnalysisMeshArtifact) -> MeshCadEvidence {
    MeshCadEvidence {
        topology_source: mesh.backend.cad_topology_source.clone(),
        evaluation_source: mesh.backend.cad_evaluation_source.clone(),
        vertex_count: mesh.backend.cad_vertex_count,
        edge_count: mesh.backend.cad_edge_count,
        face_count: mesh.backend.cad_face_count,
        shell_count: mesh.backend.cad_shell_count,
        volume_count: mesh.backend.cad_volume_count,
        imported_face_count: mesh.backend.cad_imported_face_count,
        evaluator_face_count: mesh.backend.cad_evaluation_evaluator_face_count,
        live_query_face_count: mesh.backend.cad_evaluation_live_query_face_count,
        exact_query_face_count: mesh.backend.cad_evaluation_exact_query_face_count,
        missing_exact_query_face_count: mesh.backend.cad_evaluation_missing_exact_query_face_count,
        missing_derivative_query_face_count: mesh
            .backend
            .cad_evaluation_missing_derivative_query_face_count,
        missing_curvature_query_face_count: mesh
            .backend
            .cad_evaluation_missing_curvature_query_face_count,
        point_evaluation_supported_face_count: mesh
            .backend
            .cad_evaluation_point_supported_face_count,
        projection_supported_face_count: mesh
            .backend
            .cad_evaluation_projection_supported_face_count,
        normal_supported_face_count: mesh.backend.cad_evaluation_normal_supported_face_count,
        derivative_supported_face_count: mesh
            .backend
            .cad_evaluation_derivative_supported_face_count,
        curvature_supported_face_count: mesh.backend.cad_evaluation_curvature_supported_face_count,
        evaluator_sample_count: mesh.backend.cad_evaluation_sample_count,
        evaluator_rejected_sample_count: mesh.backend.cad_evaluation_rejected_sample_count,
        projection_query_count: mesh.backend.cad_projection_query_count,
        derivative_query_count: mesh.backend.cad_derivative_query_count,
        curvature_query_count: mesh.backend.cad_curvature_query_count,
        uv_domain_face_count: mesh.backend.cad_uv_domain_face_count,
        uv_projection_out_of_bounds_count: mesh.backend.cad_uv_projection_out_of_bounds_count,
        max_projection_error_m: mesh.backend.cad_max_projection_error_m,
        max_normal_deviation: mesh.backend.cad_max_normal_deviation,
        max_curvature_estimate_1_per_m: mesh.backend.cad_max_curvature_estimate_1_per_m,
        surface_cad_face_count: mesh.backend.surface_cad_face_count,
        surface_source_edge_loop_count: mesh.backend.surface_source_edge_loop_count,
        surface_closed_edge_loop_count: mesh.backend.surface_closed_edge_loop_count,
        surface_conforming_source_edge_count: mesh.backend.surface_conforming_source_edge_count,
        surface_missing_source_edge_count: mesh.backend.surface_missing_source_edge_count,
        surface_exact_cad_sample_node_count: mesh.backend.surface_exact_cad_sample_node_count,
        surface_rejected_exact_cad_sample_count: mesh
            .backend
            .surface_rejected_exact_cad_sample_count,
        surface_max_projection_error_m: mesh.backend.surface_max_cad_projection_error_m,
    }
}
