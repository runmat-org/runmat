use std::collections::BTreeMap;

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
    pub cad_evaluation_live_query_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_exact_query_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_sample_count: usize,
    #[serde(default)]
    pub cad_projection_query_count: usize,
    #[serde(default)]
    pub cad_derivative_query_count: usize,
    #[serde(default)]
    pub cad_curvature_query_count: usize,
    #[serde(default)]
    pub cad_uv_domain_face_count: usize,
    #[serde(default)]
    pub cad_uv_projection_out_of_bounds_count: usize,
    #[serde(default)]
    pub cad_max_projection_error_m: f64,
    #[serde(default)]
    pub cad_max_normal_deviation: f64,
    #[serde(default)]
    pub cad_max_curvature_estimate_1_per_m: f64,
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
    pub surface_exact_cad_sample_node_count: usize,
    #[serde(default)]
    pub surface_rejected_exact_cad_sample_count: usize,
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
    pub tet_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tet_accepted_requested_refinement_candidate_count: usize,
    #[serde(default)]
    pub tet_accepted_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tet_accepted_requested_refinement_surrogate_point_count: usize,
    #[serde(default)]
    pub tet_rejected_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tet_dropped_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tet_max_radius_edge_ratio: f64,
    #[serde(default)]
    pub tet_sizing_violation_count: usize,
    #[serde(default)]
    pub tet_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub tet_exact_scaled_jacobian_below_threshold_count: usize,
    #[serde(default)]
    pub tet_exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    #[serde(default)]
    pub tet_optimization_pass_count: usize,
    #[serde(default)]
    pub tet_smoothed_point_count: usize,
    #[serde(default)]
    pub tet_sliver_candidate_count: usize,
    #[serde(default)]
    pub tet_sliver_removed_count: usize,
    #[serde(default)]
    pub tet_optimization_target_seed_count: usize,
    #[serde(default)]
    pub tet_optimization_skipped_target_seed_count: usize,
    #[serde(default)]
    pub tet_optimization_rejected_edit_count: usize,
    #[serde(default)]
    pub tet_optimization_initial_max_aspect_ratio: f64,
    #[serde(default)]
    pub tet_optimization_final_max_aspect_ratio: f64,
    #[serde(default)]
    pub tet_optimization_initial_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub tet_optimization_final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub tet_untangling_pass_count: usize,
    #[serde(default)]
    pub tet_untangling_initial_near_singular_count: usize,
    #[serde(default)]
    pub tet_untangling_final_near_singular_count: usize,
    #[serde(default)]
    pub tet_untangling_relocated_seed_count: usize,
    #[serde(default)]
    pub tet_untangling_reconnected_edge_star_count: usize,
    #[serde(default)]
    pub tet_untangling_reconnected_boundary_adjacent_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_repair_pass_count: usize,
    #[serde(default)]
    pub tet_exact_quality_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_reconnection_quality_gain_count: usize,
    #[serde(default)]
    pub tet_exact_quality_face_neighbor_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_boundary_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_expanded_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_split_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_seed_star_collapse_count: usize,
    #[serde(default)]
    pub tet_exact_quality_seed_star_relocation_count: usize,
    #[serde(default)]
    pub tet_exact_quality_unrepaired_total_count: usize,
    #[serde(default)]
    pub tet_exact_quality_unrepaired_general_cavity_count: usize,
    #[serde(default)]
    pub tet_exact_quality_unrepaired_boundary_adjacent_count: usize,
    #[serde(default)]
    pub tet_exact_quality_unrepaired_interior_seed_count: usize,
    #[serde(default)]
    pub tet_exact_quality_unrepaired_edge_star_count: usize,
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
            cad_evaluation_live_query_face_count: 0,
            cad_evaluation_exact_query_face_count: 0,
            cad_evaluation_sample_count: 0,
            cad_projection_query_count: 0,
            cad_derivative_query_count: 0,
            cad_curvature_query_count: 0,
            cad_uv_domain_face_count: 0,
            cad_uv_projection_out_of_bounds_count: 0,
            cad_max_projection_error_m: 0.0,
            cad_max_normal_deviation: 0.0,
            cad_max_curvature_estimate_1_per_m: 0.0,
            curve_element_count: 0,
            surface_element_count: 0,
            surface_source_edge_loop_count: 0,
            surface_closed_edge_loop_count: 0,
            surface_projection_error_m: 0.0,
            surface_face_coverage_ratio: 0.0,
            surface_cad_face_count: 0,
            surface_exact_cad_sample_node_count: 0,
            surface_rejected_exact_cad_sample_count: 0,
            surface_max_cad_projection_error_m: 0.0,
            volume_candidate_count: 0,
            interior_seed_point_count: 0,
            tet_candidate_count: 0,
            tet_recovered_component_ratio: 0.0,
            tet_fan_fallback_component_count: 0,
            tet_candidate_volume_ratio: 0.0,
            tet_refinement_pass_count: 0,
            tet_refinement_point_count: 0,
            tet_requested_refinement_point_count: 0,
            tet_accepted_requested_refinement_candidate_count: 0,
            tet_accepted_requested_refinement_point_count: 0,
            tet_accepted_requested_refinement_surrogate_point_count: 0,
            tet_rejected_requested_refinement_point_count: 0,
            tet_dropped_requested_refinement_point_count: 0,
            tet_max_radius_edge_ratio: 0.0,
            tet_sizing_violation_count: 0,
            tet_min_exact_scaled_jacobian: 1.0,
            tet_exact_scaled_jacobian_below_threshold_count: 0,
            tet_exact_scaled_jacobian_bins: BTreeMap::new(),
            tet_optimization_pass_count: 0,
            tet_smoothed_point_count: 0,
            tet_sliver_candidate_count: 0,
            tet_sliver_removed_count: 0,
            tet_optimization_target_seed_count: 0,
            tet_optimization_skipped_target_seed_count: 0,
            tet_optimization_rejected_edit_count: 0,
            tet_optimization_initial_max_aspect_ratio: 0.0,
            tet_optimization_final_max_aspect_ratio: 0.0,
            tet_optimization_initial_min_exact_scaled_jacobian: 0.0,
            tet_optimization_final_min_exact_scaled_jacobian: 0.0,
            tet_untangling_pass_count: 0,
            tet_untangling_initial_near_singular_count: 0,
            tet_untangling_final_near_singular_count: 0,
            tet_untangling_relocated_seed_count: 0,
            tet_untangling_reconnected_edge_star_count: 0,
            tet_untangling_reconnected_boundary_adjacent_cavity_count: 0,
            tet_exact_quality_repair_pass_count: 0,
            tet_exact_quality_reconnected_cavity_count: 0,
            tet_exact_quality_reconnection_quality_gain_count: 0,
            tet_exact_quality_face_neighbor_reconnected_cavity_count: 0,
            tet_exact_quality_connected_reconnected_cavity_count: 0,
            tet_exact_quality_boundary_adjacent_reconnected_cavity_count: 0,
            tet_exact_quality_expanded_connected_reconnected_cavity_count: 0,
            tet_exact_quality_split_cavity_count: 0,
            tet_exact_quality_seed_star_collapse_count: 0,
            tet_exact_quality_seed_star_relocation_count: 0,
            tet_exact_quality_unrepaired_total_count: 0,
            tet_exact_quality_unrepaired_general_cavity_count: 0,
            tet_exact_quality_unrepaired_boundary_adjacent_count: 0,
            tet_exact_quality_unrepaired_interior_seed_count: 0,
            tet_exact_quality_unrepaired_edge_star_count: 0,
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
