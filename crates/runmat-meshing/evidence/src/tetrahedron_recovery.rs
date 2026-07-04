use serde::{Deserialize, Serialize};

use runmat_meshing_core::contracts::AnalysisMeshArtifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshTetrahedronRecoveryEvidence {
    #[serde(default)]
    pub plc_input_node_count: usize,
    #[serde(default)]
    pub plc_input_facet_count: usize,
    #[serde(default)]
    pub plc_input_protected_edge_count: usize,
    #[serde(default)]
    pub plc_input_boundary_component_count: usize,
    #[serde(default)]
    pub plc_input_boundary_component_node_count: usize,
    #[serde(default)]
    pub plc_input_max_boundary_component_node_count: usize,
    #[serde(default)]
    pub plc_input_shell_nesting_classified: bool,
    #[serde(default)]
    pub plc_input_outer_shell_count: usize,
    #[serde(default)]
    pub plc_input_nested_shell_count: usize,
    #[serde(default)]
    pub plc_input_max_shell_nesting_depth: usize,
    pub element_count: usize,
    pub recovered_component_ratio: f64,
    pub unrecovered_tetrahedron_component_count: usize,
    pub volume_coverage_ratio: f64,
    pub refinement_pass_count: usize,
    pub refinement_point_count: usize,
    pub optimization_pass_count: usize,
    pub smoothed_point_count: usize,
    pub sliver_count: usize,
    #[serde(default)]
    pub sliver_removed_count: usize,
    #[serde(default)]
    pub optimization_target_seed_count: usize,
    #[serde(default)]
    pub optimization_skipped_target_seed_count: usize,
    #[serde(default)]
    pub optimization_rejected_edit_count: usize,
    #[serde(default)]
    pub optimization_initial_max_aspect_ratio: f64,
    #[serde(default)]
    pub optimization_final_max_aspect_ratio: f64,
    #[serde(default)]
    pub optimization_initial_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub optimization_final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub untangling_pass_count: usize,
    #[serde(default)]
    pub untangling_initial_near_singular_count: usize,
    #[serde(default)]
    pub untangling_final_near_singular_count: usize,
    #[serde(default)]
    pub untangling_relocated_seed_count: usize,
    #[serde(default)]
    pub untangling_reconnected_edge_star_count: usize,
    #[serde(default)]
    pub untangling_reconnected_boundary_adjacent_cavity_count: usize,
    #[serde(default)]
    pub untangling_reconnected_node_adjacent_cavity_count: usize,
    pub exact_quality_repair_pass_count: usize,
    pub exact_quality_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_reconnection_quality_gain_count: usize,
    #[serde(default)]
    pub exact_quality_face_neighbor_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_node_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_boundary_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_expanded_connected_reconnected_cavity_count: usize,
    pub exact_quality_split_cavity_count: usize,
    pub exact_quality_seed_star_collapse_count: usize,
    #[serde(default)]
    pub exact_quality_seed_star_relocation_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_total_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_general_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_boundary_adjacent_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_node_adjacent_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_interior_seed_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_edge_star_count: usize,
}

pub(super) fn tetrahedron_recovery_evidence(
    mesh: &AnalysisMeshArtifact,
) -> MeshTetrahedronRecoveryEvidence {
    MeshTetrahedronRecoveryEvidence {
        plc_input_node_count: mesh.backend.plc_input_node_count,
        plc_input_facet_count: mesh.backend.plc_input_facet_count,
        plc_input_protected_edge_count: mesh.backend.plc_input_protected_edge_count,
        plc_input_boundary_component_count: mesh.backend.plc_input_boundary_component_count,
        plc_input_boundary_component_node_count: mesh
            .backend
            .plc_input_boundary_component_node_count,
        plc_input_max_boundary_component_node_count: mesh
            .backend
            .plc_input_max_boundary_component_node_count,
        plc_input_shell_nesting_classified: mesh.backend.plc_input_shell_nesting_classified,
        plc_input_outer_shell_count: mesh.backend.plc_input_outer_shell_count,
        plc_input_nested_shell_count: mesh.backend.plc_input_nested_shell_count,
        plc_input_max_shell_nesting_depth: mesh.backend.plc_input_max_shell_nesting_depth,
        element_count: mesh.backend.tetrahedron_element_count,
        recovered_component_ratio: mesh.backend.tetrahedron_recovered_component_ratio,
        unrecovered_tetrahedron_component_count: mesh
            .backend
            .tetrahedron_unrecovered_component_count,
        volume_coverage_ratio: mesh.backend.tetrahedron_volume_coverage_ratio,
        refinement_pass_count: mesh.backend.tetrahedron_refinement_pass_count,
        refinement_point_count: mesh.backend.tetrahedron_refinement_point_count,
        optimization_pass_count: mesh.backend.tetrahedron_optimization_pass_count,
        smoothed_point_count: mesh.backend.tetrahedron_smoothed_point_count,
        sliver_count: mesh.backend.tetrahedron_sliver_count,
        sliver_removed_count: mesh.backend.tetrahedron_sliver_removed_count,
        optimization_target_seed_count: mesh.backend.tetrahedron_optimization_target_seed_count,
        optimization_skipped_target_seed_count: mesh
            .backend
            .tetrahedron_optimization_skipped_target_seed_count,
        optimization_rejected_edit_count: mesh.backend.tetrahedron_optimization_rejected_edit_count,
        optimization_initial_max_aspect_ratio: mesh
            .backend
            .tetrahedron_optimization_initial_max_aspect_ratio,
        optimization_final_max_aspect_ratio: mesh
            .backend
            .tetrahedron_optimization_final_max_aspect_ratio,
        optimization_initial_min_exact_scaled_jacobian: mesh
            .backend
            .tetrahedron_optimization_initial_min_exact_scaled_jacobian,
        optimization_final_min_exact_scaled_jacobian: mesh
            .backend
            .tetrahedron_optimization_final_min_exact_scaled_jacobian,
        untangling_pass_count: mesh.backend.tetrahedron_untangling_pass_count,
        untangling_initial_near_singular_count: mesh
            .backend
            .tetrahedron_untangling_initial_near_singular_count,
        untangling_final_near_singular_count: mesh
            .backend
            .tetrahedron_untangling_final_near_singular_count,
        untangling_relocated_seed_count: mesh.backend.tetrahedron_untangling_relocated_seed_count,
        untangling_reconnected_edge_star_count: mesh
            .backend
            .tetrahedron_untangling_reconnected_edge_star_count,
        untangling_reconnected_boundary_adjacent_cavity_count: mesh
            .backend
            .tetrahedron_untangling_reconnected_boundary_adjacent_cavity_count,
        untangling_reconnected_node_adjacent_cavity_count: mesh
            .backend
            .tetrahedron_untangling_reconnected_node_adjacent_cavity_count,
        exact_quality_repair_pass_count: mesh.backend.tetrahedron_exact_quality_repair_pass_count,
        exact_quality_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_reconnected_cavity_count,
        exact_quality_reconnection_quality_gain_count: mesh
            .backend
            .tetrahedron_exact_quality_reconnection_quality_gain_count,
        exact_quality_face_neighbor_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_face_neighbor_reconnected_cavity_count,
        exact_quality_connected_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_connected_reconnected_cavity_count,
        exact_quality_node_adjacent_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_node_adjacent_reconnected_cavity_count,
        exact_quality_boundary_adjacent_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_boundary_adjacent_reconnected_cavity_count,
        exact_quality_expanded_connected_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_expanded_connected_reconnected_cavity_count,
        exact_quality_split_cavity_count: mesh.backend.tetrahedron_exact_quality_split_cavity_count,
        exact_quality_seed_star_collapse_count: mesh
            .backend
            .tetrahedron_exact_quality_seed_star_collapse_count,
        exact_quality_seed_star_relocation_count: mesh
            .backend
            .tetrahedron_exact_quality_seed_star_relocation_count,
        exact_quality_unrepaired_total_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_total_count,
        exact_quality_unrepaired_general_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_general_cavity_count,
        exact_quality_unrepaired_boundary_adjacent_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_boundary_adjacent_count,
        exact_quality_unrepaired_node_adjacent_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_node_adjacent_count,
        exact_quality_unrepaired_interior_seed_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_interior_seed_count,
        exact_quality_unrepaired_edge_star_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_edge_star_count,
    }
}
