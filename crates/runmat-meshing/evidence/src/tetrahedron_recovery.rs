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
    #[serde(default)]
    pub recovery_item_count: usize,
    #[serde(default)]
    pub recovered_item_count: usize,
    #[serde(default)]
    pub missing_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_boundary_face_count: usize,
    #[serde(default)]
    pub recovered_protected_edge_boundary_face_count: usize,
    #[serde(default)]
    pub volume_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub boundary_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub boundary_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub volume_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub deferred_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub attempted_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub reconnected_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_adjacent_facet_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_adjacent_facet_topology_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_current_boundary_face_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_element_topology_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_material_region_mismatch_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_quality_gate_count: usize,
    #[serde(default)]
    pub recovered_absent_source_edge_boundary_face_count: usize,
    #[serde(default)]
    pub attempted_source_face_diagonal_recovery_pair_count: usize,
    #[serde(default)]
    pub recovered_source_face_diagonal_pair_count: usize,
    #[serde(default)]
    pub recovered_source_face_diagonal_boundary_face_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_recovery_pair_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_adjacent_facet_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_adjacent_facet_topology_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_current_boundary_face_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_element_topology_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_material_region_mismatch_count: usize,
    #[serde(default)]
    pub rejected_source_face_diagonal_quality_gate_count: usize,
    #[serde(default)]
    pub repaired_boundary_face_identity_count: usize,
    #[serde(default)]
    pub removed_redundant_boundary_face_count: usize,
    #[serde(default)]
    pub removed_unsupported_boundary_face_count: usize,
    #[serde(default)]
    pub repaired_source_face_provenance_count: usize,
    #[serde(default)]
    pub repaired_source_edge_provenance_count: usize,
    #[serde(default)]
    pub repaired_material_interface_element_count: usize,
    #[serde(default)]
    pub attempted_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub global_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub boundary_owned_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub interior_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_material_interface_missing_boundary_ownership_count: usize,
    #[serde(default)]
    pub rejected_material_interface_ambiguous_boundary_ownership_count: usize,
    #[serde(default)]
    pub attempted_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub inserted_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub inserted_absent_material_partition_element_count: usize,
    #[serde(default)]
    pub inserted_absent_material_partition_boundary_face_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub rolled_back_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub rolled_back_absent_material_partition_element_count: usize,
    #[serde(default)]
    pub rolled_back_absent_material_partition_boundary_face_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_facet_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_facet_topology_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_element_exists_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_interior_face_topology_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_quality_gate_count: usize,
    #[serde(default)]
    pub rejected_absent_material_partition_post_insertion_audit_count: usize,
    #[serde(default)]
    pub source_face_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_topology_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_provenance_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_boundary_face_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_volume_face_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_absent_face_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_face_recovery_ids: Vec<String>,
    #[serde(default)]
    pub omitted_missing_source_face_recovery_id_count: usize,
    #[serde(default)]
    pub source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_topology_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_provenance_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_volume_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_absent_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_recovery_ids: Vec<String>,
    #[serde(default)]
    pub omitted_missing_source_edge_recovery_id_count: usize,
    #[serde(default)]
    pub material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub missing_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub missing_material_interface_boundary_owned_recovery_item_count: usize,
    #[serde(default)]
    pub missing_material_interface_interior_face_recovery_item_count: usize,
    #[serde(default)]
    pub missing_material_interface_absent_partition_recovery_item_count: usize,
    #[serde(default)]
    pub missing_material_interface_recovery_ids: Vec<String>,
    #[serde(default)]
    pub omitted_missing_material_interface_recovery_id_count: usize,
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
        recovery_item_count: mesh.backend.tetrahedron_recovery_item_count,
        recovered_item_count: mesh.backend.tetrahedron_recovered_item_count,
        missing_recovery_item_count: mesh.backend.tetrahedron_missing_recovery_item_count,
        recovered_boundary_face_count: mesh.backend.tetrahedron_recovered_boundary_face_count,
        recovered_protected_edge_boundary_face_count: mesh
            .backend
            .tetrahedron_recovered_protected_edge_boundary_face_count,
        volume_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_volume_edge_source_edge_recovery_item_count,
        boundary_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_boundary_edge_source_edge_recovery_item_count,
        boundary_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_boundary_face_source_face_recovery_item_count,
        volume_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_volume_face_source_face_recovery_item_count,
        deferred_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_deferred_absent_source_edge_recovery_item_count,
        attempted_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_attempted_absent_source_edge_recovery_item_count,
        reconnected_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_reconnected_absent_source_edge_recovery_item_count,
        rejected_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_recovery_item_count,
        rejected_absent_source_edge_adjacent_facet_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_adjacent_facet_count,
        rejected_absent_source_edge_adjacent_facet_topology_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_adjacent_facet_topology_count,
        rejected_absent_source_edge_current_boundary_face_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_current_boundary_face_count,
        rejected_absent_source_edge_element_topology_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_element_topology_count,
        rejected_absent_source_edge_material_region_mismatch_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_material_region_mismatch_count,
        rejected_absent_source_edge_quality_gate_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_quality_gate_count,
        recovered_absent_source_edge_boundary_face_count: mesh
            .backend
            .tetrahedron_recovered_absent_source_edge_boundary_face_count,
        attempted_source_face_diagonal_recovery_pair_count: mesh
            .backend
            .tetrahedron_attempted_source_face_diagonal_recovery_pair_count,
        recovered_source_face_diagonal_pair_count: mesh
            .backend
            .tetrahedron_recovered_source_face_diagonal_pair_count,
        recovered_source_face_diagonal_boundary_face_count: mesh
            .backend
            .tetrahedron_recovered_source_face_diagonal_boundary_face_count,
        rejected_source_face_diagonal_recovery_pair_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_recovery_pair_count,
        rejected_source_face_diagonal_adjacent_facet_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_adjacent_facet_count,
        rejected_source_face_diagonal_adjacent_facet_topology_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_adjacent_facet_topology_count,
        rejected_source_face_diagonal_current_boundary_face_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_current_boundary_face_count,
        rejected_source_face_diagonal_element_topology_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_element_topology_count,
        rejected_source_face_diagonal_material_region_mismatch_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_material_region_mismatch_count,
        rejected_source_face_diagonal_quality_gate_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_quality_gate_count,
        repaired_boundary_face_identity_count: mesh
            .backend
            .tetrahedron_repaired_boundary_face_identity_count,
        removed_redundant_boundary_face_count: mesh
            .backend
            .tetrahedron_removed_redundant_boundary_face_count,
        removed_unsupported_boundary_face_count: mesh
            .backend
            .tetrahedron_removed_unsupported_boundary_face_count,
        repaired_source_face_provenance_count: mesh
            .backend
            .tetrahedron_repaired_source_face_provenance_count,
        repaired_source_edge_provenance_count: mesh
            .backend
            .tetrahedron_repaired_source_edge_provenance_count,
        repaired_material_interface_element_count: mesh
            .backend
            .tetrahedron_repaired_material_interface_element_count,
        attempted_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_attempted_material_interface_recovery_item_count,
        rejected_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_recovery_item_count,
        global_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_global_material_interface_recovery_item_count,
        boundary_owned_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_boundary_owned_material_interface_recovery_item_count,
        interior_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_interior_material_interface_recovery_item_count,
        rejected_material_interface_missing_boundary_ownership_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_missing_boundary_ownership_count,
        rejected_material_interface_ambiguous_boundary_ownership_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_ambiguous_boundary_ownership_count,
        attempted_absent_material_partition_recovery_item_count: mesh
            .backend
            .tetrahedron_attempted_absent_material_partition_recovery_item_count,
        inserted_absent_material_partition_recovery_item_count: mesh
            .backend
            .tetrahedron_inserted_absent_material_partition_recovery_item_count,
        inserted_absent_material_partition_element_count: mesh
            .backend
            .tetrahedron_inserted_absent_material_partition_element_count,
        inserted_absent_material_partition_boundary_face_count: mesh
            .backend
            .tetrahedron_inserted_absent_material_partition_boundary_face_count,
        rejected_absent_material_partition_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_recovery_item_count,
        rolled_back_absent_material_partition_recovery_item_count: mesh
            .backend
            .tetrahedron_rolled_back_absent_material_partition_recovery_item_count,
        rolled_back_absent_material_partition_element_count: mesh
            .backend
            .tetrahedron_rolled_back_absent_material_partition_element_count,
        rolled_back_absent_material_partition_boundary_face_count: mesh
            .backend
            .tetrahedron_rolled_back_absent_material_partition_boundary_face_count,
        rejected_absent_material_partition_facet_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_facet_count,
        rejected_absent_material_partition_facet_topology_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_facet_topology_count,
        rejected_absent_material_partition_element_exists_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_element_exists_count,
        rejected_absent_material_partition_interior_face_topology_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_interior_face_topology_count,
        rejected_absent_material_partition_quality_gate_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_quality_gate_count,
        rejected_absent_material_partition_post_insertion_audit_count: mesh
            .backend
            .tetrahedron_rejected_absent_material_partition_post_insertion_audit_count,
        source_face_recovery_item_count: mesh.backend.tetrahedron_source_face_recovery_item_count,
        recovered_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_source_face_recovery_item_count,
        missing_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_recovery_item_count,
        missing_source_face_topology_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_topology_recovery_item_count,
        missing_source_face_provenance_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_provenance_recovery_item_count,
        missing_source_face_boundary_face_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_boundary_face_recovery_item_count,
        missing_source_face_volume_face_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_volume_face_recovery_item_count,
        missing_source_face_absent_face_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_absent_face_recovery_item_count,
        missing_source_face_recovery_ids: mesh
            .backend
            .tetrahedron_missing_source_face_recovery_ids
            .clone(),
        omitted_missing_source_face_recovery_id_count: mesh
            .backend
            .tetrahedron_omitted_missing_source_face_recovery_id_count,
        source_edge_recovery_item_count: mesh.backend.tetrahedron_source_edge_recovery_item_count,
        recovered_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_source_edge_recovery_item_count,
        missing_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_edge_recovery_item_count,
        missing_source_edge_topology_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_edge_topology_recovery_item_count,
        missing_source_edge_provenance_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_edge_provenance_recovery_item_count,
        missing_source_edge_volume_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_edge_volume_edge_recovery_item_count,
        missing_source_edge_absent_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_edge_absent_edge_recovery_item_count,
        missing_source_edge_recovery_ids: mesh
            .backend
            .tetrahedron_missing_source_edge_recovery_ids
            .clone(),
        omitted_missing_source_edge_recovery_id_count: mesh
            .backend
            .tetrahedron_omitted_missing_source_edge_recovery_id_count,
        material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_material_interface_recovery_item_count,
        recovered_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_material_interface_recovery_item_count,
        missing_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_material_interface_recovery_item_count,
        missing_material_interface_boundary_owned_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_material_interface_boundary_owned_recovery_item_count,
        missing_material_interface_interior_face_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_material_interface_interior_face_recovery_item_count,
        missing_material_interface_absent_partition_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_material_interface_absent_partition_recovery_item_count,
        missing_material_interface_recovery_ids: mesh
            .backend
            .tetrahedron_missing_material_interface_recovery_ids
            .clone(),
        omitted_missing_material_interface_recovery_id_count: mesh
            .backend
            .tetrahedron_omitted_missing_material_interface_recovery_id_count,
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
