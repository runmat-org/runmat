use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
    #[serde(default)]
    pub plc_input_material_region_count: usize,
    #[serde(default)]
    pub plc_input_material_region_facet_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_boundary_source_edge_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_boundary_segment_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_imported_edge_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_evaluator_edge_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_evaluator_sample_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_live_query_edge_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_live_query_sample_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_rejected_evaluator_sample_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_curvature_sized_edge_count: usize,
    #[serde(default)]
    pub plc_input_cad_curve_curvature_sample_count: usize,
    #[serde(default)]
    pub plc_input_surface_boundary_node_count: usize,
    #[serde(default)]
    pub generation_family: String,
    #[serde(default)]
    pub generation_attempted_family_count: usize,
    #[serde(default)]
    pub generation_rejected_family_count: usize,
    #[serde(default)]
    pub generation_selected_family_index: usize,
    #[serde(default)]
    pub generation_interior_support_candidate_count: usize,
    #[serde(default)]
    pub generation_interior_support_accepted_count: usize,
    #[serde(default)]
    pub generation_nested_shell_outer_node_count: usize,
    #[serde(default)]
    pub generation_nested_shell_inner_node_count: usize,
    #[serde(default)]
    pub generation_nested_shell_generated_node_count: usize,
    #[serde(default)]
    pub generation_nested_shell_refill_boundary_face_count: usize,
    #[serde(default)]
    pub generation_nested_shell_boundary_centroid_refinement_attempt_count: usize,
    #[serde(default)]
    pub generation_nested_shell_boundary_centroid_refinement_rejected_count: usize,
    #[serde(default)]
    pub generation_nested_shell_boundary_exact_cover_refill_count: usize,
    #[serde(default)]
    pub generation_nested_shell_boundary_centroid_refinement_refill_count: usize,
    #[serde(default)]
    pub generation_nested_shell_barycentric_partition_refill_count: usize,
    #[serde(default)]
    pub generation_nested_shell_outer_facet_count: usize,
    #[serde(default)]
    pub generation_nested_shell_inner_facet_count: usize,
    pub element_count: usize,
    #[serde(default)]
    pub material_region_count: usize,
    #[serde(default)]
    pub unclassified_material_element_count: usize,
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
    pub recovered_cad_curve_protected_edge_boundary_face_count: usize,
    #[serde(default)]
    pub attempted_protected_edge_boundary_face_restoration_item_count: usize,
    #[serde(default)]
    pub attempted_cad_curve_protected_edge_boundary_face_restoration_item_count: usize,
    #[serde(default)]
    pub rejected_protected_edge_boundary_face_restoration_item_count: usize,
    #[serde(default)]
    pub rejected_cad_curve_protected_edge_boundary_face_restoration_item_count: usize,
    #[serde(default)]
    pub rejected_protected_edge_boundary_face_restoration_volume_face_topology_count: usize,
    #[serde(default)]
    pub volume_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_volume_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub boundary_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_boundary_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub interior_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_interior_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub cad_curve_interior_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_cad_curve_interior_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub attempted_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub attempted_cad_curve_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub accepted_source_edge_split_refill_candidate_item_count: usize,
    #[serde(default)]
    pub accepted_cad_curve_source_edge_split_refill_candidate_item_count: usize,
    #[serde(default)]
    pub post_repair_attempted_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub post_repair_attempted_cad_curve_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub applied_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub applied_cad_curve_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub post_repair_rejected_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub post_repair_rejected_cad_curve_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub rejected_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub rejected_cad_curve_source_edge_split_refill_item_count: usize,
    #[serde(default)]
    pub absent_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_absent_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub boundary_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_boundary_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub interior_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_interior_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub volume_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_volume_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub attempted_volume_face_source_face_boundary_restoration_item_count: usize,
    #[serde(default)]
    pub rejected_volume_face_source_face_boundary_restoration_item_count: usize,
    #[serde(default)]
    pub rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count: usize,
    #[serde(default)]
    pub absent_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_absent_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub deferred_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub attempted_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub attempted_cad_curve_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub reconnected_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub reconnected_cad_curve_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_cad_curve_absent_source_edge_recovery_item_count: usize,
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
    pub rejected_source_face_diagonal_recovery_item_count: usize,
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
    pub rejected_source_face_diagonal_unpaired_source_face_count: usize,
    #[serde(default)]
    pub repaired_boundary_face_identity_count: usize,
    #[serde(default)]
    pub removed_redundant_boundary_face_count: usize,
    #[serde(default)]
    pub removed_unsupported_boundary_face_count: usize,
    #[serde(default)]
    pub attempted_boundary_leak_recovery_item_count: usize,
    #[serde(default)]
    pub removed_exterior_leaked_element_count: usize,
    #[serde(default)]
    pub exposed_interior_source_face_count: usize,
    #[serde(default)]
    pub inserted_exposed_interior_boundary_face_count: usize,
    #[serde(default)]
    pub rejected_boundary_leak_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_boundary_leak_adjacent_element_count: usize,
    #[serde(default)]
    pub rejected_boundary_leak_material_region_mismatch_count: usize,
    #[serde(default)]
    pub rejected_boundary_leak_outside_classification_count: usize,
    #[serde(default)]
    pub rejected_boundary_leak_closed_surface_coordinate_count: usize,
    #[serde(default)]
    pub repaired_source_face_provenance_count: usize,
    #[serde(default)]
    pub repaired_source_edge_provenance_count: usize,
    #[serde(default)]
    pub repaired_cad_curve_source_edge_provenance_count: usize,
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
    pub recovered_boundary_owned_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub interior_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_interior_face_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_absent_partition_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub rejected_material_interface_missing_boundary_ownership_count: usize,
    #[serde(default)]
    pub rejected_material_interface_missing_interior_ownership_count: usize,
    #[serde(default)]
    pub rejected_material_interface_ambiguous_boundary_ownership_count: usize,
    #[serde(default)]
    pub rejected_material_interface_absent_partition_count: usize,
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
    pub missing_source_face_interior_face_recovery_item_count: usize,
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
    pub missing_source_edge_interior_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_absent_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_source_edge_recovery_ids: Vec<String>,
    #[serde(default)]
    pub omitted_missing_source_edge_recovery_id_count: usize,
    #[serde(default)]
    pub cad_curve_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub recovered_cad_curve_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_cad_curve_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub missing_cad_curve_source_edge_topology_recovery_item_count: usize,
    #[serde(default)]
    pub missing_cad_curve_source_edge_provenance_recovery_item_count: usize,
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
    #[serde(default)]
    pub optimization_budget_limited_count: usize,
    pub smoothed_point_count: usize,
    #[serde(default)]
    pub optimization_interior_smoothing_attempt_count: usize,
    #[serde(default)]
    pub optimization_interior_smoothing_accepted_count: usize,
    #[serde(default)]
    pub optimization_interior_smoothing_rejected_count: usize,
    #[serde(default)]
    pub optimization_interior_smoothing_budget_limited_count: usize,
    #[serde(default)]
    pub optimization_interior_smoothing_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub optimization_boundary_smoothing_attempt_count: usize,
    #[serde(default)]
    pub optimization_boundary_smoothing_accepted_count: usize,
    #[serde(default)]
    pub optimization_boundary_smoothing_rejected_count: usize,
    #[serde(default)]
    pub optimization_boundary_smoothing_budget_limited_count: usize,
    #[serde(default)]
    pub optimization_boundary_smoothing_rejected_by_reason: BTreeMap<String, usize>,
    pub sliver_count: usize,
    #[serde(default)]
    pub sliver_removed_count: usize,
    #[serde(default)]
    pub optimization_sliver_removal_attempt_count: usize,
    #[serde(default)]
    pub optimization_sliver_removal_accepted_count: usize,
    #[serde(default)]
    pub optimization_sliver_removal_rejected_count: usize,
    #[serde(default)]
    pub optimization_sliver_removal_budget_limited_count: usize,
    #[serde(default)]
    pub optimization_sliver_removal_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub optimization_target_seed_count: usize,
    #[serde(default)]
    pub optimization_skipped_target_seed_count: usize,
    #[serde(default)]
    pub optimization_rejected_edit_count: usize,
    #[serde(default)]
    pub optimization_local_reconnection_attempt_count: usize,
    #[serde(default)]
    pub optimization_local_reconnection_accepted_count: usize,
    #[serde(default)]
    pub optimization_local_reconnection_rejected_count: usize,
    #[serde(default)]
    pub optimization_local_reconnection_budget_limited_count: usize,
    #[serde(default)]
    pub optimization_local_reconnection_rejected_by_reason: BTreeMap<String, usize>,
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
