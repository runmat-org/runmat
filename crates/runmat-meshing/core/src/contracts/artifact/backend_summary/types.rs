use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
    pub cad_evaluation_point_supported_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_projection_supported_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_normal_supported_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_derivative_supported_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_curvature_supported_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_missing_exact_query_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_missing_derivative_query_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_missing_curvature_query_face_count: usize,
    #[serde(default)]
    pub cad_evaluation_sample_count: usize,
    #[serde(default)]
    pub cad_evaluation_rejected_sample_count: usize,
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
    pub surface_conforming_source_edge_count: usize,
    #[serde(default)]
    pub surface_missing_source_edge_count: usize,
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
    pub volume_component_count: usize,
    #[serde(default)]
    pub interior_seed_point_count: usize,
    #[serde(default)]
    pub tetrahedron_element_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_component_ratio: f64,
    #[serde(default)]
    pub tetrahedron_unrecovered_component_count: usize,
    #[serde(default)]
    pub tetrahedron_volume_coverage_ratio: f64,
    #[serde(default)]
    pub tetrahedron_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_protected_edge_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_protected_edge_boundary_face_restoration_volume_face_topology_count:
        usize,
    #[serde(default)]
    pub tetrahedron_volume_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_boundary_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_interior_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_absent_edge_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_boundary_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_interior_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_volume_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count:
        usize,
    #[serde(default)]
    pub tetrahedron_absent_face_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_deferred_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_reconnected_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_adjacent_facet_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_adjacent_facet_topology_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_current_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_element_topology_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_material_region_mismatch_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_source_edge_quality_gate_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_absent_source_edge_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_source_face_diagonal_recovery_pair_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_source_face_diagonal_pair_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_source_face_diagonal_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_recovery_pair_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_adjacent_facet_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_adjacent_facet_topology_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_current_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_element_topology_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_material_region_mismatch_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_quality_gate_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_source_face_diagonal_unpaired_source_face_count: usize,
    #[serde(default)]
    pub tetrahedron_repaired_boundary_face_identity_count: usize,
    #[serde(default)]
    pub tetrahedron_removed_redundant_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_removed_unsupported_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_boundary_leak_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_removed_exterior_leaked_element_count: usize,
    #[serde(default)]
    pub tetrahedron_exposed_interior_source_face_count: usize,
    #[serde(default)]
    pub tetrahedron_inserted_exposed_interior_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_boundary_leak_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_boundary_leak_adjacent_element_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_boundary_leak_material_region_mismatch_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_boundary_leak_outside_classification_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_boundary_leak_closed_surface_coordinate_count: usize,
    #[serde(default)]
    pub tetrahedron_repaired_source_face_provenance_count: usize,
    #[serde(default)]
    pub tetrahedron_repaired_source_edge_provenance_count: usize,
    #[serde(default)]
    pub tetrahedron_repaired_material_interface_element_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_global_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_boundary_owned_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_interior_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_material_interface_missing_boundary_ownership_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_material_interface_ambiguous_boundary_ownership_count: usize,
    #[serde(default)]
    pub tetrahedron_attempted_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_inserted_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_inserted_absent_material_partition_element_count: usize,
    #[serde(default)]
    pub tetrahedron_inserted_absent_material_partition_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rolled_back_absent_material_partition_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_rolled_back_absent_material_partition_element_count: usize,
    #[serde(default)]
    pub tetrahedron_rolled_back_absent_material_partition_boundary_face_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_facet_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_facet_topology_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_element_exists_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_interior_face_topology_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_quality_gate_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_absent_material_partition_post_insertion_audit_count: usize,
    #[serde(default)]
    pub tetrahedron_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_topology_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_provenance_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_boundary_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_volume_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_interior_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_absent_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_face_recovery_ids: Vec<String>,
    #[serde(default)]
    pub tetrahedron_omitted_missing_source_face_recovery_id_count: usize,
    #[serde(default)]
    pub tetrahedron_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_topology_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_provenance_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_volume_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_interior_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_absent_edge_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_source_edge_recovery_ids: Vec<String>,
    #[serde(default)]
    pub tetrahedron_omitted_missing_source_edge_recovery_id_count: usize,
    #[serde(default)]
    pub tetrahedron_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_recovered_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_material_interface_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_material_interface_boundary_owned_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_material_interface_interior_face_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_material_interface_absent_partition_recovery_item_count: usize,
    #[serde(default)]
    pub tetrahedron_missing_material_interface_recovery_ids: Vec<String>,
    #[serde(default)]
    pub tetrahedron_omitted_missing_material_interface_recovery_id_count: usize,
    #[serde(default)]
    pub tetrahedron_refinement_pass_count: usize,
    #[serde(default)]
    pub tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub tetrahedron_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tetrahedron_accepted_requested_refinement_location_count: usize,
    #[serde(default)]
    pub tetrahedron_accepted_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tetrahedron_accepted_requested_refinement_surrogate_point_count: usize,
    #[serde(default)]
    pub tetrahedron_rejected_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tetrahedron_requested_refinement_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub tetrahedron_dropped_requested_refinement_point_count: usize,
    #[serde(default)]
    pub tetrahedron_requested_refinement_dropped_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub tetrahedron_max_radius_edge_ratio: f64,
    #[serde(default)]
    pub tetrahedron_sizing_violation_count: usize,
    #[serde(default)]
    pub tetrahedron_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub tetrahedron_exact_scaled_jacobian_below_threshold_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    #[serde(default)]
    pub tetrahedron_optimization_pass_count: usize,
    #[serde(default)]
    pub tetrahedron_smoothed_point_count: usize,
    #[serde(default)]
    pub tetrahedron_sliver_count: usize,
    #[serde(default)]
    pub tetrahedron_sliver_removed_count: usize,
    #[serde(default)]
    pub tetrahedron_optimization_target_seed_count: usize,
    #[serde(default)]
    pub tetrahedron_optimization_skipped_target_seed_count: usize,
    #[serde(default)]
    pub tetrahedron_optimization_rejected_edit_count: usize,
    #[serde(default)]
    pub tetrahedron_optimization_initial_max_aspect_ratio: f64,
    #[serde(default)]
    pub tetrahedron_optimization_final_max_aspect_ratio: f64,
    #[serde(default)]
    pub tetrahedron_optimization_initial_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub tetrahedron_optimization_final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub tetrahedron_untangling_pass_count: usize,
    #[serde(default)]
    pub tetrahedron_untangling_initial_near_singular_count: usize,
    #[serde(default)]
    pub tetrahedron_untangling_final_near_singular_count: usize,
    #[serde(default)]
    pub tetrahedron_untangling_relocated_seed_count: usize,
    #[serde(default)]
    pub tetrahedron_untangling_reconnected_edge_star_count: usize,
    #[serde(default)]
    pub tetrahedron_untangling_reconnected_boundary_adjacent_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_untangling_reconnected_node_adjacent_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_repair_pass_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_reconnection_quality_gain_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_face_neighbor_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_node_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_boundary_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_expanded_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_split_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_seed_star_collapse_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_seed_star_relocation_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_unrepaired_total_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_unrepaired_general_cavity_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_unrepaired_boundary_adjacent_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_unrepaired_node_adjacent_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_unrepaired_interior_seed_count: usize,
    #[serde(default)]
    pub tetrahedron_exact_quality_unrepaired_edge_star_count: usize,
    #[serde(default)]
    pub boundary_face_recovery_ratio: f64,
    #[serde(default)]
    pub boundary_edge_recovery_ratio: f64,
}
