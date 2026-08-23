use runmat_meshing_core::contracts::AnalysisMeshArtifact;

use super::MeshTetrahedronRecoveryEvidence;

pub(crate) fn tetrahedron_recovery_evidence(
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
        plc_input_material_region_count: mesh.backend.plc_input_material_region_count,
        plc_input_material_region_facet_count: mesh.backend.plc_input_material_region_facet_count,
        plc_input_cad_curve_boundary_source_edge_count: mesh
            .backend
            .plc_input_cad_curve_boundary_source_edge_count,
        plc_input_cad_curve_boundary_segment_count: mesh
            .backend
            .plc_input_cad_curve_boundary_segment_count,
        plc_input_cad_curve_imported_edge_count: mesh
            .backend
            .plc_input_cad_curve_imported_edge_count,
        plc_input_cad_curve_evaluator_edge_count: mesh
            .backend
            .plc_input_cad_curve_evaluator_edge_count,
        plc_input_cad_curve_evaluator_sample_count: mesh
            .backend
            .plc_input_cad_curve_evaluator_sample_count,
        plc_input_cad_curve_live_query_edge_count: mesh
            .backend
            .plc_input_cad_curve_live_query_edge_count,
        plc_input_cad_curve_live_query_sample_count: mesh
            .backend
            .plc_input_cad_curve_live_query_sample_count,
        plc_input_cad_curve_rejected_evaluator_sample_count: mesh
            .backend
            .plc_input_cad_curve_rejected_evaluator_sample_count,
        plc_input_cad_curve_curvature_sized_edge_count: mesh
            .backend
            .plc_input_cad_curve_curvature_sized_edge_count,
        plc_input_cad_curve_curvature_sample_count: mesh
            .backend
            .plc_input_cad_curve_curvature_sample_count,
        plc_input_surface_boundary_node_count: mesh.backend.plc_input_surface_boundary_node_count,
        element_count: mesh.backend.tetrahedron_element_count,
        material_region_count: mesh.backend.tetrahedron_material_region_count,
        unclassified_material_element_count: mesh
            .backend
            .tetrahedron_unclassified_material_element_count,
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
        recovered_cad_curve_protected_edge_boundary_face_count: mesh
            .backend
            .tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count,
        attempted_protected_edge_boundary_face_restoration_item_count: mesh
            .backend
            .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count,
        attempted_cad_curve_protected_edge_boundary_face_restoration_item_count: mesh
            .backend
            .tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count,
        rejected_protected_edge_boundary_face_restoration_item_count: mesh
            .backend
            .tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count,
        rejected_cad_curve_protected_edge_boundary_face_restoration_item_count: mesh
            .backend
            .tetrahedron_rejected_cad_curve_protected_edge_boundary_face_restoration_item_count,
        rejected_protected_edge_boundary_face_restoration_volume_face_topology_count: mesh
            .backend
            .tetrahedron_rejected_protected_edge_boundary_face_restoration_volume_face_topology_count,
        volume_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_volume_edge_source_edge_recovery_item_count,
        recovered_volume_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_volume_edge_source_edge_recovery_item_count,
        boundary_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_boundary_edge_source_edge_recovery_item_count,
        recovered_boundary_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_boundary_edge_source_edge_recovery_item_count,
        interior_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_interior_edge_source_edge_recovery_item_count,
        recovered_interior_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_interior_edge_source_edge_recovery_item_count,
        cad_curve_interior_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count,
        recovered_cad_curve_interior_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count,
        attempted_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_attempted_source_edge_split_refill_item_count,
        attempted_cad_curve_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count,
        accepted_source_edge_split_refill_candidate_item_count: mesh
            .backend
            .tetrahedron_accepted_source_edge_split_refill_candidate_item_count,
        accepted_cad_curve_source_edge_split_refill_candidate_item_count: mesh
            .backend
            .tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count,
        post_repair_attempted_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_post_repair_attempted_source_edge_split_refill_item_count,
        post_repair_attempted_cad_curve_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count,
        applied_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_applied_source_edge_split_refill_item_count,
        applied_cad_curve_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_applied_cad_curve_source_edge_split_refill_item_count,
        post_repair_rejected_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_post_repair_rejected_source_edge_split_refill_item_count,
        post_repair_rejected_cad_curve_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_post_repair_rejected_cad_curve_source_edge_split_refill_item_count,
        rejected_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_rejected_source_edge_split_refill_item_count,
        rejected_cad_curve_source_edge_split_refill_item_count: mesh
            .backend
            .tetrahedron_rejected_cad_curve_source_edge_split_refill_item_count,
        absent_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_absent_edge_source_edge_recovery_item_count,
        recovered_absent_edge_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_absent_edge_source_edge_recovery_item_count,
        boundary_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_boundary_face_source_face_recovery_item_count,
        recovered_boundary_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_boundary_face_source_face_recovery_item_count,
        interior_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_interior_face_source_face_recovery_item_count,
        recovered_interior_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_interior_face_source_face_recovery_item_count,
        volume_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_volume_face_source_face_recovery_item_count,
        recovered_volume_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_volume_face_source_face_recovery_item_count,
        attempted_volume_face_source_face_boundary_restoration_item_count: mesh
            .backend
            .tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count,
        rejected_volume_face_source_face_boundary_restoration_item_count: mesh
            .backend
            .tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count,
        rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count: mesh
            .backend
            .tetrahedron_rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count,
        absent_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_absent_face_source_face_recovery_item_count,
        recovered_absent_face_source_face_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_absent_face_source_face_recovery_item_count,
        deferred_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_deferred_absent_source_edge_recovery_item_count,
        attempted_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_attempted_absent_source_edge_recovery_item_count,
        attempted_cad_curve_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count,
        reconnected_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_reconnected_absent_source_edge_recovery_item_count,
        reconnected_cad_curve_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count,
        rejected_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_absent_source_edge_recovery_item_count,
        rejected_cad_curve_absent_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_cad_curve_absent_source_edge_recovery_item_count,
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
        rejected_source_face_diagonal_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_recovery_item_count,
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
        rejected_source_face_diagonal_unpaired_source_face_count: mesh
            .backend
            .tetrahedron_rejected_source_face_diagonal_unpaired_source_face_count,
        repaired_boundary_face_identity_count: mesh
            .backend
            .tetrahedron_repaired_boundary_face_identity_count,
        removed_redundant_boundary_face_count: mesh
            .backend
            .tetrahedron_removed_redundant_boundary_face_count,
        removed_unsupported_boundary_face_count: mesh
            .backend
            .tetrahedron_removed_unsupported_boundary_face_count,
        attempted_boundary_leak_recovery_item_count: mesh
            .backend
            .tetrahedron_attempted_boundary_leak_recovery_item_count,
        removed_exterior_leaked_element_count: mesh
            .backend
            .tetrahedron_removed_exterior_leaked_element_count,
        exposed_interior_source_face_count: mesh
            .backend
            .tetrahedron_exposed_interior_source_face_count,
        inserted_exposed_interior_boundary_face_count: mesh
            .backend
            .tetrahedron_inserted_exposed_interior_boundary_face_count,
        rejected_boundary_leak_recovery_item_count: mesh
            .backend
            .tetrahedron_rejected_boundary_leak_recovery_item_count,
        rejected_boundary_leak_adjacent_element_count: mesh
            .backend
            .tetrahedron_rejected_boundary_leak_adjacent_element_count,
        rejected_boundary_leak_material_region_mismatch_count: mesh
            .backend
            .tetrahedron_rejected_boundary_leak_material_region_mismatch_count,
        rejected_boundary_leak_outside_classification_count: mesh
            .backend
            .tetrahedron_rejected_boundary_leak_outside_classification_count,
        rejected_boundary_leak_closed_surface_coordinate_count: mesh
            .backend
            .tetrahedron_rejected_boundary_leak_closed_surface_coordinate_count,
        repaired_source_face_provenance_count: mesh
            .backend
            .tetrahedron_repaired_source_face_provenance_count,
        repaired_source_edge_provenance_count: mesh
            .backend
            .tetrahedron_repaired_source_edge_provenance_count,
        repaired_cad_curve_source_edge_provenance_count: mesh
            .backend
            .tetrahedron_repaired_cad_curve_source_edge_provenance_count,
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
        recovered_boundary_owned_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_boundary_owned_material_interface_recovery_item_count,
        interior_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_interior_material_interface_recovery_item_count,
        recovered_interior_face_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_interior_face_material_interface_recovery_item_count,
        recovered_absent_partition_material_interface_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_absent_partition_material_interface_recovery_item_count,
        rejected_material_interface_missing_boundary_ownership_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_missing_boundary_ownership_count,
        rejected_material_interface_missing_interior_ownership_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_missing_interior_ownership_count,
        rejected_material_interface_ambiguous_boundary_ownership_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_ambiguous_boundary_ownership_count,
        rejected_material_interface_absent_partition_count: mesh
            .backend
            .tetrahedron_rejected_material_interface_absent_partition_count,
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
        missing_source_face_interior_face_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_face_interior_face_recovery_item_count,
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
        missing_source_edge_interior_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_source_edge_interior_edge_recovery_item_count,
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
        cad_curve_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_cad_curve_source_edge_recovery_item_count,
        recovered_cad_curve_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_recovered_cad_curve_source_edge_recovery_item_count,
        missing_cad_curve_source_edge_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_cad_curve_source_edge_recovery_item_count,
        missing_cad_curve_source_edge_topology_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_cad_curve_source_edge_topology_recovery_item_count,
        missing_cad_curve_source_edge_provenance_recovery_item_count: mesh
            .backend
            .tetrahedron_missing_cad_curve_source_edge_provenance_recovery_item_count,
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
        optimization_budget_limited_count: mesh
            .backend
            .tetrahedron_optimization_budget_limited_count,
        smoothed_point_count: mesh.backend.tetrahedron_smoothed_point_count,
        optimization_interior_smoothing_attempt_count: mesh
            .backend
            .tetrahedron_optimization_interior_smoothing_attempt_count,
        optimization_interior_smoothing_accepted_count: mesh
            .backend
            .tetrahedron_optimization_interior_smoothing_accepted_count,
        optimization_interior_smoothing_rejected_count: mesh
            .backend
            .tetrahedron_optimization_interior_smoothing_rejected_count,
        optimization_interior_smoothing_budget_limited_count: mesh
            .backend
            .tetrahedron_optimization_interior_smoothing_budget_limited_count,
        optimization_interior_smoothing_rejected_by_reason: mesh
            .backend
            .tetrahedron_optimization_interior_smoothing_rejected_by_reason
            .clone(),
        optimization_boundary_smoothing_attempt_count: mesh
            .backend
            .tetrahedron_optimization_boundary_smoothing_attempt_count,
        optimization_boundary_smoothing_accepted_count: mesh
            .backend
            .tetrahedron_optimization_boundary_smoothing_accepted_count,
        optimization_boundary_smoothing_rejected_count: mesh
            .backend
            .tetrahedron_optimization_boundary_smoothing_rejected_count,
        optimization_boundary_smoothing_budget_limited_count: mesh
            .backend
            .tetrahedron_optimization_boundary_smoothing_budget_limited_count,
        optimization_boundary_smoothing_rejected_by_reason: mesh
            .backend
            .tetrahedron_optimization_boundary_smoothing_rejected_by_reason
            .clone(),
        sliver_count: mesh.backend.tetrahedron_sliver_count,
        sliver_removed_count: mesh.backend.tetrahedron_sliver_removed_count,
        optimization_sliver_removal_attempt_count: mesh
            .backend
            .tetrahedron_optimization_sliver_removal_attempt_count,
        optimization_sliver_removal_accepted_count: mesh
            .backend
            .tetrahedron_optimization_sliver_removal_accepted_count,
        optimization_sliver_removal_rejected_count: mesh
            .backend
            .tetrahedron_optimization_sliver_removal_rejected_count,
        optimization_sliver_removal_budget_limited_count: mesh
            .backend
            .tetrahedron_optimization_sliver_removal_budget_limited_count,
        optimization_sliver_removal_rejected_by_reason: mesh
            .backend
            .tetrahedron_optimization_sliver_removal_rejected_by_reason
            .clone(),
        optimization_target_seed_count: mesh.backend.tetrahedron_optimization_target_seed_count,
        optimization_skipped_target_seed_count: mesh
            .backend
            .tetrahedron_optimization_skipped_target_seed_count,
        optimization_rejected_edit_count: mesh.backend.tetrahedron_optimization_rejected_edit_count,
        optimization_local_reconnection_attempt_count: mesh
            .backend
            .tetrahedron_optimization_local_reconnection_attempt_count,
        optimization_local_reconnection_accepted_count: mesh
            .backend
            .tetrahedron_optimization_local_reconnection_accepted_count,
        optimization_local_reconnection_rejected_count: mesh
            .backend
            .tetrahedron_optimization_local_reconnection_rejected_count,
        optimization_local_reconnection_budget_limited_count: mesh
            .backend
            .tetrahedron_optimization_local_reconnection_budget_limited_count,
        optimization_local_reconnection_rejected_by_reason: mesh
            .backend
            .tetrahedron_optimization_local_reconnection_rejected_by_reason
            .clone(),
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
