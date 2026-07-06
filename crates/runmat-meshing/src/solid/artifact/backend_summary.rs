use runmat_meshing_core::contracts::{
    MeshBackendSummary, TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT,
    TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT,
    TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT,
    TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX,
    TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT,
    TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT, TETRAHEDRON_UNTANGLING_PASS_COUNT,
    TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT,
};
use runmat_meshing_tetrahedron::{
    generate::TetrahedronMesh,
    recover::{TetrahedronRecoveryKind, TetrahedronRecoveryQueue},
};

use super::backend_counts::{
    bounded_missing_recovery_ids, recovery_entity_count, tetrahedron_entity_count,
    tetrahedron_material_region_count, tetrahedron_rejection_counts_by_prefix,
    tetrahedron_unclassified_material_element_count,
};
use super::backend_quality::{optimization_target_evidence, BackendQualityEvidence};

pub(super) const SOLID_PLC_TETRAHEDRON_ALGORITHM: &str = "plc_tetrahedron/v1";

pub(super) struct BackendSummaryInput<'a> {
    pub(super) surface_element_count: usize,
    pub(super) tetrahedron_mesh: &'a TetrahedronMesh,
    pub(super) recovery_queue: &'a TetrahedronRecoveryQueue,
    pub(super) initial_backend_quality: &'a BackendQualityEvidence,
    pub(super) backend_quality: BackendQualityEvidence,
}

pub(super) fn build_backend_summary(input: BackendSummaryInput<'_>) -> MeshBackendSummary {
    let tetrahedron_mesh = input.tetrahedron_mesh;
    let recovery_queue = input.recovery_queue;
    let backend_quality = input.backend_quality;
    let initial_backend_quality = input.initial_backend_quality;
    let optimization_targets =
        optimization_target_evidence(initial_backend_quality, &backend_quality);
    let missing_source_face_recovery =
        bounded_missing_recovery_ids(recovery_queue, TetrahedronRecoveryKind::SourceFace);
    let missing_source_edge_recovery =
        bounded_missing_recovery_ids(recovery_queue, TetrahedronRecoveryKind::SourceEdge);
    let missing_material_interface_recovery =
        bounded_missing_recovery_ids(recovery_queue, TetrahedronRecoveryKind::MaterialInterface);

    MeshBackendSummary {
    backend: "solid".to_string(),
    algorithm: SOLID_PLC_TETRAHEDRON_ALGORITHM.to_string(),
    surface_element_count: input.surface_element_count,
    plc_input_node_count: tetrahedron_entity_count(tetrahedron_mesh, "input_plc_nodes"),
    plc_input_facet_count: tetrahedron_entity_count(tetrahedron_mesh, "input_plc_facets"),
    plc_input_protected_edge_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_protected_edges",
    ),
    plc_input_boundary_component_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_boundary_components",
    ),
    plc_input_boundary_component_node_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_boundary_component_nodes",
    ),
    plc_input_max_boundary_component_node_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_max_boundary_component_nodes",
    ),
    plc_input_shell_nesting_classified: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_shell_nesting_classified",
    ) > 0,
    plc_input_outer_shell_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_outer_shells",
    ),
    plc_input_nested_shell_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_nested_shells",
    ),
    plc_input_max_shell_nesting_depth: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_max_shell_nesting_depth",
    ),
    plc_input_material_region_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_material_regions",
    ),
    plc_input_material_region_facet_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_material_region_facets",
    ),
    plc_input_cad_curve_boundary_source_edge_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_boundary_source_edges",
    ),
    plc_input_cad_curve_boundary_segment_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_boundary_segments",
    ),
    plc_input_cad_curve_imported_edge_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_imported_edges",
    ),
    plc_input_cad_curve_evaluator_edge_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_evaluator_edges",
    ),
    plc_input_cad_curve_evaluator_sample_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_evaluator_samples",
    ),
    plc_input_cad_curve_live_query_edge_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_live_query_edges",
    ),
    plc_input_cad_curve_live_query_sample_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_live_query_samples",
    ),
    plc_input_cad_curve_rejected_evaluator_sample_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_rejected_evaluator_samples",
    ),
    plc_input_cad_curve_curvature_sized_edge_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_curvature_sized_edges",
    ),
    plc_input_cad_curve_curvature_sample_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_cad_curve_curvature_samples",
    ),
    plc_input_surface_boundary_node_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "input_plc_surface_boundary_nodes",
    ),
    tetrahedron_generation_family: tetrahedron_mesh.tetrahedron_generation_family.clone(),
    tetrahedron_generation_attempted_family_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "solver_generation_attempted_families",
    ),
    tetrahedron_generation_rejected_family_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "solver_generation_rejected_families",
    ),
    tetrahedron_generation_selected_family_index: tetrahedron_entity_count(
        tetrahedron_mesh,
        "solver_generation_selected_family_index",
    ),
    tetrahedron_generation_interior_support_candidate_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "interior_support_candidate_points",
    ),
    tetrahedron_generation_interior_support_accepted_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "interior_support_accepted_points",
    ),
    tetrahedron_generation_nested_shell_outer_node_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "nested_tetrahedron_shell_outer_nodes",
    ),
    tetrahedron_generation_nested_shell_inner_node_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "nested_tetrahedron_shell_inner_nodes",
    ),
    tetrahedron_generation_nested_shell_generated_node_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "nested_tetrahedron_shell_generated_nodes",
    ),
    tetrahedron_generation_nested_shell_refill_boundary_face_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_refill_boundary_faces",
        ),
    tetrahedron_generation_nested_shell_boundary_centroid_refinement_attempt_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_boundary_centroid_refinement_attempts",
        ),
    tetrahedron_generation_nested_shell_boundary_centroid_refinement_rejected_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_boundary_centroid_refinement_rejected",
        ),
    tetrahedron_generation_nested_shell_boundary_exact_cover_refill_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_boundary_exact_cover_refills",
        ),
    tetrahedron_generation_nested_shell_boundary_centroid_refinement_refill_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_boundary_centroid_refinement_refills",
        ),
    tetrahedron_generation_nested_shell_barycentric_partition_refill_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_barycentric_partition_refills",
        ),
    tetrahedron_generation_nested_shell_outer_facet_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "nested_tetrahedron_shell_outer_facets",
    ),
    tetrahedron_generation_nested_shell_inner_facet_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        "nested_tetrahedron_shell_inner_facets",
    ),
    tetrahedron_element_count: tetrahedron_mesh.elements.len(),
    tetrahedron_material_region_count: tetrahedron_material_region_count(
        tetrahedron_mesh,
    ),
    tetrahedron_unclassified_material_element_count:
        tetrahedron_unclassified_material_element_count(tetrahedron_mesh),
    tetrahedron_min_exact_scaled_jacobian: backend_quality.min_exact_scaled_jacobian,
    tetrahedron_exact_scaled_jacobian_below_threshold_count: backend_quality
        .exact_scaled_jacobian_below_threshold_count,
    tetrahedron_exact_scaled_jacobian_bins: backend_quality.exact_scaled_jacobian_bins,
    boundary_face_recovery_ratio: 1.0,
    boundary_edge_recovery_ratio: 1.0,
    volume_component_count: 1,
    tetrahedron_recovered_component_ratio: 1.0,
    tetrahedron_volume_coverage_ratio: 1.0,
    tetrahedron_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "recovery_items",
    ),
    tetrahedron_recovered_item_count: recovery_entity_count(
        recovery_queue,
        "recovered_items",
    ),
    tetrahedron_missing_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_items",
    ),
    tetrahedron_recovered_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "recovered_missing_boundary_faces",
    ),
    tetrahedron_recovered_protected_edge_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "recovered_protected_edge_boundary_faces",
    ),
    tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_cad_curve_protected_edge_boundary_faces",
        ),
    tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count:
        recovery_entity_count(
            recovery_queue,
            "attempted_protected_edge_boundary_face_restoration_items",
        ),
    tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count:
        recovery_entity_count(
            recovery_queue,
            "attempted_cad_curve_protected_edge_boundary_face_restoration_items",
        ),
    tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_protected_edge_boundary_face_restoration_items",
        ),
    tetrahedron_rejected_cad_curve_protected_edge_boundary_face_restoration_item_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_cad_curve_protected_edge_boundary_face_restoration_items",
        ),
    tetrahedron_rejected_protected_edge_boundary_face_restoration_volume_face_topology_count:
        recovery_entity_count(
            recovery_queue,
            "protected_edge_rejected_boundary_face_restoration_volume_face_topology",
        ),
    tetrahedron_volume_edge_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "volume_edge_source_edge_recovery_items",
    ),
    tetrahedron_recovered_volume_edge_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_volume_edge_source_edge_items",
        ),
    tetrahedron_boundary_edge_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "boundary_edge_source_edge_recovery_items",
    ),
    tetrahedron_recovered_boundary_edge_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_boundary_edge_source_edge_items",
        ),
    tetrahedron_interior_edge_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "interior_edge_source_edge_recovery_items",
    ),
    tetrahedron_recovered_interior_edge_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_interior_edge_source_edge_items",
        ),
    tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "cad_curve_interior_edge_source_edge_recovery_items",
        ),
    tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_cad_curve_interior_edge_source_edge_items",
        ),
    tetrahedron_attempted_source_edge_split_refill_item_count: recovery_entity_count(
        recovery_queue,
        "attempted_source_edge_split_refill_items",
    ),
    tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count:
        recovery_entity_count(
            recovery_queue,
            "attempted_cad_curve_source_edge_split_refill_items",
        ),
    tetrahedron_accepted_source_edge_split_refill_candidate_item_count:
        recovery_entity_count(
            recovery_queue,
            "accepted_source_edge_split_refill_candidate_items",
        ),
    tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count:
        recovery_entity_count(
            recovery_queue,
            "accepted_cad_curve_source_edge_split_refill_candidate_items",
        ),
    tetrahedron_applied_source_edge_split_refill_item_count: recovery_entity_count(
        recovery_queue,
        "applied_source_edge_split_refill_items",
    ),
    tetrahedron_applied_cad_curve_source_edge_split_refill_item_count:
        recovery_entity_count(
            recovery_queue,
            "applied_cad_curve_source_edge_split_refill_items",
        ),
    tetrahedron_rejected_source_edge_split_refill_item_count: recovery_entity_count(
        recovery_queue,
        "rejected_source_edge_split_refill_items",
    ),
    tetrahedron_rejected_cad_curve_source_edge_split_refill_item_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_cad_curve_source_edge_split_refill_items",
        ),
    tetrahedron_absent_edge_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "absent_edge_source_edge_recovery_items",
    ),
    tetrahedron_recovered_absent_edge_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_absent_edge_source_edge_items",
        ),
    tetrahedron_boundary_face_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "boundary_face_source_face_recovery_items",
    ),
    tetrahedron_recovered_boundary_face_source_face_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_boundary_face_source_face_items",
        ),
    tetrahedron_interior_face_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "interior_face_source_face_recovery_items",
    ),
    tetrahedron_recovered_interior_face_source_face_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_interior_face_source_face_items",
        ),
    tetrahedron_volume_face_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "volume_face_source_face_recovery_items",
    ),
    tetrahedron_recovered_volume_face_source_face_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_volume_face_source_face_items",
        ),
    tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count:
        recovery_entity_count(
            recovery_queue,
            "attempted_volume_face_source_face_boundary_restoration_items",
        ),
    tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_volume_face_source_face_boundary_restoration_items",
        ),
    tetrahedron_rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count:
        recovery_entity_count(
            recovery_queue,
            "source_face_rejected_boundary_face_restoration_volume_face_topology",
        ),
    tetrahedron_absent_face_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "absent_face_source_face_recovery_items",
    ),
    tetrahedron_recovered_absent_face_source_face_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_absent_face_source_face_items",
        ),
    tetrahedron_deferred_absent_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "deferred_absent_source_edge_recovery_items",
    ),
    tetrahedron_attempted_absent_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "attempted_absent_source_edge_recovery_items",
    ),
    tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "attempted_cad_curve_absent_source_edge_recovery_items",
        ),
    tetrahedron_reconnected_absent_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "reconnected_absent_source_edge_items",
    ),
    tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "reconnected_cad_curve_absent_source_edge_items",
        ),
    tetrahedron_rejected_absent_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "rejected_absent_source_edge_recovery_items",
    ),
    tetrahedron_rejected_cad_curve_absent_source_edge_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_cad_curve_absent_source_edge_recovery_items",
        ),
    tetrahedron_rejected_absent_source_edge_adjacent_facet_count: recovery_entity_count(
        recovery_queue,
        "rejected_absent_source_edge_recovery_adjacent_facet_count",
    ),
    tetrahedron_rejected_absent_source_edge_adjacent_facet_topology_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_source_edge_recovery_adjacent_facet_topology",
        ),
    tetrahedron_rejected_absent_source_edge_current_boundary_face_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_source_edge_recovery_current_boundary_faces",
        ),
    tetrahedron_rejected_absent_source_edge_element_topology_count: recovery_entity_count(
        recovery_queue,
        "rejected_absent_source_edge_recovery_element_topology",
    ),
    tetrahedron_rejected_absent_source_edge_material_region_mismatch_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_source_edge_recovery_material_region_mismatch",
        ),
    tetrahedron_rejected_absent_source_edge_quality_gate_count: recovery_entity_count(
        recovery_queue,
        "rejected_absent_source_edge_recovery_quality_gate",
    ),
    tetrahedron_recovered_absent_source_edge_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "recovered_absent_source_edge_boundary_faces",
    ),
    tetrahedron_attempted_source_face_diagonal_recovery_pair_count: recovery_entity_count(
        recovery_queue,
        "attempted_source_face_diagonal_recovery_pairs",
    ),
    tetrahedron_recovered_source_face_diagonal_pair_count: recovery_entity_count(
        recovery_queue,
        "recovered_source_face_diagonal_pairs",
    ),
    tetrahedron_recovered_source_face_diagonal_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "recovered_source_face_diagonal_boundary_faces",
    ),
    tetrahedron_rejected_source_face_diagonal_recovery_pair_count: recovery_entity_count(
        recovery_queue,
        "rejected_source_face_diagonal_recovery_pairs",
    ),
    tetrahedron_rejected_source_face_diagonal_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "rejected_source_face_diagonal_recovery_items",
    ),
    tetrahedron_rejected_source_face_diagonal_adjacent_facet_count: recovery_entity_count(
        recovery_queue,
        "rejected_source_face_diagonal_recovery_adjacent_facet_count",
    ),
    tetrahedron_rejected_source_face_diagonal_adjacent_facet_topology_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_source_face_diagonal_recovery_adjacent_facet_topology",
        ),
    tetrahedron_rejected_source_face_diagonal_current_boundary_face_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_source_face_diagonal_recovery_current_boundary_faces",
        ),
    tetrahedron_rejected_source_face_diagonal_element_topology_count: recovery_entity_count(
        recovery_queue,
        "rejected_source_face_diagonal_recovery_element_topology",
    ),
    tetrahedron_rejected_source_face_diagonal_material_region_mismatch_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_source_face_diagonal_recovery_material_region_mismatch",
        ),
    tetrahedron_rejected_source_face_diagonal_quality_gate_count: recovery_entity_count(
        recovery_queue,
        "rejected_source_face_diagonal_recovery_quality_gate",
    ),
    tetrahedron_rejected_source_face_diagonal_unpaired_source_face_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_source_face_diagonal_recovery_unpaired_source_face",
        ),
    tetrahedron_repaired_boundary_face_identity_count: recovery_entity_count(
        recovery_queue,
        "repaired_boundary_face_identity_items",
    ),
    tetrahedron_removed_redundant_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "removed_redundant_boundary_faces",
    ),
    tetrahedron_removed_unsupported_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "removed_unsupported_boundary_faces",
    ),
    tetrahedron_attempted_boundary_leak_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "attempted_boundary_leak_recovery_items",
    ),
    tetrahedron_removed_exterior_leaked_element_count: recovery_entity_count(
        recovery_queue,
        "removed_exterior_leaked_elements",
    ),
    tetrahedron_exposed_interior_source_face_count: recovery_entity_count(
        recovery_queue,
        "exposed_interior_source_faces",
    ),
    tetrahedron_inserted_exposed_interior_boundary_face_count: recovery_entity_count(
        recovery_queue,
        "inserted_exposed_interior_boundary_faces",
    ),
    tetrahedron_rejected_boundary_leak_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "rejected_boundary_leak_recovery_items",
    ),
    tetrahedron_rejected_boundary_leak_adjacent_element_count: recovery_entity_count(
        recovery_queue,
        "rejected_boundary_leak_adjacent_element_count",
    ),
    tetrahedron_rejected_boundary_leak_material_region_mismatch_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_boundary_leak_material_region_mismatch",
        ),
    tetrahedron_rejected_boundary_leak_outside_classification_count: recovery_entity_count(
        recovery_queue,
        "rejected_boundary_leak_outside_classification",
    ),
    tetrahedron_rejected_boundary_leak_closed_surface_coordinate_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_boundary_leak_closed_surface_coordinates",
        ),
    tetrahedron_repaired_source_face_provenance_count: recovery_entity_count(
        recovery_queue,
        "repaired_source_face_provenance_items",
    ),
    tetrahedron_repaired_source_edge_provenance_count: recovery_entity_count(
        recovery_queue,
        "repaired_source_edge_provenance_items",
    ),
    tetrahedron_repaired_cad_curve_source_edge_provenance_count: recovery_entity_count(
        recovery_queue,
        "repaired_cad_curve_source_edge_provenance_items",
    ),
    tetrahedron_repaired_material_interface_element_count: recovery_entity_count(
        recovery_queue,
        "repaired_material_interface_elements",
    ),
    tetrahedron_attempted_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "attempted_material_interface_recovery_items",
    ),
    tetrahedron_rejected_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "rejected_material_interface_recovery_items",
    ),
    tetrahedron_global_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "global_material_interface_recovery_items",
    ),
    tetrahedron_boundary_owned_material_interface_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "boundary_owned_material_interface_recovery_items",
        ),
    tetrahedron_recovered_boundary_owned_material_interface_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_boundary_owned_material_interface_items",
        ),
    tetrahedron_interior_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "interior_material_interface_recovery_items",
    ),
    tetrahedron_recovered_interior_face_material_interface_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_interior_face_material_interface_items",
        ),
    tetrahedron_recovered_absent_partition_material_interface_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "recovered_absent_partition_material_interface_items",
        ),
    tetrahedron_rejected_material_interface_missing_boundary_ownership_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_material_interface_missing_boundary_ownership",
        ),
    tetrahedron_rejected_material_interface_ambiguous_boundary_ownership_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_material_interface_ambiguous_boundary_ownership",
        ),
    tetrahedron_attempted_absent_material_partition_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "attempted_absent_material_partition_recovery_items",
        ),
    tetrahedron_inserted_absent_material_partition_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "inserted_absent_material_partition_recovery_items",
        ),
    tetrahedron_inserted_absent_material_partition_element_count: recovery_entity_count(
        recovery_queue,
        "inserted_absent_material_partition_elements",
    ),
    tetrahedron_inserted_absent_material_partition_boundary_face_count:
        recovery_entity_count(
            recovery_queue,
            "inserted_absent_material_partition_boundary_faces",
        ),
    tetrahedron_rejected_absent_material_partition_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_material_partition_recovery_items",
        ),
    tetrahedron_rolled_back_absent_material_partition_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "rolled_back_absent_material_partition_recovery_items",
        ),
    tetrahedron_rolled_back_absent_material_partition_element_count: recovery_entity_count(
        recovery_queue,
        "rolled_back_absent_material_partition_elements",
    ),
    tetrahedron_rolled_back_absent_material_partition_boundary_face_count:
        recovery_entity_count(
            recovery_queue,
            "rolled_back_absent_material_partition_boundary_faces",
        ),
    tetrahedron_rejected_absent_material_partition_facet_count: recovery_entity_count(
        recovery_queue,
        "rejected_absent_material_partition_facet_count",
    ),
    tetrahedron_rejected_absent_material_partition_facet_topology_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_material_partition_facet_topology",
        ),
    tetrahedron_rejected_absent_material_partition_element_exists_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_material_partition_element_exists",
        ),
    tetrahedron_rejected_absent_material_partition_interior_face_topology_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_material_partition_interior_face_topology",
        ),
    tetrahedron_rejected_absent_material_partition_quality_gate_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_material_partition_quality_gate",
        ),
    tetrahedron_rejected_absent_material_partition_post_insertion_audit_count:
        recovery_entity_count(
            recovery_queue,
            "rejected_absent_material_partition_post_insertion_audit",
        ),
    tetrahedron_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "source_face_items",
    ),
    tetrahedron_recovered_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "recovered_source_face_items",
    ),
    tetrahedron_missing_source_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_face_items",
    ),
    tetrahedron_missing_source_face_topology_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_face_topology_items",
    ),
    tetrahedron_missing_source_face_provenance_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_face_provenance_items",
    ),
    tetrahedron_missing_source_face_boundary_face_recovery_item_count:
        recovery_entity_count(recovery_queue, "missing_source_face_boundary_face_items"),
    tetrahedron_missing_source_face_volume_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_face_volume_face_items",
    ),
    tetrahedron_missing_source_face_interior_face_recovery_item_count:
        recovery_entity_count(recovery_queue, "missing_source_face_interior_face_items"),
    tetrahedron_missing_source_face_absent_face_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_face_absent_face_items",
    ),
    tetrahedron_missing_source_face_recovery_ids: missing_source_face_recovery.ids,
    tetrahedron_omitted_missing_source_face_recovery_id_count: missing_source_face_recovery
        .omitted_count,
    tetrahedron_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "source_edge_items",
    ),
    tetrahedron_recovered_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "recovered_source_edge_items",
    ),
    tetrahedron_missing_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_edge_items",
    ),
    tetrahedron_missing_source_edge_topology_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_edge_topology_items",
    ),
    tetrahedron_missing_source_edge_provenance_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_edge_provenance_items",
    ),
    tetrahedron_missing_source_edge_volume_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_edge_volume_edge_items",
    ),
    tetrahedron_missing_source_edge_interior_edge_recovery_item_count:
        recovery_entity_count(recovery_queue, "missing_source_edge_interior_edge_items"),
    tetrahedron_missing_source_edge_absent_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_source_edge_absent_edge_items",
    ),
    tetrahedron_missing_source_edge_recovery_ids: missing_source_edge_recovery.ids,
    tetrahedron_omitted_missing_source_edge_recovery_id_count: missing_source_edge_recovery
        .omitted_count,
    tetrahedron_cad_curve_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "cad_curve_source_edge_items",
    ),
    tetrahedron_recovered_cad_curve_source_edge_recovery_item_count:
        recovery_entity_count(recovery_queue, "recovered_cad_curve_source_edge_items"),
    tetrahedron_missing_cad_curve_source_edge_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_cad_curve_source_edge_items",
    ),
    tetrahedron_missing_cad_curve_source_edge_topology_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "missing_cad_curve_source_edge_topology_items",
        ),
    tetrahedron_missing_cad_curve_source_edge_provenance_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "missing_cad_curve_source_edge_provenance_items",
        ),
    tetrahedron_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "material_interface_items",
    ),
    tetrahedron_recovered_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "recovered_material_interface_items",
    ),
    tetrahedron_missing_material_interface_recovery_item_count: recovery_entity_count(
        recovery_queue,
        "missing_material_interface_items",
    ),
    tetrahedron_missing_material_interface_boundary_owned_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "missing_material_interface_boundary_owned_items",
        ),
    tetrahedron_missing_material_interface_interior_face_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "missing_material_interface_interior_face_items",
        ),
    tetrahedron_missing_material_interface_absent_partition_recovery_item_count:
        recovery_entity_count(
            recovery_queue,
            "missing_material_interface_absent_partition_items",
        ),
    tetrahedron_missing_material_interface_recovery_ids:
        missing_material_interface_recovery.ids,
    tetrahedron_omitted_missing_material_interface_recovery_id_count:
        missing_material_interface_recovery.omitted_count,
    tetrahedron_optimization_pass_count: usize::from(tetrahedron_mesh.quality_optimized),
    tetrahedron_optimization_budget_limited_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
    ) + tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
    ) + tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
    ) + tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
    ),
    tetrahedron_smoothed_point_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
    ) + tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
    ),
    tetrahedron_sliver_count: backend_quality.sliver_count,
    tetrahedron_sliver_removed_count: optimization_targets.sliver_removed_count,
    tetrahedron_optimization_sliver_removal_attempt_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT,
    ),
    tetrahedron_optimization_sliver_removal_accepted_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT,
    ),
    tetrahedron_optimization_sliver_removal_rejected_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT,
    ),
    tetrahedron_optimization_sliver_removal_budget_limited_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
    ),
    tetrahedron_optimization_sliver_removal_rejected_by_reason:
        tetrahedron_rejection_counts_by_prefix(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX,
        ),
    tetrahedron_optimization_target_seed_count: optimization_targets.target_seed_count,
    tetrahedron_optimization_skipped_target_seed_count: optimization_targets
        .skipped_target_seed_count,
    tetrahedron_optimization_interior_smoothing_attempt_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
    ),
    tetrahedron_optimization_interior_smoothing_accepted_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
    ),
    tetrahedron_optimization_interior_smoothing_rejected_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
    ),
    tetrahedron_optimization_interior_smoothing_budget_limited_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
        ),
    tetrahedron_optimization_interior_smoothing_rejected_by_reason:
        tetrahedron_rejection_counts_by_prefix(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
        ),
    tetrahedron_optimization_boundary_smoothing_attempt_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
    ),
    tetrahedron_optimization_boundary_smoothing_accepted_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
    ),
    tetrahedron_optimization_boundary_smoothing_rejected_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
    ),
    tetrahedron_optimization_boundary_smoothing_budget_limited_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
        ),
    tetrahedron_optimization_boundary_smoothing_rejected_by_reason:
        tetrahedron_rejection_counts_by_prefix(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
        ),
    tetrahedron_optimization_local_reconnection_attempt_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
    ),
    tetrahedron_optimization_local_reconnection_accepted_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
    ),
    tetrahedron_optimization_local_reconnection_rejected_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
    ),
    tetrahedron_optimization_local_reconnection_budget_limited_count:
        tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
        ),
    tetrahedron_optimization_local_reconnection_rejected_by_reason:
        tetrahedron_rejection_counts_by_prefix(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
        ),
    tetrahedron_optimization_initial_max_aspect_ratio: initial_backend_quality
        .max_aspect_ratio,
    tetrahedron_optimization_final_max_aspect_ratio: backend_quality.max_aspect_ratio,
    tetrahedron_optimization_initial_min_exact_scaled_jacobian: initial_backend_quality
        .min_exact_scaled_jacobian,
    tetrahedron_optimization_final_min_exact_scaled_jacobian: backend_quality
        .min_exact_scaled_jacobian,
    tetrahedron_untangling_pass_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_UNTANGLING_PASS_COUNT,
    ),
    tetrahedron_untangling_initial_near_singular_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT,
    ),
    tetrahedron_untangling_final_near_singular_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT,
    ),
    tetrahedron_untangling_relocated_seed_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT,
    ),
    tetrahedron_exact_quality_repair_pass_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT,
    ),
    tetrahedron_exact_quality_seed_star_relocation_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT,
    ),
    tetrahedron_exact_quality_unrepaired_total_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT,
    ),
    tetrahedron_exact_quality_unrepaired_interior_seed_count: tetrahedron_entity_count(
        tetrahedron_mesh,
        TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT,
    ),
    ..MeshBackendSummary::default()
}
}
