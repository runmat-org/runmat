use super::*;
use crate::{
    MeshCadEvidence, MeshEvidenceArtifact, MeshSizingEvidence, MeshTetrahedronRecoveryEvidence,
    MeshValidationEvidence, MESH_EVIDENCE_SCHEMA_VERSION,
};

pub(super) fn assert_schema_and_rolled_back_partition_readiness_failure(
    evidence: &MeshEvidenceArtifact,
) {
    assert_eq!(evidence.schema_version, MESH_EVIDENCE_SCHEMA_VERSION);
    assert!(!evidence.validation.solve_ready);
    assert_eq!(
        evidence.validation.validation_error_code.as_deref(),
        Some("rolled_back_material_interface_partition_recovery_present")
    );
    assert!(evidence
        .validation
        .validation_error_message
        .as_deref()
        .is_some_and(
            |message| message.contains("RolledBackMaterialInterfacePartitionRecoveryPresent")
        ));
}

pub(super) fn assert_cad_evidence(cad: &MeshCadEvidence) {
    assert_eq!(cad.topology_source, "semantic_cad");
    assert_eq!(cad.evaluation_source, "imported_evaluator_samples");
    assert_eq!(cad.imported_face_count, 3);
    assert_eq!(cad.exact_query_face_count, 1);
    assert_eq!(cad.missing_exact_query_face_count, 1);
    assert_eq!(cad.missing_derivative_query_face_count, 2);
    assert_eq!(cad.missing_curvature_query_face_count, 1);
    assert_eq!(cad.point_evaluation_supported_face_count, 2);
    assert_eq!(cad.projection_supported_face_count, 2);
    assert_eq!(cad.normal_supported_face_count, 2);
    assert_eq!(cad.derivative_supported_face_count, 2);
    assert_eq!(cad.curvature_supported_face_count, 1);
    assert_eq!(cad.evaluator_sample_count, 8);
    assert_eq!(cad.evaluator_rejected_sample_count, 9);
    assert_eq!(cad.projection_query_count, 12);
    assert_eq!(cad.derivative_query_count, 6);
    assert_eq!(cad.curvature_query_count, 5);
    assert_eq!(cad.uv_domain_face_count, 10);
    assert_eq!(cad.uv_projection_out_of_bounds_count, 2);
    assert_eq!(cad.max_projection_error_m, 2.0e-6);
    assert_eq!(cad.max_normal_deviation, 1.0e-5);
    assert_eq!(cad.max_curvature_estimate_1_per_m, 0.125);
    assert_eq!(cad.surface_source_edge_loop_count, 2);
    assert_eq!(cad.surface_closed_edge_loop_count, 1);
    assert_eq!(cad.surface_conforming_source_edge_count, 5);
    assert_eq!(cad.surface_missing_source_edge_count, 1);
    assert_eq!(cad.surface_max_projection_error_m, 3.0e-6);
    assert_eq!(cad.surface_exact_cad_sample_node_count, 4);
    assert_eq!(cad.surface_rejected_exact_cad_sample_count, 5);
}

pub(super) fn assert_topology_and_adaptive_evidence(evidence: &MeshEvidenceArtifact) {
    assert_eq!(evidence.topology.node_count, 4);
    assert_eq!(evidence.adaptive.iteration_count, 0);
    assert_eq!(evidence.adaptive.latest_convergence_status, None);
    assert_eq!(evidence.adaptive.marker_count, 0);
    assert_eq!(evidence.adaptive.sizing_update_sample_count, 0);
}

pub(super) fn assert_validation_evidence(validation: &MeshValidationEvidence) {
    assert_eq!(validation.volume_element_count, 1);
    assert_eq!(validation.max_volume_element_count, Some(7));
    assert_eq!(validation.volume_component_count, 1);
    assert_eq!(validation.volume_component_element_counts, vec![1]);
    assert_eq!(validation.max_volume_component_count, Some(1));
    assert_eq!(validation.coverage_sample_count, 1);
    assert_eq!(validation.covered_coverage_sample_count, 1);
    assert_eq!(validation.coverage_sample_ratio, Some(1.0));
    assert_eq!(validation.min_coverage_sample_ratio, 1.0);
    assert_eq!(validation.coverage_sample_points_m, vec![[0.1, 0.1, 0.1]]);
    assert!(validation.require_no_unrecovered_tetrahedron_components);
    assert!(!validation.require_boundary_source_edge_provenance);
    assert!(!validation.require_no_unrepaired_exact_quality);
    assert_eq!(validation.unrecovered_tetrahedron_component_count, 0);
    assert_eq!(validation.unrepaired_exact_quality_total_count, 9);
    assert_eq!(validation.unrepaired_exact_quality_general_cavity_count, 1);
    assert_eq!(
        validation.unrepaired_exact_quality_boundary_adjacent_count,
        6
    );
    assert_eq!(validation.unrepaired_exact_quality_node_adjacent_count, 10);
    assert_eq!(validation.unrepaired_exact_quality_interior_seed_count, 7);
    assert_eq!(validation.unrepaired_exact_quality_edge_star_count, 8);
    assert_eq!(
        validation.boundary_recovery.boundary_edge_recovery_ratio,
        1.0
    );
}

pub(super) fn assert_sizing_evidence(sizing: &MeshSizingEvidence) {
    assert_eq!(sizing.inserted_breakpoint_count, 2);
    assert_eq!(sizing.requested_tetrahedron_refinement_point_count, 5);
    assert_eq!(
        sizing.accepted_requested_tetrahedron_refinement_location_count,
        5
    );
    assert_eq!(
        sizing.accepted_requested_tetrahedron_refinement_point_count,
        3
    );
    assert_eq!(
        sizing.rejected_requested_tetrahedron_refinement_point_count,
        1
    );
    assert_eq!(
        sizing
            .requested_tetrahedron_refinement_rejected_by_reason
            .get("quality_or_recovery"),
        Some(&1)
    );
    assert_eq!(
        sizing.dropped_requested_tetrahedron_refinement_point_count,
        2
    );
    assert_eq!(
        sizing
            .requested_tetrahedron_refinement_dropped_by_reason
            .get("not_retained_after_repair"),
        Some(&2)
    );
    assert_eq!(
        sizing.accepted_requested_tetrahedron_refinement_surrogate_point_count,
        2
    );
    assert_eq!(
        sizing.accepted_requested_tetrahedron_refinement_exact_point_count,
        1
    );
    assert_eq!(
        sizing.requested_tetrahedron_refinement_acceptance_ratio,
        Some(0.6)
    );
    assert_eq!(
        sizing.requested_tetrahedron_refinement_rejection_ratio,
        Some(0.2)
    );
    assert_eq!(
        sizing.requested_tetrahedron_refinement_surrogate_ratio,
        Some(2.0 / 3.0)
    );
    assert_eq!(sizing.sample_count, 3);
    assert_eq!(sizing.generated_cad_sample_count, 2);
    assert_eq!(sizing.anisotropic_sample_count, 2);
    assert_eq!(sizing.valid_anisotropic_sample_count, 1);
    assert_eq!(sizing.invalid_anisotropic_sample_count, 1);
    assert_eq!(sizing.anisotropic_by_reason.get("boundary_layer"), Some(&1));
    assert_eq!(
        sizing.invalid_anisotropic_by_reason.get("cad.proximity"),
        Some(&1)
    );
    assert_eq!(
        sizing.generated_cad_by_reason.get("cad.curvature"),
        Some(&1)
    );
    assert_eq!(
        sizing.generated_cad_by_reason.get("cad.interface"),
        Some(&1)
    );
    assert_eq!(sizing.applied_by_reason.get("load_region"), Some(&1));
    assert_eq!(
        sizing.inserted_breakpoint_by_reason.get("load_region"),
        Some(&2)
    );
    assert_eq!(
        sizing.uninserted_sample_by_reason.get("cad.curvature"),
        Some(&1)
    );
    assert_eq!(sizing.growth_rate, Some(1.4));
    assert_eq!(sizing.rejected_by_status.get("outside_bounds"), Some(&1));
}

pub(super) fn assert_tetrahedron_recovery_evidence(recovery: &MeshTetrahedronRecoveryEvidence) {
    assert_eq!(recovery.plc_input_node_count, 4);
    assert_eq!(recovery.plc_input_facet_count, 4);
    assert_eq!(recovery.plc_input_protected_edge_count, 6);
    assert_eq!(recovery.plc_input_boundary_component_count, 1);
    assert_eq!(recovery.plc_input_boundary_component_node_count, 4);
    assert_eq!(recovery.plc_input_max_boundary_component_node_count, 4);
    assert!(recovery.plc_input_shell_nesting_classified);
    assert_eq!(recovery.plc_input_outer_shell_count, 1);
    assert_eq!(recovery.plc_input_nested_shell_count, 0);
    assert_eq!(recovery.plc_input_max_shell_nesting_depth, 0);
    assert_eq!(recovery.element_count, 12);
    assert_eq!(recovery.recovered_component_ratio, 1.0);
    assert_eq!(recovery.volume_coverage_ratio, 0.99);
    assert_eq!(recovery.recovery_item_count, 9);
    assert_eq!(recovery.recovered_item_count, 7);
    assert_eq!(recovery.missing_recovery_item_count, 2);
    assert_eq!(recovery.recovered_boundary_face_count, 3);
    assert_eq!(recovery.recovered_protected_edge_boundary_face_count, 2);
    assert_eq!(
        recovery.attempted_protected_edge_boundary_face_restoration_item_count,
        3
    );
    assert_eq!(
        recovery.rejected_protected_edge_boundary_face_restoration_item_count,
        1
    );
    assert_eq!(
        recovery.rejected_protected_edge_boundary_face_restoration_volume_face_topology_count,
        1
    );
    assert_eq!(recovery.volume_edge_source_edge_recovery_item_count, 1);
    assert_eq!(
        recovery.recovered_volume_edge_source_edge_recovery_item_count,
        1
    );
    assert_eq!(recovery.boundary_edge_source_edge_recovery_item_count, 1);
    assert_eq!(
        recovery.recovered_boundary_edge_source_edge_recovery_item_count,
        1
    );
    assert_eq!(recovery.interior_edge_source_edge_recovery_item_count, 1);
    assert_eq!(
        recovery.recovered_interior_edge_source_edge_recovery_item_count,
        1
    );
    assert_eq!(recovery.absent_edge_source_edge_recovery_item_count, 2);
    assert_eq!(
        recovery.recovered_absent_edge_source_edge_recovery_item_count,
        1
    );
    assert_eq!(recovery.boundary_face_source_face_recovery_item_count, 1);
    assert_eq!(
        recovery.recovered_boundary_face_source_face_recovery_item_count,
        1
    );
    assert_eq!(recovery.interior_face_source_face_recovery_item_count, 1);
    assert_eq!(
        recovery.recovered_interior_face_source_face_recovery_item_count,
        1
    );
    assert_eq!(recovery.volume_face_source_face_recovery_item_count, 2);
    assert_eq!(
        recovery.recovered_volume_face_source_face_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.attempted_volume_face_source_face_boundary_restoration_item_count,
        2
    );
    assert_eq!(
        recovery.rejected_volume_face_source_face_boundary_restoration_item_count,
        1
    );
    assert_eq!(
        recovery.rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count,
        1
    );
    assert_eq!(recovery.absent_face_source_face_recovery_item_count, 3);
    assert_eq!(
        recovery.recovered_absent_face_source_face_recovery_item_count,
        2
    );
    assert_eq!(recovery.deferred_absent_source_edge_recovery_item_count, 0);
    assert_eq!(recovery.attempted_absent_source_edge_recovery_item_count, 1);
    assert_eq!(
        recovery.reconnected_absent_source_edge_recovery_item_count,
        1
    );
    assert_eq!(recovery.rejected_absent_source_edge_recovery_item_count, 0);
    assert_eq!(recovery.rejected_absent_source_edge_adjacent_facet_count, 0);
    assert_eq!(
        recovery.rejected_absent_source_edge_adjacent_facet_topology_count,
        0
    );
    assert_eq!(
        recovery.rejected_absent_source_edge_current_boundary_face_count,
        0
    );
    assert_eq!(
        recovery.rejected_absent_source_edge_element_topology_count,
        0
    );
    assert_eq!(
        recovery.rejected_absent_source_edge_material_region_mismatch_count,
        0
    );
    assert_eq!(recovery.rejected_absent_source_edge_quality_gate_count, 0);
    assert_eq!(recovery.recovered_absent_source_edge_boundary_face_count, 2);
    assert_eq!(
        recovery.attempted_source_face_diagonal_recovery_pair_count,
        2
    );
    assert_eq!(recovery.recovered_source_face_diagonal_pair_count, 1);
    assert_eq!(
        recovery.recovered_source_face_diagonal_boundary_face_count,
        2
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_recovery_pair_count,
        1
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_adjacent_facet_count,
        0
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_adjacent_facet_topology_count,
        0
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_current_boundary_face_count,
        1
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_element_topology_count,
        0
    );
    assert_eq!(
        recovery.rejected_source_face_diagonal_material_region_mismatch_count,
        0
    );
    assert_eq!(recovery.rejected_source_face_diagonal_quality_gate_count, 0);
    assert_eq!(
        recovery.rejected_source_face_diagonal_unpaired_source_face_count,
        1
    );
    assert_eq!(recovery.repaired_boundary_face_identity_count, 1);
    assert_eq!(recovery.removed_redundant_boundary_face_count, 1);
    assert_eq!(recovery.removed_unsupported_boundary_face_count, 1);
    assert_eq!(recovery.attempted_boundary_leak_recovery_item_count, 2);
    assert_eq!(recovery.removed_exterior_leaked_element_count, 1);
    assert_eq!(recovery.exposed_interior_source_face_count, 1);
    assert_eq!(recovery.inserted_exposed_interior_boundary_face_count, 1);
    assert_eq!(recovery.rejected_boundary_leak_recovery_item_count, 4);
    assert_eq!(recovery.rejected_boundary_leak_adjacent_element_count, 1);
    assert_eq!(
        recovery.rejected_boundary_leak_material_region_mismatch_count,
        1
    );
    assert_eq!(
        recovery.rejected_boundary_leak_outside_classification_count,
        1
    );
    assert_eq!(
        recovery.rejected_boundary_leak_closed_surface_coordinate_count,
        1
    );
    assert_eq!(recovery.repaired_source_face_provenance_count, 1);
    assert_eq!(recovery.repaired_source_edge_provenance_count, 2);
    assert_eq!(recovery.repaired_material_interface_element_count, 3);
    assert_eq!(recovery.attempted_material_interface_recovery_item_count, 2);
    assert_eq!(recovery.rejected_material_interface_recovery_item_count, 1);
    assert_eq!(recovery.global_material_interface_recovery_item_count, 1);
    assert_eq!(
        recovery.boundary_owned_material_interface_recovery_item_count,
        1
    );
    assert_eq!(recovery.interior_material_interface_recovery_item_count, 1);
    assert_eq!(
        recovery.rejected_material_interface_missing_boundary_ownership_count,
        1
    );
    assert_eq!(
        recovery.rejected_material_interface_ambiguous_boundary_ownership_count,
        0
    );
    assert_eq!(
        recovery.attempted_absent_material_partition_recovery_item_count,
        2
    );
    assert_eq!(
        recovery.inserted_absent_material_partition_recovery_item_count,
        1
    );
    assert_eq!(recovery.inserted_absent_material_partition_element_count, 1);
    assert_eq!(
        recovery.inserted_absent_material_partition_boundary_face_count,
        3
    );
    assert_eq!(
        recovery.rejected_absent_material_partition_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.rolled_back_absent_material_partition_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.rolled_back_absent_material_partition_element_count,
        1
    );
    assert_eq!(
        recovery.rolled_back_absent_material_partition_boundary_face_count,
        2
    );
    assert_eq!(recovery.rejected_absent_material_partition_facet_count, 0);
    assert_eq!(
        recovery.rejected_absent_material_partition_facet_topology_count,
        1
    );
    assert_eq!(
        recovery.rejected_absent_material_partition_element_exists_count,
        0
    );
    assert_eq!(
        recovery.rejected_absent_material_partition_interior_face_topology_count,
        0
    );
    assert_eq!(
        recovery.rejected_absent_material_partition_quality_gate_count,
        0
    );
    assert_eq!(
        recovery.rejected_absent_material_partition_post_insertion_audit_count,
        1
    );
    assert_eq!(recovery.source_face_recovery_item_count, 5);
    assert_eq!(recovery.recovered_source_face_recovery_item_count, 3);
    assert_eq!(recovery.missing_source_face_recovery_item_count, 2);
    assert_eq!(recovery.missing_source_face_topology_recovery_item_count, 1);
    assert_eq!(
        recovery.missing_source_face_provenance_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_source_face_boundary_face_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_source_face_volume_face_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_source_face_interior_face_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_source_face_absent_face_recovery_item_count,
        0
    );
    assert_eq!(
        recovery.missing_source_face_recovery_ids,
        vec!["source_face_missing_1"]
    );
    assert_eq!(recovery.omitted_missing_source_face_recovery_id_count, 0);
    assert_eq!(recovery.source_edge_recovery_item_count, 4);
    assert_eq!(recovery.recovered_source_edge_recovery_item_count, 3);
    assert_eq!(recovery.missing_source_edge_recovery_item_count, 1);
    assert_eq!(recovery.missing_source_edge_topology_recovery_item_count, 1);
    assert_eq!(
        recovery.missing_source_edge_provenance_recovery_item_count,
        0
    );
    assert_eq!(
        recovery.missing_source_edge_volume_edge_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_source_edge_interior_edge_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_source_edge_absent_edge_recovery_item_count,
        0
    );
    assert_eq!(
        recovery.missing_source_edge_recovery_ids,
        vec!["source_edge_missing_1".to_string()]
    );
    assert_eq!(recovery.omitted_missing_source_edge_recovery_id_count, 0);
    assert_eq!(recovery.material_interface_recovery_item_count, 1);
    assert_eq!(recovery.recovered_material_interface_recovery_item_count, 0);
    assert_eq!(recovery.missing_material_interface_recovery_item_count, 1);
    assert_eq!(
        recovery.missing_material_interface_boundary_owned_recovery_item_count,
        1
    );
    assert_eq!(
        recovery.missing_material_interface_interior_face_recovery_item_count,
        2
    );
    assert_eq!(
        recovery.missing_material_interface_absent_partition_recovery_item_count,
        3
    );
    assert_eq!(
        recovery.missing_material_interface_recovery_ids,
        vec!["material_interface_missing_1".to_string()]
    );
    assert_eq!(
        recovery.omitted_missing_material_interface_recovery_id_count,
        0
    );
    assert_eq!(recovery.refinement_pass_count, 2);
    assert_eq!(recovery.refinement_point_count, 5);
    assert_eq!(recovery.optimization_pass_count, 1);
    assert_eq!(recovery.smoothed_point_count, 2);
    assert_eq!(recovery.sliver_count, 1);
    assert_eq!(recovery.sliver_removed_count, 2);
    assert_eq!(recovery.optimization_target_seed_count, 7);
    assert_eq!(recovery.optimization_skipped_target_seed_count, 4);
    assert_eq!(recovery.optimization_rejected_edit_count, 3);
    assert_eq!(recovery.optimization_initial_max_aspect_ratio, 6.0);
    assert_eq!(recovery.optimization_final_max_aspect_ratio, 4.0);
    assert_eq!(
        recovery.optimization_initial_min_exact_scaled_jacobian,
        0.32
    );
    assert_eq!(recovery.optimization_final_min_exact_scaled_jacobian, 0.40);
    assert_eq!(recovery.untangling_pass_count, 2);
    assert_eq!(recovery.untangling_initial_near_singular_count, 6);
    assert_eq!(recovery.untangling_final_near_singular_count, 1);
    assert_eq!(recovery.untangling_relocated_seed_count, 3);
    assert_eq!(recovery.untangling_reconnected_edge_star_count, 4);
    assert_eq!(
        recovery.untangling_reconnected_boundary_adjacent_cavity_count,
        5
    );
    assert_eq!(
        recovery.untangling_reconnected_node_adjacent_cavity_count,
        11
    );
    assert_eq!(recovery.exact_quality_repair_pass_count, 1);
    assert_eq!(recovery.exact_quality_reconnected_cavity_count, 2);
    assert_eq!(recovery.exact_quality_reconnection_quality_gain_count, 1);
    assert_eq!(
        recovery.exact_quality_face_neighbor_reconnected_cavity_count,
        6
    );
    assert_eq!(recovery.exact_quality_connected_reconnected_cavity_count, 7);
    assert_eq!(
        recovery.exact_quality_node_adjacent_reconnected_cavity_count,
        12
    );
    assert_eq!(
        recovery.exact_quality_boundary_adjacent_reconnected_cavity_count,
        8
    );
    assert_eq!(
        recovery.exact_quality_expanded_connected_reconnected_cavity_count,
        9
    );
    assert_eq!(recovery.exact_quality_split_cavity_count, 3);
    assert_eq!(recovery.exact_quality_seed_star_collapse_count, 4);
    assert_eq!(recovery.exact_quality_seed_star_relocation_count, 5);
    assert_eq!(recovery.exact_quality_unrepaired_total_count, 9);
    assert_eq!(recovery.exact_quality_unrepaired_general_cavity_count, 1);
    assert_eq!(recovery.exact_quality_unrepaired_boundary_adjacent_count, 6);
    assert_eq!(recovery.exact_quality_unrepaired_node_adjacent_count, 10);
    assert_eq!(recovery.exact_quality_unrepaired_interior_seed_count, 7);
    assert_eq!(recovery.exact_quality_unrepaired_edge_star_count, 8);
}

pub(super) fn assert_region_and_quality_evidence(evidence: &MeshEvidenceArtifact) {
    assert_eq!(
        evidence.regions.boundary_region_face_counts.get("fixed"),
        Some(&1)
    );
    assert_eq!(
        evidence.regions.material_region_volume_m3.get("solid"),
        Some(&(1.0 / 6.0))
    );
    assert_eq!(
        evidence
            .regions
            .boundary_region_recovered_face_counts
            .get("fixed"),
        Some(&1)
    );
    assert_eq!(evidence.quality.min_exact_scaled_jacobian, 0.45);
    assert_eq!(evidence.quality.scaled_jacobian_p05, Some(0.5));
    assert_eq!(evidence.quality.scaled_jacobian_p50, Some(0.5));
    assert_eq!(evidence.quality.scaled_jacobian_p95, Some(0.5));
    assert_eq!(evidence.quality.exact_scaled_jacobian_p05, Some(0.45));
    assert_eq!(evidence.quality.exact_scaled_jacobian_p50, Some(0.45));
    assert_eq!(evidence.quality.exact_scaled_jacobian_p95, Some(0.45));
    assert_eq!(evidence.quality.aspect_ratio_p50, Some(2.0));
    assert_eq!(evidence.quality.aspect_ratio_p95, Some(2.0));
    assert_eq!(
        evidence
            .quality
            .exact_scaled_jacobian_bins
            .get("0_35_to_0_65"),
        Some(&1)
    );
}

pub(super) fn assert_serialized_evidence(evidence: &MeshEvidenceArtifact) {
    let encoded = serde_json::to_value(evidence).expect("serialize evidence");
    assert!(encoded.get("sizing").is_some());
    assert!(encoded.get("debug").is_none());
    assert_eq!(
        encoded["cad"]["evaluation_source"],
        serde_json::Value::String("imported_evaluator_samples".to_string())
    );
    assert!(!encoded
        .to_string()
        .contains("sample detail should not be copied"));
}

pub(super) fn assert_failed_validation_is_reported(mesh: &AnalysisMeshArtifact) {
    let failed_validation = AnalysisMeshValidationOptions {
        max_volume_element_count: Some(0),
        ..AnalysisMeshValidationOptions::default()
    };
    let failed_evidence = build_mesh_evidence_artifact(mesh, &failed_validation);
    assert!(!failed_evidence.validation.solve_ready);
    assert_eq!(
        failed_evidence.validation.validation_error_code.as_deref(),
        Some("element_budget_exceeded")
    );
    assert!(failed_evidence
        .validation
        .validation_error_message
        .as_deref()
        .is_some_and(|message| message.contains("ElementBudgetExceeded")));
}

pub(super) fn assert_validation_refresh(
    mesh: &AnalysisMeshArtifact,
    evidence: &MeshEvidenceArtifact,
) {
    let mut stale_validation = evidence.validation.clone();
    stale_validation.solve_ready = false;
    stale_validation.validation_error_code = Some("stale".to_string());
    stale_validation.validation_error_message = Some("stale".to_string());
    stale_validation.volume_element_count = 999;
    stale_validation.volume_component_count = 999;
    stale_validation
        .boundary_recovery
        .boundary_edge_recovery_ratio = 0.0;

    let refreshed_evidence =
        build_mesh_evidence_artifact_with_validation_evidence(mesh, stale_validation);
    assert!(!refreshed_evidence.validation.solve_ready);
    assert_eq!(
        refreshed_evidence
            .validation
            .validation_error_code
            .as_deref(),
        Some("rolled_back_material_interface_partition_recovery_present")
    );
    assert_eq!(refreshed_evidence.validation.volume_element_count, 1);
    assert_eq!(refreshed_evidence.validation.volume_component_count, 1);
    assert_eq!(
        refreshed_evidence
            .validation
            .boundary_recovery
            .boundary_edge_recovery_ratio,
        1.0
    );
}
