use runmat_meshing_core::contracts::MeshBackendSummary;
use runmat_meshing_tetrahedron::{
    generate::TetrahedronMesh,
    recover::{TetrahedronRecoveryKind, TetrahedronRecoveryQueue},
};

use super::backend_counts::{
    bounded_missing_recovery_ids, recovery_entity_count, tetrahedron_material_region_count,
    tetrahedron_unclassified_material_element_count,
};
use super::backend_generation::plc_input_and_generation_summary;
use super::backend_optimization::optimization_summary;
use super::backend_quality::BackendQualityEvidence;

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
    let optimization_summary =
        optimization_summary(tetrahedron_mesh, initial_backend_quality, &backend_quality);
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
    ..plc_input_and_generation_summary(tetrahedron_mesh, optimization_summary)
}
}
