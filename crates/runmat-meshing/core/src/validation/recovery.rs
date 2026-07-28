use std::collections::BTreeSet;

use crate::contracts::AnalysisMeshArtifact;

use super::{connectivity::boundary_face_edges, AnalysisMeshValidationError};

pub(super) fn validate_no_unrecovered_tetrahedron_components(
    mesh: &AnalysisMeshArtifact,
    require_no_unrecovered_tetrahedron_components: bool,
) -> Result<(), AnalysisMeshValidationError> {
    if require_no_unrecovered_tetrahedron_components
        && mesh.backend.tetrahedron_unrecovered_component_count > 0
    {
        return Err(
            AnalysisMeshValidationError::UnrecoveredTetrahedronComponentsPresent {
                component_count: mesh.backend.tetrahedron_unrecovered_component_count,
            },
        );
    }
    Ok(())
}

pub(super) fn validate_no_rolled_back_material_interface_partitions(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let recovery_item_count = mesh
        .backend
        .tetrahedron_rolled_back_absent_material_partition_recovery_item_count;
    let element_count = mesh
        .backend
        .tetrahedron_rolled_back_absent_material_partition_element_count;
    let boundary_face_count = mesh
        .backend
        .tetrahedron_rolled_back_absent_material_partition_boundary_face_count;
    let post_insertion_audit_rejection_count = mesh
        .backend
        .tetrahedron_rejected_absent_material_partition_post_insertion_audit_count;

    if recovery_item_count > 0
        || element_count > 0
        || boundary_face_count > 0
        || post_insertion_audit_rejection_count > 0
    {
        return Err(
            AnalysisMeshValidationError::RolledBackMaterialInterfacePartitionRecoveryPresent {
                recovery_item_count,
                element_count,
                boundary_face_count,
                post_insertion_audit_rejection_count,
            },
        );
    }
    Ok(())
}

pub(super) fn validate_tetrahedron_recovery_complete(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let source_face_item_count = mesh
        .backend
        .tetrahedron_missing_source_face_recovery_item_count;
    let source_edge_item_count = mesh
        .backend
        .tetrahedron_missing_source_edge_recovery_item_count;
    let material_interface_item_count = mesh
        .backend
        .tetrahedron_missing_material_interface_recovery_item_count;
    let total_item_count = mesh
        .backend
        .tetrahedron_missing_recovery_item_count
        .max(source_face_item_count + source_edge_item_count + material_interface_item_count);

    if total_item_count > 0 {
        return Err(
            AnalysisMeshValidationError::IncompleteTetrahedronRecoveryPresent {
                missing_item_count: total_item_count,
                missing_source_face_item_count: source_face_item_count,
                missing_source_edge_item_count: source_edge_item_count,
                missing_material_interface_item_count: material_interface_item_count,
            },
        );
    }
    Ok(())
}

pub(super) fn validate_recovery_evidence_consistency(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let backend = &mesh.backend;
    validate_recovery_item_count(
        "attempted_protected_edge_boundary_face_restoration",
        backend.tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count,
        backend
            .tetrahedron_volume_edge_source_edge_recovery_item_count
            .saturating_mul(2),
    )?;
    validate_recovery_item_count(
        "attempted_cad_curve_protected_edge_boundary_face_restoration",
        backend.tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count,
        backend.tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_recovery_item_count(
        "recovered_cad_curve_protected_edge_boundary_face",
        backend.tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count,
        backend.tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_recovery_item_count(
        "recovered_protected_edge_boundary_face",
        backend.tetrahedron_recovered_protected_edge_boundary_face_count,
        backend.tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_protected_edge_boundary_face_restoration",
        backend.tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count,
        backend.tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_cad_curve_protected_edge_boundary_face_restoration",
        backend.tetrahedron_rejected_cad_curve_protected_edge_boundary_face_restoration_item_count,
        backend.tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_aggregate_recovery_count(
        "protected_edge_boundary_face_restoration_status_items",
        backend.tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count,
        backend.tetrahedron_recovered_protected_edge_boundary_face_count
            + backend.tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_aggregate_recovery_count(
        "cad_curve_protected_edge_boundary_face_restoration_status_items",
        backend.tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count,
        backend.tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count
            + backend
                .tetrahedron_rejected_cad_curve_protected_edge_boundary_face_restoration_item_count,
    )?;
    validate_aggregate_recovery_count(
        "protected_edge_boundary_face_restoration_rejection_reason_items",
        backend.tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count,
        backend
            .tetrahedron_rejected_protected_edge_boundary_face_restoration_volume_face_topology_count,
    )?;
    validate_recovered_count(
        "volume_edge_source_edge",
        backend.tetrahedron_recovered_volume_edge_source_edge_recovery_item_count,
        backend.tetrahedron_volume_edge_source_edge_recovery_item_count,
    )?;
    validate_recovered_count(
        "boundary_edge_source_edge",
        backend.tetrahedron_recovered_boundary_edge_source_edge_recovery_item_count,
        backend.tetrahedron_boundary_edge_source_edge_recovery_item_count,
    )?;
    validate_recovered_count(
        "interior_edge_source_edge",
        backend.tetrahedron_recovered_interior_edge_source_edge_recovery_item_count,
        backend.tetrahedron_interior_edge_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "cad_curve_interior_edge_source_edge",
        backend.tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count,
        backend.tetrahedron_interior_edge_source_edge_recovery_item_count,
    )?;
    validate_recovered_count(
        "cad_curve_interior_edge_source_edge",
        backend.tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count,
        backend.tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "recovered_cad_curve_interior_edge_source_edge",
        backend.tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count,
        backend.tetrahedron_recovered_interior_edge_source_edge_recovery_item_count,
    )?;
    let source_edge_split_refill_input_count = backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count
        + backend.tetrahedron_interior_edge_source_edge_recovery_item_count;
    validate_recovery_item_count(
        "attempted_source_edge_split_refill",
        backend.tetrahedron_attempted_source_edge_split_refill_item_count,
        source_edge_split_refill_input_count,
    )?;
    validate_recovery_item_count(
        "attempted_cad_curve_source_edge_split_refill",
        backend.tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_attempted_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "accepted_source_edge_split_refill_candidate",
        backend.tetrahedron_accepted_source_edge_split_refill_candidate_item_count,
        backend.tetrahedron_attempted_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "accepted_cad_curve_source_edge_split_refill_candidate",
        backend.tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count,
        backend.tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "accepted_cad_curve_source_edge_split_refill_candidate",
        backend.tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count,
        backend.tetrahedron_accepted_source_edge_split_refill_candidate_item_count,
    )?;
    validate_recovery_item_count(
        "applied_source_edge_split_refill",
        backend.tetrahedron_applied_source_edge_split_refill_item_count,
        backend.tetrahedron_accepted_source_edge_split_refill_candidate_item_count,
    )?;
    validate_recovery_item_count(
        "post_repair_attempted_source_edge_split_refill",
        backend.tetrahedron_post_repair_attempted_source_edge_split_refill_item_count,
        backend.tetrahedron_accepted_source_edge_split_refill_candidate_item_count,
    )?;
    validate_recovery_item_count(
        "applied_source_edge_split_refill",
        backend.tetrahedron_applied_source_edge_split_refill_item_count,
        backend.tetrahedron_post_repair_attempted_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "post_repair_rejected_source_edge_split_refill",
        backend.tetrahedron_post_repair_rejected_source_edge_split_refill_item_count,
        backend.tetrahedron_post_repair_attempted_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "applied_cad_curve_source_edge_split_refill",
        backend.tetrahedron_applied_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count,
    )?;
    validate_recovery_item_count(
        "post_repair_attempted_cad_curve_source_edge_split_refill",
        backend.tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count,
    )?;
    validate_recovery_item_count(
        "post_repair_attempted_cad_curve_source_edge_split_refill",
        backend.tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_post_repair_attempted_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "applied_cad_curve_source_edge_split_refill",
        backend.tetrahedron_applied_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "post_repair_rejected_cad_curve_source_edge_split_refill",
        backend.tetrahedron_post_repair_rejected_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "applied_cad_curve_source_edge_split_refill",
        backend.tetrahedron_applied_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_applied_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_source_edge_split_refill",
        backend.tetrahedron_rejected_source_edge_split_refill_item_count,
        backend.tetrahedron_attempted_source_edge_split_refill_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_cad_curve_source_edge_split_refill",
        backend.tetrahedron_rejected_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count,
    )?;
    validate_aggregate_recovery_count(
        "source_edge_split_refill_status_items",
        backend.tetrahedron_attempted_source_edge_split_refill_item_count,
        backend.tetrahedron_accepted_source_edge_split_refill_candidate_item_count
            + backend.tetrahedron_rejected_source_edge_split_refill_item_count,
    )?;
    validate_aggregate_recovery_count(
        "cad_curve_source_edge_split_refill_status_items",
        backend.tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count
            + backend.tetrahedron_rejected_cad_curve_source_edge_split_refill_item_count,
    )?;
    validate_aggregate_recovery_count(
        "post_repair_source_edge_split_refill_status_items",
        backend.tetrahedron_post_repair_attempted_source_edge_split_refill_item_count,
        backend.tetrahedron_applied_source_edge_split_refill_item_count
            + backend.tetrahedron_post_repair_rejected_source_edge_split_refill_item_count,
    )?;
    validate_aggregate_recovery_count(
        "post_repair_cad_curve_source_edge_split_refill_status_items",
        backend.tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count,
        backend.tetrahedron_applied_cad_curve_source_edge_split_refill_item_count
            + backend
                .tetrahedron_post_repair_rejected_cad_curve_source_edge_split_refill_item_count,
    )?;
    validate_recovered_count(
        "absent_edge_source_edge",
        backend.tetrahedron_recovered_absent_edge_source_edge_recovery_item_count,
        backend.tetrahedron_absent_edge_source_edge_recovery_item_count,
    )?;
    validate_recovered_count(
        "boundary_face_source_face",
        backend.tetrahedron_recovered_boundary_face_source_face_recovery_item_count,
        backend.tetrahedron_boundary_face_source_face_recovery_item_count,
    )?;
    validate_recovered_count(
        "interior_face_source_face",
        backend.tetrahedron_recovered_interior_face_source_face_recovery_item_count,
        backend.tetrahedron_interior_face_source_face_recovery_item_count,
    )?;
    validate_recovered_count(
        "volume_face_source_face",
        backend.tetrahedron_recovered_volume_face_source_face_recovery_item_count,
        backend.tetrahedron_volume_face_source_face_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "attempted_volume_face_source_face_boundary_restoration",
        backend.tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count,
        backend.tetrahedron_volume_face_source_face_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_volume_face_source_face_boundary_restoration",
        backend.tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count,
        backend.tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count,
    )?;
    validate_aggregate_recovery_count(
        "volume_face_source_face_boundary_restoration_status_items",
        backend.tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count,
        backend.tetrahedron_recovered_volume_face_source_face_recovery_item_count
            + backend.tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count,
    )?;
    validate_aggregate_recovery_count(
        "volume_face_source_face_boundary_restoration_rejection_reason_items",
        backend.tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count,
        backend
            .tetrahedron_rejected_volume_face_source_face_boundary_restoration_volume_face_topology_count,
    )?;
    validate_recovered_count(
        "absent_face_source_face",
        backend.tetrahedron_recovered_absent_face_source_face_recovery_item_count,
        backend.tetrahedron_absent_face_source_face_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "deferred_absent_source_edge",
        backend.tetrahedron_deferred_absent_source_edge_recovery_item_count,
        backend.tetrahedron_absent_edge_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "attempted_absent_source_edge",
        backend.tetrahedron_attempted_absent_source_edge_recovery_item_count,
        backend.tetrahedron_absent_edge_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "attempted_cad_curve_absent_source_edge",
        backend.tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count,
        backend.tetrahedron_attempted_absent_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "reconnected_absent_source_edge",
        backend.tetrahedron_reconnected_absent_source_edge_recovery_item_count,
        backend.tetrahedron_attempted_absent_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "reconnected_cad_curve_absent_source_edge",
        backend.tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count,
        backend.tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_absent_source_edge",
        backend.tetrahedron_rejected_absent_source_edge_recovery_item_count,
        backend.tetrahedron_attempted_absent_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_cad_curve_absent_source_edge",
        backend.tetrahedron_rejected_cad_curve_absent_source_edge_recovery_item_count,
        backend.tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "absent_source_edge_status_items",
        backend.tetrahedron_attempted_absent_source_edge_recovery_item_count,
        backend.tetrahedron_reconnected_absent_source_edge_recovery_item_count
            + backend.tetrahedron_rejected_absent_source_edge_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "cad_curve_absent_source_edge_status_items",
        backend.tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count,
        backend.tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count
            + backend.tetrahedron_rejected_cad_curve_absent_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "attempted_source_face_diagonal_pair",
        backend.tetrahedron_attempted_source_face_diagonal_recovery_pair_count,
        backend.tetrahedron_absent_face_source_face_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "recovered_source_face_diagonal_pair",
        backend.tetrahedron_recovered_source_face_diagonal_pair_count,
        backend.tetrahedron_attempted_source_face_diagonal_recovery_pair_count,
    )?;
    validate_recovery_item_count(
        "rejected_source_face_diagonal_pair",
        backend.tetrahedron_rejected_source_face_diagonal_recovery_pair_count,
        backend.tetrahedron_attempted_source_face_diagonal_recovery_pair_count,
    )?;
    validate_recovery_item_count(
        "rejected_source_face_diagonal_item",
        backend.tetrahedron_rejected_source_face_diagonal_recovery_item_count,
        backend.tetrahedron_absent_face_source_face_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "source_face_diagonal_pair_status_items",
        backend.tetrahedron_attempted_source_face_diagonal_recovery_pair_count,
        backend.tetrahedron_recovered_source_face_diagonal_pair_count
            + backend.tetrahedron_rejected_source_face_diagonal_recovery_pair_count,
    )?;
    validate_recovery_item_count(
        "attempted_material_interface",
        backend.tetrahedron_attempted_material_interface_recovery_item_count,
        backend.tetrahedron_material_interface_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_material_interface",
        backend.tetrahedron_rejected_material_interface_recovery_item_count,
        backend.tetrahedron_attempted_material_interface_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "material_interface_status_items",
        backend.tetrahedron_attempted_material_interface_recovery_item_count,
        backend.tetrahedron_recovered_material_interface_recovery_item_count
            + backend.tetrahedron_rejected_material_interface_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "material_interface_rejection_reason_items",
        backend.tetrahedron_rejected_material_interface_recovery_item_count,
        backend.tetrahedron_rejected_material_interface_missing_boundary_ownership_count
            + backend.tetrahedron_rejected_material_interface_missing_interior_ownership_count
            + backend.tetrahedron_rejected_material_interface_ambiguous_boundary_ownership_count
            + backend.tetrahedron_rejected_material_interface_absent_partition_count,
    )?;
    validate_recovered_count(
        "material_interface",
        backend.tetrahedron_recovered_material_interface_recovery_item_count,
        backend.tetrahedron_material_interface_recovery_item_count,
    )?;
    validate_recovered_count(
        "boundary_owned_material_interface",
        backend.tetrahedron_recovered_boundary_owned_material_interface_recovery_item_count,
        backend.tetrahedron_boundary_owned_material_interface_recovery_item_count,
    )?;
    validate_recovered_count(
        "interior_face_material_interface",
        backend.tetrahedron_recovered_interior_face_material_interface_recovery_item_count,
        backend.tetrahedron_interior_material_interface_recovery_item_count,
    )?;
    validate_recovered_count(
        "absent_partition_material_interface",
        backend.tetrahedron_recovered_absent_partition_material_interface_recovery_item_count,
        backend.tetrahedron_missing_material_interface_absent_partition_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "attempted_absent_material_partition",
        backend.tetrahedron_attempted_absent_material_partition_recovery_item_count,
        backend.tetrahedron_missing_material_interface_absent_partition_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "inserted_absent_material_partition",
        backend.tetrahedron_inserted_absent_material_partition_recovery_item_count,
        backend.tetrahedron_attempted_absent_material_partition_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_absent_material_partition",
        backend.tetrahedron_rejected_absent_material_partition_recovery_item_count,
        backend.tetrahedron_attempted_absent_material_partition_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rolled_back_absent_material_partition",
        backend.tetrahedron_rolled_back_absent_material_partition_recovery_item_count,
        backend.tetrahedron_attempted_absent_material_partition_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "absent_material_partition_status_items",
        backend.tetrahedron_attempted_absent_material_partition_recovery_item_count,
        backend.tetrahedron_inserted_absent_material_partition_recovery_item_count
            + backend.tetrahedron_rejected_absent_material_partition_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "absent_material_partition_rejection_reason_items",
        backend.tetrahedron_rejected_absent_material_partition_recovery_item_count,
        backend.tetrahedron_rejected_absent_material_partition_facet_count
            + backend.tetrahedron_rejected_absent_material_partition_facet_topology_count
            + backend.tetrahedron_rejected_absent_material_partition_element_exists_count
            + backend.tetrahedron_rejected_absent_material_partition_interior_face_topology_count
            + backend.tetrahedron_rejected_absent_material_partition_quality_gate_count
            + backend.tetrahedron_rejected_absent_material_partition_post_insertion_audit_count,
    )?;
    let typed_recovery_item_count = backend.tetrahedron_source_face_recovery_item_count
        + backend.tetrahedron_source_edge_recovery_item_count
        + backend.tetrahedron_material_interface_recovery_item_count;
    validate_aggregate_recovery_count(
        "recovery_items",
        backend.tetrahedron_recovery_item_count,
        typed_recovery_item_count,
    )?;
    let typed_missing_item_count = backend.tetrahedron_missing_source_face_recovery_item_count
        + backend.tetrahedron_missing_source_edge_recovery_item_count
        + backend.tetrahedron_missing_material_interface_recovery_item_count;
    validate_aggregate_recovery_count(
        "missing_items",
        backend.tetrahedron_missing_recovery_item_count,
        typed_missing_item_count,
    )?;
    validate_missing_recovery_reason_counts(mesh)?;
    validate_aggregate_recovery_count(
        "recovery_status_items",
        backend.tetrahedron_recovery_item_count,
        backend.tetrahedron_recovered_item_count + backend.tetrahedron_missing_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "recovered_items",
        backend.tetrahedron_recovered_item_count,
        backend.tetrahedron_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "missing_items",
        backend.tetrahedron_missing_recovery_item_count,
        backend.tetrahedron_recovery_item_count,
    )?;
    validate_recovered_count(
        "cad_curve_source_edge",
        backend.tetrahedron_recovered_cad_curve_source_edge_recovery_item_count,
        backend.tetrahedron_cad_curve_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "missing_cad_curve_source_edge",
        backend.tetrahedron_missing_cad_curve_source_edge_recovery_item_count,
        backend.tetrahedron_cad_curve_source_edge_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "cad_curve_source_edge_status_items",
        backend.tetrahedron_cad_curve_source_edge_recovery_item_count,
        backend.tetrahedron_recovered_cad_curve_source_edge_recovery_item_count
            + backend.tetrahedron_missing_cad_curve_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "missing_cad_curve_source_edge_topology",
        backend.tetrahedron_missing_cad_curve_source_edge_topology_recovery_item_count,
        backend.tetrahedron_missing_cad_curve_source_edge_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "missing_cad_curve_source_edge_provenance",
        backend.tetrahedron_missing_cad_curve_source_edge_provenance_recovery_item_count,
        backend.tetrahedron_missing_cad_curve_source_edge_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "cad_curve_source_edge_missing_reason_items",
        backend.tetrahedron_missing_cad_curve_source_edge_recovery_item_count,
        backend.tetrahedron_missing_cad_curve_source_edge_topology_recovery_item_count
            + backend.tetrahedron_missing_cad_curve_source_edge_provenance_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "rejected_boundary_leak",
        backend.tetrahedron_rejected_boundary_leak_recovery_item_count,
        backend.tetrahedron_attempted_boundary_leak_recovery_item_count,
    )?;
    validate_recovery_item_count(
        "exposed_boundary_leak_source_face",
        backend.tetrahedron_exposed_interior_source_face_count,
        backend.tetrahedron_attempted_boundary_leak_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "boundary_leak_status_items",
        backend.tetrahedron_attempted_boundary_leak_recovery_item_count,
        backend.tetrahedron_exposed_interior_source_face_count
            + backend.tetrahedron_rejected_boundary_leak_recovery_item_count,
    )?;
    validate_aggregate_recovery_count(
        "boundary_leak_rejection_reason_items",
        backend.tetrahedron_rejected_boundary_leak_recovery_item_count,
        backend.tetrahedron_rejected_boundary_leak_adjacent_element_count
            + backend.tetrahedron_rejected_boundary_leak_material_region_mismatch_count
            + backend.tetrahedron_rejected_boundary_leak_outside_classification_count
            + backend.tetrahedron_rejected_boundary_leak_closed_surface_coordinate_count,
    )?;
    Ok(())
}

fn validate_missing_recovery_reason_counts(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let backend = &mesh.backend;
    for (family, item_count) in [
        (
            "missing_source_face_topology",
            backend.tetrahedron_missing_source_face_topology_recovery_item_count,
        ),
        (
            "missing_source_face_provenance",
            backend.tetrahedron_missing_source_face_provenance_recovery_item_count,
        ),
        (
            "missing_source_face_boundary_face",
            backend.tetrahedron_missing_source_face_boundary_face_recovery_item_count,
        ),
        (
            "missing_source_face_volume_face",
            backend.tetrahedron_missing_source_face_volume_face_recovery_item_count,
        ),
        (
            "missing_source_face_interior_face",
            backend.tetrahedron_missing_source_face_interior_face_recovery_item_count,
        ),
        (
            "missing_source_face_absent_face",
            backend.tetrahedron_missing_source_face_absent_face_recovery_item_count,
        ),
    ] {
        validate_recovery_item_count(
            family,
            item_count,
            backend.tetrahedron_missing_source_face_recovery_item_count,
        )?;
    }
    for (family, item_count) in [
        (
            "missing_source_edge_topology",
            backend.tetrahedron_missing_source_edge_topology_recovery_item_count,
        ),
        (
            "missing_source_edge_provenance",
            backend.tetrahedron_missing_source_edge_provenance_recovery_item_count,
        ),
        (
            "missing_source_edge_volume_edge",
            backend.tetrahedron_missing_source_edge_volume_edge_recovery_item_count,
        ),
        (
            "missing_source_edge_interior_edge",
            backend.tetrahedron_missing_source_edge_interior_edge_recovery_item_count,
        ),
        (
            "missing_source_edge_absent_edge",
            backend.tetrahedron_missing_source_edge_absent_edge_recovery_item_count,
        ),
    ] {
        validate_recovery_item_count(
            family,
            item_count,
            backend.tetrahedron_missing_source_edge_recovery_item_count,
        )?;
    }
    for (family, item_count) in [
        (
            "missing_material_interface_boundary_owned",
            backend.tetrahedron_missing_material_interface_boundary_owned_recovery_item_count,
        ),
        (
            "missing_material_interface_interior_face",
            backend.tetrahedron_missing_material_interface_interior_face_recovery_item_count,
        ),
        (
            "missing_material_interface_absent_partition",
            backend.tetrahedron_missing_material_interface_absent_partition_recovery_item_count,
        ),
    ] {
        validate_recovery_item_count(
            family,
            item_count,
            backend.tetrahedron_missing_material_interface_recovery_item_count,
        )?;
    }
    Ok(())
}

fn validate_aggregate_recovery_count(
    family: &str,
    aggregate_count: usize,
    typed_count: usize,
) -> Result<(), AnalysisMeshValidationError> {
    if aggregate_count != typed_count {
        return Err(
            AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
                family: family.to_string(),
                aggregate_count,
                typed_count,
            },
        );
    }
    Ok(())
}

fn validate_recovery_item_count(
    family: &str,
    item_count: usize,
    input_count: usize,
) -> Result<(), AnalysisMeshValidationError> {
    if item_count > input_count {
        return Err(
            AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
                family: family.to_string(),
                item_count,
                input_count,
            },
        );
    }
    Ok(())
}

fn validate_recovered_count(
    family: &str,
    recovered_count: usize,
    input_count: usize,
) -> Result<(), AnalysisMeshValidationError> {
    if recovered_count > input_count {
        return Err(
            AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
                family: family.to_string(),
                recovered_count,
                input_count,
            },
        );
    }
    Ok(())
}

pub(super) fn validate_no_unrepaired_exact_quality(
    mesh: &AnalysisMeshArtifact,
    require_no_unrepaired_exact_quality: bool,
) -> Result<(), AnalysisMeshValidationError> {
    if !require_no_unrepaired_exact_quality {
        return Ok(());
    }
    let boundary_adjacent_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_boundary_adjacent_count;
    let general_cavity_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_general_cavity_count;
    let interior_seed_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_interior_seed_count;
    let node_adjacent_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_node_adjacent_count;
    let edge_star_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_edge_star_count;
    let categorized_lower_bound = [
        boundary_adjacent_count,
        node_adjacent_count,
        interior_seed_count,
        edge_star_count,
        general_cavity_count,
    ]
    .into_iter()
    .max()
    .unwrap_or_default();
    let total_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_total_count
        .max(categorized_lower_bound);
    if total_count > 0 {
        return Err(AnalysisMeshValidationError::UnrepairedExactQualityPresent {
            total_count,
            general_cavity_count,
            boundary_adjacent_count,
            node_adjacent_count,
            interior_seed_count,
            edge_star_count,
        });
    }
    Ok(())
}

pub(super) fn validate_boundary_face_recovery(
    mesh: &AnalysisMeshArtifact,
    min_boundary_face_recovery_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.boundary_faces.is_empty()
        || !min_boundary_face_recovery_ratio.is_finite()
        || min_boundary_face_recovery_ratio <= 0.0
    {
        return Ok(());
    }
    let recovered_count = mesh
        .boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .count();
    let recovery_ratio = recovered_count as f64 / mesh.boundary_faces.len() as f64;
    if recovery_ratio + 1.0e-9 < min_boundary_face_recovery_ratio {
        return Err(AnalysisMeshValidationError::BoundaryFaceRecoveryFailed {
            recovery_ratio: format!("{recovery_ratio:.6}"),
            required_ratio: format!("{min_boundary_face_recovery_ratio:.6}"),
        });
    }
    Ok(())
}

pub(super) fn validate_boundary_edge_recovery(
    mesh: &AnalysisMeshArtifact,
    recovered_boundary_edges: &BTreeSet<[u32; 2]>,
    min_boundary_edge_recovery_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.boundary_faces.is_empty()
        || !min_boundary_edge_recovery_ratio.is_finite()
        || min_boundary_edge_recovery_ratio <= 0.0
    {
        return Ok(());
    }
    let expected_edges = boundary_face_edges(mesh);
    if expected_edges.is_empty() {
        return Ok(());
    }
    let recovered_count = expected_edges
        .iter()
        .filter(|edge| recovered_boundary_edges.contains(*edge))
        .count();
    let recovery_ratio = recovered_count as f64 / expected_edges.len() as f64;
    if recovery_ratio + 1.0e-9 < min_boundary_edge_recovery_ratio {
        return Err(AnalysisMeshValidationError::BoundaryEdgeRecoveryFailed {
            recovery_ratio: format!("{recovery_ratio:.6}"),
            required_ratio: format!("{min_boundary_edge_recovery_ratio:.6}"),
        });
    }
    Ok(())
}
