mod absent_edges;
mod boundary_diagonal;
mod boundary_faces;
mod boundary_leaks;
pub mod boundary_queue;
mod input_validation;
mod material_interfaces;
mod material_partitions;
mod queue;
mod source_edges;
mod source_face_coverage;
mod source_faces;
mod topology;
mod types;

use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TetrahedronMesh};

use absent_edges::recover_absent_protected_edges_by_boundary_diagonal_flip;
use boundary_faces::{
    recover_missing_protected_edge_boundary_faces, recover_volume_face_source_face_boundary_faces,
    remove_redundant_boundary_faces, remove_unsupported_boundary_faces,
    repair_boundary_face_identity, repair_boundary_source_edge_provenance,
    repair_boundary_source_face_provenance,
};
use boundary_leaks::remove_exterior_elements_across_interior_source_faces;
use material_interfaces::recover_material_interface_regions;
use material_partitions::recover_absent_material_interface_partitions;
pub use queue::build_recovery_queue_from_plc;
use source_edges::{
    apply_source_edge_split_refill_recovery, evaluate_source_edge_split_refill_recovery,
};
use source_faces::recover_source_faces_by_boundary_diagonal_flip;
pub use types::{
    TetrahedronMaterialInterfaceTopology, TetrahedronProtectedEdgeTopology,
    TetrahedronRecoveryError, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryQueueItem, TetrahedronRecoveryResult, TetrahedronRecoveryStatus,
    TetrahedronSourceFaceTopology,
};

pub const MODULE_PURPOSE: &str = "source-edge, source-face, and material-interface recovery queues";

pub fn mark_tetrahedron_mesh_recovery_state(
    tetrahedron_mesh: &mut TetrahedronMesh,
    recovery_queue: &TetrahedronRecoveryQueue,
) {
    tetrahedron_mesh.recovery_complete = recovery_queue
        .items
        .iter()
        .all(|item| item.status == TetrahedronRecoveryStatus::Recovered);
}

pub fn recover_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
    mut tetrahedron_mesh: TetrahedronMesh,
) -> Result<TetrahedronRecoveryResult, TetrahedronRecoveryError> {
    let initial_recovery_queue = build_recovery_queue_from_plc(plc, &tetrahedron_mesh)?;
    let volume_edge_source_edge_recovery_item_count = recovery_source_edge_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronProtectedEdgeTopology::VolumeEdge,
    );
    let boundary_edge_source_edge_recovery_item_count = recovery_source_edge_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronProtectedEdgeTopology::BoundaryEdge,
    );
    let interior_edge_source_edge_recovery_item_count = recovery_source_edge_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronProtectedEdgeTopology::InteriorEdge,
    );
    let cad_curve_interior_edge_source_edge_recovery_item_count =
        recovery_cad_curve_source_edge_item_count_by_topology(
            &initial_recovery_queue,
            plc,
            TetrahedronProtectedEdgeTopology::InteriorEdge,
        );
    let absent_edge_source_edge_recovery_item_count = recovery_source_edge_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronProtectedEdgeTopology::Absent,
    );
    let deferred_absent_source_edge_recovery_item_count =
        absent_edge_source_edge_recovery_item_count;
    let volume_face_source_face_recovery_item_count = recovery_source_face_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronSourceFaceTopology::VolumeFace,
    );
    let boundary_face_source_face_recovery_item_count = recovery_source_face_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronSourceFaceTopology::BoundaryFace,
    );
    let interior_face_source_face_recovery_item_count = recovery_source_face_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronSourceFaceTopology::InteriorFace,
    );
    let absent_face_source_face_recovery_item_count = recovery_source_face_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronSourceFaceTopology::Absent,
    );
    let boundary_owned_material_interface_recovery_item_count =
        recovery_material_interface_item_count_by_topology(
            &initial_recovery_queue,
            TetrahedronMaterialInterfaceTopology::BoundaryOwned,
        );
    let interior_face_material_interface_recovery_item_count =
        recovery_material_interface_item_count_by_topology(
            &initial_recovery_queue,
            TetrahedronMaterialInterfaceTopology::InteriorFace,
        );
    let absent_partition_material_interface_recovery_item_count =
        recovery_material_interface_item_count_by_topology(
            &initial_recovery_queue,
            TetrahedronMaterialInterfaceTopology::AbsentPartition,
        );
    let recovered_boundary_leaks = remove_exterior_elements_across_interior_source_faces(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let source_edge_split_refill_recovery =
        evaluate_source_edge_split_refill_recovery(plc, &initial_recovery_queue, &tetrahedron_mesh);
    let recovered_material_partitions = recover_absent_material_interface_partitions(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let recovered_absent_source_edges = recover_absent_protected_edges_by_boundary_diagonal_flip(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let recovered_source_faces = recover_source_faces_by_boundary_diagonal_flip(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let protected_edge_boundary_faces = recover_missing_protected_edge_boundary_faces(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let source_face_boundary_faces = recover_volume_face_source_face_boundary_faces(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let removed_unsupported_boundary_face_count =
        remove_unsupported_boundary_faces(&mut tetrahedron_mesh);
    let removed_redundant_boundary_face_count =
        remove_redundant_boundary_faces(plc, &mut tetrahedron_mesh);
    let repaired_boundary_face_identity_count =
        repair_boundary_face_identity(plc, &mut tetrahedron_mesh);
    let repaired_source_face_provenance_count =
        repair_boundary_source_face_provenance(&initial_recovery_queue, &mut tetrahedron_mesh);
    let repaired_source_edge_provenance =
        repair_boundary_source_edge_provenance(plc, &mut tetrahedron_mesh);
    let split_refill_application_queue = build_recovery_queue_from_plc(plc, &tetrahedron_mesh)?;
    let applied_source_edge_split_refill_recovery = apply_source_edge_split_refill_recovery(
        plc,
        &split_refill_application_queue,
        &mut tetrahedron_mesh,
    );
    let material_interface_recovery_queue = build_recovery_queue_from_plc(plc, &tetrahedron_mesh)?;
    let material_interface_recovery = recover_material_interface_regions(
        plc,
        &material_interface_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let mut recovery_queue = build_recovery_queue_from_plc(plc, &tetrahedron_mesh)?;
    record_recovered_queue_item_counts(&initial_recovery_queue, &mut recovery_queue);
    let recovered_cad_curve_interior_edge_source_edge_recovery_item_count =
        cad_curve_interior_edge_source_edge_recovery_item_count.saturating_sub(
            recovery_cad_curve_source_edge_item_count_by_topology(
                &recovery_queue,
                plc,
                TetrahedronProtectedEdgeTopology::InteriorEdge,
            ),
        );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_missing_boundary_faces".to_string(),
        protected_edge_boundary_faces.recovered_boundary_face_count
            + source_face_boundary_faces.recovered_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_protected_edge_boundary_faces".to_string(),
        protected_edge_boundary_faces.recovered_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_cad_curve_protected_edge_boundary_faces".to_string(),
        protected_edge_boundary_faces.recovered_cad_curve_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_protected_edge_boundary_face_restoration_items".to_string(),
        protected_edge_boundary_faces.attempted_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_cad_curve_protected_edge_boundary_face_restoration_items".to_string(),
        protected_edge_boundary_faces.attempted_cad_curve_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_protected_edge_boundary_face_restoration_items".to_string(),
        protected_edge_boundary_faces.rejected_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_cad_curve_protected_edge_boundary_face_restoration_items".to_string(),
        protected_edge_boundary_faces.rejected_cad_curve_boundary_face_count,
    );
    for (reason_key, count) in protected_edge_boundary_faces.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(format!("protected_edge_{reason_key}"), count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "volume_edge_source_edge_recovery_items".to_string(),
        volume_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "boundary_edge_source_edge_recovery_items".to_string(),
        boundary_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "interior_edge_source_edge_recovery_items".to_string(),
        interior_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "cad_curve_interior_edge_source_edge_recovery_items".to_string(),
        cad_curve_interior_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_cad_curve_interior_edge_source_edge_items".to_string(),
        recovered_cad_curve_interior_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_source_edge_split_refill_items".to_string(),
        source_edge_split_refill_recovery.attempted_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_cad_curve_source_edge_split_refill_items".to_string(),
        source_edge_split_refill_recovery.attempted_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "accepted_source_edge_split_refill_candidate_items".to_string(),
        source_edge_split_refill_recovery.accepted_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "accepted_cad_curve_source_edge_split_refill_candidate_items".to_string(),
        source_edge_split_refill_recovery.accepted_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_source_edge_split_refill_items".to_string(),
        source_edge_split_refill_recovery.rejected_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_cad_curve_source_edge_split_refill_items".to_string(),
        source_edge_split_refill_recovery.rejected_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "post_repair_attempted_source_edge_split_refill_items".to_string(),
        applied_source_edge_split_refill_recovery.attempted_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "post_repair_attempted_cad_curve_source_edge_split_refill_items".to_string(),
        applied_source_edge_split_refill_recovery.attempted_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "applied_source_edge_split_refill_items".to_string(),
        applied_source_edge_split_refill_recovery.applied_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "applied_cad_curve_source_edge_split_refill_items".to_string(),
        applied_source_edge_split_refill_recovery.applied_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "post_repair_rejected_source_edge_split_refill_items".to_string(),
        applied_source_edge_split_refill_recovery.rejected_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "post_repair_rejected_cad_curve_source_edge_split_refill_items".to_string(),
        applied_source_edge_split_refill_recovery.rejected_cad_curve_source_edge_count,
    );
    for (reason_key, count) in applied_source_edge_split_refill_recovery.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(format!("post_repair_{reason_key}"), count);
    }
    for (reason_key, count) in source_edge_split_refill_recovery.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(reason_key, count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "absent_edge_source_edge_recovery_items".to_string(),
        absent_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "volume_face_source_face_recovery_items".to_string(),
        volume_face_source_face_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_volume_face_source_face_boundary_restoration_items".to_string(),
        source_face_boundary_faces.attempted_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_volume_face_source_face_boundary_restoration_items".to_string(),
        source_face_boundary_faces.rejected_boundary_face_count,
    );
    for (reason_key, count) in source_face_boundary_faces.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(format!("source_face_{reason_key}"), count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "boundary_face_source_face_recovery_items".to_string(),
        boundary_face_source_face_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "interior_face_source_face_recovery_items".to_string(),
        interior_face_source_face_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "absent_face_source_face_recovery_items".to_string(),
        absent_face_source_face_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "deferred_absent_source_edge_recovery_items".to_string(),
        deferred_absent_source_edge_recovery_item_count
            .saturating_sub(recovered_absent_source_edges.source_edge_count),
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_absent_source_edge_recovery_items".to_string(),
        recovered_absent_source_edges.attempted_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_cad_curve_absent_source_edge_recovery_items".to_string(),
        recovered_absent_source_edges.attempted_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "reconnected_absent_source_edge_items".to_string(),
        recovered_absent_source_edges.source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "reconnected_cad_curve_absent_source_edge_items".to_string(),
        recovered_absent_source_edges.cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_absent_source_edge_recovery_items".to_string(),
        recovered_absent_source_edges.rejected_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_cad_curve_absent_source_edge_recovery_items".to_string(),
        recovered_absent_source_edges.rejected_cad_curve_source_edge_count,
    );
    for (reason_key, count) in recovered_absent_source_edges.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(reason_key.to_string(), count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "recovered_absent_source_edge_boundary_faces".to_string(),
        recovered_absent_source_edges.boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_source_face_diagonal_recovery_pairs".to_string(),
        recovered_source_faces.attempted_source_face_pair_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_source_face_diagonal_pairs".to_string(),
        recovered_source_faces.source_face_pair_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_source_face_diagonal_boundary_faces".to_string(),
        recovered_source_faces.boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_source_face_diagonal_recovery_pairs".to_string(),
        recovered_source_faces.rejected_source_face_pair_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_source_face_diagonal_recovery_items".to_string(),
        recovered_source_faces.rejected_source_face_count,
    );
    for (reason_key, count) in recovered_source_faces.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(reason_key.to_string(), count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "repaired_boundary_face_identity_items".to_string(),
        repaired_boundary_face_identity_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "removed_redundant_boundary_faces".to_string(),
        removed_redundant_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "removed_unsupported_boundary_faces".to_string(),
        removed_unsupported_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "repaired_source_face_provenance_items".to_string(),
        repaired_source_face_provenance_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "repaired_source_edge_provenance_items".to_string(),
        repaired_source_edge_provenance.repaired_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "repaired_cad_curve_source_edge_provenance_items".to_string(),
        repaired_source_edge_provenance.repaired_cad_curve_source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_boundary_leak_recovery_items".to_string(),
        recovered_boundary_leaks.attempted_source_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "removed_exterior_leaked_elements".to_string(),
        recovered_boundary_leaks.removed_element_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "exposed_interior_source_faces".to_string(),
        recovered_boundary_leaks.exposed_source_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "inserted_exposed_interior_boundary_faces".to_string(),
        recovered_boundary_leaks.inserted_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_boundary_leak_recovery_items".to_string(),
        recovered_boundary_leaks.rejected_source_face_count,
    );
    for (reason_key, count) in recovered_boundary_leaks.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(reason_key.to_string(), count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "repaired_material_interface_elements".to_string(),
        material_interface_recovery.repaired_element_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_material_interface_recovery_items".to_string(),
        material_interface_recovery.attempted_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_absent_material_partition_recovery_items".to_string(),
        recovered_material_partitions.attempted_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "inserted_absent_material_partition_recovery_items".to_string(),
        recovered_material_partitions.inserted_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "inserted_absent_material_partition_elements".to_string(),
        recovered_material_partitions.inserted_element_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "inserted_absent_material_partition_boundary_faces".to_string(),
        recovered_material_partitions.inserted_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_absent_material_partition_recovery_items".to_string(),
        recovered_material_partitions.rejected_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rolled_back_absent_material_partition_recovery_items".to_string(),
        recovered_material_partitions.rolled_back_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rolled_back_absent_material_partition_elements".to_string(),
        recovered_material_partitions.rolled_back_element_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rolled_back_absent_material_partition_boundary_faces".to_string(),
        recovered_material_partitions.rolled_back_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "absent_material_partition_topology_candidate_items".to_string(),
        recovered_material_partitions.topology_candidate_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "absent_material_partition_usable_candidate_items".to_string(),
        recovered_material_partitions.usable_candidate_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_absent_material_partition_existing_candidate_items".to_string(),
        recovered_material_partitions.rejected_existing_candidate_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_absent_material_partition_quality_candidate_items".to_string(),
        recovered_material_partitions.rejected_quality_candidate_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_absent_material_partition_interior_candidate_sets".to_string(),
        recovered_material_partitions.rejected_interior_candidate_set_count,
    );
    for (reason_key, count) in recovered_material_partitions.rejection_counts {
        recovery_queue
            .evidence
            .entity_counts
            .insert(reason_key.to_string(), count);
    }
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_recovery_items".to_string(),
        material_interface_recovery.rejected_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "global_material_interface_recovery_items".to_string(),
        material_interface_recovery.global_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "boundary_owned_material_interface_recovery_items".to_string(),
        material_interface_recovery.boundary_owned_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "interior_material_interface_recovery_items".to_string(),
        material_interface_recovery.interior_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "boundary_owned_material_interface_recovery_input_items".to_string(),
        boundary_owned_material_interface_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "interior_face_material_interface_recovery_input_items".to_string(),
        interior_face_material_interface_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "absent_partition_material_interface_recovery_items".to_string(),
        material_interface_recovery.absent_partition_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "absent_partition_material_interface_recovery_input_items".to_string(),
        absent_partition_material_interface_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_missing_boundary_ownership".to_string(),
        material_interface_recovery.missing_boundary_ownership_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_missing_interior_ownership".to_string(),
        material_interface_recovery.missing_interior_ownership_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_ambiguous_boundary_ownership".to_string(),
        material_interface_recovery.ambiguous_boundary_ownership_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_absent_partition".to_string(),
        material_interface_recovery.absent_partition_rejection_count,
    );
    mark_tetrahedron_mesh_recovery_state(&mut tetrahedron_mesh, &recovery_queue);
    if !tetrahedron_mesh.recovery_complete {
        return Err(TetrahedronRecoveryError::IncompleteRecovery {
            missing_item_count: recovery_entity_count(&recovery_queue, "missing_items"),
            missing_source_face_item_count: recovery_entity_count(
                &recovery_queue,
                "missing_source_face_items",
            ),
            missing_source_edge_item_count: recovery_entity_count(
                &recovery_queue,
                "missing_source_edge_items",
            ),
            missing_material_interface_item_count: recovery_entity_count(
                &recovery_queue,
                "missing_material_interface_items",
            ),
            recovery_evidence: recovery_queue.evidence.clone(),
        });
    }
    Ok(TetrahedronRecoveryResult {
        tetrahedron_mesh,
        recovery_queue,
    })
}

fn record_recovered_queue_item_counts(
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    recovery_queue: &mut TetrahedronRecoveryQueue,
) {
    for (recovered_key, missing_key) in [
        ("recovered_source_face_items", "missing_source_face_items"),
        ("recovered_source_edge_items", "missing_source_edge_items"),
        (
            "recovered_material_interface_items",
            "missing_material_interface_items",
        ),
    ] {
        let recovered_count = recovery_entity_count(initial_recovery_queue, missing_key)
            .saturating_sub(recovery_entity_count(recovery_queue, missing_key));
        recovery_queue
            .evidence
            .entity_counts
            .insert(recovered_key.to_string(), recovered_count);
    }
    for (recovered_key, topology) in [
        (
            "recovered_boundary_edge_source_edge_items",
            TetrahedronProtectedEdgeTopology::BoundaryEdge,
        ),
        (
            "recovered_volume_edge_source_edge_items",
            TetrahedronProtectedEdgeTopology::VolumeEdge,
        ),
        (
            "recovered_interior_edge_source_edge_items",
            TetrahedronProtectedEdgeTopology::InteriorEdge,
        ),
        (
            "recovered_absent_edge_source_edge_items",
            TetrahedronProtectedEdgeTopology::Absent,
        ),
    ] {
        let recovered_count =
            recovery_source_edge_item_count_by_topology(initial_recovery_queue, topology)
                .saturating_sub(recovery_source_edge_item_count_by_topology(
                    recovery_queue,
                    topology,
                ));
        recovery_queue
            .evidence
            .entity_counts
            .insert(recovered_key.to_string(), recovered_count);
    }
    for (recovered_key, topology) in [
        (
            "recovered_boundary_face_source_face_items",
            TetrahedronSourceFaceTopology::BoundaryFace,
        ),
        (
            "recovered_volume_face_source_face_items",
            TetrahedronSourceFaceTopology::VolumeFace,
        ),
        (
            "recovered_interior_face_source_face_items",
            TetrahedronSourceFaceTopology::InteriorFace,
        ),
        (
            "recovered_absent_face_source_face_items",
            TetrahedronSourceFaceTopology::Absent,
        ),
    ] {
        let recovered_count =
            recovery_source_face_item_count_by_topology(initial_recovery_queue, topology)
                .saturating_sub(recovery_source_face_item_count_by_topology(
                    recovery_queue,
                    topology,
                ));
        recovery_queue
            .evidence
            .entity_counts
            .insert(recovered_key.to_string(), recovered_count);
    }
    for (recovered_key, topology) in [
        (
            "recovered_boundary_owned_material_interface_items",
            TetrahedronMaterialInterfaceTopology::BoundaryOwned,
        ),
        (
            "recovered_interior_face_material_interface_items",
            TetrahedronMaterialInterfaceTopology::InteriorFace,
        ),
        (
            "recovered_absent_partition_material_interface_items",
            TetrahedronMaterialInterfaceTopology::AbsentPartition,
        ),
    ] {
        let recovered_count =
            recovery_material_interface_item_count_by_topology(initial_recovery_queue, topology)
                .saturating_sub(recovery_material_interface_item_count_by_topology(
                    recovery_queue,
                    topology,
                ));
        recovery_queue
            .evidence
            .entity_counts
            .insert(recovered_key.to_string(), recovered_count);
    }
}

fn recovery_entity_count(recovery_queue: &TetrahedronRecoveryQueue, key: &str) -> usize {
    recovery_queue
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

fn recovery_source_edge_item_count_by_topology(
    recovery_queue: &TetrahedronRecoveryQueue,
    topology: TetrahedronProtectedEdgeTopology,
) -> usize {
    recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceEdge
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.protected_edge_topology == Some(topology)
        })
        .count()
}

fn recovery_cad_curve_source_edge_item_count_by_topology(
    recovery_queue: &TetrahedronRecoveryQueue,
    plc: &ProtectedBoundaryComplex,
    topology: TetrahedronProtectedEdgeTopology,
) -> usize {
    let cad_curve_source_edge_ids = plc
        .protected_edges
        .iter()
        .filter(|edge| edge.cad_curve_boundary.is_some())
        .map(|edge| edge.source_edge_id.clone())
        .collect::<BTreeSet<_>>();
    recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceEdge
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.protected_edge_topology == Some(topology)
                && item
                    .source_entity_id
                    .as_ref()
                    .is_some_and(|source_edge_id| {
                        cad_curve_source_edge_ids.contains(source_edge_id)
                    })
        })
        .count()
}

fn recovery_source_face_item_count_by_topology(
    recovery_queue: &TetrahedronRecoveryQueue,
    topology: TetrahedronSourceFaceTopology,
) -> usize {
    recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceFace
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.source_face_topology == Some(topology)
        })
        .count()
}

fn recovery_material_interface_item_count_by_topology(
    recovery_queue: &TetrahedronRecoveryQueue,
    topology: TetrahedronMaterialInterfaceTopology,
) -> usize {
    recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::MaterialInterface
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.material_interface_topology == Some(topology)
        })
        .count()
}

#[cfg(test)]
mod tests;
