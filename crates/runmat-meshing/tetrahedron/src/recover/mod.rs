mod absent_edges;
mod boundary_diagonal;
mod boundary_faces;
pub mod boundary_queue;
mod material_interfaces;
mod source_faces;
mod topology;
mod types;

use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{
    MeshingStage, ProtectedBoundaryComplex, StageEvidence, StageEvidenceStatus, TetrahedronMesh,
    TopologyEntityId,
};
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

use absent_edges::recover_absent_protected_edges_by_boundary_diagonal_flip;
use boundary_faces::{
    boundary_face_source_edges, recover_missing_protected_edge_boundary_faces,
    recover_volume_face_source_face_boundary_faces, repair_boundary_face_identity,
    repair_boundary_source_edge_provenance, repair_boundary_source_face_provenance,
};
use material_interfaces::recover_single_material_interface_region;
use source_faces::recover_source_faces_by_boundary_diagonal_flip;
use topology::sorted_topology_ids;
pub use types::{
    TetrahedronProtectedEdgeTopology, TetrahedronRecoveryError, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryQueueItem, TetrahedronRecoveryResult,
    TetrahedronRecoveryStatus, TetrahedronSourceFaceTopology,
};

pub const MODULE_PURPOSE: &str = "source-edge, source-face, and material-interface recovery queues";

pub fn build_recovery_queue_from_plc(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &TetrahedronMesh,
) -> Result<TetrahedronRecoveryQueue, TetrahedronRecoveryError> {
    validate_protected_boundary_complex(plc)
        .map_err(|error| TetrahedronRecoveryError::InvalidProtectedBoundaryComplex { error })?;
    if tetrahedron_mesh.nodes.is_empty() || tetrahedron_mesh.elements.is_empty() {
        return Err(TetrahedronRecoveryError::EmptyTetrahedronMesh);
    }

    let recovered_face_keys = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            (
                face.source_face_id.clone(),
                sorted_topology_ids(face.node_ids.clone()),
            )
        })
        .collect::<BTreeSet<_>>();
    let recovered_boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_volume_faces = tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_faces(element.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_boundary_source_edges = tetrahedron_mesh
        .boundary_faces
        .iter()
        .flat_map(boundary_face_source_edges)
        .filter_map(|(edge_key, source_edge_id)| {
            source_edge_id.map(|source_edge_id| (edge_key, source_edge_id))
        })
        .collect::<BTreeSet<_>>();
    let recovered_boundary_edges = tetrahedron_mesh
        .boundary_faces
        .iter()
        .flat_map(boundary_face_source_edges)
        .map(|(edge_key, _)| edge_key)
        .collect::<BTreeSet<_>>();
    let recovered_volume_edges = tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_edges(element.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_material_interfaces = tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.clone())
        .collect::<BTreeSet<_>>();

    let mut items = Vec::<TetrahedronRecoveryQueueItem>::new();
    for facet in &plc.facets {
        let face_node_ids = sorted_topology_ids(facet.node_ids.clone());
        let face_key = (facet.source_face_id.clone(), face_node_ids.clone());
        let source_face_topology = if recovered_boundary_faces.contains(&face_node_ids) {
            TetrahedronSourceFaceTopology::BoundaryFace
        } else if recovered_volume_faces.contains(&face_node_ids) {
            TetrahedronSourceFaceTopology::VolumeFace
        } else {
            TetrahedronSourceFaceTopology::Absent
        };
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_face:{}", facet.facet_id.id),
            kind: TetrahedronRecoveryKind::SourceFace,
            status: if recovered_face_keys.contains(&face_key) {
                TetrahedronRecoveryStatus::Recovered
            } else {
                TetrahedronRecoveryStatus::Missing
            },
            source_entity_id: Some(facet.source_face_id.clone()),
            source_face_node_ids: Some(face_node_ids),
            source_face_topology: Some(source_face_topology),
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_id: None,
        });
    }

    for protected_edge in &plc.protected_edges {
        let edge_key = sorted_topology_ids(protected_edge.node_ids.clone());
        let protected_edge_topology = if recovered_boundary_edges.contains(&edge_key) {
            TetrahedronProtectedEdgeTopology::BoundaryEdge
        } else if recovered_volume_edges.contains(&edge_key) {
            TetrahedronProtectedEdgeTopology::VolumeEdge
        } else {
            TetrahedronProtectedEdgeTopology::Absent
        };
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_edge:{}", protected_edge.edge_id.id),
            kind: TetrahedronRecoveryKind::SourceEdge,
            status: if recovered_boundary_source_edges
                .contains(&(edge_key.clone(), protected_edge.source_edge_id.clone()))
            {
                TetrahedronRecoveryStatus::Recovered
            } else {
                TetrahedronRecoveryStatus::Missing
            },
            source_entity_id: Some(protected_edge.source_edge_id.clone()),
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: Some(edge_key),
            protected_edge_topology: Some(protected_edge_topology),
            material_interface_id: None,
        });
    }

    let material_interfaces = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    for material_interface_id in material_interfaces {
        let status = if recovered_material_interfaces.contains(&material_interface_id) {
            TetrahedronRecoveryStatus::Recovered
        } else {
            TetrahedronRecoveryStatus::Missing
        };
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("material_interface:{material_interface_id}"),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status,
            source_entity_id: None,
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_id: Some(material_interface_id),
        });
    }

    let mut evidence = StageEvidence::complete(MeshingStage::ConstraintRecovery);
    if items
        .iter()
        .any(|item| item.status == TetrahedronRecoveryStatus::Missing)
    {
        evidence.status = StageEvidenceStatus::Failed;
    }
    evidence
        .entity_counts
        .insert("recovery_items".to_string(), items.len());
    evidence.entity_counts.insert(
        "source_face_items".to_string(),
        items
            .iter()
            .filter(|item| item.kind == TetrahedronRecoveryKind::SourceFace)
            .count(),
    );
    evidence.entity_counts.insert(
        "source_edge_items".to_string(),
        items
            .iter()
            .filter(|item| item.kind == TetrahedronRecoveryKind::SourceEdge)
            .count(),
    );
    evidence.entity_counts.insert(
        "material_interface_items".to_string(),
        items
            .iter()
            .filter(|item| item.kind == TetrahedronRecoveryKind::MaterialInterface)
            .count(),
    );
    evidence.entity_counts.insert(
        "recovered_items".to_string(),
        items
            .iter()
            .filter(|item| item.status == TetrahedronRecoveryStatus::Recovered)
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_items".to_string(),
        items
            .iter()
            .filter(|item| item.status == TetrahedronRecoveryStatus::Missing)
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_face_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_face_topology_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.source_face_topology.as_ref().is_some_and(|topology| {
                        *topology != TetrahedronSourceFaceTopology::BoundaryFace
                    })
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_face_provenance_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.source_face_topology
                        == Some(TetrahedronSourceFaceTopology::BoundaryFace)
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_face_boundary_face_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.source_face_topology
                        == Some(TetrahedronSourceFaceTopology::BoundaryFace)
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_face_volume_face_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.source_face_topology == Some(TetrahedronSourceFaceTopology::VolumeFace)
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_face_absent_face_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.source_face_topology == Some(TetrahedronSourceFaceTopology::Absent)
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_edge_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_edge_topology_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item
                        .protected_edge_topology
                        .as_ref()
                        .is_some_and(|topology| {
                            *topology != TetrahedronProtectedEdgeTopology::BoundaryEdge
                        })
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_edge_provenance_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item
                        .protected_edge_topology
                        .as_ref()
                        .is_some_and(|topology| {
                            *topology == TetrahedronProtectedEdgeTopology::BoundaryEdge
                        })
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_edge_volume_edge_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.protected_edge_topology
                        == Some(TetrahedronProtectedEdgeTopology::VolumeEdge)
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_source_edge_absent_edge_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.protected_edge_topology
                        == Some(TetrahedronProtectedEdgeTopology::Absent)
            })
            .count(),
    );
    evidence.entity_counts.insert(
        "missing_material_interface_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::MaterialInterface
                    && item.status == TetrahedronRecoveryStatus::Missing
            })
            .count(),
    );

    Ok(TetrahedronRecoveryQueue { items, evidence })
}

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
    let deferred_absent_source_edge_recovery_item_count =
        recovery_source_edge_item_count_by_topology(
            &initial_recovery_queue,
            TetrahedronProtectedEdgeTopology::Absent,
        );
    let volume_face_source_face_recovery_item_count = recovery_source_face_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronSourceFaceTopology::VolumeFace,
    );
    let boundary_face_source_face_recovery_item_count = recovery_source_face_item_count_by_topology(
        &initial_recovery_queue,
        TetrahedronSourceFaceTopology::BoundaryFace,
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
    let recovered_protected_edge_boundary_face_count =
        recover_missing_protected_edge_boundary_faces(
            plc,
            &initial_recovery_queue,
            &mut tetrahedron_mesh,
        );
    let recovered_boundary_face_count = recover_volume_face_source_face_boundary_faces(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let repaired_boundary_face_identity_count =
        repair_boundary_face_identity(plc, &mut tetrahedron_mesh);
    let repaired_source_face_provenance_count =
        repair_boundary_source_face_provenance(&initial_recovery_queue, &mut tetrahedron_mesh);
    let repaired_source_edge_provenance_count =
        repair_boundary_source_edge_provenance(plc, &mut tetrahedron_mesh);
    let material_interface_recovery = recover_single_material_interface_region(
        plc,
        &initial_recovery_queue,
        &mut tetrahedron_mesh,
    );
    let mut recovery_queue = build_recovery_queue_from_plc(plc, &tetrahedron_mesh)?;
    record_recovered_queue_item_counts(&initial_recovery_queue, &mut recovery_queue);
    recovery_queue.evidence.entity_counts.insert(
        "recovered_missing_boundary_faces".to_string(),
        recovered_protected_edge_boundary_face_count + recovered_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "recovered_protected_edge_boundary_faces".to_string(),
        recovered_protected_edge_boundary_face_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "volume_edge_source_edge_recovery_items".to_string(),
        volume_edge_source_edge_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "volume_face_source_face_recovery_items".to_string(),
        volume_face_source_face_recovery_item_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "boundary_face_source_face_recovery_items".to_string(),
        boundary_face_source_face_recovery_item_count,
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
        "reconnected_absent_source_edge_items".to_string(),
        recovered_absent_source_edges.source_edge_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_absent_source_edge_recovery_items".to_string(),
        recovered_absent_source_edges.rejected_source_edge_count,
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
        "repaired_source_face_provenance_items".to_string(),
        repaired_source_face_provenance_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "repaired_source_edge_provenance_items".to_string(),
        repaired_source_edge_provenance_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "repaired_material_interface_elements".to_string(),
        material_interface_recovery.repaired_element_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "attempted_material_interface_recovery_items".to_string(),
        material_interface_recovery.attempted_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_recovery_items".to_string(),
        material_interface_recovery.rejected_material_interface_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_missing_boundary_ownership".to_string(),
        material_interface_recovery.missing_boundary_ownership_count,
    );
    recovery_queue.evidence.entity_counts.insert(
        "rejected_material_interface_ambiguous_boundary_ownership".to_string(),
        material_interface_recovery.ambiguous_boundary_ownership_count,
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

fn tetrahedron_edges(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 2]; 6] {
    [
        sorted_topology_ids([node_ids[0].clone(), node_ids[1].clone()]),
        sorted_topology_ids([node_ids[0].clone(), node_ids[2].clone()]),
        sorted_topology_ids([node_ids[0].clone(), node_ids[3].clone()]),
        sorted_topology_ids([node_ids[1].clone(), node_ids[2].clone()]),
        sorted_topology_ids([node_ids[1].clone(), node_ids[3].clone()]),
        sorted_topology_ids([node_ids[2].clone(), node_ids[3].clone()]),
    ]
}

fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
    ]
}

#[cfg(test)]
mod tests;
