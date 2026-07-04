pub mod boundary_queue;
mod topology;
mod types;

use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{
    MeshingStage, ProtectedBoundaryComplex, StageEvidence, StageEvidenceStatus, TetrahedronMesh,
};
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

use topology::{sorted_topology_ids, topology_face_edges};
pub use types::{
    TetrahedronRecoveryError, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryQueueItem, TetrahedronRecoveryResult, TetrahedronRecoveryStatus,
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
    let recovered_boundary_edges = tetrahedron_mesh
        .boundary_faces
        .iter()
        .flat_map(|face| topology_face_edges(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_material_interfaces = tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.clone())
        .collect::<BTreeSet<_>>();

    let mut items = Vec::<TetrahedronRecoveryQueueItem>::new();
    for facet in &plc.facets {
        let face_key = (
            facet.source_face_id.clone(),
            sorted_topology_ids(facet.node_ids.clone()),
        );
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_face:{}", facet.facet_id.id),
            kind: TetrahedronRecoveryKind::SourceFace,
            status: if recovered_face_keys.contains(&face_key) {
                TetrahedronRecoveryStatus::Recovered
            } else {
                TetrahedronRecoveryStatus::Missing
            },
            source_entity_id: Some(facet.source_face_id.clone()),
            material_interface_id: None,
        });
    }

    for protected_edge in &plc.protected_edges {
        let edge_key = sorted_topology_ids(protected_edge.node_ids.clone());
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_edge:{}", protected_edge.edge_id.id),
            kind: TetrahedronRecoveryKind::SourceEdge,
            status: if recovered_boundary_edges.contains(&edge_key) {
                TetrahedronRecoveryStatus::Recovered
            } else {
                TetrahedronRecoveryStatus::Missing
            },
            source_entity_id: Some(protected_edge.source_edge_id.clone()),
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
    let repaired_source_face_provenance_count =
        repair_boundary_source_face_provenance(plc, &mut tetrahedron_mesh);
    let mut recovery_queue = build_recovery_queue_from_plc(plc, &tetrahedron_mesh)?;
    recovery_queue.evidence.entity_counts.insert(
        "repaired_source_face_provenance_items".to_string(),
        repaired_source_face_provenance_count,
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
        });
    }
    Ok(TetrahedronRecoveryResult {
        tetrahedron_mesh,
        recovery_queue,
    })
}

fn repair_boundary_source_face_provenance(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let expected_source_face_by_nodes = plc
        .facets
        .iter()
        .map(|facet| {
            (
                sorted_topology_ids(facet.node_ids.clone()),
                facet.source_face_id.clone(),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut repaired_count = 0;
    for boundary_face in &mut tetrahedron_mesh.boundary_faces {
        let face_key = sorted_topology_ids(boundary_face.node_ids.clone());
        let Some(expected_source_face_id) = expected_source_face_by_nodes.get(&face_key) else {
            continue;
        };
        if &boundary_face.source_face_id != expected_source_face_id {
            boundary_face.source_face_id = expected_source_face_id.clone();
            repaired_count += 1;
        }
    }
    repaired_count
}

fn recovery_entity_count(recovery_queue: &TetrahedronRecoveryQueue, key: &str) -> usize {
    recovery_queue
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
