pub mod boundary_queue;
mod topology;
mod types;

use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{
    MeshingStage, ProtectedBoundaryComplex, StageEvidence, TetrahedronMesh,
};
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

use topology::{sorted_topology_ids, topology_face_edges};
pub use types::{
    TetrahedronRecoveryError, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryQueueItem, TetrahedronRecoveryStatus,
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
        if !recovered_face_keys.contains(&face_key) {
            return Err(TetrahedronRecoveryError::MissingSourceFaceRecovery {
                face_id: facet.source_face_id.id.clone(),
            });
        }
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_face:{}", facet.facet_id.id),
            kind: TetrahedronRecoveryKind::SourceFace,
            status: TetrahedronRecoveryStatus::Recovered,
            source_entity_id: Some(facet.source_face_id.clone()),
            material_interface_id: None,
        });
    }

    for protected_edge in &plc.protected_edges {
        let edge_key = sorted_topology_ids(protected_edge.node_ids.clone());
        if !recovered_boundary_edges.contains(&edge_key) {
            return Err(TetrahedronRecoveryError::MissingSourceEdgeRecovery {
                edge_id: protected_edge.source_edge_id.id.clone(),
            });
        }
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_edge:{}", protected_edge.edge_id.id),
            kind: TetrahedronRecoveryKind::SourceEdge,
            status: TetrahedronRecoveryStatus::Recovered,
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
        if !recovered_material_interfaces.contains(&material_interface_id) {
            return Err(TetrahedronRecoveryError::MissingMaterialInterfaceRecovery {
                material_interface_id,
            });
        }
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("material_interface:{material_interface_id}"),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status: TetrahedronRecoveryStatus::Recovered,
            source_entity_id: None,
            material_interface_id: Some(material_interface_id),
        });
    }

    let mut evidence = StageEvidence::complete(MeshingStage::ConstraintRecovery);
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

    Ok(TetrahedronRecoveryQueue { items, evidence })
}

#[cfg(test)]
mod tests;
