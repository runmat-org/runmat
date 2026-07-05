use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{
    MeshingStage, StageEvidence, StageEvidenceStatus, TopologyEntityId,
};

use super::super::{
    TetrahedronMaterialInterfaceTopology, TetrahedronProtectedEdgeTopology,
    TetrahedronRecoveryKind, TetrahedronRecoveryQueueItem, TetrahedronRecoveryStatus,
    TetrahedronSourceFaceTopology,
};

pub(super) fn build_queue_evidence(
    items: &[TetrahedronRecoveryQueueItem],
    cad_curve_source_edge_ids: &BTreeSet<TopologyEntityId>,
) -> StageEvidence {
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
        "missing_source_face_interior_face_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceFace
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.source_face_topology
                        == Some(TetrahedronSourceFaceTopology::InteriorFace)
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
        "missing_source_edge_interior_edge_items".to_string(),
        items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && item.protected_edge_topology
                        == Some(TetrahedronProtectedEdgeTopology::InteriorEdge)
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
        "cad_curve_source_edge_items".to_string(),
        source_edge_item_count_by_cad_curve_boundary(items, cad_curve_source_edge_ids),
    );
    evidence.entity_counts.insert(
        "recovered_cad_curve_source_edge_items".to_string(),
        source_edge_item_count_by_cad_curve_boundary_and_status(
            items,
            cad_curve_source_edge_ids,
            TetrahedronRecoveryStatus::Recovered,
        ),
    );
    evidence.entity_counts.insert(
        "missing_cad_curve_source_edge_items".to_string(),
        source_edge_item_count_by_cad_curve_boundary_and_status(
            items,
            cad_curve_source_edge_ids,
            TetrahedronRecoveryStatus::Missing,
        ),
    );
    evidence.entity_counts.insert(
        "missing_cad_curve_source_edge_topology_items".to_string(),
        source_edge_item_count_by_cad_curve_boundary_status_and_topology(
            items,
            cad_curve_source_edge_ids,
            TetrahedronRecoveryStatus::Missing,
            |topology| topology != TetrahedronProtectedEdgeTopology::BoundaryEdge,
        ),
    );
    evidence.entity_counts.insert(
        "missing_cad_curve_source_edge_provenance_items".to_string(),
        source_edge_item_count_by_cad_curve_boundary_status_and_topology(
            items,
            cad_curve_source_edge_ids,
            TetrahedronRecoveryStatus::Missing,
            |topology| topology == TetrahedronProtectedEdgeTopology::BoundaryEdge,
        ),
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
    evidence.entity_counts.insert(
        "missing_material_interface_boundary_owned_items".to_string(),
        missing_material_interface_item_count_by_topology(
            items,
            TetrahedronMaterialInterfaceTopology::BoundaryOwned,
        ),
    );
    evidence.entity_counts.insert(
        "missing_material_interface_interior_face_items".to_string(),
        missing_material_interface_item_count_by_topology(
            items,
            TetrahedronMaterialInterfaceTopology::InteriorFace,
        ),
    );
    evidence.entity_counts.insert(
        "missing_material_interface_absent_partition_items".to_string(),
        missing_material_interface_item_count_by_topology(
            items,
            TetrahedronMaterialInterfaceTopology::AbsentPartition,
        ),
    );

    evidence
}

fn source_edge_item_count_by_cad_curve_boundary(
    items: &[TetrahedronRecoveryQueueItem],
    cad_curve_source_edge_ids: &BTreeSet<TopologyEntityId>,
) -> usize {
    items
        .iter()
        .filter(|item| source_edge_item_has_cad_curve_boundary(item, cad_curve_source_edge_ids))
        .count()
}

fn source_edge_item_count_by_cad_curve_boundary_and_status(
    items: &[TetrahedronRecoveryQueueItem],
    cad_curve_source_edge_ids: &BTreeSet<TopologyEntityId>,
    status: TetrahedronRecoveryStatus,
) -> usize {
    items
        .iter()
        .filter(|item| {
            item.status == status
                && source_edge_item_has_cad_curve_boundary(item, cad_curve_source_edge_ids)
        })
        .count()
}

fn source_edge_item_count_by_cad_curve_boundary_status_and_topology(
    items: &[TetrahedronRecoveryQueueItem],
    cad_curve_source_edge_ids: &BTreeSet<TopologyEntityId>,
    status: TetrahedronRecoveryStatus,
    topology_matches: impl Fn(TetrahedronProtectedEdgeTopology) -> bool,
) -> usize {
    items
        .iter()
        .filter(|item| {
            item.status == status
                && source_edge_item_has_cad_curve_boundary(item, cad_curve_source_edge_ids)
                && item
                    .protected_edge_topology
                    .is_some_and(|topology| topology_matches(topology))
        })
        .count()
}

fn source_edge_item_has_cad_curve_boundary(
    item: &TetrahedronRecoveryQueueItem,
    cad_curve_source_edge_ids: &BTreeSet<TopologyEntityId>,
) -> bool {
    item.kind == TetrahedronRecoveryKind::SourceEdge
        && item
            .source_entity_id
            .as_ref()
            .is_some_and(|source_edge_id| cad_curve_source_edge_ids.contains(source_edge_id))
}

fn missing_material_interface_item_count_by_topology(
    items: &[TetrahedronRecoveryQueueItem],
    topology: TetrahedronMaterialInterfaceTopology,
) -> usize {
    items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::MaterialInterface
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.material_interface_topology == Some(topology)
        })
        .count()
}
