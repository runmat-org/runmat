use super::super::cavity::ConstrainedCavityBoundaryFace;
use topology::{
    boundary_edge_set, boundary_face_map, boundary_face_source_edges, sorted_region_ids,
    sorted_u32_ids,
};
pub use types::{
    BoundaryRecoveryPriority, BoundaryRecoveryQueue, BoundaryRecoveryQueueError,
    BoundaryRecoveryQueueItem, BoundaryRecoveryReason,
};

mod topology;
mod types;

pub fn build_boundary_recovery_queue(
    expected_faces: &[ConstrainedCavityBoundaryFace],
    candidate_faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BoundaryRecoveryQueue, BoundaryRecoveryQueueError> {
    let expected = boundary_face_map(expected_faces)?;
    let candidate = boundary_face_map(candidate_faces)?;
    let candidate_edges = boundary_edge_set(candidate_faces)?;
    let mut items = Vec::<BoundaryRecoveryQueueItem>::new();

    for (face_key, expected_face) in &expected {
        if !candidate.contains_key(face_key) {
            for (edge_key, source_edge_id) in boundary_face_source_edges(expected_face) {
                if !candidate_edges.contains(&edge_key) {
                    items.push(BoundaryRecoveryQueueItem {
                        priority: BoundaryRecoveryPriority::Edge,
                        reason: BoundaryRecoveryReason::MissingEdge,
                        face_node_ids: Some(*face_key),
                        edge_node_ids: Some(edge_key),
                        source_face_id: expected_face.source_face_id,
                        source_edge_id,
                        outside_tetrahedron_ids: sorted_u32_ids(
                            &expected_face.outside_tetrahedron_ids,
                        ),
                        region_ids: sorted_region_ids(&expected_face.region_ids),
                    });
                }
            }
            items.push(BoundaryRecoveryQueueItem {
                priority: BoundaryRecoveryPriority::Face,
                reason: BoundaryRecoveryReason::MissingFace,
                face_node_ids: Some(*face_key),
                edge_node_ids: None,
                source_face_id: expected_face.source_face_id,
                source_edge_id: None,
                outside_tetrahedron_ids: sorted_u32_ids(&expected_face.outside_tetrahedron_ids),
                region_ids: sorted_region_ids(&expected_face.region_ids),
            });
            continue;
        }

        let candidate_face = candidate
            .get(face_key)
            .expect("candidate face exists after contains_key check");
        let expected_outside_tetrahedron_ids =
            sorted_u32_ids(&expected_face.outside_tetrahedron_ids);
        let candidate_outside_tetrahedron_ids =
            sorted_u32_ids(&candidate_face.outside_tetrahedron_ids);
        if expected_outside_tetrahedron_ids != candidate_outside_tetrahedron_ids {
            items.push(BoundaryRecoveryQueueItem {
                priority: BoundaryRecoveryPriority::Face,
                reason: BoundaryRecoveryReason::OutsideTetrahedronMismatch,
                face_node_ids: Some(*face_key),
                edge_node_ids: None,
                source_face_id: expected_face.source_face_id,
                source_edge_id: None,
                outside_tetrahedron_ids: expected_outside_tetrahedron_ids,
                region_ids: sorted_region_ids(&expected_face.region_ids),
            });
        }
        if expected_face.source_face_id != candidate_face.source_face_id {
            items.push(BoundaryRecoveryQueueItem {
                priority: BoundaryRecoveryPriority::Provenance,
                reason: BoundaryRecoveryReason::SourceFaceMismatch,
                face_node_ids: Some(*face_key),
                edge_node_ids: None,
                source_face_id: expected_face.source_face_id,
                source_edge_id: None,
                outside_tetrahedron_ids: sorted_u32_ids(&expected_face.outside_tetrahedron_ids),
                region_ids: sorted_region_ids(&expected_face.region_ids),
            });
        }
        let candidate_edge_sources = boundary_face_source_edges(candidate_face);
        for (edge_key, source_edge_id) in boundary_face_source_edges(expected_face) {
            if candidate_edge_sources.get(&edge_key).copied().flatten() != source_edge_id {
                items.push(BoundaryRecoveryQueueItem {
                    priority: BoundaryRecoveryPriority::Provenance,
                    reason: BoundaryRecoveryReason::SourceEdgeMismatch,
                    face_node_ids: Some(*face_key),
                    edge_node_ids: Some(edge_key),
                    source_face_id: expected_face.source_face_id,
                    source_edge_id,
                    outside_tetrahedron_ids: sorted_u32_ids(&expected_face.outside_tetrahedron_ids),
                    region_ids: sorted_region_ids(&expected_face.region_ids),
                });
            }
        }
        if sorted_region_ids(&expected_face.region_ids)
            != sorted_region_ids(&candidate_face.region_ids)
        {
            items.push(BoundaryRecoveryQueueItem {
                priority: BoundaryRecoveryPriority::Provenance,
                reason: BoundaryRecoveryReason::RegionMismatch,
                face_node_ids: Some(*face_key),
                edge_node_ids: None,
                source_face_id: expected_face.source_face_id,
                source_edge_id: None,
                outside_tetrahedron_ids: sorted_u32_ids(&expected_face.outside_tetrahedron_ids),
                region_ids: sorted_region_ids(&expected_face.region_ids),
            });
        }
    }

    items.sort_by(boundary_recovery_item_order);
    let missing_edge_count = items
        .iter()
        .filter(|item| item.reason == BoundaryRecoveryReason::MissingEdge)
        .count();
    let missing_face_count = items
        .iter()
        .filter(|item| item.reason == BoundaryRecoveryReason::MissingFace)
        .count();
    let interface_mismatch_count = items
        .iter()
        .filter(|item| item.reason == BoundaryRecoveryReason::OutsideTetrahedronMismatch)
        .count();
    let provenance_mismatch_count = items
        .iter()
        .filter(|item| item.priority == BoundaryRecoveryPriority::Provenance)
        .count();
    Ok(BoundaryRecoveryQueue {
        items,
        missing_edge_count,
        missing_face_count,
        interface_mismatch_count,
        provenance_mismatch_count,
    })
}

fn boundary_recovery_item_order(
    left: &BoundaryRecoveryQueueItem,
    right: &BoundaryRecoveryQueueItem,
) -> std::cmp::Ordering {
    left.priority
        .cmp(&right.priority)
        .then_with(|| left.face_node_ids.cmp(&right.face_node_ids))
        .then_with(|| left.edge_node_ids.cmp(&right.edge_node_ids))
        .then_with(|| left.reason.cmp(&right.reason))
}

#[cfg(test)]
mod tests;
