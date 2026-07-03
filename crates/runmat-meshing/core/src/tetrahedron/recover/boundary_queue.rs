use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::tetrahedron::cavity::ConstrainedCavityBoundaryFace;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRecoveryPriority {
    Edge,
    Face,
    Provenance,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRecoveryReason {
    MissingEdge,
    MissingFace,
    OutsideTetrahedronMismatch,
    SourceEdgeMismatch,
    SourceFaceMismatch,
    RegionMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryRecoveryQueueItem {
    pub priority: BoundaryRecoveryPriority,
    pub reason: BoundaryRecoveryReason,
    #[serde(default)]
    pub face_node_ids: Option<[u32; 3]>,
    #[serde(default)]
    pub edge_node_ids: Option<[u32; 2]>,
    #[serde(default)]
    pub source_face_id: Option<u32>,
    #[serde(default)]
    pub source_edge_id: Option<u32>,
    #[serde(default)]
    pub outside_tetrahedron_ids: Vec<u32>,
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryRecoveryQueue {
    pub items: Vec<BoundaryRecoveryQueueItem>,
    pub missing_edge_count: usize,
    pub missing_face_count: usize,
    #[serde(default)]
    pub interface_mismatch_count: usize,
    pub provenance_mismatch_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRecoveryQueueError {
    DegenerateBoundaryFace { node_ids: [u32; 3] },
    DuplicateBoundaryFace { node_ids: [u32; 3] },
}

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

fn boundary_face_map(
    faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BTreeMap<[u32; 3], &ConstrainedCavityBoundaryFace>, BoundaryRecoveryQueueError> {
    let mut map = BTreeMap::<[u32; 3], &ConstrainedCavityBoundaryFace>::new();
    for face in faces {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(BoundaryRecoveryQueueError::DegenerateBoundaryFace {
                node_ids: face.node_ids,
            });
        }
        let key = sorted_face(face.node_ids);
        if map.insert(key, face).is_some() {
            return Err(BoundaryRecoveryQueueError::DuplicateBoundaryFace { node_ids: key });
        }
    }
    Ok(map)
}

fn boundary_edge_set(
    faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BTreeSet<[u32; 2]>, BoundaryRecoveryQueueError> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in faces {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(BoundaryRecoveryQueueError::DegenerateBoundaryFace {
                node_ids: face.node_ids,
            });
        }
        for edge in face_edges(face.node_ids) {
            edges.insert(sorted_edge(edge));
        }
    }
    Ok(edges)
}

fn boundary_face_source_edges(
    face: &ConstrainedCavityBoundaryFace,
) -> BTreeMap<[u32; 2], Option<u32>> {
    face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
        .map(|(edge, source_edge_id)| (sorted_edge(edge), source_edge_id))
        .collect()
}

fn sorted_region_ids(region_ids: &[String]) -> Vec<String> {
    let mut sorted = region_ids.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
}

fn sorted_u32_ids(ids: &[u32]) -> Vec<u32> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
}

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

fn sorted_edge(mut node_ids: [u32; 2]) -> [u32; 2] {
    node_ids.sort();
    node_ids
}

fn face_edges(node_ids: [u32; 3]) -> [[u32; 2]; 3] {
    [
        [node_ids[0], node_ids[1]],
        [node_ids[1], node_ids[2]],
        [node_ids[2], node_ids[0]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_queue_reports_complete_boundary_recovery() {
        let faces = tetrahedron_faces();

        let queue = build_boundary_recovery_queue(&faces, &faces)
            .expect("matching boundary faces should queue no work");

        assert!(queue.items.is_empty());
        assert_eq!(queue.missing_edge_count, 0);
        assert_eq!(queue.missing_face_count, 0);
        assert_eq!(queue.interface_mismatch_count, 0);
        assert_eq!(queue.provenance_mismatch_count, 0);
    }

    #[test]
    fn missing_face_queues_face_recovery_when_edges_are_still_present() {
        let expected = tetrahedron_faces();
        let candidate = expected[1..].to_vec();

        let queue = build_boundary_recovery_queue(&expected, &candidate)
            .expect("missing face should produce recovery work");

        assert_eq!(queue.missing_edge_count, 0);
        assert_eq!(queue.missing_face_count, 1);
        assert_eq!(queue.interface_mismatch_count, 0);
        assert_eq!(queue.items[0].priority, BoundaryRecoveryPriority::Face);
        assert_eq!(queue.items[0].reason, BoundaryRecoveryReason::MissingFace);
        assert_eq!(queue.items[0].face_node_ids, Some([0, 1, 2]));
    }

    #[test]
    fn missing_edges_are_prioritized_before_face_recovery() {
        let expected = vec![face([0, 1, 2], 10, [100, 101, 102], &["loaded"])];
        let candidate = Vec::new();

        let queue = build_boundary_recovery_queue(&expected, &candidate)
            .expect("missing face and edges should produce recovery work");

        assert_eq!(queue.missing_edge_count, 3);
        assert_eq!(queue.missing_face_count, 1);
        assert_eq!(queue.interface_mismatch_count, 0);
        assert_eq!(
            queue
                .items
                .iter()
                .map(|item| item.priority)
                .collect::<Vec<_>>(),
            vec![
                BoundaryRecoveryPriority::Edge,
                BoundaryRecoveryPriority::Edge,
                BoundaryRecoveryPriority::Edge,
                BoundaryRecoveryPriority::Face,
            ]
        );
    }

    #[test]
    fn provenance_mismatch_queues_source_and_region_repairs() {
        let expected = tetrahedron_faces();
        let mut candidate = expected.clone();
        candidate[0].source_face_id = Some(99);
        candidate[0].source_edge_ids[1] = Some(88);
        candidate[0].region_ids = vec!["other".to_string()];

        let queue = build_boundary_recovery_queue(&expected, &candidate)
            .expect("provenance mismatch should queue repair work");

        assert_eq!(queue.missing_edge_count, 0);
        assert_eq!(queue.missing_face_count, 0);
        assert_eq!(queue.interface_mismatch_count, 0);
        assert_eq!(queue.provenance_mismatch_count, 3);
        assert_eq!(
            queue
                .items
                .iter()
                .map(|item| item.reason.clone())
                .collect::<Vec<_>>(),
            vec![
                BoundaryRecoveryReason::SourceFaceMismatch,
                BoundaryRecoveryReason::RegionMismatch,
                BoundaryRecoveryReason::SourceEdgeMismatch,
            ]
        );
    }

    #[test]
    fn outside_neighbor_mismatch_queues_face_recovery() {
        let mut expected = tetrahedron_faces();
        expected[0].outside_tetrahedron_ids = vec![42, 24, 42];
        let mut candidate = expected.clone();
        candidate[0].outside_tetrahedron_ids = vec![42];

        let queue = build_boundary_recovery_queue(&expected, &candidate)
            .expect("outside-neighbor mismatch should queue face recovery work");

        assert_eq!(queue.missing_edge_count, 0);
        assert_eq!(queue.missing_face_count, 0);
        assert_eq!(queue.interface_mismatch_count, 1);
        assert_eq!(queue.provenance_mismatch_count, 0);
        assert_eq!(queue.items.len(), 1);
        assert_eq!(queue.items[0].priority, BoundaryRecoveryPriority::Face);
        assert_eq!(
            queue.items[0].reason,
            BoundaryRecoveryReason::OutsideTetrahedronMismatch
        );
        assert_eq!(queue.items[0].face_node_ids, Some([0, 1, 2]));
        assert_eq!(queue.items[0].outside_tetrahedron_ids, vec![24, 42]);
    }

    #[test]
    fn duplicate_expected_faces_are_rejected() {
        let mut faces = tetrahedron_faces();
        faces[1].node_ids = faces[0].node_ids;

        let err = build_boundary_recovery_queue(&faces, &[])
            .expect_err("duplicate expected faces should fail");

        assert_eq!(
            err,
            BoundaryRecoveryQueueError::DuplicateBoundaryFace {
                node_ids: [0, 1, 2]
            }
        );
    }

    fn tetrahedron_faces() -> Vec<ConstrainedCavityBoundaryFace> {
        vec![
            face([0, 1, 2], 10, [100, 101, 102], &["loaded", "fixed"]),
            face([0, 3, 1], 11, [103, 104, 100], &["fixed"]),
            face([1, 3, 2], 12, [104, 105, 101], &["solid"]),
            face([2, 3, 0], 13, [105, 103, 102], &["solid"]),
        ]
    }

    fn face(
        node_ids: [u32; 3],
        source_face_id: u32,
        source_edge_ids: [u32; 3],
        region_ids: &[&str],
    ) -> ConstrainedCavityBoundaryFace {
        ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: Some(source_face_id),
            source_edge_ids: source_edge_ids.map(Some),
            region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
        }
    }
}
