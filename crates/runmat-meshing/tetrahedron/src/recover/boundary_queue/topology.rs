use std::collections::{BTreeMap, BTreeSet};

use super::{BoundaryRecoveryQueueError, ConstrainedCavityBoundaryFace};

pub(super) fn boundary_face_map(
    faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BTreeMap<[u32; 3], &ConstrainedCavityBoundaryFace>, BoundaryRecoveryQueueError> {
    let mut map = BTreeMap::<[u32; 3], &ConstrainedCavityBoundaryFace>::new();
    for face in faces {
        if face_is_degenerate(face.node_ids) {
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

pub(super) fn boundary_edge_set(
    faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BTreeSet<[u32; 2]>, BoundaryRecoveryQueueError> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in faces {
        if face_is_degenerate(face.node_ids) {
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

pub(super) fn boundary_face_source_edges(
    face: &ConstrainedCavityBoundaryFace,
) -> BTreeMap<[u32; 2], Option<u32>> {
    face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
        .map(|(edge, source_edge_id)| (sorted_edge(edge), source_edge_id))
        .collect()
}

pub(super) fn sorted_region_ids(region_ids: &[String]) -> Vec<String> {
    let mut sorted = region_ids.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
}

pub(super) fn sorted_u32_ids(ids: &[u32]) -> Vec<u32> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
}

fn face_is_degenerate(node_ids: [u32; 3]) -> bool {
    node_ids[0] == node_ids[1] || node_ids[0] == node_ids[2] || node_ids[1] == node_ids[2]
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
