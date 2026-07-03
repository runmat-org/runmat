use std::collections::BTreeMap;

use super::{
    ConstrainedCavityBoundaryFace, ConstrainedCavityValidationError,
    ConstrainedCavityValidationError::*,
};

pub(super) fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

pub(super) fn sorted_tet_nodes(mut node_ids: [u32; 4]) -> [u32; 4] {
    node_ids.sort();
    node_ids
}

pub(super) fn sorted_edge(mut node_ids: [u32; 2]) -> [u32; 2] {
    node_ids.sort();
    node_ids
}

pub(super) fn common_tet_edges(tets: [[u32; 4]; 3]) -> Vec<[u32; 2]> {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for tet in tets {
        for edge in tet_edges(tet) {
            *edge_counts.entry(sorted_edge(edge)).or_default() += 1;
        }
    }
    edge_counts
        .into_iter()
        .filter_map(|(edge, count)| (count == 3).then_some(edge))
        .collect()
}

pub(super) fn tet_edges(node_ids: [u32; 4]) -> [[u32; 2]; 6] {
    [
        [node_ids[0], node_ids[1]],
        [node_ids[0], node_ids[2]],
        [node_ids[0], node_ids[3]],
        [node_ids[1], node_ids[2]],
        [node_ids[1], node_ids[3]],
        [node_ids[2], node_ids[3]],
    ]
}

pub(super) fn face_edges(node_ids: [u32; 3]) -> [[u32; 2]; 3] {
    [
        [node_ids[0], node_ids[1]],
        [node_ids[1], node_ids[2]],
        [node_ids[2], node_ids[0]],
    ]
}

pub(super) fn boundary_face_map(
    faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BTreeMap<[u32; 3], &ConstrainedCavityBoundaryFace>, ConstrainedCavityValidationError> {
    let mut map = BTreeMap::<[u32; 3], &ConstrainedCavityBoundaryFace>::new();
    for (face_index, face) in faces.iter().enumerate() {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(DegenerateBoundaryFace {
                face_index,
                node_ids: face.node_ids,
            });
        }
        let key = sorted_face(face.node_ids);
        if map.insert(key, face).is_some() {
            return Err(DuplicateBoundaryFace { node_ids: key });
        }
    }
    Ok(map)
}

pub(super) fn boundary_face_source_edges(
    face: &ConstrainedCavityBoundaryFace,
) -> Result<BTreeMap<[u32; 2], Option<u32>>, ConstrainedCavityValidationError> {
    let mut edge_sources = BTreeMap::<[u32; 2], Option<u32>>::new();
    for (edge, source_edge_id) in face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
    {
        let key = sorted_edge(edge);
        if edge_sources.insert(key, source_edge_id).is_some() {
            return Err(DegenerateBoundaryFace {
                face_index: 0,
                node_ids: face.node_ids,
            });
        }
    }
    Ok(edge_sources)
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

pub(super) fn tet_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[2], node_ids[1]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
        [node_ids[2], node_ids[0], node_ids[3]],
    ]
}
