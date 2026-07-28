use std::collections::{BTreeMap, BTreeSet};

pub fn local_tetrahedron_boundary_faces(tetrahedra: &[[u32; 4]]) -> BTreeSet<[u32; 3]> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(*tetrahedron) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

pub(super) fn shared_face(left: [u32; 4], right: [u32; 4]) -> Option<[u32; 3]> {
    let right_nodes = right.into_iter().collect::<BTreeSet<_>>();
    let shared = left
        .into_iter()
        .filter(|node_id| right_nodes.contains(node_id))
        .collect::<Vec<_>>();
    (shared.len() == 3).then(|| sorted_face([shared[0], shared[1], shared[2]]))
}

pub(super) fn opposite_node(node_ids: [u32; 4], face: &[u32; 3]) -> Option<u32> {
    node_ids.into_iter().find(|node_id| !face.contains(node_id))
}

pub(super) fn ring_edges_form_cycle(
    ring_nodes: &BTreeSet<u32>,
    ring_edges: &BTreeSet<[u32; 2]>,
) -> bool {
    let mut degree = BTreeMap::<u32, usize>::new();
    for edge in ring_edges {
        *degree.entry(edge[0]).or_default() += 1;
        *degree.entry(edge[1]).or_default() += 1;
    }
    ring_nodes
        .iter()
        .all(|node_id| degree.get(node_id).copied().unwrap_or_default() == 2)
}

pub(super) fn sorted_removed_tetrahedron_ids<const N: usize>(
    mut tetrahedron_ids: [u32; N],
) -> Vec<u32> {
    tetrahedron_ids.sort();
    tetrahedron_ids.to_vec()
}

pub(super) fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

pub(super) fn sorted_edge(mut node_ids: [u32; 2]) -> [u32; 2] {
    node_ids.sort();
    node_ids
}

fn tetrahedron_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[2], node_ids[1]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
        [node_ids[2], node_ids[0], node_ids[3]],
    ]
}
