use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTet {
    pub tet_id: u32,
    pub node_ids: [u32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTetFlipKind {
    TwoToThreeFace,
    ThreeToTwoEdge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTetFlipCandidate {
    pub kind: LocalTetFlipKind,
    pub removed_tet_ids: Vec<u32>,
    pub created_tets: Vec<[u32; 4]>,
    #[serde(default)]
    pub shared_face: Option<[u32; 3]>,
    #[serde(default)]
    pub shared_edge: Option<[u32; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTetFlipError {
    DegenerateTet { tet_id: u32, node_ids: [u32; 4] },
    NoSharedFace,
    NoSharedEdge,
    InvalidEdgeRing,
}

pub fn two_to_three_face_flip_candidate(
    left: LocalTet,
    right: LocalTet,
) -> Result<LocalTetFlipCandidate, LocalTetFlipError> {
    validate_tet(left)?;
    validate_tet(right)?;
    let Some(shared_face) = shared_face(left.node_ids, right.node_ids) else {
        return Err(LocalTetFlipError::NoSharedFace);
    };
    let Some(left_apex) = opposite_node(left.node_ids, &shared_face) else {
        return Err(LocalTetFlipError::NoSharedFace);
    };
    let Some(right_apex) = opposite_node(right.node_ids, &shared_face) else {
        return Err(LocalTetFlipError::NoSharedFace);
    };

    Ok(LocalTetFlipCandidate {
        kind: LocalTetFlipKind::TwoToThreeFace,
        removed_tet_ids: sorted_removed_tet_ids([left.tet_id, right.tet_id]),
        created_tets: vec![
            [left_apex, right_apex, shared_face[0], shared_face[1]],
            [left_apex, right_apex, shared_face[1], shared_face[2]],
            [left_apex, right_apex, shared_face[2], shared_face[0]],
        ],
        shared_face: Some(shared_face),
        shared_edge: Some(sorted_edge([left_apex, right_apex])),
    })
}

pub fn three_to_two_edge_flip_candidate(
    tets: [LocalTet; 3],
    edge: [u32; 2],
) -> Result<LocalTetFlipCandidate, LocalTetFlipError> {
    for tet in tets {
        validate_tet(tet)?;
    }
    let edge = sorted_edge(edge);
    let mut ring_edges = BTreeSet::<[u32; 2]>::new();
    let mut ring_nodes = BTreeSet::<u32>::new();
    for tet in tets {
        if !tet.node_ids.contains(&edge[0]) || !tet.node_ids.contains(&edge[1]) {
            return Err(LocalTetFlipError::NoSharedEdge);
        }
        let opposite = tet
            .node_ids
            .into_iter()
            .filter(|node_id| !edge.contains(node_id))
            .collect::<Vec<_>>();
        if opposite.len() != 2 {
            return Err(LocalTetFlipError::InvalidEdgeRing);
        }
        ring_nodes.insert(opposite[0]);
        ring_nodes.insert(opposite[1]);
        ring_edges.insert(sorted_edge([opposite[0], opposite[1]]));
    }
    if ring_nodes.len() != 3
        || ring_edges.len() != 3
        || !ring_edges_form_cycle(&ring_nodes, &ring_edges)
    {
        return Err(LocalTetFlipError::InvalidEdgeRing);
    }
    let ring = ring_nodes.into_iter().collect::<Vec<_>>();
    Ok(LocalTetFlipCandidate {
        kind: LocalTetFlipKind::ThreeToTwoEdge,
        removed_tet_ids: sorted_removed_tet_ids([tets[0].tet_id, tets[1].tet_id, tets[2].tet_id]),
        created_tets: vec![
            [edge[0], ring[0], ring[1], ring[2]],
            [edge[1], ring[0], ring[2], ring[1]],
        ],
        shared_face: Some(sorted_face([ring[0], ring[1], ring[2]])),
        shared_edge: Some(edge),
    })
}

pub fn local_tet_boundary_faces(tets: &[[u32; 4]]) -> BTreeSet<[u32; 3]> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tet in tets {
        for face in tet_faces(*tet) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

fn validate_tet(tet: LocalTet) -> Result<(), LocalTetFlipError> {
    let unique = tet.node_ids.into_iter().collect::<BTreeSet<_>>();
    if unique.len() != 4 {
        return Err(LocalTetFlipError::DegenerateTet {
            tet_id: tet.tet_id,
            node_ids: tet.node_ids,
        });
    }
    Ok(())
}

fn shared_face(left: [u32; 4], right: [u32; 4]) -> Option<[u32; 3]> {
    let right_nodes = right.into_iter().collect::<BTreeSet<_>>();
    let shared = left
        .into_iter()
        .filter(|node_id| right_nodes.contains(node_id))
        .collect::<Vec<_>>();
    (shared.len() == 3).then(|| sorted_face([shared[0], shared[1], shared[2]]))
}

fn opposite_node(node_ids: [u32; 4], face: &[u32; 3]) -> Option<u32> {
    node_ids.into_iter().find(|node_id| !face.contains(node_id))
}

fn ring_edges_form_cycle(ring_nodes: &BTreeSet<u32>, ring_edges: &BTreeSet<[u32; 2]>) -> bool {
    let mut degree = BTreeMap::<u32, usize>::new();
    for edge in ring_edges {
        *degree.entry(edge[0]).or_default() += 1;
        *degree.entry(edge[1]).or_default() += 1;
    }
    ring_nodes
        .iter()
        .all(|node_id| degree.get(node_id).copied().unwrap_or_default() == 2)
}

fn sorted_removed_tet_ids<const N: usize>(mut tet_ids: [u32; N]) -> Vec<u32> {
    tet_ids.sort();
    tet_ids.to_vec()
}

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

fn sorted_edge(mut node_ids: [u32; 2]) -> [u32; 2] {
    node_ids.sort();
    node_ids
}

fn tet_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[2], node_ids[1]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
        [node_ids[2], node_ids[0], node_ids[3]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_to_three_face_flip_preserves_local_boundary_faces() {
        let left = LocalTet {
            tet_id: 4,
            node_ids: [0, 1, 2, 3],
        };
        let right = LocalTet {
            tet_id: 9,
            node_ids: [0, 2, 1, 4],
        };

        let candidate =
            two_to_three_face_flip_candidate(left, right).expect("shared face should flip");

        assert_eq!(candidate.kind, LocalTetFlipKind::TwoToThreeFace);
        assert_eq!(candidate.removed_tet_ids, vec![4, 9]);
        assert_eq!(candidate.shared_face, Some([0, 1, 2]));
        assert_eq!(candidate.shared_edge, Some([3, 4]));
        assert_eq!(candidate.created_tets.len(), 3);
        assert_eq!(
            local_tet_boundary_faces(&candidate.created_tets),
            local_tet_boundary_faces(&[left.node_ids, right.node_ids])
        );
    }

    #[test]
    fn three_to_two_edge_flip_preserves_local_boundary_faces() {
        let tets = [
            LocalTet {
                tet_id: 1,
                node_ids: [0, 3, 4, 5],
            },
            LocalTet {
                tet_id: 2,
                node_ids: [0, 4, 3, 6],
            },
            LocalTet {
                tet_id: 3,
                node_ids: [0, 5, 6, 3],
            },
        ];

        let candidate =
            three_to_two_edge_flip_candidate(tets, [0, 3]).expect("edge ring should flip");

        assert_eq!(candidate.kind, LocalTetFlipKind::ThreeToTwoEdge);
        assert_eq!(candidate.removed_tet_ids, vec![1, 2, 3]);
        assert_eq!(candidate.shared_edge, Some([0, 3]));
        assert_eq!(candidate.created_tets.len(), 2);
        assert_eq!(
            local_tet_boundary_faces(&candidate.created_tets),
            local_tet_boundary_faces(&tets.map(|tet| tet.node_ids))
        );
    }

    #[test]
    fn two_to_three_face_flip_rejects_non_neighbors() {
        let err = two_to_three_face_flip_candidate(
            LocalTet {
                tet_id: 1,
                node_ids: [0, 1, 2, 3],
            },
            LocalTet {
                tet_id: 2,
                node_ids: [4, 5, 6, 7],
            },
        )
        .expect_err("non-neighbor tets should not flip");

        assert_eq!(err, LocalTetFlipError::NoSharedFace);
    }

    #[test]
    fn three_to_two_edge_flip_rejects_invalid_ring() {
        let err = three_to_two_edge_flip_candidate(
            [
                LocalTet {
                    tet_id: 1,
                    node_ids: [0, 1, 2, 3],
                },
                LocalTet {
                    tet_id: 2,
                    node_ids: [0, 1, 3, 4],
                },
                LocalTet {
                    tet_id: 3,
                    node_ids: [0, 1, 5, 6],
                },
            ],
            [0, 1],
        )
        .expect_err("edge ring with more than three opposite nodes should fail");

        assert_eq!(err, LocalTetFlipError::InvalidEdgeRing);
    }

    #[test]
    fn three_to_two_edge_flip_rejects_noncyclic_ring_edges() {
        let err = three_to_two_edge_flip_candidate(
            [
                LocalTet {
                    tet_id: 1,
                    node_ids: [0, 1, 2, 3],
                },
                LocalTet {
                    tet_id: 2,
                    node_ids: [0, 1, 3, 2],
                },
                LocalTet {
                    tet_id: 3,
                    node_ids: [0, 1, 2, 4],
                },
            ],
            [0, 1],
        )
        .expect_err("duplicate opposite edges do not form a three-edge ring");

        assert_eq!(err, LocalTetFlipError::InvalidEdgeRing);
    }

    #[test]
    fn degenerate_tets_are_rejected_before_flip_generation() {
        let err = two_to_three_face_flip_candidate(
            LocalTet {
                tet_id: 1,
                node_ids: [0, 0, 2, 3],
            },
            LocalTet {
                tet_id: 2,
                node_ids: [0, 2, 1, 4],
            },
        )
        .expect_err("degenerate tet should fail");

        assert_eq!(
            err,
            LocalTetFlipError::DegenerateTet {
                tet_id: 1,
                node_ids: [0, 0, 2, 3]
            }
        );
    }
}
