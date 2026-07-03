use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::predicate::{
    tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_signed_volume, Point3,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTetrahedron {
    pub tetrahedron_id: u32,
    pub node_ids: [u32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTetrahedronFlipKind {
    TwoToThreeFace,
    ThreeToTwoEdge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipCandidate {
    pub kind: LocalTetrahedronFlipKind,
    pub removed_tetrahedron_ids: Vec<u32>,
    pub created_tetrahedra: Vec<[u32; 4]>,
    #[serde(default)]
    pub shared_face: Option<[u32; 3]>,
    #[serde(default)]
    pub shared_edge: Option<[u32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipQualityThresholds {
    pub min_volume_m3: f64,
    pub min_scaled_jacobian: f64,
}

impl Default for LocalTetrahedronFlipQualityThresholds {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            min_scaled_jacobian: 0.15,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipQualityReport {
    pub created_tetrahedron_count: usize,
    pub total_volume_m3: f64,
    pub min_volume_m3: f64,
    pub min_scaled_jacobian: f64,
    pub max_aspect_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTetrahedronFlipError {
    DegenerateTetrahedron {
        tetrahedron_id: u32,
        node_ids: [u32; 4],
    },
    NoSharedFace,
    NoSharedEdge,
    InvalidEdgeRing,
    InvalidQualityThresholds,
    MissingNode {
        node_id: u32,
    },
    NonPositiveVolume {
        node_ids: [u32; 4],
    },
    VolumeBelowThreshold {
        node_ids: [u32; 4],
        volume_m3: String,
    },
    ScaledJacobianBelowThreshold {
        node_ids: [u32; 4],
        scaled_jacobian: String,
    },
}

pub fn two_to_three_face_flip_candidate(
    left: LocalTetrahedron,
    right: LocalTetrahedron,
) -> Result<LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError> {
    validate_tetrahedron(left)?;
    validate_tetrahedron(right)?;
    let Some(shared_face) = shared_face(left.node_ids, right.node_ids) else {
        return Err(LocalTetrahedronFlipError::NoSharedFace);
    };
    let Some(left_apex) = opposite_node(left.node_ids, &shared_face) else {
        return Err(LocalTetrahedronFlipError::NoSharedFace);
    };
    let Some(right_apex) = opposite_node(right.node_ids, &shared_face) else {
        return Err(LocalTetrahedronFlipError::NoSharedFace);
    };

    Ok(LocalTetrahedronFlipCandidate {
        kind: LocalTetrahedronFlipKind::TwoToThreeFace,
        removed_tetrahedron_ids: sorted_removed_tetrahedron_ids([
            left.tetrahedron_id,
            right.tetrahedron_id,
        ]),
        created_tetrahedra: vec![
            [left_apex, right_apex, shared_face[0], shared_face[1]],
            [left_apex, right_apex, shared_face[1], shared_face[2]],
            [left_apex, right_apex, shared_face[2], shared_face[0]],
        ],
        shared_face: Some(shared_face),
        shared_edge: Some(sorted_edge([left_apex, right_apex])),
    })
}

pub fn three_to_two_edge_flip_candidate(
    tetrahedra: [LocalTetrahedron; 3],
    edge: [u32; 2],
) -> Result<LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError> {
    for tetrahedron in tetrahedra {
        validate_tetrahedron(tetrahedron)?;
    }
    let edge = sorted_edge(edge);
    let mut ring_edges = BTreeSet::<[u32; 2]>::new();
    let mut ring_nodes = BTreeSet::<u32>::new();
    for tetrahedron in tetrahedra {
        if !tetrahedron.node_ids.contains(&edge[0]) || !tetrahedron.node_ids.contains(&edge[1]) {
            return Err(LocalTetrahedronFlipError::NoSharedEdge);
        }
        let opposite = tetrahedron
            .node_ids
            .into_iter()
            .filter(|node_id| !edge.contains(node_id))
            .collect::<Vec<_>>();
        if opposite.len() != 2 {
            return Err(LocalTetrahedronFlipError::InvalidEdgeRing);
        }
        ring_nodes.insert(opposite[0]);
        ring_nodes.insert(opposite[1]);
        ring_edges.insert(sorted_edge([opposite[0], opposite[1]]));
    }
    if ring_nodes.len() != 3
        || ring_edges.len() != 3
        || !ring_edges_form_cycle(&ring_nodes, &ring_edges)
    {
        return Err(LocalTetrahedronFlipError::InvalidEdgeRing);
    }
    let ring = ring_nodes.into_iter().collect::<Vec<_>>();
    Ok(LocalTetrahedronFlipCandidate {
        kind: LocalTetrahedronFlipKind::ThreeToTwoEdge,
        removed_tetrahedron_ids: sorted_removed_tetrahedron_ids([
            tetrahedra[0].tetrahedron_id,
            tetrahedra[1].tetrahedron_id,
            tetrahedra[2].tetrahedron_id,
        ]),
        created_tetrahedra: vec![
            [edge[0], ring[0], ring[1], ring[2]],
            [edge[1], ring[0], ring[2], ring[1]],
        ],
        shared_face: Some(sorted_face([ring[0], ring[1], ring[2]])),
        shared_edge: Some(edge),
    })
}

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

pub fn evaluate_local_tetrahedron_flip_quality(
    candidate: &LocalTetrahedronFlipCandidate,
    node_coordinates: &BTreeMap<u32, Point3>,
    thresholds: LocalTetrahedronFlipQualityThresholds,
) -> Result<LocalTetrahedronFlipQualityReport, LocalTetrahedronFlipError> {
    if !thresholds.min_volume_m3.is_finite()
        || thresholds.min_volume_m3 < 0.0
        || !thresholds.min_scaled_jacobian.is_finite()
        || thresholds.min_scaled_jacobian < 0.0
    {
        return Err(LocalTetrahedronFlipError::InvalidQualityThresholds);
    }

    let mut total_volume_m3 = 0.0_f64;
    let mut min_volume_m3 = f64::INFINITY;
    let mut min_scaled_jacobian = f64::INFINITY;
    let mut max_aspect_ratio = 0.0_f64;
    for node_ids in &candidate.created_tetrahedra {
        let points = [
            *node_coordinates
                .get(&node_ids[0])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[0],
                })?,
            *node_coordinates
                .get(&node_ids[1])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[1],
                })?,
            *node_coordinates
                .get(&node_ids[2])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[2],
                })?,
            *node_coordinates
                .get(&node_ids[3])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[3],
                })?,
        ];
        let volume_m3 = tetrahedron_signed_volume(points).abs();
        if !volume_m3.is_finite() || volume_m3 <= 0.0 {
            return Err(LocalTetrahedronFlipError::NonPositiveVolume {
                node_ids: *node_ids,
            });
        }
        if volume_m3 < thresholds.min_volume_m3 {
            return Err(LocalTetrahedronFlipError::VolumeBelowThreshold {
                node_ids: *node_ids,
                volume_m3: stable_float(volume_m3),
            });
        }
        let scaled_jacobian = tetrahedron_scaled_jacobian(points);
        if !scaled_jacobian.is_finite() || scaled_jacobian < thresholds.min_scaled_jacobian {
            return Err(LocalTetrahedronFlipError::ScaledJacobianBelowThreshold {
                node_ids: *node_ids,
                scaled_jacobian: stable_float(scaled_jacobian),
            });
        }
        let aspect_ratio = tetrahedron_edge_aspect_ratio(points);
        total_volume_m3 += volume_m3;
        min_volume_m3 = min_volume_m3.min(volume_m3);
        min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
        max_aspect_ratio = max_aspect_ratio.max(aspect_ratio);
    }

    Ok(LocalTetrahedronFlipQualityReport {
        created_tetrahedron_count: candidate.created_tetrahedra.len(),
        total_volume_m3,
        min_volume_m3,
        min_scaled_jacobian,
        max_aspect_ratio,
    })
}

fn validate_tetrahedron(tetrahedron: LocalTetrahedron) -> Result<(), LocalTetrahedronFlipError> {
    let unique = tetrahedron.node_ids.into_iter().collect::<BTreeSet<_>>();
    if unique.len() != 4 {
        return Err(LocalTetrahedronFlipError::DegenerateTetrahedron {
            tetrahedron_id: tetrahedron.tetrahedron_id,
            node_ids: tetrahedron.node_ids,
        });
    }
    Ok(())
}

fn stable_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12e}")
    } else {
        value.to_string()
    }
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

fn sorted_removed_tetrahedron_ids<const N: usize>(mut tetrahedron_ids: [u32; N]) -> Vec<u32> {
    tetrahedron_ids.sort();
    tetrahedron_ids.to_vec()
}

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

fn sorted_edge(mut node_ids: [u32; 2]) -> [u32; 2] {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_to_three_face_flip_preserves_local_boundary_faces() {
        let left = LocalTetrahedron {
            tetrahedron_id: 4,
            node_ids: [0, 1, 2, 3],
        };
        let right = LocalTetrahedron {
            tetrahedron_id: 9,
            node_ids: [0, 2, 1, 4],
        };

        let candidate =
            two_to_three_face_flip_candidate(left, right).expect("shared face should flip");

        assert_eq!(candidate.kind, LocalTetrahedronFlipKind::TwoToThreeFace);
        assert_eq!(candidate.removed_tetrahedron_ids, vec![4, 9]);
        assert_eq!(candidate.shared_face, Some([0, 1, 2]));
        assert_eq!(candidate.shared_edge, Some([3, 4]));
        assert_eq!(candidate.created_tetrahedra.len(), 3);
        assert_eq!(
            local_tetrahedron_boundary_faces(&candidate.created_tetrahedra),
            local_tetrahedron_boundary_faces(&[left.node_ids, right.node_ids])
        );
    }

    #[test]
    fn local_flip_quality_accepts_well_shaped_created_tetrahedra() {
        let left = LocalTetrahedron {
            tetrahedron_id: 4,
            node_ids: [0, 1, 2, 3],
        };
        let right = LocalTetrahedron {
            tetrahedron_id: 9,
            node_ids: [0, 2, 1, 4],
        };
        let candidate =
            two_to_three_face_flip_candidate(left, right).expect("shared face should flip");
        let node_coordinates = BTreeMap::from([
            (0, [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [1.0 / 3.0, 1.0 / 3.0, 1.0]),
            (4, [1.0 / 3.0, 1.0 / 3.0, -1.0]),
        ]);

        let report = evaluate_local_tetrahedron_flip_quality(
            &candidate,
            &node_coordinates,
            LocalTetrahedronFlipQualityThresholds {
                min_volume_m3: 1.0e-12,
                min_scaled_jacobian: 0.05,
            },
        )
        .expect("well shaped flip should pass quality gates");

        assert_eq!(report.created_tetrahedron_count, 3);
        assert!(report.total_volume_m3 > 0.0);
        assert!(report.min_volume_m3 > 0.0);
        assert!(report.min_scaled_jacobian >= 0.05);
        assert!(report.max_aspect_ratio.is_finite());
    }

    #[test]
    fn local_flip_quality_rejects_missing_nodes() {
        let candidate = two_to_three_face_flip_candidate(
            LocalTetrahedron {
                tetrahedron_id: 4,
                node_ids: [0, 1, 2, 3],
            },
            LocalTetrahedron {
                tetrahedron_id: 9,
                node_ids: [0, 2, 1, 4],
            },
        )
        .expect("shared face should flip");
        let node_coordinates = BTreeMap::from([
            (0, [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [1.0 / 3.0, 1.0 / 3.0, 1.0]),
        ]);

        let err = evaluate_local_tetrahedron_flip_quality(
            &candidate,
            &node_coordinates,
            LocalTetrahedronFlipQualityThresholds::default(),
        )
        .expect_err("missing node should fail quality evaluation");

        assert_eq!(err, LocalTetrahedronFlipError::MissingNode { node_id: 4 });
    }

    #[test]
    fn local_flip_quality_rejects_low_scaled_jacobian() {
        let candidate = two_to_three_face_flip_candidate(
            LocalTetrahedron {
                tetrahedron_id: 4,
                node_ids: [0, 1, 2, 3],
            },
            LocalTetrahedron {
                tetrahedron_id: 9,
                node_ids: [0, 2, 1, 4],
            },
        )
        .expect("shared face should flip");
        let node_coordinates = BTreeMap::from([
            (0, [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [1.0 / 3.0, 1.0 / 3.0, 1.0]),
            (4, [1.0 / 3.0, 1.0 / 3.0, 1.0 + 1.0e-8]),
        ]);

        let err = evaluate_local_tetrahedron_flip_quality(
            &candidate,
            &node_coordinates,
            LocalTetrahedronFlipQualityThresholds {
                min_volume_m3: 1.0e-18,
                min_scaled_jacobian: 0.15,
            },
        )
        .expect_err("sliver created tetrahedron should fail quality gates");

        assert!(matches!(
            err,
            LocalTetrahedronFlipError::ScaledJacobianBelowThreshold { .. }
        ));
    }

    #[test]
    fn three_to_two_edge_flip_preserves_local_boundary_faces() {
        let tetrahedra = [
            LocalTetrahedron {
                tetrahedron_id: 1,
                node_ids: [0, 3, 4, 5],
            },
            LocalTetrahedron {
                tetrahedron_id: 2,
                node_ids: [0, 4, 3, 6],
            },
            LocalTetrahedron {
                tetrahedron_id: 3,
                node_ids: [0, 5, 6, 3],
            },
        ];

        let candidate =
            three_to_two_edge_flip_candidate(tetrahedra, [0, 3]).expect("edge ring should flip");

        assert_eq!(candidate.kind, LocalTetrahedronFlipKind::ThreeToTwoEdge);
        assert_eq!(candidate.removed_tetrahedron_ids, vec![1, 2, 3]);
        assert_eq!(candidate.shared_edge, Some([0, 3]));
        assert_eq!(candidate.created_tetrahedra.len(), 2);
        assert_eq!(
            local_tetrahedron_boundary_faces(&candidate.created_tetrahedra),
            local_tetrahedron_boundary_faces(&tetrahedra.map(|tetrahedron| tetrahedron.node_ids))
        );
    }

    #[test]
    fn two_to_three_face_flip_rejects_non_neighbors() {
        let err = two_to_three_face_flip_candidate(
            LocalTetrahedron {
                tetrahedron_id: 1,
                node_ids: [0, 1, 2, 3],
            },
            LocalTetrahedron {
                tetrahedron_id: 2,
                node_ids: [4, 5, 6, 7],
            },
        )
        .expect_err("non-neighbor tetrahedra should not flip");

        assert_eq!(err, LocalTetrahedronFlipError::NoSharedFace);
    }

    #[test]
    fn three_to_two_edge_flip_rejects_invalid_ring() {
        let err = three_to_two_edge_flip_candidate(
            [
                LocalTetrahedron {
                    tetrahedron_id: 1,
                    node_ids: [0, 1, 2, 3],
                },
                LocalTetrahedron {
                    tetrahedron_id: 2,
                    node_ids: [0, 1, 3, 4],
                },
                LocalTetrahedron {
                    tetrahedron_id: 3,
                    node_ids: [0, 1, 5, 6],
                },
            ],
            [0, 1],
        )
        .expect_err("edge ring with more than three opposite nodes should fail");

        assert_eq!(err, LocalTetrahedronFlipError::InvalidEdgeRing);
    }

    #[test]
    fn three_to_two_edge_flip_rejects_noncyclic_ring_edges() {
        let err = three_to_two_edge_flip_candidate(
            [
                LocalTetrahedron {
                    tetrahedron_id: 1,
                    node_ids: [0, 1, 2, 3],
                },
                LocalTetrahedron {
                    tetrahedron_id: 2,
                    node_ids: [0, 1, 3, 2],
                },
                LocalTetrahedron {
                    tetrahedron_id: 3,
                    node_ids: [0, 1, 2, 4],
                },
            ],
            [0, 1],
        )
        .expect_err("duplicate opposite edges do not form a three-edge ring");

        assert_eq!(err, LocalTetrahedronFlipError::InvalidEdgeRing);
    }

    #[test]
    fn degenerate_tetrahedra_are_rejected_before_flip_generation() {
        let err = two_to_three_face_flip_candidate(
            LocalTetrahedron {
                tetrahedron_id: 1,
                node_ids: [0, 0, 2, 3],
            },
            LocalTetrahedron {
                tetrahedron_id: 2,
                node_ids: [0, 2, 1, 4],
            },
        )
        .expect_err("degenerate tetrahedron should fail");

        assert_eq!(
            err,
            LocalTetrahedronFlipError::DegenerateTetrahedron {
                tetrahedron_id: 1,
                node_ids: [0, 0, 2, 3]
            }
        );
    }
}
