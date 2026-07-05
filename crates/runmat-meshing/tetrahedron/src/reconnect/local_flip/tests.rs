use super::*;
use std::collections::BTreeMap;

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

    let candidate = two_to_three_face_flip_candidate(left, right).expect("shared face should flip");

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
    let candidate = two_to_three_face_flip_candidate(left, right).expect("shared face should flip");
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
fn local_flip_improvement_accepts_quality_improving_flip() {
    let left = LocalTetrahedron {
        tetrahedron_id: 9,
        node_ids: [0, 1, 2, 3],
    };
    let right = LocalTetrahedron {
        tetrahedron_id: 4,
        node_ids: [0, 2, 1, 4],
    };
    let candidate = two_to_three_face_flip_candidate(left, right).expect("shared face should flip");
    let node_coordinates = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.0, 1.0, 0.0]),
        (
            3,
            [0.5357890395018374, -0.18744218273792734, 0.3827030431131124],
        ),
        (
            4,
            [0.4852929197428164, 1.1962618573152073, -0.10539356221586194],
        ),
    ]);

    let report = evaluate_local_tetrahedron_flip_improvement(
        &[left, right],
        &candidate,
        &node_coordinates,
        LocalTetrahedronFlipQualityThresholds {
            min_volume_m3: 1.0e-12,
            min_scaled_jacobian: 0.01,
        },
    )
    .expect("candidate should improve the worst local scaled-Jacobian");

    assert_eq!(report.removed_tetrahedron_count, 2);
    assert_eq!(report.created_tetrahedron_count, 3);
    assert!(report.candidate_min_scaled_jacobian > report.current_min_scaled_jacobian);
}

#[test]
fn local_flip_improvement_rejects_candidate_that_does_not_improve_quality_or_count() {
    let left = LocalTetrahedron {
        tetrahedron_id: 4,
        node_ids: [0, 1, 2, 3],
    };
    let right = LocalTetrahedron {
        tetrahedron_id: 9,
        node_ids: [0, 2, 1, 4],
    };
    let candidate = two_to_three_face_flip_candidate(left, right).expect("shared face should flip");
    let node_coordinates = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.0, 1.0, 0.0]),
        (3, [1.0 / 3.0, 1.0 / 3.0, 1.0]),
        (4, [1.0 / 3.0, 1.0 / 3.0, -1.0]),
    ]);

    let err = evaluate_local_tetrahedron_flip_improvement(
        &[left, right],
        &candidate,
        &node_coordinates,
        LocalTetrahedronFlipQualityThresholds {
            min_volume_m3: 1.0e-12,
            min_scaled_jacobian: 0.05,
        },
    )
    .expect_err("quality-neutral count-increasing flip should be rejected");

    assert_eq!(err, LocalTetrahedronFlipError::QualityDoesNotImprove);
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
