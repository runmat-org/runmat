use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, DelaunaySegmentRecoveryOptions, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};
use runmat_meshing_core::NeverCancelled;

fn patch(interior: bool) -> TopologyMutation {
    let mut points = vec![
        [0., 0., 0.],
        [2., 0., 0.],
        [2., 1., 0.],
        [0., 1., 0.],
        [1., 0.5, 2.],
    ];
    let mut cells = vec![[0, 1, 2, 4], [0, 2, 3, 4]];
    if interior {
        points.push([1., 0.5, -2.]);
        cells.extend([[0, 1, 2, 5], [0, 2, 3, 5]]);
    }
    let topology = build_delaunay_volume_topology(
        points
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes([(index + 1) as u8; 32]),
                    point,
                )
            })
            .collect(),
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut topology = TopologyMutation::from_canonical(topology);
    let mut identity = [0; 32];
    identity[31] = 1;
    topology
        .append_node(DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(identity),
            [10., 10., 10.],
        ))
        .unwrap();
    topology
}

#[test]
fn local_diagonals_preserve_physical_endpoints_across_appended_node_ordering() {
    for interior in [false, true] {
        let mut topology = patch(interior);
        let points = topology.nodes().to_vec();
        let mut work =
            RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
        assert!(recover(
            &mut topology,
            [1, 3],
            &BTreeSet::new(),
            &[],
            if interior {
                DiagonalKind::Interior
            } else {
                DiagonalKind::Boundary
            },
            0,
            &mut work
        )
        .unwrap());
        assert_eq!(topology.nodes(), points);
        assert!(edge_exists(
            &topology,
            points[1].identity,
            points[3].identity,
            0,
            &mut work
        )
        .unwrap());
        topology
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap();
    }
}

#[test]
fn local_diagonal_barriers_and_resource_rejection_preserve_the_transaction() {
    for interior in [false, true] {
        let mut topology = patch(interior);
        let original = topology
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap();
        let mut work =
            RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
        assert!(!recover(
            &mut topology,
            [1, 3],
            &BTreeSet::from([[0, 2]]),
            &[],
            if interior {
                DiagonalKind::Interior
            } else {
                DiagonalKind::Boundary
            },
            0,
            &mut work
        )
        .unwrap());
        assert_eq!(
            topology
                .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
                .unwrap(),
            original
        );
        let mut limited = RecoveryWork::new(
            DelaunaySegmentRecoveryOptions {
                maximum_search_steps: 1,
                ..Default::default()
            },
            &NeverCancelled,
        );
        assert_eq!(
            recover(
                &mut topology,
                [1, 3],
                &BTreeSet::new(),
                &[],
                if interior {
                    DiagonalKind::Interior
                } else {
                    DiagonalKind::Boundary
                },
                0,
                &mut limited
            )
            .unwrap_err()
            .kind,
            DelaunaySegmentRecoveryErrorKind::ResourceLimit
        );
        assert_eq!(
            topology
                .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
                .unwrap(),
            original
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A four-cell planar cavity reconnects through exact local replacement admission,
//   retains original physical points and passes independent exhaustive empty-sphere validation.
//   This component fixture does not establish arbitrary recovery, native heat acceptance,
//   whole-task memory or asymptotic scaling.
#[test]
fn admitted_planar_reconnection_preserves_all_original_points_and_empty_spheres() {
    let mut topology = patch(true);
    let points = topology.nodes().to_vec();
    let mut work = RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
    assert!(
        try_recover_admitted_planar_diagonal(&mut topology, 1, 3, &[], &[], 0, &mut work).unwrap()
    );
    assert_eq!(topology.nodes(), points);
    let admitted = topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    crate::cdt::validate_delaunay_volume_topology(&admitted, Default::default(), &NeverCancelled)
        .unwrap();
}

#[test]
fn planar_reconnection_preserves_protected_internal_faces_beyond_the_planar_patch() {
    for interior in [false, true] {
        let mut topology = patch(interior);
        let original = topology
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap();
        let protected = stable_face(&topology, [0, 2, 4]);
        let mut work =
            RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
        assert!(!recover(
            &mut topology,
            [1, 3],
            &BTreeSet::new(),
            &[protected],
            if interior {
                DiagonalKind::Interior
            } else {
                DiagonalKind::Boundary
            },
            0,
            &mut work
        )
        .unwrap());
        assert_eq!(
            topology
                .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
                .unwrap(),
            original
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Exact opposite-edge face lookup preserves every candidate and its original Cartesian
//   order across endpoint orderings and missing partners. Existing geometric admission, barrier and
//   atomic rejection tests remain required. This candidate-equivalence witness does not establish
//   arbitrary valence bounds, complete recovery termination, native timing or whole-task admission.
#[test]
fn indexed_planar_partners_match_every_exhaustive_pair_in_original_order() {
    for (left, right) in [(0, 1), (7, 2), (12, 11)] {
        let mut left_faces = Vec::new();
        let mut right_faces = BTreeMap::new();
        for first in 0..16 {
            for second in first + 1..16 {
                if [first, second].contains(&left) || [first, second].contains(&right) {
                    continue;
                }
                let mut face = [left, first, second];
                face.sort_unstable();
                left_faces.push(face);
                if (first + second) % 3 != 0 {
                    let mut face = [right, first, second];
                    face.sort_unstable();
                    right_faces.insert(face, ());
                }
            }
        }
        left_faces.sort_unstable();
        let expected = left_faces
            .iter()
            .flat_map(|left_face| {
                right_faces.keys().filter_map(move |right_face| {
                    (left_face
                        .iter()
                        .filter(|vertex| right_face.contains(vertex))
                        .count()
                        == 2)
                        .then_some((*left_face, *right_face))
                })
            })
            .collect::<Vec<_>>();
        let actual = left_faces
            .iter()
            .filter_map(|left_face| {
                let right_face = partner_face(*left_face, left, right);
                right_faces
                    .contains_key(&right_face)
                    .then_some((*left_face, right_face))
            })
            .collect::<Vec<_>>();
        assert!(!expected.is_empty());
        assert_eq!(actual, expected);
    }
}
