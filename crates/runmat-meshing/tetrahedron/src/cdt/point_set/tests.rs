use runmat_meshing_core::{NeverCancelled, StableDigest};

use super::*;

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode {
        identity: StableDigest::from_bytes([identity; 32]),
        coordinates_m,
    }
}

fn point_set() -> Vec<DelaunayVolumeNode> {
    vec![
        node(10, [0.0, 0.0, 0.0]),
        node(20, [2.0, 0.0, 0.0]),
        node(30, [0.0, 2.0, 0.0]),
        node(40, [0.0, 0.0, 2.0]),
        node(50, [0.35, 0.41, 0.29]),
    ]
}

#[test]
fn point_set_bootstrap_is_canonical_and_removes_enclosing_nodes() {
    let expected_nodes = point_set();
    let mut reversed = expected_nodes.clone();
    reversed.reverse();
    let forward = build_delaunay_volume_point_set(
        expected_nodes.clone(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let backward = build_delaunay_volume_point_set(
        reversed,
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(forward, backward);
    assert_eq!(forward.nodes, expected_nodes);
    assert!(forward.tetrahedra.len() >= 4);
    assert!(forward.tetrahedra.iter().all(|tetrahedron| tetrahedron
        .vertex_indices
        .iter()
        .all(|vertex| (*vertex as usize) < expected_nodes.len())));
    validate_delaunay_volume_topology(
        &forward,
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn point_set_bootstrap_preserves_a_single_tetrahedron() {
    let topology = build_delaunay_volume_point_set(
        point_set()[..4].to_vec(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(topology.tetrahedra.len(), 1);
    assert_eq!(topology.tetrahedra[0].vertex_indices, [1, 0, 2, 3]);
}

#[test]
fn point_set_bootstrap_resolves_cospherical_topology_by_stable_identity() {
    let points = vec![
        node(10, [-1.0, 0.0, 0.0]),
        node(20, [0.0, -1.0, 0.0]),
        node(30, [0.0, 0.0, -1.0]),
        node(40, [0.0, 0.0, 1.0]),
        node(50, [0.0, 1.0, 0.0]),
        node(60, [1.0, 0.0, 0.0]),
    ];
    let mut permuted = points.clone();
    permuted.rotate_left(2);
    permuted.reverse();
    let expected = build_delaunay_volume_point_set(
        points,
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let actual = build_delaunay_volume_point_set(
        permuted,
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.tetrahedra.len(), 4);
}

#[test]
fn point_set_bootstrap_rejects_degenerate_and_duplicate_inputs() {
    let coplanar = vec![
        node(10, [0.0, 0.0, 0.0]),
        node(20, [1.0, 0.0, 0.0]),
        node(30, [0.0, 1.0, 0.0]),
        node(40, [1.0, 1.0, 0.0]),
    ];
    assert_eq!(
        build_delaunay_volume_point_set(
            coplanar,
            DelaunayPointSetOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayPointSetErrorKind::InsufficientDimension
    );

    let mut duplicate = point_set();
    duplicate[1].coordinates_m = [-0.0, 0.0, 0.0];
    assert_eq!(
        build_delaunay_volume_point_set(
            duplicate,
            DelaunayPointSetOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayPointSetErrorKind::InvalidNode
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn point_set_bootstrap_enforces_limits_and_cancellation() {
    let bounded = DelaunayPointSetOptions {
        insertion: DelaunayInsertionOptions {
            topology: super::super::DelaunayTopologyOptions {
                maximum_nodes: 4,
                ..super::super::DelaunayTopologyOptions::default()
            },
            ..DelaunayInsertionOptions::default()
        },
        ..DelaunayPointSetOptions::default()
    };
    assert_eq!(
        build_delaunay_volume_point_set(point_set(), bounded, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayPointSetErrorKind::ResourceLimit
    );
    let bounded_intermediate = DelaunayPointSetOptions {
        insertion: DelaunayInsertionOptions {
            topology: super::super::DelaunayTopologyOptions {
                maximum_tetrahedra: 1,
                ..super::super::DelaunayTopologyOptions::default()
            },
            ..DelaunayInsertionOptions::default()
        },
        ..DelaunayPointSetOptions::default()
    };
    assert_eq!(
        build_delaunay_volume_point_set(point_set(), bounded_intermediate, &NeverCancelled,)
            .unwrap_err()
            .kind,
        DelaunayPointSetErrorKind::ResourceLimit
    );
    assert_eq!(
        build_delaunay_volume_point_set(
            point_set(),
            DelaunayPointSetOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayPointSetErrorKind::Cancelled
    );
}

#[test]
fn point_set_bootstrap_handles_construction_identity_collision_and_extreme_range() {
    let mut collision = point_set()[..4].to_vec();
    let mut reserved = [0; 32];
    reserved[31] = 1;
    collision[0].identity = StableDigest::from_bytes(reserved);
    let topology = build_delaunay_volume_point_set(
        collision,
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(topology.tetrahedra.len(), 1);

    let extreme = vec![
        node(10, [-f64::MAX, 0.0, 0.0]),
        node(20, [f64::MAX, 0.0, 0.0]),
        node(30, [0.0, 1.0, 0.0]),
        node(40, [0.0, 0.0, 1.0]),
    ];
    assert_eq!(
        build_delaunay_volume_point_set(
            extreme,
            DelaunayPointSetOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayPointSetErrorKind::ResourceLimit
    );
}
