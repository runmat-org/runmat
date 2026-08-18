use runmat_meshing_core::{NeverCancelled, StableDigest};

use super::*;
use crate::cdt::build_delaunay_volume_topology;

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode {
        identity: StableDigest::from_bytes([identity; 32]),
        coordinates_m,
    }
}

fn enclosing_tetrahedron() -> DelaunayVolumeTopology {
    build_delaunay_volume_topology(
        vec![
            node(10, [-2.0, -2.0, -2.0]),
            node(20, [2.0, -2.0, -2.0]),
            node(30, [0.0, 2.0, -2.0]),
            node(40, [0.0, 0.0, 2.0]),
        ],
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn insertion_retriangulates_the_connected_cavity_canonically() {
    let topology = insert_delaunay_volume_node(
        enclosing_tetrahedron(),
        node(25, [0.0, 0.0, 0.0]),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(
        topology.nodes[2].identity,
        StableDigest::from_bytes([25; 32])
    );
    assert_eq!(topology.tetrahedra.len(), 4);
    assert!(topology
        .tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.vertex_indices.contains(&2)));
    validate_delaunay_volume_topology(
        &topology,
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn insertion_is_independent_of_admitted_point_order() {
    let insert = |first, second| {
        let topology = insert_delaunay_volume_node(
            enclosing_tetrahedron(),
            first,
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        insert_delaunay_volume_node(
            topology,
            second,
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap()
    };
    let left = node(22, [-0.37, -0.11, 0.08]);
    let right = node(28, [0.29, 0.17, -0.21]);
    assert_eq!(insert(left, right), insert(right, left));
}

#[test]
fn insertion_retriangulates_an_exact_shared_face_without_zero_volume_cells() {
    let topology = build_delaunay_volume_topology(
        vec![
            node(10, [0.0, 0.0, 0.0]),
            node(20, [2.0, 0.0, 0.0]),
            node(30, [0.0, 2.0, 0.0]),
            node(40, [0.0, 0.0, 2.0]),
            node(50, [0.0, 0.0, -2.0]),
        ],
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let topology = insert_delaunay_volume_node(
        topology,
        node(25, [0.5, 0.5, 0.0]),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(topology.tetrahedra.len(), 6);
    validate_delaunay_volume_topology(
        &topology,
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn insertion_rejects_outside_repeated_and_tampered_inputs() {
    let topology = enclosing_tetrahedron();
    assert_eq!(
        insert_delaunay_volume_node(
            topology.clone(),
            node(25, [10.0, 10.0, 10.0]),
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::PointOutsideTopology
    );
    assert_eq!(
        insert_delaunay_volume_node(
            topology.clone(),
            node(25, topology.nodes[0].coordinates_m),
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::InvalidNode
    );
    let mut tampered = topology;
    tampered.tetrahedra[0].neighbors[0] = Some(0);
    assert_eq!(
        validate_delaunay_volume_topology(
            &tampered,
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::InvalidTopology
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn insertion_enforces_search_budgets_and_cancellation() {
    let limited = DelaunayInsertionOptions {
        maximum_predicate_evaluations: 1,
        ..DelaunayInsertionOptions::default()
    };
    assert_eq!(
        insert_delaunay_volume_node(
            enclosing_tetrahedron(),
            node(25, [0.0, 0.0, 0.0]),
            limited,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::ResourceLimit
    );

    let bounded_output = DelaunayInsertionOptions {
        topology: DelaunayTopologyOptions {
            maximum_tetrahedra: 3,
            ..DelaunayTopologyOptions::default()
        },
        ..DelaunayInsertionOptions::default()
    };
    assert_eq!(
        insert_delaunay_volume_node(
            enclosing_tetrahedron(),
            node(25, [0.0, 0.0, 0.0]),
            bounded_output,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::ResourceLimit
    );

    let bounded_boundary = DelaunayInsertionOptions {
        maximum_cavity_boundary_faces: 3,
        ..DelaunayInsertionOptions::default()
    };
    assert_eq!(
        insert_delaunay_volume_node(
            enclosing_tetrahedron(),
            node(25, [0.0, 0.0, 0.0]),
            bounded_boundary,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::ResourceLimit
    );

    let immediate = DelaunayInsertionOptions {
        topology: DelaunayTopologyOptions {
            cancellation_check_interval: 1,
            ..DelaunayTopologyOptions::default()
        },
        ..DelaunayInsertionOptions::default()
    };
    assert_eq!(
        insert_delaunay_volume_node(
            enclosing_tetrahedron(),
            node(25, [0.0, 0.0, 0.0]),
            immediate,
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::Cancelled
    );
}
