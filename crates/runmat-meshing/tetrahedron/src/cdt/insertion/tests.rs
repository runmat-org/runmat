use runmat_meshing_core::{
    quality::predicate::{Point3Construction, PredicateSign},
    NeverCancelled, StableDigest,
};

use super::*;
#[path = "tests/incremental_admission.rs"]
mod incremental_admission;
use crate::cdt::topology::insphere_materialized_volume_nodes;
use crate::cdt::{
    build_delaunay_volume_point_set, build_delaunay_volume_topology, DelaunayPointSetOptions,
};

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
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

fn torus_nodes(count: u32) -> Vec<DelaunayVolumeNode> {
    const MINOR_COUNT: u32 = 6;
    (0..count)
        .map(|index| {
            let major = std::f64::consts::TAU * (index / MINOR_COUNT) as f64 / 12.0;
            let minor = std::f64::consts::TAU * (index % MINOR_COUNT) as f64 / MINOR_COUNT as f64;
            let radius = 2.0 + 0.5 * minor.cos();
            node(
                (index + 1) as u8,
                [
                    radius * major.cos(),
                    radius * major.sin(),
                    0.5 * minor.sin(),
                ],
            )
        })
        .collect()
}

fn topology_has_edge(topology: &DelaunayVolumeTopology, identities: [StableDigest; 2]) -> bool {
    let indices = identities.map(|identity| {
        topology
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .unwrap() as u32
    });
    topology.incidence.vertex_stars[indices[0] as usize]
        .iter()
        .any(|tetrahedron| {
            topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .contains(&indices[1])
        })
}

fn global_delaunay_violation(
    topology: &DelaunayVolumeTopology,
) -> Option<(usize, usize, [u32; 4])> {
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        let tetrahedron_nodes = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize]);
        for (node_index, query) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.vertex_indices.contains(&(node_index as u32)) {
                continue;
            }
            let relation = insphere_materialized_volume_nodes([
                tetrahedron_nodes[0],
                tetrahedron_nodes[1],
                tetrahedron_nodes[2],
                tetrahedron_nodes[3],
                query,
            ])
            .unwrap();
            if relation == PredicateSign::Positive {
                return Some((node_index, tetrahedron_index, tetrahedron.vertex_indices));
            }
        }
    }
    None
}

fn assert_globally_delaunay(topology: &DelaunayVolumeTopology) {
    assert_eq!(global_delaunay_violation(topology), None);
}

#[test]
fn insertion_preserves_a_strong_torus_gabriel_edge() {
    let nodes = torus_nodes(54);
    let topology = build_delaunay_volume_point_set(
        nodes.clone(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let edge = [nodes[2].identity, nodes[3].identity];
    assert!(topology_has_edge(&topology, edge));
}

#[test]
fn point_set_satisfies_the_global_empty_sphere_property() {
    for count in [12, 25, 45, 72] {
        let topology = build_delaunay_volume_point_set(
            torus_nodes(count),
            DelaunayPointSetOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        assert_eq!(
            global_delaunay_violation(&topology),
            None,
            "point-set prefix {count} is not globally Delaunay",
        );
    }
}

#[test]
fn point_set_preserves_a_strong_edge_from_an_exact_segment_midpoint() {
    let mut nodes = torus_nodes(72);
    let endpoints = [nodes[33], nodes[40]];
    let coordinates_m = std::array::from_fn(|axis| {
        endpoints[0].coordinates_m.canonical_value()[axis] * 0.5
            + endpoints[1].coordinates_m.canonical_value()[axis] * 0.5
    });
    let midpoint = DelaunayVolumeNode {
        identity: StableDigest::from_bytes([
            218, 103, 220, 161, 211, 32, 103, 189, 211, 173, 228, 197, 173, 232, 205, 167, 71, 149,
            126, 172, 50, 144, 28, 177, 163, 151, 195, 224, 218, 82, 154, 70,
        ]),
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                coordinates_m,
            ),
        construction: Point3Construction::Segment {
            endpoints: endpoints.map(|node| node.coordinates_m.canonical_value()),
            parameter_numerator: 1,
            parameter_exponent: 1,
        },
    };
    let explicit_midpoint =
        DelaunayVolumeNode::explicit(midpoint.identity, midpoint.coordinates_m.canonical_value());
    let mut explicit_nodes = torus_nodes(72);
    explicit_nodes.push(explicit_midpoint);
    let explicit_topology = build_delaunay_volume_point_set(
        explicit_nodes,
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_globally_delaunay(&explicit_topology);
    assert!(topology_has_edge(
        &explicit_topology,
        [explicit_midpoint.identity, endpoints[1].identity],
    ));
    nodes.push(midpoint);
    let topology =
        build_delaunay_volume_point_set(nodes, DelaunayPointSetOptions::default(), &NeverCancelled)
            .unwrap();

    assert!(topology_has_edge(
        &topology,
        [midpoint.identity, endpoints[1].identity],
    ));
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
fn insertion_retriangulates_an_exact_hull_face_without_zero_volume_cells() {
    let topology = insert_delaunay_volume_node(
        enclosing_tetrahedron(),
        node(25, [0.0, -2.0 / 3.0, -2.0]),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(topology.tetrahedra.len(), 3);
    validate_delaunay_volume_topology(
        &topology,
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn insertion_retriangulates_an_exact_hull_edge_without_zero_volume_cells() {
    let topology = insert_delaunay_volume_node(
        enclosing_tetrahedron(),
        node(25, [0.0, -2.0, -2.0]),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(topology.tetrahedra.len(), 2);
    validate_delaunay_volume_topology(
        &topology,
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn insertion_shrinks_a_nonmanifold_conflict_cavity_to_a_visible_star() {
    let topology = build_delaunay_volume_topology(
        vec![
            node(10, [0.0, 0.0, -1.0]),
            node(20, [0.0, 0.0, 1.0]),
            node(30, [1.0, 0.0, 0.0]),
            node(40, [0.0, 1.0, 0.0]),
            node(50, [-1.0, 0.0, 0.0]),
            node(60, [0.0, -1.0, 0.0]),
        ],
        vec![[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut cavity = BTreeSet::from([0, 3]);
    let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);

    let inserted = node(70, [0.2, 0.1, 0.1]);
    regularize_cavity_boundary(&topology, &mut cavity, inserted, &mut work).unwrap();

    assert!(cavity_is_star_shaped(&topology, &cavity, inserted, &mut work).unwrap());
    assert_eq!(cavity, BTreeSet::from([0]));
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
            node(25, topology.nodes[0].coordinates_m.canonical_value()),
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

// witness: FEA-MESH-EXECUTION-001/budgets_and_cancellation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Volume insertion enforces search budgets and cancellation.
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
