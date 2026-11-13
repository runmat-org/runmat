use runmat_meshing_core::{NeverCancelled, StableDigest};

use super::*;
use crate::cdt::{insert_delaunay_volume_node, validate_delaunay_volume_topology};

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
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
fn point_set_bootstrap_is_canonical_and_closes_the_hull() {
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
fn point_set_bootstrap_closes_the_convex_hull_of_a_torus_point_cloud() {
    let mut points = Vec::new();
    let mut identity = 1_u8;
    for major_index in 0..12 {
        let major = std::f64::consts::TAU * major_index as f64 / 12.0;
        for minor_index in 0..6 {
            let minor = std::f64::consts::TAU * minor_index as f64 / 6.0;
            let radius = 2.0 + 0.5 * minor.cos();
            points.push(node(
                identity,
                [
                    radius * major.cos(),
                    radius * major.sin(),
                    0.5 * minor.sin(),
                ],
            ));
            identity += 1;
        }
    }
    let first = points[3].coordinates_m.canonical_value();
    let last = points[9].coordinates_m.canonical_value();
    let midpoint = std::array::from_fn(|axis| first[axis] * 0.5 + last[axis] * 0.5);
    let topology = build_delaunay_volume_point_set(
        points,
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    insert_delaunay_volume_node(
        topology,
        node(250, midpoint),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
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
    duplicate[1].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([-0.0, 0.0, 0.0]);
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
fn point_set_bootstrap_handles_low_identity_and_extreme_range() {
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

#[test]
fn admitted_cpu_counts_preserve_complete_point_set_and_independent_validation() {
    let mut state = 31_u64;
    let mut points = point_set();
    for index in 0..192_u32 {
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as f64 / u32::MAX as f64
        };
        points.push(DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(
                <sha2::Sha256 as sha2::Digest>::digest(index.to_be_bytes()).into(),
            ),
            [4. * random() - 1., 4. * random() - 1., 4. * random() - 1.],
        ));
    }
    let mut reference = None;
    for threads in [1, 2, 4] {
        let start = std::time::Instant::now();
        let mesh = runmat_cpu_execution::with_cpu_pool(threads, || {
            build_delaunay_volume_point_set(
                points.clone(),
                DelaunayPointSetOptions::default(),
                &NeverCancelled,
            )
        })
        .unwrap()
        .unwrap();
        validate_delaunay_volume_topology(
            &mesh,
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        if let Some(expected) = &reference {
            assert_eq!(&mesh, expected);
        } else {
            reference = Some(mesh);
        }
        eprintln!(
            "CPU_POINT_SET threads={threads} nodes={} wall_seconds={:.6}",
            points.len(),
            start.elapsed().as_secs_f64()
        );
    }
}

/// The first 177 cap points of a 1 mm wire, in the canonical-frame order the volume mesher used,
/// and the first point below the cap (the seed apex). The last cap point lies on the flat hull
/// cap, inside the circumcircle of cap faces it does not see; the cells under those faces must
/// join its cavity.
// witness: FEA-MESH-DISCRETIZATION-001/volume_conformity @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Regression: a point on a flat hull cap replaces exactly the cells whose circumcircle
//   contains it in the incremental point-set construction; facet recovery and carving are separate
//   tests.
#[test]
fn a_point_on_a_flat_hull_cap_replaces_the_cells_whose_circumcircle_holds_it() {
    let nodes = include_str!("fixtures/wire_cap_insertion_order.txt")
        .lines()
        .map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            let mut identity = [0_u8; 32];
            for (index, byte) in identity.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&fields[0][2 * index..2 * index + 2], 16).unwrap();
            }
            let coordinate = |field: &str| f64::from_bits(u64::from_str_radix(field, 16).unwrap());
            DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [
                    coordinate(fields[1]),
                    coordinate(fields[2]),
                    coordinate(fields[3]),
                ],
            )
        })
        .collect::<Vec<_>>();
    let order = nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (node.identity, position))
        .collect::<std::collections::BTreeMap<_, _>>();
    let topology = build_delaunay_volume_point_set_with_order(
        nodes.clone(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
        |left, right| order[&left.identity].cmp(&order[&right.identity]),
    )
    .unwrap();
    assert_eq!(topology.nodes.len(), nodes.len());
    validate_delaunay_volume_topology(
        &topology,
        crate::cdt::DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_conformity @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Regression (EV battery module, 22 shared-interface boxes): boundary nodes of
//   axis-aligned boxes land on hull edges between collinear nodes; whatever cell the walk
//   returns, the point set stays Delaunay. Facet recovery and carving are separate tests.
#[test]
fn box_surface_lattice_stays_delaunay_in_any_insertion_order() {
    let mut nodes = Vec::new();
    for x in 0..5 {
        for y in 0..4 {
            for z in 0..3 {
                if x == 0 || x == 4 || y == 0 || y == 3 || z == 0 || z == 2 {
                    let identity = nodes.len() as u8 + 1;
                    nodes.push(node(identity, [f64::from(x), f64::from(y), f64::from(z)]));
                }
            }
        }
    }
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..64 {
        let mut order = (0..nodes.len()).collect::<Vec<_>>();
        for slot in (1..order.len()).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            order.swap(slot, (state >> 33) as usize % (slot + 1));
        }
        let rank = order
            .iter()
            .enumerate()
            .map(|(position, index)| (nodes[*index].identity, position))
            .collect::<std::collections::BTreeMap<_, _>>();
        let topology = build_delaunay_volume_point_set_with_order(
            nodes.clone(),
            DelaunayPointSetOptions::default(),
            &NeverCancelled,
            |left, right| rank[&left.identity].cmp(&rank[&right.identity]),
        )
        .unwrap_or_else(|failure| panic!("order {order:?}: {failure}"));
        assert_eq!(topology.nodes.len(), nodes.len());
        validate_delaunay_volume_topology(
            &topology,
            crate::cdt::DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_or_else(|failure| panic!("order {order:?}: {failure}"));
    }
}
