use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::NeverCancelled;

use super::*;
#[path = "tests/acute_family.rs"]
mod acute_family;
#[path = "tests/parameter.rs"]
mod parameter;
#[path = "tests/reconnection.rs"]
mod reconnection;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_point_set,
    build_delaunay_volume_topology, DelaunayConstraintNode, DelaunayConstraintSegment,
    DelaunayPointSetOptions, DelaunayTopologyOptions,
};

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, value.to_owned(), Vec::new())
}

pub(super) fn octahedron_constraints() -> DelaunayConstraints {
    let coordinates = [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    DelaunayConstraints {
        nodes: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 10) as u8; 32]),
                source_vertex_id: None,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
            })
            .collect(),
        segments: vec![DelaunayConstraintSegment {
            vertex_indices: [1, 4],
            source_edge_id: None,
            source_edge_parameters: None,
        }],
        facets: Vec::new(),
    }
}

fn octahedron_topology(constraints: &DelaunayConstraints) -> DelaunayVolumeTopology {
    build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn triangulated_torus_constraints() -> DelaunayConstraints {
    const MAJOR_COUNT: u32 = 12;
    const MINOR_COUNT: u32 = 6;
    let mut nodes = Vec::new();
    for major_index in 0..MAJOR_COUNT {
        let major = std::f64::consts::TAU * major_index as f64 / MAJOR_COUNT as f64;
        for minor_index in 0..MINOR_COUNT {
            let minor = std::f64::consts::TAU * minor_index as f64 / MINOR_COUNT as f64;
            let radius = 2.0 + 0.5 * minor.cos();
            let index = major_index * MINOR_COUNT + minor_index;
            nodes.push(DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
                source_vertex_id: None,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([
                    radius * major.cos(),
                    radius * major.sin(),
                    0.5 * minor.sin(),
                ]),
            });
        }
    }
    let node_index =
        |major: u32, minor: u32| (major % MAJOR_COUNT) * MINOR_COUNT + minor % MINOR_COUNT;
    let mut edges = std::collections::BTreeSet::new();
    for major in 0..MAJOR_COUNT {
        for minor in 0..MINOR_COUNT {
            let lower_left = node_index(major, minor);
            let lower_right = node_index(major + 1, minor);
            let upper_left = node_index(major, minor + 1);
            let upper_right = node_index(major + 1, minor + 1);
            for mut edge in [
                [lower_left, lower_right],
                [lower_left, upper_left],
                [lower_left, upper_right],
            ] {
                edge.sort_unstable();
                edges.insert(edge);
            }
        }
    }
    DelaunayConstraints {
        nodes,
        segments: edges
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: None,
                source_edge_parameters: None,
            })
            .collect(),
        facets: Vec::new(),
    }
}

fn cospherical_bipyramid() -> (DelaunayConstraints, DelaunayVolumeTopology) {
    let coordinates = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [-3.0, -4.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ];
    let constraints = DelaunayConstraints {
        nodes: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 30) as u8; 32]),
                source_vertex_id: None,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
            })
            .collect(),
        segments: vec![
            DelaunayConstraintSegment {
                vertex_indices: [0, 1],
                source_edge_id: Some(entity(PersistentEntityKind::Edge, "curve:base")),
                source_edge_parameters: Some([0.0, 1.0]),
            },
            DelaunayConstraintSegment {
                vertex_indices: [3, 4],
                source_edge_id: None,
                source_edge_parameters: None,
            },
        ],
        facets: Vec::new(),
    };
    let topology = build_delaunay_volume_topology(
        constraints.volume_nodes(),
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    (constraints, topology)
}

#[test]
fn recovery_prefers_a_checked_reconnection_over_steiner_splitting() {
    let constraints = octahedron_constraints();
    let topology = octahedron_topology(&constraints);
    let recovered = recover_delaunay_segments(
        topology.clone(),
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = recover_delaunay_segments(
        topology,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(recovered, repeated);
    assert_eq!(recovered.recovery_passes, 1);
    assert!(recovered.steiner_node_identities.is_empty());
    assert_eq!(recovered.segments[0].nodes.len(), 2);
    validate_delaunay_segment_recovery(
        &recovered,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn recovery_scales_to_a_closed_triangulated_torus() {
    let constraints = triangulated_torus_constraints();
    let options = DelaunaySegmentRecoveryOptions::default();
    let topology = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let recovered =
        recover_delaunay_segments(topology.clone(), &constraints, options, &NeverCancelled)
            .unwrap();

    assert_eq!(recovered.segments.len(), constraints.segments.len());
    assert!(recovered
        .segments
        .iter()
        .all(|segment| segment.nodes.len() >= 2));
    assert!(!recovered.steiner_node_identities.is_empty());
    validate_delaunay_segment_recovery(&recovered, &constraints, options, &NeverCancelled).unwrap();

    let split = recovered
        .segments
        .iter()
        .find(|segment| segment.nodes.len() > 2)
        .expect("the torus fixture exercises Steiner segment recovery");
    let segment = &constraints.segments[split.constraint_index as usize];
    let endpoints = segment
        .vertex_indices
        .map(|index| constraints.nodes[index as usize].identity);
    let mut support = [
        endpoints[0],
        endpoints[1],
        constraints.nodes[(segment.vertex_indices[1] as usize + 1) % constraints.nodes.len()]
            .identity,
    ];
    support.sort_unstable();
    let temporary = recover_temporary_delaunay_segment(
        &topology,
        TemporarySegmentRequest {
            endpoints,
            protected_edges: &[],
            protected_faces: &[],
            constraint_index: split.constraint_index,
            options: DelaunaySegmentRecoveryOptions {
                maximum_steiner_nodes: 64,
                maximum_split_depth: 16,
                ..options
            },
            cancellation: &NeverCancelled,
        },
        |numerator, exponent, _coordinates_m| {
            crate::cdt::facet_recovery::facet_steiner_identity(
                support,
                0,
                (u64::from(exponent) << 48) | numerator,
            )
        },
    )
    .unwrap();
    assert_eq!(temporary.chain.first().unwrap().identity, endpoints[0]);
    assert_eq!(temporary.chain.last().unwrap().identity, endpoints[1]);
    assert!(temporary.inserted_nodes.len() <= 64);
    assert!(temporary.conflicting_protected_edges.is_empty());
    validate_constrained_delaunay_volume_topology(
        &temporary.topology,
        &[],
        options.insertion,
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn recovery_coalesces_only_at_coordinate_resolution() {
    let point: [f64; 3] = [-0.2591172135939178, -2.365808596882859, -0.309347234130807];
    let adjacent = [f64::from_bits(point[0].to_bits() + 1), point[1], point[2]];
    assert!(coordinates_share_representable_location(point, adjacent));
    assert!(!coordinates_share_representable_location(
        point,
        [point[0] + 1.0e-12, point[1], point[2]],
    ));
}

#[test]
fn recovery_evidence_is_independently_rejected_after_tampering() {
    let constraints = octahedron_constraints();
    let mut recovered = recover_delaunay_segments(
        octahedron_topology(&constraints),
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    recovered.segments[0].nodes[1].parameter_numerator = 0;
    assert_eq!(
        validate_delaunay_segment_recovery(
            &recovered,
            &constraints,
            DelaunaySegmentRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySegmentRecoveryErrorKind::InvalidConstraints
    );

    let first = constraints.nodes[1].identity;
    let last = constraints.nodes[4].identity;
    let split_constraints = DelaunayConstraints {
        nodes: vec![constraints.nodes[1].clone(), constraints.nodes[4].clone()],
        segments: vec![DelaunayConstraintSegment {
            vertex_indices: [0, 1],
            source_edge_id: None,
            source_edge_parameters: None,
        }],
        facets: Vec::new(),
    };
    let midpoint = StableDigest::from_bytes([99; 32]);
    let midpoint_coordinates = [0.0, 0.0, 0.0];
    let nodes = vec![
        DelaunayVolumeNode::explicit(first, constraints.nodes[1].coordinates_m.canonical_value()),
        DelaunayVolumeNode::explicit(last, constraints.nodes[4].coordinates_m.canonical_value()),
        DelaunayVolumeNode {
            identity: midpoint,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(midpoint_coordinates),
            construction: Point3Construction::Segment {
                endpoints: [
                    constraints.nodes[1].coordinates_m.canonical_value(),
                    constraints.nodes[4].coordinates_m.canonical_value(),
                ],
                parameter_numerator: 1,
                parameter_exponent: 1,
            },
        },
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([100; 32]), [0.0, 0.0, 1.0]),
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([101; 32]), [1.0, 0.0, 0.0]),
    ];
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 2, 3, 4], [2, 1, 3, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut recovered = DelaunaySegmentRecovery {
        topology,
        segments: vec![DelaunayRecoveredSegment {
            constraint_index: 0,
            nodes: vec![
                DelaunayRecoveredSegmentNode {
                    identity: first,
                    parameter_numerator: 0,
                    parameter_exponent: 0,
                },
                DelaunayRecoveredSegmentNode {
                    identity: midpoint,
                    parameter_numerator: 1,
                    parameter_exponent: 1,
                },
                DelaunayRecoveredSegmentNode {
                    identity: last,
                    parameter_numerator: 1,
                    parameter_exponent: 0,
                },
            ],
        }],
        steiner_node_identities: vec![midpoint],
        recovery_passes: 1,
    };
    recovered.steiner_node_identities.clear();
    assert_eq!(
        validate_delaunay_segment_recovery(
            &recovered,
            &split_constraints,
            DelaunaySegmentRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySegmentRecoveryErrorKind::InvalidConstraints
    );
}

#[test]
fn recovery_preserves_existing_edges_without_steiner_nodes() {
    let mut constraints = octahedron_constraints();
    let topology = octahedron_topology(&constraints);
    let mut edge = [
        topology.tetrahedra[0].vertex_indices[0],
        topology.tetrahedra[0].vertex_indices[1],
    ];
    edge.sort_unstable();
    constraints.segments[0].vertex_indices = edge;

    let recovered = recover_delaunay_segments(
        topology.clone(),
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(recovered.topology, topology);
    assert!(recovered.steiner_node_identities.is_empty());
    assert_eq!(recovered.segments[0].nodes.len(), 2);
}

#[test]
fn recovery_uses_a_checked_face_flip_before_steiner_splitting() {
    let (constraints, topology) = cospherical_bipyramid();

    let recovered = recover_delaunay_segments(
        topology,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert!(recovered.steiner_node_identities.is_empty());
    assert_eq!(recovered.segments[0].nodes.len(), 2);
    assert_eq!(recovered.segments[1].nodes.len(), 2);
    assert_eq!(recovered.topology.tetrahedra.len(), 3);
    validate_delaunay_segment_recovery(
        &recovered,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let search_limited = DelaunaySegmentRecoveryOptions {
        maximum_search_steps: 1,
        ..DelaunaySegmentRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_segments(
            cospherical_bipyramid().1,
            &constraints,
            search_limited,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySegmentRecoveryErrorKind::ResourceLimit
    );
}

#[test]
fn recovery_rejects_mismatched_constraints_and_assigned_regions() {
    let constraints = octahedron_constraints();
    let topology = octahedron_topology(&constraints);
    let mut mismatched = constraints.clone();
    {
        let quantity_change = 0.25;
        let quantity_index = 0;
        let quantity_slot = &mut mismatched.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert_eq!(
        recover_delaunay_segments(
            topology.clone(),
            &mismatched,
            DelaunaySegmentRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySegmentRecoveryErrorKind::InvalidTopology
    );

    let region = PersistentEntityId::new(
        PersistentEntityKind::Region,
        "region:assigned".to_owned(),
        Vec::new(),
    );
    let tetrahedron_count = topology.tetrahedra.len();
    let assigned = assign_delaunay_volume_regions(
        topology,
        vec![region; tetrahedron_count],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(
        recover_delaunay_segments(
            assigned,
            &constraints,
            DelaunaySegmentRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySegmentRecoveryErrorKind::InvalidTopology
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn recovery_enforces_search_work_steiner_and_cancellation_limits() {
    let constraints = octahedron_constraints();
    let topology = octahedron_topology(&constraints);
    let bounded = DelaunaySegmentRecoveryOptions {
        maximum_search_steps: 1,
        ..DelaunaySegmentRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_segments(topology.clone(), &constraints, bounded, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunaySegmentRecoveryErrorKind::ResourceLimit
    );
    let cancelled = DelaunaySegmentRecoveryOptions {
        constraints: DelaunayConstraintOptions {
            cancellation_check_interval: 1,
            ..DelaunayConstraintOptions::default()
        },
        ..DelaunaySegmentRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_segments(topology, &constraints, cancelled, &Cancelled)
            .unwrap_err()
            .kind,
        DelaunaySegmentRecoveryErrorKind::Cancelled
    );
}

#[path = "tests/encroached_reconnection.rs"]
mod encroached_reconnection;
#[path = "tests/indexed_encroachment.rs"]
mod indexed_encroachment;
