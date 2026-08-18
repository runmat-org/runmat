use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::quality::predicate::tetrahedron_volume;
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};

use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, recover_delaunay_segments, validate_delaunay_segment_recovery,
    DelaunayConstraintFacet, DelaunayConstraintFacetSide, DelaunayConstraintNode,
    DelaunayConstraintSegment, DelaunayTopologyOptions, DelaunayVolumeNode,
};

fn constraints(include_crossing_segment: bool) -> DelaunayConstraints {
    let coordinates = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [-3.0, -4.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ];
    let mut segments = vec![[0, 1], [0, 2], [1, 2]];
    if include_crossing_segment {
        segments.push([3, 4]);
    }
    DelaunayConstraints {
        nodes: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 50) as u8; 32]),
                source_vertex_id: None,
                coordinates_m,
            })
            .collect(),
        segments: segments
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: None,
                source_edge_parameters: None,
            })
            .collect(),
        facets: vec![DelaunayConstraintFacet {
            facet_id: StableDigest::from_bytes([90; 32]),
            vertex_indices: [0, 1, 2],
            source_face_id: entity(PersistentEntityKind::Face, "face:base"),
            positive_side: DelaunayConstraintFacetSide::Exterior,
            negative_side: DelaunayConstraintFacetSide::Region(entity(
                PersistentEntityKind::Region,
                "solid",
            )),
            contact_ids: Vec::new(),
        }],
    }
}

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn topology(
    around_central_edge: bool,
    constraints: &DelaunayConstraints,
) -> DelaunayVolumeTopology {
    let tetrahedra = if around_central_edge {
        vec![[3, 4, 0, 1], [3, 4, 1, 2], [3, 4, 2, 0]]
    } else {
        vec![[0, 1, 2, 3], [0, 2, 1, 4]]
    };
    build_delaunay_volume_topology(
        constraints.volume_nodes(),
        tetrahedra,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn segment_recovery(
    around_central_edge: bool,
    constraints: &DelaunayConstraints,
) -> DelaunaySegmentRecovery {
    recover_delaunay_segments(
        topology(around_central_edge, constraints),
        constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn split_boundary_segment_recovery(
    constraints: &DelaunayConstraints,
) -> (DelaunaySegmentRecovery, StableDigest) {
    let mut midpoint_identity = [50; 32];
    midpoint_identity[31] = 51;
    let midpoint_identity = StableDigest::from_bytes(midpoint_identity);
    let mut nodes = vec![DelaunayVolumeNode {
        identity: constraints.nodes[0].identity,
        coordinates_m: constraints.nodes[0].coordinates_m,
    }];
    nodes.push(DelaunayVolumeNode {
        identity: midpoint_identity,
        coordinates_m: [2.5, 2.5, 0.0],
    });
    nodes.extend(
        constraints
            .nodes
            .iter()
            .skip(1)
            .map(|node| DelaunayVolumeNode {
                identity: node.identity,
                coordinates_m: node.coordinates_m,
            }),
    );
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 3, 4], [1, 2, 3, 4], [0, 3, 1, 5], [1, 3, 2, 5]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    (
        recover_delaunay_segments(
            topology,
            constraints,
            DelaunaySegmentRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap(),
        midpoint_identity,
    )
}

#[test]
fn facet_recovery_uses_a_checked_edge_star_flip() {
    let constraints = constraints(false);
    let recovered = recover_delaunay_facets(
        segment_recovery(true, &constraints),
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(recovered.segment_recovery.topology.tetrahedra.len(), 2);
    assert_eq!(recovered.facets[0].constraint_index, 0);
    assert_eq!(recovered.facets[0].triangles.len(), 1);
    validate_delaunay_facet_recovery(
        &recovered,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn facet_edge_star_cavity_retriangulates_both_exact_sides() {
    let constraints = constraints(false);
    let segments = segment_recovery(true, &constraints);
    let mut work = FacetRecoveryWork::new(DelaunayFacetRecoveryOptions::default(), &NeverCancelled);
    let recovered = super::cavity::try_recover_facet_with_edge_star_cavity(
        &segments,
        constraints.facets[0]
            .vertex_indices
            .map(|index| constraints.nodes[index as usize].identity),
        &[],
        0,
        &mut work,
    )
    .unwrap()
    .expect("the bipyramid edge-star cavity should be recoverable");

    assert_eq!(recovered.tetrahedra.len(), 2);
    let mut candidate = segments;
    candidate.topology = recovered;
    validate_delaunay_segment_recovery(
        &candidate,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn facet_edge_star_cavity_enforces_its_tetrahedron_budget() {
    let constraints = constraints(false);
    let segments = segment_recovery(true, &constraints);
    let mut work = FacetRecoveryWork::new(
        DelaunayFacetRecoveryOptions {
            maximum_cavity_tetrahedra: 1,
            ..DelaunayFacetRecoveryOptions::default()
        },
        &NeverCancelled,
    );
    assert_eq!(
        super::cavity::try_recover_facet_with_edge_star_cavity(
            &segments,
            constraints.facets[0]
                .vertex_indices
                .map(|index| constraints.nodes[index as usize].identity),
            &[],
            0,
            &mut work,
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::ResourceLimit
    );
}

#[test]
fn facet_star_refill_exceeds_small_exact_cover_inventory() {
    const RING_NODES: usize = 24;
    let mut nodes = vec![DelaunayVolumeNode {
        identity: StableDigest::from_bytes([1; 32]),
        coordinates_m: [0.0, 0.0, 0.0],
    }];
    for index in 0..RING_NODES {
        let angle = std::f64::consts::TAU * index as f64 / RING_NODES as f64;
        nodes.push(DelaunayVolumeNode {
            identity: StableDigest::from_bytes([(index + 2) as u8; 32]),
            coordinates_m: [angle.cos(), angle.sin(), 0.0],
        });
    }
    nodes.push(DelaunayVolumeNode {
        identity: StableDigest::from_bytes([RING_NODES as u8 + 2; 32]),
        coordinates_m: [0.0, 0.0, 1.0],
    });
    let apex = RING_NODES as u32 + 1;
    let tetrahedra = (0..RING_NODES)
        .map(|index| {
            [
                0,
                index as u32 + 1,
                (index as u32 + 1) % RING_NODES as u32 + 1,
                apex,
            ]
        })
        .collect::<Vec<_>>();
    let topology = build_delaunay_volume_topology(
        nodes,
        tetrahedra,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let target_volume_m3 = topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            tetrahedron_volume(
                tetrahedron
                    .vertex_indices
                    .map(|node| topology.nodes[node as usize].coordinates_m),
            )
        })
        .sum();
    let cavity = crate::cavity::constrained::ConstrainedCavity {
        removed_tetrahedron_ids: (0..RING_NODES as u32).collect(),
        boundary_faces: topology
            .incidence
            .boundary_facets
            .iter()
            .map(
                |facet| crate::cavity::constrained::ConstrainedCavityBoundaryFace {
                    node_ids: facet.vertex_indices,
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id: None,
                    source_edge_ids: [None; 3],
                    region_ids: Vec::new(),
                },
            )
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3,
    };
    assert!(topology.nodes.len() > 20);
    assert!(cavity.boundary_faces.len() > 40);
    let mut work = FacetRecoveryWork::new(DelaunayFacetRecoveryOptions::default(), &NeverCancelled);

    let refill = super::cavity::star::star_refill(&cavity, &topology, 0, &mut work)
        .unwrap()
        .expect("the large convex pyramid has a boundary apex star");
    assert_eq!(refill.len(), RING_NODES);
}

#[test]
fn facet_recovery_is_a_noop_for_existing_support_and_rejects_tampering() {
    let constraints = constraints(false);
    let segments = segment_recovery(false, &constraints);
    let topology = segments.topology.clone();
    let mut recovered = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(recovered.segment_recovery.topology, topology);

    recovered.facets[0].triangles[0].node_identities.swap(0, 1);
    assert_eq!(
        validate_delaunay_facet_recovery(
            &recovered,
            &constraints,
            DelaunayFacetRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::InvalidConstraints
    );
}

#[test]
fn facet_validation_treats_recovered_faces_as_delaunay_barriers() {
    let mut constraints = constraints(false);
    let mut segments = segment_recovery(false, &constraints);
    constraints.nodes[4].coordinates_m = [0.0, 0.0, -1.0];
    segments.topology = build_delaunay_volume_topology(
        constraints.volume_nodes(),
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(
        validate_delaunay_segment_recovery(
            &segments,
            &constraints,
            DelaunaySegmentRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySegmentRecoveryErrorKind::InvalidTopology
    );
    let recovery = DelaunayFacetRecovery {
        segment_recovery: segments,
        facets: vec![DelaunayRecoveredFacet {
            constraint_index: 0,
            triangles: vec![DelaunayRecoveredFacetTriangle {
                node_identities: constraints.facets[0]
                    .vertex_indices
                    .map(|index| constraints.nodes[index as usize].identity),
            }],
        }],
    };

    validate_delaunay_facet_recovery(
        &recovery,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn facet_recovery_triangulates_subdivided_segment_chains_canonically() {
    let constraints = constraints(false);
    let (segments, midpoint_identity) = split_boundary_segment_recovery(&constraints);

    let recovered = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = recover_delaunay_facets(
        split_boundary_segment_recovery(&constraints).0,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(recovered, repeated);
    assert_eq!(recovered.facets[0].triangles.len(), 2);
    assert!(recovered.facets[0]
        .triangles
        .iter()
        .all(|triangle| triangle.node_identities.contains(&midpoint_identity)));
    validate_delaunay_facet_recovery(
        &recovered,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn facet_recovery_never_removes_a_recovered_segment() {
    let constraints = constraints(true);
    assert_eq!(
        recover_delaunay_facets(
            segment_recovery(true, &constraints),
            &constraints,
            DelaunayFacetRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::UnsatisfiableConstraint
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn facet_recovery_enforces_search_limits_and_cancellation() {
    let constraints = constraints(false);
    let segments = segment_recovery(true, &constraints);
    let invalid = DelaunayFacetRecoveryOptions {
        maximum_cavity_apex_attempts: 0,
        ..DelaunayFacetRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_facets(segments.clone(), &constraints, invalid, &NeverCancelled,)
            .unwrap_err()
            .kind,
        DelaunayFacetRecoveryErrorKind::InvalidOptions
    );
    let bounded = DelaunayFacetRecoveryOptions {
        maximum_search_steps: 1,
        ..DelaunayFacetRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_facets(segments.clone(), &constraints, bounded, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayFacetRecoveryErrorKind::ResourceLimit
    );
    let support_bounded = DelaunayFacetRecoveryOptions {
        maximum_support_steps: 1,
        ..DelaunayFacetRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_facets(
            split_boundary_segment_recovery(&constraints).0,
            &constraints,
            support_bounded,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::ResourceLimit
    );
    let cancelled = DelaunayFacetRecoveryOptions {
        segment_recovery: DelaunaySegmentRecoveryOptions {
            constraints: super::super::DelaunayConstraintOptions {
                cancellation_check_interval: 1,
                ..super::super::DelaunayConstraintOptions::default()
            },
            ..DelaunaySegmentRecoveryOptions::default()
        },
        ..DelaunayFacetRecoveryOptions::default()
    };
    assert_eq!(
        recover_delaunay_facets(segments, &constraints, cancelled, &Cancelled)
            .unwrap_err()
            .kind,
        DelaunayFacetRecoveryErrorKind::Cancelled
    );
}
