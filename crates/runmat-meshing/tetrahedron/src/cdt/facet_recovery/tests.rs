use runmat_meshing_core::{
    contracts::{MeshingStage, TopologyEntityId},
    MeshingCancellationSignal, NeverCancelled, StableDigest,
};

use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, recover_delaunay_segments, DelaunayConstraintFacet,
    DelaunayConstraintNode, DelaunayConstraintSegment, DelaunayTopologyOptions, DelaunayVolumeNode,
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
                source_node_id: id(
                    MeshingStage::ProtectedBoundaryComplex,
                    &format!("node:{index}"),
                ),
                coordinates_m,
            })
            .collect(),
        segments: segments
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                protected_edge_id: None,
                source_edge_id: None,
            })
            .collect(),
        facets: vec![DelaunayConstraintFacet {
            facet_id: id(MeshingStage::ProtectedBoundaryComplex, "facet:base"),
            vertex_indices: [0, 1, 2],
            source_face_id: id(MeshingStage::SurfaceMesh, "face:base"),
            material_interface_ids: vec!["material:interface".to_owned()],
        }],
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

fn id(stage: MeshingStage, value: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: value.to_owned(),
    }
}
