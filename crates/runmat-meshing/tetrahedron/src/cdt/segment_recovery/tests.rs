use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::contracts::{MeshingStage, TopologyEntityId};
use runmat_meshing_core::NeverCancelled;

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_point_set, DelaunayConstraintNode,
    DelaunayConstraintSegment, DelaunayPointSetOptions, DelaunayTopologyOptions,
};

fn octahedron_constraints() -> DelaunayConstraints {
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
                source_node_id: TopologyEntityId {
                    stage: MeshingStage::ProtectedBoundaryComplex,
                    id: format!("node:{index}"),
                },
                coordinates_m,
            })
            .collect(),
        segments: vec![DelaunayConstraintSegment {
            vertex_indices: [1, 4],
            protected_edge_id: None,
            source_edge_id: None,
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

#[test]
fn recovery_constructs_a_missing_interior_segment_with_dyadic_steiner_nodes() {
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
    assert_eq!(recovered.steiner_node_identities.len(), 1);
    assert_eq!(recovered.segments[0].nodes.len(), 3);
    assert_eq!(recovered.segments[0].nodes[1].parameter_numerator, 1);
    assert_eq!(recovered.segments[0].nodes[1].parameter_exponent, 1);
    validate_delaunay_segment_recovery(
        &recovered,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
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

    let mut recovered = recover_delaunay_segments(
        octahedron_topology(&constraints),
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    recovered.steiner_node_identities.clear();
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
fn recovery_rejects_mismatched_constraints_and_assigned_regions() {
    let constraints = octahedron_constraints();
    let topology = octahedron_topology(&constraints);
    let mut mismatched = constraints.clone();
    mismatched.nodes[0].coordinates_m[0] += 0.25;
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

    let region = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: "region:assigned".to_owned(),
        assembly_path: Vec::new(),
    };
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
        maximum_recovery_steps: 1,
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
