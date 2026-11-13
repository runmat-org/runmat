//! Boundary-edge preservation and independent Delaunay-restoration witnesses.
use super::*;
use crate::cdt::{
    insertion::validate_constrained_delaunay_volume_topology, mutation::TopologyMutation,
};

fn neighborhood() -> DelaunayVolumeTopology {
    let coordinates = [
        [
            0.026286795735359192,
            0.19175226986408234,
            -0.05752567946910858,
        ],
        [
            0.026286795735359192,
            0.19175226986408234,
            0.05752567946910858,
        ],
        [
            0.06823260337114334,
            0.3451540879905224,
            -0.05752567946910858,
        ],
        [0.06823260337114334, 0.3451540879905224, 0.05752567946910858],
        [
            0.023290663957595825,
            0.3451540879905224,
            -0.05752567946910858,
        ],
        [
            0.023290663957595825,
            0.3451540879905224,
            0.05752567946910858,
        ],
        [0.0008196961134672165, 0.3451540879905224, 0.0],
        [
            0.04426356963813305,
            0.26845317892730236,
            -0.05752567946910858,
        ],
        [
            0.06523647531867027,
            0.19175226986408234,
            -0.05752567946910858,
        ],
    ];
    let nodes = coordinates
        .into_iter()
        .enumerate()
        .map(|(index, point)| DelaunayVolumeNode::explicit(identity(index), point))
        .collect();
    build_delaunay_volume_topology(
        nodes,
        vec![
            [1, 0, 3, 7],
            [0, 1, 6, 7],
            [4, 0, 6, 7],
            [3, 0, 2, 7],
            [1, 3, 5, 7],
            [1, 5, 6, 7],
            [3, 4, 5, 7],
            [4, 3, 2, 7],
            [5, 4, 6, 7],
            [3, 0, 8, 2],
            [0, 3, 8, 1],
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn identity(index: usize) -> StableDigest {
    StableDigest::from_bytes([(index + 1) as u8; 32])
}

#[test]
fn boundary_reconnection_preserves_protected_edges() {
    let topology = neighborhood();
    let options = DelaunaySegmentRecoveryOptions::default();
    validate_constrained_delaunay_volume_topology(
        &topology,
        &[],
        options.insertion,
        &NeverCancelled,
    )
    .unwrap_err();
    let protected = [[identity(4), identity(7)], [identity(2), identity(3)]];
    let mut raw = TopologyMutation::from_canonical(topology.clone());
    assert!(flip::try_recover_edge_with_face_flip_mutation(
        &mut raw,
        identity(7),
        identity(8),
        &protected,
        &[],
        0,
        &mut RecoveryWork::new(options, &NeverCancelled),
    )
    .unwrap());
    let raw = raw
        .freeze(options.insertion.topology, &NeverCancelled)
        .unwrap();
    // A raw reconnection proves connectivity and barrier preservation only.
    // Independent empty-sphere admission belongs to the checked transaction.
    let mut raw_work = RecoveryWork::new(options, &NeverCancelled);
    for [left, right] in protected.into_iter().chain([[identity(7), identity(8)]]) {
        assert!(edge_exists(&raw, left, right, 0, &mut raw_work).unwrap());
    }

    let mut work = RecoveryWork::new(options, &NeverCancelled);
    let mut recovered = TopologyMutation::from_canonical(topology.clone());
    assert!(flip::try_recover_edge_with_face_flip(
        &mut recovered,
        identity(7),
        identity(8),
        &protected,
        &[],
        0,
        &mut work,
    )
    .unwrap());
    let recovered = recovered
        .freeze(options.insertion.topology, &NeverCancelled)
        .unwrap();
    assert_eq!(recovered.nodes, topology.nodes);
    validate_constrained_delaunay_volume_topology(
        &recovered,
        &[],
        options.insertion,
        &NeverCancelled,
    )
    .unwrap();
    for [left, right] in protected.into_iter().chain([[identity(7), identity(8)]]) {
        assert!(edge_exists(&recovered, left, right, 0, &mut work).unwrap());
    }
}

#[test]
fn reconnection_propagates_predicate_budget_exhaustion() {
    let mut options = DelaunaySegmentRecoveryOptions::default();
    options.insertion.maximum_predicate_evaluations = 1;
    let failure = flip::try_recover_edge_with_face_flip(
        &mut TopologyMutation::from_canonical(neighborhood()),
        identity(7),
        identity(8),
        &[],
        &[],
        0,
        &mut RecoveryWork::new(options, &NeverCancelled),
    )
    .unwrap_err();
    assert_eq!(
        failure.kind,
        DelaunaySegmentRecoveryErrorKind::ResourceLimit
    );
}

#[test]
fn reconnection_propagates_cancellation() {
    struct Cancelled;
    impl runmat_meshing_core::MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let mut options = DelaunaySegmentRecoveryOptions::default();
    options.insertion.topology.cancellation_check_interval = 1;
    let failure = candidate::admit(
        neighborhood(),
        [identity(7), identity(8)],
        &[],
        &[],
        0,
        &mut RecoveryWork::new(options, &Cancelled),
    )
    .unwrap_err();
    assert_eq!(failure.kind, DelaunaySegmentRecoveryErrorKind::Cancelled);
}

#[test]
fn candidate_admission_restores_empty_spheres_and_preserves_edges() {
    let topology = build_delaunay_volume_topology(
        vec![
            DelaunayVolumeNode::explicit(identity(0), [0.0, 0.0, 0.0]),
            DelaunayVolumeNode::explicit(identity(1), [2.0, 0.0, 0.0]),
            DelaunayVolumeNode::explicit(identity(2), [0.0, 2.0, 0.0]),
            DelaunayVolumeNode::explicit(identity(3), [0.5, 0.5, 0.2]),
            DelaunayVolumeNode::explicit(identity(4), [0.5, 0.5, -0.2]),
        ],
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunaySegmentRecoveryOptions::default();
    assert_eq!(
        validate_constrained_delaunay_volume_topology(
            &topology,
            &[],
            options.insertion,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::InvalidTopology
    );
    let protected = [[identity(0), identity(1)], [identity(2), identity(4)]];
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    let repaired = candidate::admit(
        topology.clone(),
        [identity(0), identity(3)],
        &protected,
        &[],
        0,
        &mut work,
    )
    .unwrap()
    .expect("bounded legalization must restore this convex cavity");
    assert_eq!(repaired.nodes, topology.nodes);
    validate_constrained_delaunay_volume_topology(
        &repaired,
        &[],
        options.insertion,
        &NeverCancelled,
    )
    .unwrap();
    for [left, right] in protected.into_iter().chain([[identity(0), identity(3)]]) {
        assert!(edge_exists(&repaired, left, right, 0, &mut work).unwrap());
    }
}

fn illegal_edge_star() -> DelaunayVolumeTopology {
    build_delaunay_volume_topology(
        [
            [0., 0., -2.],
            [0., 0., 2.],
            [1., 0., 0.],
            [-0.5, 1., 0.],
            [-0.5, -1., 0.],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, point)| DelaunayVolumeNode::explicit(identity(index), point))
        .collect(),
        vec![[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn candidate_admission_repairs_an_illegal_edge_star_without_losing_required_edges() {
    let original = illegal_edge_star();
    let options = DelaunaySegmentRecoveryOptions::default();
    validate_constrained_delaunay_volume_topology(
        &original,
        &[],
        options.insertion,
        &NeverCancelled,
    )
    .unwrap_err();
    crate::cdt::insertion::validate_delaunay_volume_topology(
        &original,
        options.insertion,
        &NeverCancelled,
    )
    .unwrap_err();
    let protected = [[identity(0), identity(3)], [identity(1), identity(4)]];
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    let repaired = candidate::admit(
        original.clone(),
        [identity(0), identity(2)],
        &protected,
        &[],
        0,
        &mut work,
    )
    .unwrap()
    .expect("three-to-two repair preserves these boundary edges");
    assert_eq!(repaired.nodes, original.nodes);
    assert_eq!(repaired.tetrahedra.len(), 2);
    crate::cdt::insertion::validate_delaunay_volume_topology(
        &repaired,
        options.insertion,
        &NeverCancelled,
    )
    .unwrap();
    for [left, right] in protected.into_iter().chain([[identity(0), identity(2)]]) {
        assert!(edge_exists(&repaired, left, right, 0, &mut work).unwrap());
    }
}

#[test]
fn candidate_admission_rejects_repair_that_removes_a_protected_edge() {
    let original = illegal_edge_star();
    let options = DelaunaySegmentRecoveryOptions::default();
    let result = candidate::admit(
        original,
        [identity(0), identity(2)],
        &[[identity(0), identity(1)]],
        &[],
        0,
        &mut RecoveryWork::new(options, &NeverCancelled),
    )
    .unwrap();
    assert!(result.is_none());
}
