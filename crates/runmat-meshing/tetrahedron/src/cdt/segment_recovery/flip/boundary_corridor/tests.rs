use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, DelaunaySegmentRecoveryOptions, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};
use runmat_meshing_core::NeverCancelled;
mod candidates;

fn fan() -> DelaunayVolumeTopology {
    let points = [
        [0., 0., 0.],
        [1., -1., 0.],
        [2., 0., 0.],
        [2., 1., 0.],
        [1., 2., 0.],
        [0., 1., 0.],
        [1., 0.5, 1.],
    ];
    build_delaunay_volume_topology(
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
        vec![[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 4, 6], [0, 4, 5, 6]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn corridor_recovers_nonadjacent_boundary_vertices_without_steiner_points() {
    let original = fan();
    let mut mutation = TopologyMutation::from_canonical(original.clone());
    let mut work = RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
    assert!(recover(&mut mutation, 1, 5, &BTreeSet::new(), &[], 0, &mut work).unwrap());
    let actual = mutation
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(actual.nodes, original.nodes);
    assert_eq!(actual.tetrahedra.len(), original.tetrahedra.len());
    assert!(edge_exists(
        &actual,
        original.nodes[1].identity,
        original.nodes[5].identity,
        0,
        &mut work
    )
    .unwrap());
    let side_faces = |topology: &DelaunayVolumeTopology| {
        topology
            .incidence
            .boundary_facets
            .iter()
            .filter(|facet| facet.vertex_indices.contains(&6))
            .map(|facet| facet.vertex_indices)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(side_faces(&actual), side_faces(&original));
}

#[test]
fn corridor_rejection_preserves_the_input_and_all_barriers() {
    let original = fan();
    let mut mutation = TopologyMutation::from_canonical(original.clone());
    let mut work = RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
    assert!(!recover(
        &mut mutation,
        1,
        5,
        &BTreeSet::from([[0, 2], [0, 3], [0, 4]]),
        &[],
        0,
        &mut work
    )
    .unwrap());
    assert_eq!(
        mutation
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        original
    );
}

#[test]
fn corridor_resource_rejection_is_typed_and_atomic() {
    let original = fan();
    let mut mutation = TopologyMutation::from_canonical(original.clone());
    let options = DelaunaySegmentRecoveryOptions {
        maximum_search_steps: 1,
        ..Default::default()
    };
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    let failure = recover(&mut mutation, 1, 5, &BTreeSet::new(), &[], 0, &mut work).unwrap_err();
    assert_eq!(
        failure.kind,
        DelaunaySegmentRecoveryErrorKind::ResourceLimit
    );
    assert_eq!(
        mutation
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        original
    );
}

#[test]
fn corridor_preserves_endpoint_indices_when_new_nodes_sort_before_original_nodes() {
    let original = fan();
    let mut mutation = TopologyMutation::from_canonical(original.clone());
    let mut identity = [0; 32];
    identity[31] = 1;
    mutation
        .append_node(DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(identity),
            [10.0, 10.0, 10.0],
        ))
        .unwrap();
    let physical_nodes = mutation.nodes().to_vec();
    let mut work = RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
    assert!(recover(&mut mutation, 1, 5, &BTreeSet::new(), &[], 0, &mut work).unwrap());
    assert_eq!(mutation.nodes(), physical_nodes);
    assert!(edge_exists(
        &mutation,
        original.nodes[1].identity,
        original.nodes[5].identity,
        0,
        &mut work
    )
    .unwrap());
    mutation
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An interior endpoint rejects a hull-only transaction using its four-cell local star
//   under a five-step search budget, preserving original topology. Necessary-condition pruning
//   only; not full recovery, asymptotic scaling or whole-task memory evidence.
#[test]
fn interior_endpoint_rejects_hull_corridor_before_transaction_search() {
    let original = build_delaunay_volume_topology(
        [
            [0., 0., 0.],
            [3., 0., 0.],
            [0., 3., 0.],
            [0., 0., 3.],
            [0.5, 0.5, 0.5],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([(index + 1) as u8; 32]), point)
        })
        .collect(),
        vec![[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut mutation = TopologyMutation::from_canonical(original.clone());
    let mut work = RecoveryWork::new(
        DelaunaySegmentRecoveryOptions {
            maximum_search_steps: 5,
            ..Default::default()
        },
        &NeverCancelled,
    );
    assert!(!recover(&mut mutation, 4, 0, &BTreeSet::new(), &[], 0, &mut work).unwrap());
    assert_eq!(
        mutation
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        original
    );
}
