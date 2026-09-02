use std::collections::BTreeSet;

use runmat_meshing_core::{NeverCancelled, StableDigest};

use super::TopologyMutation;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
}

fn triangular_bipyramid() -> TopologyMutation {
    let topology = build_delaunay_volume_topology(
        vec![
            node(10, [0.0, 0.0, 0.0]),
            node(20, [2.0, 0.0, 0.0]),
            node(30, [0.0, 2.0, 0.0]),
            node(40, [0.5, 0.5, 1.0]),
            node(50, [0.5, 0.5, -1.0]),
        ],
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    TopologyMutation::from_canonical(topology)
}

#[test]
fn cavity_replacement_commits_a_two_to_three_flip_locally() {
    let mut topology = triangular_bipyramid();

    let created = topology
        .replace_cavity(
            &BTreeSet::from([0, 1]),
            vec![
                ([0, 1, 3, 4], None),
                ([1, 2, 3, 4], None),
                ([2, 0, 3, 4], None),
            ],
        )
        .unwrap();

    assert_eq!(created, vec![0, 1, 2]);
    assert_eq!(topology.tetrahedra.iter().flatten().count(), 3);
    assert_eq!(topology.vertex_stars[3], BTreeSet::from([0, 1, 2]));
    assert_eq!(topology.vertex_stars[4], BTreeSet::from([0, 1, 2]));
    topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
}

#[test]
fn rejected_cavity_replacement_is_atomic() {
    let mut topology = triangular_bipyramid();
    let before = topology
        .clone()
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();

    let failure = topology
        .replace_cavity(&BTreeSet::from([0]), vec![([0, 1, 3, 4], None)])
        .unwrap_err();

    assert!(failure.reason.contains("retained interface"));
    let after = topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(after, before);
}

#[test]
fn working_snapshot_matches_independently_checked_freeze() {
    let mut topology = triangular_bipyramid();
    topology
        .replace_cavity(
            &BTreeSet::from([0, 1]),
            vec![
                ([0, 1, 3, 4], None),
                ([1, 2, 3, 4], None),
                ([2, 0, 3, 4], None),
            ],
        )
        .unwrap();

    let snapshot = topology
        .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    let frozen = topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(snapshot, frozen);
}

#[test]
fn working_snapshot_canonicalizes_after_identity_order_changes() {
    let topology = build_delaunay_volume_topology(
        vec![
            node(10, [0.0, 0.0, 0.0]),
            node(20, [2.0, 0.0, 0.0]),
            node(30, [0.0, 2.0, 0.0]),
            node(40, [0.0, 0.0, 2.0]),
        ],
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut topology = TopologyMutation::from_canonical(topology);
    let interior = topology.append_node(node(5, [0.25, 0.25, 0.25])).unwrap();
    topology
        .replace_cavity(
            &BTreeSet::from([0]),
            vec![
                ([0, 1, 2, interior], None),
                ([0, 1, 3, interior], None),
                ([0, 2, 3, interior], None),
                ([1, 2, 3, interior], None),
            ],
        )
        .unwrap();

    let snapshot = topology
        .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    let frozen = topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(snapshot, frozen);
}
