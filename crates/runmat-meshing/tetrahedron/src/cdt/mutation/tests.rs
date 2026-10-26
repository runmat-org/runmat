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

#[test]
fn overlapping_created_neighbors_are_rejected_before_commit() {
    let mut topology = triangular_bipyramid();
    let apex = topology.append_node(node(60, [0.25, 0.25, 0.5])).unwrap();
    let before = topology.clone();
    let failure = topology
        .replace_cavity(
            &BTreeSet::from([0, 1]),
            vec![([0, 1, 2, 3], None), ([0, 1, 2, apex], None)],
        )
        .unwrap_err();
    assert!(failure.reason.contains("overlaps across a shared face"));
    assert_eq!(
        topology
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        before
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
    );
    assert_eq!(topology.vertex_stars, before.vertex_stars);
    assert_eq!(topology.free_tetrahedra, before.free_tetrahedra);
}

#[test]
fn overlapping_retained_neighbor_is_rejected_before_commit() {
    let mut topology = triangular_bipyramid();
    let apex = topology.append_node(node(60, [0.25, 0.25, -0.5])).unwrap();
    let before = topology.clone();
    let failure = topology
        .replace_cavity(&BTreeSet::from([0]), vec![([0, 1, 2, apex], None)])
        .unwrap_err();
    assert!(failure.reason.contains("overlaps across a shared face"));
    assert_eq!(
        topology
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        before
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
    );
    assert_eq!(topology.vertex_stars, before.vertex_stars);
    assert_eq!(topology.free_tetrahedra, before.free_tetrahedra);
}

mod boundary;
mod spatial;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A local cavity retains lowest-slot reuse and independently rebuilds valid topology
//   with ten thousand inactive slots. This storage-ordering component check is not a
//   heap-allocation measurement, scaling series or whole-task resource qualification.
#[test]
fn local_replacement_reuses_lowest_slots_with_large_inactive_inventory() {
    let mut topology = triangular_bipyramid();
    topology.tetrahedra.resize(10002, None);
    topology.free_tetrahedra.extend(2..10002);
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
    assert_eq!(topology.tetrahedra.len(), 10002);
    assert_eq!(topology.free_tetrahedra.first(), Some(&3));
    let frozen = topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(frozen.tetrahedra.len(), 3);
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The consumed topology owner releases its shared spatial search index before the first
//   canonical materialization checkpoint and produces the same topology as retained independent
//   admission. This verifies ownership lifetime at the boundary, not allocator RSS, all resource
//   estimates or a complete study's scaling envelope.
#[test]
fn consuming_freeze_releases_search_indexes_before_independent_admission() {
    use runmat_meshing_core::{MeshingCancellationSignal, PointSpatialIndex};
    use std::sync::Arc;
    let owner = triangular_bipyramid();
    owner
        .prepare_point_spatial_index(&mut || Ok::<_, ()>(()))
        .unwrap();
    let index = owner.spatial_index.borrow().as_ref().unwrap().clone();
    struct ObserveRelease(Arc<PointSpatialIndex<StableDigest>>);
    impl MeshingCancellationSignal for ObserveRelease {
        fn is_cancelled(&self) -> bool {
            assert_eq!(
                Arc::strong_count(&self.0),
                1,
                "obsolete owner index retained during freeze"
            );
            false
        }
    }
    let reference = owner
        .validated_snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    let actual = owner
        .freeze(DelaunayTopologyOptions::default(), &ObserveRelease(index))
        .unwrap();
    assert_eq!(actual, reference);
}

#[test]
fn borrowed_snapshot_preserves_regions_disconnected_cells_and_rejected_transaction() {
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    let region = |name: &str| {
        Some(PersistentEntityId::new(
            PersistentEntityKind::Region,
            name.into(),
            vec!["assembly".into()],
        ))
    };
    let nodes = vec![
        node(10, [0.0, 0.0, 0.0]),
        node(20, [2.0, 0.0, 0.0]),
        node(30, [0.0, 2.0, 0.0]),
        node(40, [0.5, 0.5, 1.0]),
        node(50, [0.5, 0.5, -1.0]),
        node(60, [10.0, 0.0, 0.0]),
        node(70, [12.0, 0.0, 0.0]),
        node(80, [10.0, 2.0, 0.0]),
        node(90, [10.0, 0.0, 2.0]),
    ];
    let original = crate::cdt::topology::build_delaunay_volume_topology_with_regions(
        nodes,
        vec![
            ([0, 1, 2, 3], region("upper")),
            ([0, 2, 1, 4], region("lower")),
            ([5, 6, 7, 8], region("disconnected")),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut owner = TopologyMutation::from_canonical(original.clone());
    let snapshot = owner
        .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(snapshot, original);
    // An invalid local transaction cannot alter the retained rollback owner.
    assert!(owner
        .replace_cavity(&BTreeSet::from([0]), vec![([0, 1, 3, 4], region("upper"))])
        .is_err());
    assert_eq!(
        owner
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        snapshot
    );
    assert_eq!(
        owner
            .validated_snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        snapshot
    );
    assert_eq!(
        owner
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        snapshot
    );
}
