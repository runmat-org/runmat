use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, mutation::TopologyMutation,
    segment_recovery::tests::octahedron_constraints, topology::connectivity_search_fingerprint,
    DelaunayTopologyOptions, DelaunayVolumeTopology,
};
use runmat_meshing_core::NeverCancelled;

fn octahedron(tetrahedra: Vec<[u32; 4]>) -> DelaunayVolumeTopology {
    build_delaunay_volume_topology(
        octahedron_constraints().volume_nodes(),
        tetrahedra,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn fingerprint(topology: &TopologyMutation) -> [u8; 32] {
    topology_state(topology, &mut || Ok::<_, ()>(())).unwrap()
}

#[test]
fn inactive_slots_and_corner_order_do_not_create_new_cycle_states() {
    let canonical = octahedron(vec![[0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1]]);
    let expected = connectivity_search_fingerprint(&canonical);
    let mut mutation = TopologyMutation::from_canonical(canonical);
    assert_eq!(fingerprint(&mutation), expected);
    mutation.tetrahedra.push(None);
    for tetrahedron in mutation.tetrahedra.iter_mut().flatten() {
        // An even permutation preserves orientation and connectivity. The cycle
        // hint deliberately ignores local indexing, not geometric validity.
        tetrahedron.vertex_indices.rotate_left(2);
        tetrahedron.neighbors.rotate_left(2);
    }
    assert_eq!(fingerprint(&mutation), expected);
}

#[test]
fn distinct_valid_octahedron_triangulations_are_not_pruned_as_one_state() {
    let first = octahedron(vec![[0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1]]);
    let second = octahedron(vec![[1, 4, 0, 2], [1, 4, 2, 5], [1, 4, 5, 3], [1, 4, 3, 0]]);
    assert_ne!(
        fingerprint(&TopologyMutation::from_canonical(first)),
        fingerprint(&TopologyMutation::from_canonical(second)),
    );
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: One valid octahedron cavity replacement compares the maintained ordered-key
//   fingerprint with independent canonical reconstruction; a rejected duplicate-corner replacement
//   preserves it. No general spatial-candidate, growth, cancellation or whole-mesh admission claim.
#[test]
fn cavity_commit_updates_streamed_keys_and_failed_replacement_preserves_them() {
    let original = octahedron(vec![[0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1]]);
    let replacement = octahedron(vec![[1, 4, 0, 2], [1, 4, 2, 5], [1, 4, 5, 3], [1, 4, 3, 0]]);
    let expected = connectivity_search_fingerprint(&replacement);
    let created = replacement
        .tetrahedra
        .iter()
        .map(|tetrahedron| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect();
    let mut mutation = TopologyMutation::from_canonical(original);
    let original_hint = fingerprint(&mutation);
    let retained_trial = mutation.clone();
    let removed = (0..4).collect();
    mutation.replace_cavity(&removed, created).unwrap();
    assert_eq!(fingerprint(&mutation), expected);
    assert_eq!(fingerprint(&retained_trial), original_hint);
    let mut calls = 0;
    assert_eq!(
        topology_state(&mutation, &mut || {
            calls += 1;
            Ok::<_, ()>(())
        })
        .unwrap(),
        expected
    );
    assert_eq!(
        calls, 1,
        "prepared cycle hints must not rescan the inventory"
    );

    // A duplicate-corner replacement cannot commit. The independently derived
    // hint still describes the complete accepted connectivity after rejection.
    assert!(mutation
        .replace_cavity(&removed, vec![([0, 0, 1, 2], None)])
        .is_err());
    assert_eq!(fingerprint(&mutation), expected);
}

#[test]
fn rejected_fingerprint_work_returns_no_partial_hint_and_preserves_the_index() {
    let canonical = octahedron(vec![[0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1]]);
    let expected = connectivity_search_fingerprint(&canonical);
    let mutation = TopologyMutation::from_canonical(canonical);
    let mut checkpoints = 0;
    let rejected = topology_state(&mutation, &mut || {
        checkpoints += 1;
        if checkpoints == 3 {
            Err("rejected work")
        } else {
            Ok(())
        }
    });
    assert_eq!(rejected, Err("rejected work"));
    assert_eq!(checkpoints, 3);
    let mut resumed = 0;
    let complete = topology_state(&mutation, &mut || {
        resumed += 1;
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(complete, expected);
    assert_eq!(resumed, 4);
}
