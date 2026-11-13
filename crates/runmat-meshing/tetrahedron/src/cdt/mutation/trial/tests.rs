use super::*;
mod high_degree;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;
mod changes;
fn neighborhoods(count: usize) -> TopologyMutation {
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for cluster in 0..count {
        let first = nodes.len() as u32;
        for point in [
            [0., 0., 0.],
            [2., 0., 0.],
            [0., 2., 0.],
            [0.5, 0.5, 1.],
            [0.5, 0.5, -1.],
        ] {
            let mut identity = [1; 32];
            identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [point[0] + cluster as f64 * 10., point[1], point[2]],
            ));
        }
        cells.extend([
            [first, first + 1, first + 2, first + 3],
            [first, first + 2, first + 1, first + 4],
        ]);
    }
    TopologyMutation::from_canonical(
        build_delaunay_volume_topology(
            nodes,
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap(),
    )
}
fn flip(topology: &mut TopologyMutation) {
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
}
fn assert_storage(left: &TopologyMutation, right: &TopologyMutation) {
    assert_eq!(left.nodes, right.nodes);
    assert_eq!(left.node_indices, right.node_indices);
    assert_eq!(left.positions, right.positions);
    assert_eq!(left.vertex_stars, right.vertex_stars);
    assert_eq!(left.face_uses, right.face_uses);
    assert_eq!(left.boundary_faces, right.boundary_faces);
    assert_eq!(left.free_tetrahedra, right.free_tetrahedra);
    assert_eq!(left.tetrahedron_indices, right.tetrahedron_indices);
    assert_eq!(left.physical_dirty_faces, right.physical_dirty_faces);
    assert_eq!(
        left.connectivity_state
            .borrow()
            .map(|state| state.fingerprint()),
        right
            .connectivity_state
            .borrow()
            .map(|state| state.fingerprint())
    );
    assert_eq!(left.coordinate_scale_m, right.coordinate_scale_m);
    assert_eq!(
        left.snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap(),
        right
            .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap()
    );
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Committed inner changes remain reversible by their parent; rejected inner changes
//   permit a complete outer retry and commit. Flushed spatial roots, connectivity hints, dirty-face
//   inventory, cells and maintained incidence agree with original storage and independent canonical
//   reconstruction. This rollback mechanism does not independently confer physical admission or
//   complete-study qualification.
#[test]
fn nested_cavity_trials_restore_incidence_caches_and_physical_authority() {
    let mut topology = neighborhoods(4);
    topology.physical_dirty_faces = Some(BTreeSet::from([[0, 1, 2]]));
    topology
        .prepare_cell_spatial_index(&mut || Ok::<_, ()>(()))
        .unwrap();
    topology
        .connectivity_search_fingerprint(&mut || Ok::<_, ()>(()))
        .unwrap();
    let original = topology.clone();
    {
        let mut outer = topology.cavity_trial();
        {
            let mut inner = outer.cavity_trial();
            flip(&mut inner);
            inner.commit();
        }
        outer.discharge_physical_dirty_faces();
        // Queries flush changed spatial roots; rollback must restore the old root,
        // not merely truncate a now-consumed update journal.
        outer
            .prepare_cell_spatial_index(&mut || Ok::<_, ()>(()))
            .unwrap();
    }
    assert_storage(&topology, &original);
    assert_eq!(
        topology
            .spatial_tetrahedron_candidates(
                runmat_meshing_core::Aabb3 {
                    min_m: [-1.; 3],
                    max_m: [3.; 3]
                },
                &mut || Ok::<_, ()>(())
            )
            .unwrap(),
        vec![0, 1]
    );
    let mut expected = original.clone();
    flip(&mut expected);
    {
        let mut outer = topology.cavity_trial();
        {
            let mut rejected = outer.cavity_trial();
            flip(&mut rejected);
        }
        flip(&mut outer);
        outer.commit();
    }
    assert_storage(&topology, &expected);
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The same checked two-to-three cavity has identical cell, corner-key, incident-star,
//   face, boundary and free-slot undo inventories across four and sixty-four disconnected
//   neighborhoods. Rejection restores original storage and independently reconstructed canonical
//   topology. This component count excludes arbitrary vertex valence, cached pending-delta length,
//   recursion depth, whole-task memory and complete native study performance.
#[test]
fn cavity_trial_undo_inventory_is_local_across_disconnected_growth() {
    let mut costs = Vec::new();
    for size in [4, 64] {
        let mut topology = neighborhoods(size);
        let original = topology.clone();
        {
            let mut trial = topology.cavity_trial();
            flip(&mut trial);
            let j = trial.trial_journals.last().unwrap();
            costs.push((
                j.cells.len(),
                j.cell_keys.len(),
                j.stars.len(),
                j.faces.len(),
                j.boundary.len(),
                j.free.len(),
            ));
        }
        assert_storage(&topology, &original);
    }
    assert_eq!(costs[0], costs[1]);
    assert_eq!(costs[0].0, 3);
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Unwinding after a checked cavity and point append restores the original node
//   inventory, identity/position indexes, coordinate scale, cells, incidence and physical-state
//   metadata; a subsequent checked edit succeeds. This component witness does not claim arbitrary
//   whole-owner adoption inside a trial or native process cancellation.
#[test]
fn cavity_trial_rolls_back_appended_points_and_unwinding() {
    let mut topology = neighborhoods(4);
    let original = topology.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut trial = topology.cavity_trial();
        flip(&mut trial);
        trial
            .append_node(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes([253; 32]),
                [1000., 0., 0.],
            ))
            .unwrap();
        panic!("rejected trial");
    }));
    assert!(result.is_err());
    assert_storage(&topology, &original);
    assert!(topology.trial_journals.is_empty());
    flip(&mut topology); // Retry after unwinding remains a complete checked edit.
}

#[test]
fn maintained_edge_queries_match_full_cells_across_commit_and_rollback() {
    use crate::cdt::insertion::topology_access::TetrahedralTopology;
    let check = |topology: &TopologyMutation| {
        for first in 0..topology.nodes.len() as u32 {
            for second in first + 1..topology.nodes.len() as u32 {
                let expected = topology
                    .tetrahedra
                    .iter()
                    .enumerate()
                    .filter_map(|(index, cell)| {
                        cell.as_ref()
                            .filter(|cell| {
                                cell.vertex_indices.contains(&first)
                                    && cell.vertex_indices.contains(&second)
                            })
                            .map(|_| index as u32)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(topology.has_edge([first, second]), !expected.is_empty());
                assert_eq!(topology.edge_tetrahedra([first, second]), expected);
            }
        }
    };
    let mut topology = neighborhoods(3);
    check(&topology);
    {
        let mut trial = topology.cavity_trial();
        flip(&mut trial);
        check(&trial);
    }
    check(&topology);
    {
        let mut trial = topology.cavity_trial();
        flip(&mut trial);
        trial.commit();
    }
    check(&topology);
}

mod retention;
