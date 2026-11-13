use super::*;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;
use std::collections::BTreeSet;

fn mesh(count: u32) -> TopologyMutation {
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for cluster in 0..count {
        let first = nodes.len() as u32;
        for p in [
            [-1.5, 0., 0.],
            [1.4, 0., 0.],
            [0., -0.5, 0.],
            [0., 2., 2.5],
            [0., 2., -2.5],
        ] {
            let mut id = [2; 32];
            id[..4].copy_from_slice(&(nodes.len() as u32 + 1).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(id),
                [p[0] + 100.0 * cluster as f64, p[1], p[2]],
            ));
        }
        cells.extend([
            [first, first + 2, first + 3, first + 4],
            [first + 1, first + 2, first + 4, first + 3],
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
fn bounds() -> Aabb3 {
    Aabb3 {
        min_m: [-2., -1., -3.],
        max_m: [2., 3., 3.],
    }
}
fn query(mesh: &TopologyMutation, bounds: Aabb3) -> Vec<u32> {
    mesh.spatial_tetrahedron_candidates(bounds, &mut || Ok::<_, ()>(()))
        .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A persistent cell-bounds trial preserves its original owner, resumes after rejected
//   query synchronization, rejects invalid cavity replacement and retains complete candidates after
//   canonical node reordering. This component transaction test does not qualify a full native task
//   or publication path.
#[test]
fn cell_bounds_follow_atomic_cavity_trials_and_canonical_node_reordering() {
    let mut original = mesh(1);
    assert_eq!(query(&original, bounds()), vec![0, 1]);
    let mut trial = original.clone();
    trial
        .replace_cavity(
            &BTreeSet::from([0, 1]),
            vec![
                ([0, 1, 2, 3], None),
                ([0, 1, 3, 4], None),
                ([0, 1, 4, 2], None),
            ],
        )
        .unwrap();
    let mut calls = 0;
    assert!(trial
        .spatial_tetrahedron_candidates(bounds(), &mut || {
            calls += 1;
            if calls == 3 {
                Err("cancel")
            } else {
                Ok(())
            }
        })
        .is_err());
    assert_eq!(query(&original, bounds()), vec![0, 1]);
    assert_eq!(query(&trial, bounds()), vec![0, 1, 2]);
    assert!(trial
        .replace_cavity(&BTreeSet::from([0]), vec![([0, 0, 2, 3], None)])
        .is_err());
    assert_eq!(query(&trial, bounds()), vec![0, 1, 2]);
    let mut id = [0; 32];
    id[31] = 1;
    trial
        .append_node(DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(id),
            [10., 10., 10.],
        ))
        .unwrap();
    let canonical = trial
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    original
        .adopt_canonical_with_spatial_index(canonical, &mut || Ok::<_, ()>(()))
        .unwrap();
    assert_eq!(query(&original, bounds()), vec![0, 1, 2]);
    assert!(original.cell_spatial_index.borrow().is_some());
    assert!(query(
        &original,
        Aabb3 {
            min_m: [9.; 3],
            max_m: [11.; 3]
        }
    )
    .is_empty());
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared physical cell bounds retain exhaustive candidates across four and sixty-four
//   disconnected cavities with bounded query checkpoint growth. This broad-phase fixture does not
//   establish connected-mesh asymptotics, exact crossing admission, native heat recovery or
//   whole-task memory.
#[test]
fn prepared_cell_query_prunes_distant_inventory_with_complete_candidates() {
    let mut counts = Vec::new();
    for count in [4, 64] {
        let mesh = mesh(count);
        assert_eq!(query(&mesh, bounds()), vec![0, 1]);
        let mut calls = 0;
        let found = mesh
            .spatial_tetrahedron_candidates(bounds(), &mut || {
                calls += 1;
                Ok::<_, ()>(())
            })
            .unwrap();
        let expected = mesh
            .tetrahedra
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| {
                cell.as_ref()
                    .filter(|_| mesh.cell_entry(index as u32).bounds.intersects(bounds()))
                    .map(|_| index as u32)
            })
            .collect::<Vec<_>>();
        assert_eq!(found, expected);
        counts.push(calls);
    }
    assert!(
        counts[1] <= counts[0] * 4,
        "prepared 4/64 cavity query work: {counts:?}"
    );
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Sixty-four checked cavity round trips retain only five net changes at the
//   intermediate step and zero after each exact reversal. No intermediate cell reaches the spatial
//   index, while final query inventory agrees with original connectivity. Requires separate
//   cancellation, canonical-handoff and geometric admission checks; excludes arbitrary
//   changed-inventory and whole-task memory bounds.
#[test]
fn cell_change_journal_coalesces_intermediate_churn_without_changing_query_inventory() {
    let mut topology = mesh(1);
    let original_candidates = query(&topology, bounds());
    let original_cells = topology
        .tetrahedra
        .iter()
        .flatten()
        .map(|cell| (cell.vertex_indices, cell.region_id.clone()))
        .collect::<Vec<_>>();
    let mut removed = original_candidates.iter().copied().collect::<BTreeSet<_>>();
    for _ in 0..64 {
        let created = topology
            .replace_cavity(
                &removed,
                vec![
                    ([0, 1, 2, 3], None),
                    ([0, 1, 3, 4], None),
                    ([0, 1, 4, 2], None),
                ],
            )
            .unwrap();
        assert_eq!(
            topology
                .cell_spatial_index
                .borrow()
                .as_ref()
                .unwrap()
                .pending
                .len(),
            5
        );
        removed = topology
            .replace_cavity(&created.into_iter().collect(), original_cells.clone())
            .unwrap()
            .into_iter()
            .collect();
        assert!(topology
            .cell_spatial_index
            .borrow()
            .as_ref()
            .unwrap()
            .pending
            .is_empty());
    }
    let mut work = 0;
    topology
        .prepare_cell_spatial_index(&mut || {
            work += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!(
        work, 0,
        "no intermediate cell was materialized in the retained index"
    );
    let current = query(&topology, bounds());
    assert_eq!(current, removed.into_iter().collect::<Vec<_>>());
    assert_eq!(current.len(), original_candidates.len());
}

#[test]
fn coalescence_preserves_exact_bound_bits_used_by_the_spatial_order() {
    let mut cache = CellSpatialCache {
        index: BoundsSpatialIndex::from_entries_with_checkpoint(Vec::<Entry>::new(), &mut || {
            Ok::<_, ()>(())
        })
        .unwrap()
        .unwrap(),
        pending: PendingChanges::default(),
    };
    let payload = [StableDigest::from_bytes([1; 32]); 4];
    let original = Entry {
        payload,
        bounds: Aabb3 {
            min_m: [-0.; 3],
            max_m: [-0.; 3],
        },
    };
    let changed = Entry {
        payload,
        bounds: Aabb3 {
            min_m: [0.; 3],
            max_m: [0.; 3],
        },
    };
    cache.record(original.clone(), false);
    cache.record(changed.clone(), true);
    assert_eq!(
        cache.pending.len(),
        1,
        "IEEE-equivalent bounds can have different ordered bit keys"
    );
    let (removed, created) = cache.pending.get(&payload).unwrap();
    assert_eq!(removed.unwrap().min_m[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(created.unwrap().min_m[0].to_bits(), 0.0_f64.to_bits());
    cache.record(changed, false);
    cache.record(original, true);
    assert!(cache.pending.is_empty());
}

mod checkpoint;
