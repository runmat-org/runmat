use super::*;
use crate::cdt::DelaunayInsertionOptions;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use std::sync::atomic::{AtomicUsize, Ordering};

fn neighborhoods(count: usize) -> DelaunayVolumeTopology {
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for copy in 0..count {
        let offset = nodes.len() as u32;
        for point in [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0.2, 0.2, 1.],
            [0.2, 0.2, -1.],
        ] {
            let mut identity = [0; 32];
            identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [point[0] + copy as f64 * 10., point[1], point[2]],
            ));
        }
        cells.extend([
            [offset, offset + 1, offset + 2, offset + 3],
            [offset, offset + 2, offset + 1, offset + 4],
        ]);
    }
    build_delaunay_volume_topology(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}
struct Counting {
    calls: AtomicUsize,
    allowance: usize,
}
impl MeshingCancellationSignal for Counting {
    fn is_cancelled(&self) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed) >= self.allowance
    }
}
fn options() -> DelaunayInsertionOptions {
    let mut options = DelaunayInsertionOptions::default();
    options.topology.cancellation_check_interval = 1;
    options
}

fn exhaustive(topology: &DelaunayVolumeTopology, removed: &[usize], created: &[[u32; 4]]) -> bool {
    let keys = created
        .iter()
        .map(|cell| {
            let mut key = *cell;
            key.sort_unstable();
            key
        })
        .collect::<BTreeSet<_>>();
    keys.len() != created.len()
        || topology.tetrahedra.iter().enumerate().any(|(index, cell)| {
            let mut key = cell.vertex_indices;
            key.sort_unstable();
            !removed.contains(&index) && keys.contains(&key)
        })
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Replacement membership agrees with an independent exhaustive retained-cell scan
//   across removed-cell and reordered or duplicate-created-cell cases. Four and sixty-four
//   disconnected neighborhoods require exactly four checkpoints for the same two-cell query. This
//   establishes local incidence reuse for the tested membership component, not arbitrary
//   vertex-valence bounds, full trial ownership, complete recovery or native whole-task scaling.
#[test]
fn replacement_membership_matches_exhaustive_with_local_incidence_work() {
    let mut counts = Vec::new();
    for count in [4, 64] {
        let topology = neighborhoods(count);
        for removed in [vec![], vec![0], vec![0, 1]] {
            for created in [
                vec![],
                vec![[0, 1, 2, 3]],
                vec![[3, 2, 1, 0]],
                vec![[0, 1, 3, 4]],
                vec![[0, 1, 2, 3], [3, 2, 1, 0]],
                vec![[0, 1, 2, 3], [0, 2, 1, 4]],
            ] {
                let actual = replacement_duplicates_retained_tetrahedron(
                    &topology,
                    &removed,
                    &created,
                    &mut InsertionWork::new(options(), &NeverCancelled),
                )
                .unwrap();
                assert_eq!(actual, exhaustive(&topology, &removed, &created));
            }
        }
        let signal = Counting {
            calls: AtomicUsize::new(0),
            allowance: usize::MAX,
        };
        assert!(!replacement_duplicates_retained_tetrahedron(
            &topology,
            &[0, 1],
            &[[0, 1, 2, 3], [0, 2, 1, 4]],
            &mut InsertionWork::new(options(), &signal)
        )
        .unwrap());
        counts.push(signal.calls.load(Ordering::Relaxed));
    }
    assert_eq!(counts, vec![4, 4]);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Every checkpoint of the tested indexed membership query rejects without changing
//   retained cells; retry returns the original complete result. This read-only component evidence
//   does not establish worker cancellation or complete task admission.
#[test]
fn replacement_membership_cancellation_preserves_retained_topology() {
    let topology = neighborhoods(4);
    let before = topology
        .tetrahedra
        .iter()
        .map(|cell| cell.vertex_indices)
        .collect::<Vec<_>>();
    for allowance in 0..4 {
        let signal = Counting {
            calls: AtomicUsize::new(0),
            allowance,
        };
        let error = replacement_duplicates_retained_tetrahedron(
            &topology,
            &[0, 1],
            &[[0, 1, 2, 3], [0, 2, 1, 4]],
            &mut InsertionWork::new(options(), &signal),
        )
        .unwrap_err();
        assert_eq!(error.kind, DelaunayInsertionErrorKind::Cancelled);
        assert_eq!(
            before,
            topology
                .tetrahedra
                .iter()
                .map(|cell| cell.vertex_indices)
                .collect::<Vec<_>>()
        );
    }
    assert!(!replacement_duplicates_retained_tetrahedron(
        &topology,
        &[0, 1],
        &[[0, 1, 2, 3], [0, 2, 1, 4]],
        &mut InsertionWork::new(options(), &NeverCancelled)
    )
    .unwrap());
}
