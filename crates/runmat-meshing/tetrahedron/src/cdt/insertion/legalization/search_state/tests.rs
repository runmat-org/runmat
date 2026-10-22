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

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Physical restoration seeds its complete connectivity hint once. Across four and
//   sixty-four disconnected neighborhoods, prospective two-to-three cavity updates visit only two
//   removed and three created cells, match independently rebuilt complete connectivity, preserve
//   retained state and recognize the return cycle. Local and global repair use the same adapter.
//   Cycle hints are not geometry identities or admission proof; this component evidence does not
//   qualify all provider search costs, arbitrary connected-mesh scaling or native whole-task
//   memory.
#[test]
fn physical_repair_cycle_deltas_match_complete_connectivity_with_cavity_bounded_work() {
    for count in [4, 64] {
        let original = neighborhoods(count);
        let mut work = InsertionWork::new(options(), &NeverCancelled);
        let state = seed(&original, &mut work).unwrap();
        let before = state.fingerprint();
        let created = two_to_three::replacement(&original, 0, 1, [0, 1, 2]).unwrap();
        let signal = Counting {
            calls: AtomicUsize::new(0),
            allowance: usize::MAX,
        };
        let next = replacement(
            state,
            &original,
            &[0, 1],
            &created,
            &mut InsertionWork::new(options(), &signal),
        )
        .unwrap();
        assert_eq!(
            signal.calls.load(Ordering::Relaxed),
            5,
            "only removed and created cells are visited"
        );
        assert_eq!(
            state.fingerprint(),
            before,
            "prospective candidates do not mutate retained state"
        );
        let (repaired, applied) =
            apply(original.clone(), state, &[0, 1], created, &mut work).unwrap();
        assert_eq!(applied.fingerprint(), next.fingerprint());
        assert_eq!(
            next.fingerprint(),
            crate::cdt::topology::connectivity_search_fingerprint(&repaired)
        );
        let removed = (0..repaired.tetrahedra.len())
            .filter(|index| {
                repaired.tetrahedra[*index]
                    .vertex_indices
                    .iter()
                    .all(|vertex| *vertex < 5)
            })
            .collect::<Vec<_>>();
        let restored = original.tetrahedra[..2]
            .iter()
            .map(|cell| cell.vertex_indices)
            .collect::<Vec<_>>();
        let (restored, back) = apply(repaired, applied, &removed, restored, &mut work).unwrap();
        assert_eq!(restored, original);
        assert_eq!(
            back.fingerprint(),
            before,
            "returning to the original cavity retains cycle detection"
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation rejects the complete seed and each removed/created-cell checkpoint
//   without exposing a partial cycle hint or changing the retained state. This insertion-owner
//   component check does not establish native worker cancellation, geometry admission or complete
//   resource accounting.
#[test]
fn physical_repair_cycle_seed_and_each_delta_checkpoint_reject_without_partial_state() {
    let original = neighborhoods(4);
    for allowance in [0, 3, 7] {
        let signal = Counting {
            calls: AtomicUsize::new(0),
            allowance,
        };
        assert_eq!(
            seed(&original, &mut InsertionWork::new(options(), &signal))
                .unwrap_err()
                .kind,
            DelaunayInsertionErrorKind::Cancelled
        );
    }
    let state = seed(
        &original,
        &mut InsertionWork::new(options(), &NeverCancelled),
    )
    .unwrap();
    let before = state.fingerprint();
    let created = two_to_three::replacement(&original, 0, 1, [0, 1, 2]).unwrap();
    for allowance in 0..5 {
        let signal = Counting {
            calls: AtomicUsize::new(0),
            allowance,
        };
        assert_eq!(
            replacement(
                state,
                &original,
                &[0, 1],
                &created,
                &mut InsertionWork::new(options(), &signal)
            )
            .unwrap_err()
            .kind,
            DelaunayInsertionErrorKind::Cancelled
        );
        assert_eq!(state.fingerprint(), before);
    }
}
