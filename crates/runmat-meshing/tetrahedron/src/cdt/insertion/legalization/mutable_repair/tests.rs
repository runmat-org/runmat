use super::*;
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use std::sync::atomic::{AtomicUsize, Ordering};
mod admitted;

fn neighborhoods(count: usize) -> DelaunayVolumeTopology {
    let mut nodes = Vec::new();
    let mut tetrahedra = Vec::new();
    for cluster in 0..count {
        let first = nodes.len() as u32;
        for point in [
            [-1.5, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 2.0, 2.5],
            [0.0, 2.0, -2.5],
        ] {
            let mut identity = [1; 32];
            identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
            nodes.push(crate::cdt::DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [point[0] + cluster as f64 * 100.0, point[1], point[2]],
            ));
        }
        tetrahedra.extend([
            [first, first + 2, first + 3, first + 4],
            [first + 1, first + 2, first + 4, first + 3],
        ]);
    }
    crate::cdt::build_delaunay_volume_topology(
        nodes,
        tetrahedra,
        crate::cdt::DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[derive(Default)]
struct Probe(AtomicUsize);
impl MeshingCancellationSignal for Probe {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        false
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Four and sixty-four independent cavities retain original points and pass exhaustive
//   physical validation. Measured checkpoint growth bounds this disconnected repair fixture; it
//   does not establish native heat recovery, whole-task memory or arbitrary connected-mesh scaling.
#[test]
fn dirty_face_work_scales_with_independent_cavities_and_preserves_points() {
    check_repair_growth(false);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The fallback recovery repair entry retains original points and independently
//   validates four and sixty-four disconnected cavities, with bounded checkpoint growth. This
//   entry-point size series does not establish arbitrary connected-mesh scaling, original heat
//   recovery or whole-task native memory.
#[test]
fn local_recovery_entry_scales_with_independent_cavities_and_preserves_points() {
    check_repair_growth(true);
}

fn check_repair_growth(local_entry: bool) {
    let mut counts = Vec::new();
    for size in [4, 64] {
        let original = neighborhoods(size);
        let points = original.nodes.clone();
        let mut options = super::super::super::DelaunayInsertionOptions::default();
        options.topology.cancellation_check_interval = 1;
        let probe = Probe::default();
        let mut work = InsertionWork::new(options, &probe);
        let repaired = if local_entry {
            super::super::local::restore(original, &BTreeSet::new(), true, &mut work)
        } else {
            restore(original, &BTreeSet::new(), &mut work)
        }
        .unwrap();
        assert_eq!(work.legalization_flips, size as u64);
        assert_eq!(repaired.nodes, points);
        assert_eq!(repaired.tetrahedra.len(), size * 3);
        let count = probe.0.load(Ordering::Relaxed);
        counts.push(count);
        // Independent exhaustive physical empty-sphere validation is not part
        // of the repair-work measurement and uses every original point.
        crate::cdt::validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
    }
    assert!(counts[1] <= counts[0] * 20, "dirty-face work: {counts:?}");
}

#[test]
fn dirty_face_repair_preserves_barriers_and_original_work_controls() {
    let original = neighborhoods(1);
    let shared = [
        original.nodes[2].identity,
        original.nodes[3].identity,
        original.nodes[4].identity,
    ];
    let options = super::super::super::DelaunayInsertionOptions::default();
    let protected = restore(
        original.clone(),
        &BTreeSet::from([shared]),
        &mut InsertionWork::new(options, &NeverCancelled),
    )
    .unwrap();
    assert_eq!(protected, original);
    let mut limited = options;
    limited.maximum_predicate_evaluations = 1;
    assert_eq!(
        restore(
            original.clone(),
            &BTreeSet::new(),
            &mut InsertionWork::new(limited, &NeverCancelled)
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::ResourceLimit
    );
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let mut cancellable = options;
    cancellable.topology.cancellation_check_interval = 1;
    assert_eq!(
        restore(
            original,
            &BTreeSet::new(),
            &mut InsertionWork::new(cancellable, &Cancelled)
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::Cancelled
    );
}
