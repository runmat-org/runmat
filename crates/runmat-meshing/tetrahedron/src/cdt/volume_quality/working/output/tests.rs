use super::*;
use crate::cdt::volume_quality::tests::{metric_contexts, metric_request, topology};
use runmat_meshing_core::NeverCancelled;
use std::sync::atomic::{AtomicUsize, Ordering};

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Consuming quality-cell transfer polls cancellation during movement and rejects a
//   lower cell-inventory limit before output allocation. It verifies this local ownership
//   transition; it does not establish durable publication, distributed equivalence, whole-study
//   memory bounds or throughput scaling.
#[test]
fn consumed_cells_poll_mid_transfer_and_reject_lower_inventory_limits() {
    let topology = topology();
    let options = DelaunayVolumeQualityOptions {
        cancellation_check_interval: 1,
        ..DelaunayVolumeQualityOptions::default()
    };
    let report = evaluate_delaunay_volume_quality(
        &topology,
        &metric_request(),
        &metric_contexts(&topology),
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert!(report.tetrahedra.len() > 1);
    let owner =
        || WorkingVolumeQuality::new(report.clone(), &topology, options, &NeverCancelled).unwrap();
    struct CancelDuringTransfer(AtomicUsize);
    impl MeshingCancellationSignal for CancelDuringTransfer {
        fn is_cancelled(&self) -> bool {
            let count = self.0.fetch_add(1, Ordering::Relaxed) + 1;
            count >= 3
        }
    }
    let cancellation = CancelDuringTransfer(AtomicUsize::new(0));
    assert!(matches!(
        owner().into_cells(options, &cancellation),
        Err(DelaunayVolumeQualityError {
            kind: DelaunayVolumeQualityErrorKind::Cancelled,
            ..
        })
    ));
    assert_eq!(cancellation.0.load(Ordering::Relaxed), 3);
    assert!(matches!(
        owner().into_cells(
            DelaunayVolumeQualityOptions {
                maximum_tetrahedra: 1,
                ..options
            },
            &NeverCancelled
        ),
        Err(DelaunayVolumeQualityError {
            kind: DelaunayVolumeQualityErrorKind::ResourceLimit,
            ..
        })
    ));
}
