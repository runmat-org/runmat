use super::*;
use crate::cdt::volume_quality::tests::{metric_contexts, metric_request, topology};
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Complete cell reconstruction is compared exactly while retaining one quality report,
//   followed by independent alternate-arithmetic validation. The fixture rejects cell, metadata,
//   aggregate, worst-cell and inventory tampering, resource exhaustion and cancellation. It
//   establishes this handoff's acceptance semantics, not whole-study memory bounds or throughput
//   scaling.
#[test]
fn streamed_handoff_matches_full_report_and_rejects_cell_and_summary_tampering() {
    let topology = topology();
    let provenance = metric_contexts(&topology);
    let request = metric_request();
    let options = DelaunayVolumeQualityOptions::default();
    let report = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    verify_complete_handoff(
        &topology,
        &request,
        &provenance,
        &report,
        options,
        &NeverCancelled,
    )
    .unwrap();
    for tamper in 0..5 {
        let mut altered = report.clone();
        match tamper {
            0 => altered.tetrahedra[0].maximum_metric_edge_length *= 1.00000001,
            1 => altered.maximum_radius_edge_ratio *= 1.00000001,
            2 => {
                altered.worst_refinement_tetrahedron = Some([StableDigest::from_bytes([99; 32]); 4])
            }
            3 => {
                altered.tetrahedra.pop();
            }
            _ => altered.tetrahedra[0].protected_facet_count += 1,
        }
        assert!(verify_complete_handoff(
            &topology,
            &request,
            &provenance,
            &altered,
            options,
            &NeverCancelled
        )
        .is_err());
    }
    let limited = DelaunayVolumeQualityOptions {
        maximum_nodes: 1,
        ..options
    };
    assert!(verify_complete_handoff(
        &topology,
        &request,
        &provenance,
        &report,
        limited,
        &NeverCancelled
    )
    .is_err());
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert!(verify_complete_handoff(
        &topology,
        &request,
        &provenance,
        &report,
        options,
        &Cancelled
    )
    .is_err());
}
