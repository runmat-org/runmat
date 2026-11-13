//! The full converged/restart contracts keep one wire form and their rejection rules.
use crate::*;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use runmat_meshing_curve::SharedCurveMesh;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Probe {
    calls: AtomicUsize,
    cancel_at: usize,
}
impl Probe {
    fn new(cancel_at: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            cancel_at,
        }
    }
}
impl MeshingCancellationSignal for Probe {
    fn is_cancelled(&self) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst) >= self.cancel_at
    }
}

pub(super) fn verify(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) {
    let original = encode_exact_face_partition_result(result, topology, curves).unwrap();
    if let ExactFacePartitionOutcome::Converged { faces } = &result.outcome {
        let mut altered = result.clone();
        let source = &faces[0];
        let boundary_node = source.boundary_segments[0].node_ids[0];
        let ExactFacePartitionOutcome::Converged { faces } = &mut altered.outcome else {
            unreachable!()
        };
        let changed = faces[0]
            .nodes
            .iter_mut()
            .find(|node| node.node_id == boundary_node)
            .unwrap();
        changed.point_m[2] = changed.point_m[2].next_up();
        let untrusted = crate::surface_mesh::encode_unadmitted_for_test(&altered);
        let error = decode_exact_face_partition_result(&untrusted, topology, curves).unwrap_err();
        assert!(error.reason.contains("original curve state"), "{error}");
    }

    let controlled = encode_exact_face_partition_result_with_cancellation(
        result,
        topology,
        curves,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(original, controlled);
    assert_eq!(
        decode_exact_face_partition_result_with_cancellation(
            &original,
            topology,
            curves,
            &NeverCancelled
        )
        .unwrap(),
        *result
    );
    struct CancellationPulse(std::sync::atomic::AtomicBool);
    impl MeshingCancellationSignal for CancellationPulse {
        fn is_cancelled(&self) -> bool {
            self.0.swap(false, Ordering::SeqCst)
        }
    }
    // The rejection retains cancellation even if the external signal later changes.
    let pulse = CancellationPulse(std::sync::atomic::AtomicBool::new(true));
    assert_eq!(
        decode_exact_face_partition_result_with_cancellation(&original, topology, curves, &pulse)
            .unwrap_err()
            .kind,
        ExactSurfaceMeshErrorKind::Cancelled
    );
    let validation = Probe::new(usize::MAX);
    validate_exact_face_partition_result_with_cancellation(result, topology, curves, &validation)
        .unwrap();
    let probes_before_codec = validation.calls.load(Ordering::SeqCst);
    assert!(probes_before_codec > 2);
    // Cancel after semantic admission has completed, during canonical value traversal.
    let encoding = Probe::new(probes_before_codec + 4);
    assert_eq!(
        encode_exact_face_partition_result_with_cancellation(result, topology, curves, &encoding)
            .unwrap_err()
            .kind,
        ExactSurfaceMeshErrorKind::Cancelled
    );
    // Cold input must stop during decoding before it can become an admitted result.
    assert_eq!(
        decode_exact_face_partition_result_with_cancellation(
            &original,
            topology,
            curves,
            &Probe::new(4)
        )
        .unwrap_err()
        .kind,
        ExactSurfaceMeshErrorKind::Cancelled
    );
    assert_eq!(
        validate_exact_face_partition_result_with_cancellation(
            result,
            topology,
            curves,
            &Probe::new(2)
        )
        .unwrap_err()
        .kind,
        ExactSurfaceMeshErrorKind::Cancelled
    );
    let mut tampered = result.clone();
    match &mut tampered.outcome {
        ExactFacePartitionOutcome::Converged { faces } => {
            faces[0].boundary_segments[0].edge_parameters[0] += 1.0
        }
        ExactFacePartitionOutcome::RequiresCurveSplits { splits } => {
            splits[0].split_parameter = splits[0].edge_parameters[0]
        }
    }
    let original_error =
        encode_exact_face_partition_result(&tampered, topology, curves).unwrap_err();
    let controlled_error = encode_exact_face_partition_result_with_cancellation(
        &tampered,
        topology,
        curves,
        &NeverCancelled,
    )
    .unwrap_err();
    assert_eq!(original_error, controlled_error);
    assert_eq!(
        decode_exact_face_partition_result_with_cancellation(
            &original[..original.len() - 1],
            topology,
            curves,
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        ExactSurfaceMeshErrorKind::InvalidEncoding
    );
}
