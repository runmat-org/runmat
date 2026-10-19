use super::*;
use runmat_meshing_core::MeshingCancellationSignal;

pub(super) struct CancelAfter {
    pub(super) probes: AtomicU64,
    pub(super) limit: u64,
}

impl MeshingCancellationSignal for CancelAfter {
    fn is_cancelled(&self) -> bool {
        self.probes.fetch_add(1, Ordering::Relaxed) >= self.limit
    }
}

#[test]
fn cold_curve_decode_preserves_bytes_and_rejects_cancellation_and_tampering() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    let encoded = encode_shared_curve_mesh(&mesh, &topology).unwrap();
    let control = CancelAfter {
        probes: AtomicU64::new(0),
        limit: u64::MAX,
    };
    let decoded =
        decode_shared_curve_mesh_with_cancellation(&encoded, &topology, &control).unwrap();
    assert_eq!(decoded, mesh);
    assert_eq!(
        encode_shared_curve_mesh(&decoded, &topology).unwrap(),
        encoded
    );
    assert!(control.probes.load(Ordering::Relaxed) > 10);
    for limit in [0, 10] {
        let cancelled = CancelAfter {
            probes: AtomicU64::new(0),
            limit,
        };
        let error = decode_shared_curve_mesh_with_cancellation(&encoded, &topology, &cancelled)
            .unwrap_err();
        assert!(error.reason.contains("cancelled"));
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(decode_shared_curve_mesh_with_cancellation(&trailing, &topology, &control).is_err());
    let mut foreign = topology;
    foreign.edges.clear();
    assert!(decode_shared_curve_mesh_with_cancellation(&encoded, &foreign, &control,).is_err());
}

#[test]
fn semantic_curve_read_checks_cancellation_inside_node_inventory() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    let cancelled = CancelAfter {
        probes: AtomicU64::new(0),
        limit: 4,
    };
    let error = super::super::validation::validate_shared_curve_mesh_with_cancellation(
        &mesh,
        &topology,
        Some(&cancelled),
    )
    .unwrap_err();
    assert_eq!(
        error.kind,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
        )
    );
    assert_eq!(cancelled.probes.load(Ordering::Relaxed), 5);
}
