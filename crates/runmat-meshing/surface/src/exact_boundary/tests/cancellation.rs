use super::*;
use runmat_meshing_core::MeshingCancellationSignal;
use std::sync::atomic::{AtomicU64, Ordering};

struct CancelAfter {
    probes: AtomicU64,
    limit: u64,
}
impl MeshingCancellationSignal for CancelAfter {
    fn is_cancelled(&self) -> bool {
        self.probes.fetch_add(1, Ordering::Relaxed) >= self.limit
    }
}
fn control(limit: u64) -> CancelAfter {
    CancelAfter {
        probes: AtomicU64::new(0),
        limit,
    }
}

#[test]
fn cooperative_boundary_preserves_incidence_and_rejects_tampering() {
    let (topology, curves) = fixture();
    let signal = control(u64::MAX);
    let expected = build_exact_surface_boundary(&topology, &curves).unwrap();
    let mut actual =
        build_exact_surface_boundary_with_cancellation(&topology, &curves, &signal).unwrap();
    assert_eq!(actual, expected);
    validate_exact_surface_boundary_with_cancellation(&actual, &topology, &curves, &signal)
        .unwrap();
    assert!(signal.probes.load(Ordering::Relaxed) > curves.edges[0].nodes.len() as u64);
    actual.faces[0].outer_loop.segments[0].node_uv[0][0] += 1.0e-6;
    assert_eq!(
        validate_exact_surface_boundary_with_cancellation(&actual, &topology, &curves, &signal)
            .unwrap_err()
            .kind,
        ExactSurfaceBoundaryErrorKind::InvalidContract
    );
}

#[test]
fn cancellation_reaches_projection_and_independent_intersection_admission() {
    let (topology, curves) = fixture();
    let signal = control(u64::MAX);
    curves
        .validate_against_with_cancellation(&topology, &signal)
        .unwrap();
    let curve_probes = signal.probes.load(Ordering::Relaxed);
    // Pass curve admission, all three indices, the face and coedge probes;
    // cancel while projecting the first original trim segment.
    let projection_limit = curve_probes
        + 1
        + topology.wires.len() as u64
        + topology.coedges.len() as u64
        + curves.edges.len() as u64
        + 2;
    for limit in [0, projection_limit] {
        assert_eq!(
            build_exact_surface_boundary_with_cancellation(&topology, &curves, &control(limit))
                .unwrap_err()
                .kind,
            ExactSurfaceBoundaryErrorKind::Cancelled
        );
    }
    let boundary = build_exact_surface_boundary(&topology, &curves).unwrap();
    let signal = control(u64::MAX);
    validate_exact_surface_boundary_with_cancellation(&boundary, &topology, &curves, &signal)
        .unwrap();
    let probes = signal.probes.load(Ordering::Relaxed);
    // The final probes belong to the original intersection walk.
    assert!(probes > curve_probes + 4);
    assert_eq!(
        validate_exact_surface_boundary_with_cancellation(
            &boundary,
            &topology,
            &curves,
            &control(probes - 1)
        )
        .unwrap_err()
        .kind,
        ExactSurfaceBoundaryErrorKind::Cancelled
    );
}
