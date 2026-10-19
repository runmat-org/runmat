mod scopes;
mod views;
use super::*;
use runmat_meshing_core::{MeshingRequestSettings, NeverCancelled};
use std::time::Duration;

#[test]
fn time_failure_identifies_wall_and_checkpoint_envelopes_independently() {
    for wall_failure in [true, false] {
        let now = Instant::now();
        let expired = now - Duration::from_secs(2);
        let mut settings = MeshingRequestSettings::default();
        settings.resources.maximum_wall_time_ms = 1_000;
        settings.cancellation.maximum_checkpoint_latency_ms = 1_000;
        let control = MeshingGeometryEvaluationControl::new(
            &NeverCancelled,
            if wall_failure { expired } else { now },
            Arc::new(CheckpointClock::new(if wall_failure {
                now
            } else {
                expired
            })),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            &settings.resources,
            &settings.cancellation,
        );
        let failure = control.checkpoint().unwrap_err();
        assert_eq!(
            failure.kind,
            GeometryEvaluationErrorKind::TimeBudgetExceeded
        );
        let description = failure.to_string();
        assert!(description.contains("wall time"));
        assert!(description.contains("checkpoint latency"));
        assert_eq!(description.matches("/1000 ms").count(), 2);
        assert_eq!(
            control
                .maximum_observed_checkpoint_latency_ms
                .load(Ordering::Relaxed)
                >= 2_000,
            !wall_failure,
        );
    }
}

pub(crate) fn bounded_control(limit: u64) -> MeshingGeometryEvaluationControl<'static> {
    let mut settings = MeshingRequestSettings::default();
    settings.resources.maximum_memory_bytes = limit;
    let now = Instant::now();
    MeshingGeometryEvaluationControl::new(
        &NeverCancelled,
        now,
        Arc::new(CheckpointClock::new(now)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        &settings.resources,
        &settings.cancellation,
    )
}

#[test]
fn repeated_temporary_storage_preserves_peak_and_retained_result_budget() {
    use runmat_geometry_core::GeometryAllocationReservation;
    let control = bounded_control(100);
    control.consume_allocation_bytes(20).unwrap();
    for _ in 0..1000 {
        let _temporary = GeometryAllocationReservation::new(&control, 80).unwrap();
        assert_eq!(control.usage().live_allocation_bytes, 100);
        assert!(GeometryAllocationReservation::new(&control, 1).is_err());
    }
    assert_eq!(control.usage().live_allocation_bytes, 20);
    assert_eq!(control.usage().allocation_bytes, 100);
    let mut decoded = GeometryAllocationReservation::new(&control, 50).unwrap();
    decoded.retain_bytes(30);
    drop(decoded);
    assert_eq!(control.usage().live_allocation_bytes, 50);
    assert!(GeometryAllocationReservation::new(&control, 51).is_err());
}

#[test]
fn temporary_reservations_release_on_error_and_unwind() {
    use runmat_geometry_core::GeometryAllocationReservation;
    let control = bounded_control(100);
    let fail = || -> Result<(), ()> {
        let _temporary = GeometryAllocationReservation::new(&control, 100).unwrap();
        Err(())
    };
    assert!(fail().is_err());
    assert_eq!(control.usage().live_allocation_bytes, 0);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _temporary = GeometryAllocationReservation::new(&control, 100).unwrap();
        panic!("exercise cleanup");
    }));
    assert!(outcome.is_err());
    assert_eq!(control.usage().live_allocation_bytes, 0);
    assert_eq!(control.usage().allocation_bytes, 100);
}

#[test]
fn concurrent_reservations_share_one_ceiling_without_losing_peak() {
    use runmat_geometry_core::GeometryAllocationReservation;
    let control = bounded_control(100);
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let _temporary = GeometryAllocationReservation::new(&control, 50).unwrap();
                barrier.wait();
                barrier.wait();
            });
        }
        barrier.wait();
        assert_eq!(control.usage().live_allocation_bytes, 100);
        assert!(GeometryAllocationReservation::new(&control, 1).is_err());
        barrier.wait();
    });
    assert_eq!(control.usage().live_allocation_bytes, 0);
    assert_eq!(control.usage().allocation_bytes, 100);
}

#[test]
fn denied_allocation_reports_requested_live_and_maximum_without_charging() {
    let control = bounded_control(100);
    control.consume_allocation_bytes(60).unwrap();
    let error = control.consume_allocation_bytes(41).unwrap_err();
    assert_eq!(
        error.kind,
        GeometryEvaluationErrorKind::AllocationBudgetExceeded
    );
    assert_eq!(error.reason, "exact geometry allocation exceeds the execution resource budget: requested 41 bytes, live 60 bytes, maximum 100 bytes");
    assert_eq!(control.usage().live_allocation_bytes, 60);
    control.release_allocation_bytes(60);
    assert_eq!(control.usage().live_allocation_bytes, 0);
}

mod phase;
