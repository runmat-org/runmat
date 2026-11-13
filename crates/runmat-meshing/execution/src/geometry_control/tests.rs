mod scopes;
mod views;
use super::*;
use runmat_meshing_core::{MeshingRequestSettings, NeverCancelled};
use std::time::Duration;

fn control_at(
    started: Instant,
    last_probe: Instant,
    settings: &MeshingRequestSettings,
    ceilings: runmat_meshing_core::MeshingExecutionCeilings,
) -> MeshingGeometryEvaluationControl<'static> {
    MeshingGeometryEvaluationControl::new(
        &NeverCancelled,
        started,
        Arc::new(CheckpointClock::new(last_probe)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        &runmat_meshing_core::MeshingResourceBudget::new(settings.limits, ceilings),
        &settings.cancellation,
    )
}

#[test]
fn wall_time_budget_fails_exact_geometry_evaluation() {
    let mut ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    ceilings.maximum_wall_time_ms = 1_000;
    let now = Instant::now();
    let control = control_at(
        now - Duration::from_secs(2),
        now,
        &MeshingRequestSettings::default(),
        ceilings,
    );
    let failure = control.checkpoint().unwrap_err();
    assert_eq!(
        failure.kind,
        GeometryEvaluationErrorKind::TimeBudgetExceeded
    );
    assert!(failure.to_string().contains("/1000 ms"));
}

/// A checkpoint gap fails on the search work done since the previous probe, the load-independent
/// responsiveness contract; latency above the target is telemetry only.
#[test]
fn checkpoint_gap_is_bounded_by_work_done_not_by_machine_load() {
    let ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    let mut settings = MeshingRequestSettings::default();
    settings.cancellation.maximum_checkpoint_latency_ms = 1_000;
    settings.cancellation.maximum_work_units_between_checks = 10;
    let now = Instant::now();
    // A two-second gap, as a loaded machine produces, with the algorithm inside its work bound.
    let control = control_at(now, now - Duration::from_secs(2), &settings, ceilings);
    control.consume_search_work(10).unwrap();
    control.checkpoint().unwrap();
    assert!(
        control
            .maximum_observed_checkpoint_latency_ms
            .load(Ordering::Relaxed)
            >= 2_000
    );
    // A probe through the cancellation signal renews the work window.
    control.consume_search_work(10).unwrap();
    assert!(!control.is_cancelled());
    control.consume_search_work(10).unwrap();
    control.checkpoint().unwrap();
    // More work than the request admits between two probes fails, however fast it ran.
    control.consume_search_work(11).unwrap();
    let failure = control.checkpoint().unwrap_err();
    assert_eq!(
        failure.kind,
        GeometryEvaluationErrorKind::SearchWorkBudgetExceeded
    );
    assert!(failure.to_string().contains("11 search work units"));
    // The failed probe closed its window: the next gap starts from zero.
    control.checkpoint().unwrap();
}

pub(crate) fn bounded_control(limit: u64) -> MeshingGeometryEvaluationControl<'static> {
    let mut ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    let settings = MeshingRequestSettings::default();
    ceilings.maximum_memory_bytes = limit;
    let now = Instant::now();
    MeshingGeometryEvaluationControl::new(
        &NeverCancelled,
        now,
        Arc::new(CheckpointClock::new(now)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        &runmat_meshing_core::MeshingResourceBudget::new(settings.limits, ceilings),
        &settings.cancellation,
    )
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Repeated transient reservations preserve the high water mark without accumulating
//   released storage. Retained decoded storage remains charged and overlapping excess is rejected.
//   This verifies admission accounting, not allocator RSS or a whole-study envelope.
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

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Error returns and unwinding release temporary reservations without erasing peak
//   evidence. Other non-scoped allocations remain conservatively retained.
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

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Simultaneous reservations share one ceiling and preserve the high water mark after
//   both workers release storage. This bounds the reservation authority, not total process memory.
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
