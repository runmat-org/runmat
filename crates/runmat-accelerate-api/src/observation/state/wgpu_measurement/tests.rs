use super::*;
use crate::observation::WgpuQueryAllocationDisposition;

fn completed(submissions: u64) -> ProviderObservationStatus {
    ProviderObservationStatus::Completed {
        total_elapsed: Duration::ZERO,
        submissions,
    }
}

#[test]
fn missing_and_empty_submission_coverage_remain_unknown() {
    let mut aggregate = WgpuMeasurementAggregate::default();
    assert_eq!(aggregate.snapshot(0, completed(0)).dispatches, None);
    aggregate.record(None);
    aggregate.record(Some(WgpuSubmissionMeasurement {
        compute_elapsed: Some(Duration::from_nanos(3)),
        dispatches: Some(1),
        query_allocation: Some(WgpuQueryAllocationDisposition::NoComputePass),
        ..WgpuSubmissionMeasurement::default()
    }));
    let snapshot = aggregate.snapshot(2, completed(2));
    assert_eq!(snapshot.compute_elapsed, None);
    assert_eq!(snapshot.dispatches, None);
    assert_eq!(snapshot.query_allocations, None);
    assert_eq!(aggregate.snapshot(3, completed(3)).dispatches, None);
    assert_eq!(
        aggregate
            .snapshot(2, ProviderObservationStatus::Pending)
            .dispatches,
        None
    );
}

#[test]
fn each_optional_metric_fails_closed_on_its_own_checked_overflow() {
    let mut aggregate = WgpuMeasurementAggregate {
        settled: Some(0),
        compute_elapsed: Some(Duration::MAX),
        dispatches: Some(u64::MAX),
        query_allocations: Some(WgpuQueryAllocationObservation {
            query_sets: u64::MAX,
            ..WgpuQueryAllocationObservation::default()
        }),
    };
    aggregate.record(Some(WgpuSubmissionMeasurement {
        compute_elapsed: Some(Duration::from_nanos(1)),
        dispatches: Some(1),
        query_allocation: Some(WgpuQueryAllocationDisposition::Allocated {
            timestamp_slots: 1,
            resolve_requested_bytes: 8,
            staging_requested_bytes: 8,
            map_callbacks: 1,
        }),
        ..WgpuSubmissionMeasurement::default()
    }));
    let snapshot = aggregate.snapshot(1, completed(1));
    assert_eq!(snapshot.compute_elapsed, None);
    assert_eq!(snapshot.dispatches, None);
    assert_eq!(snapshot.query_allocations, None);
}
