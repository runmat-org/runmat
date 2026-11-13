use super::*;
use crate::profiling::{
    CommandProfilingRoot, CommandRouteCoverage, ProviderObservationCounts,
    ProviderObservationTerminal,
};

#[test]
fn physical_submissions_do_not_inherit_nested_provider_observation_counts() {
    let root = CommandProfilingRoot::new();
    let receipt = root.receipt();
    for _ in 0..2 {
        receipt
            .begin_provider_operation()
            .expect("nested observation")
            .complete(ProviderObservationTerminal::Completed {
                gpu_execution: Some(Duration::from_micros(5)),
                kernel_count: Some(1),
            });
    }
    receipt
        .begin_physical_submission()
        .expect("one emitted encoder")
        .complete(PhysicalSubmissionTerminal::Completed {
            gpu_execution: Some(Duration::from_micros(5)),
            kernel_count: Some(1),
        });
    let snapshot = root.seal();
    assert_eq!(snapshot.observations.expect("valid").completed, 2);
    assert_eq!(
        snapshot.physical_submissions,
        Some(PhysicalSubmissionCounts {
            completed: 1,
            ..PhysicalSubmissionCounts::default()
        })
    );
    assert_eq!(
        snapshot.physical_metric_coverage,
        Some(PhysicalMetricCoverage {
            gpu_execution: true,
            kernel_count: true,
        })
    );
    assert_eq!(
        snapshot.physical_attributed_metrics,
        Some(PhysicalAttributedMetrics {
            gpu_execution: Some(Duration::from_micros(5)),
            kernel_count: Some(1),
        })
    );
    assert_eq!(snapshot.route_coverage, CommandRouteCoverage::Incomplete);
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );
}

#[test]
fn unsupported_timestamps_do_not_discard_verified_dispatch_count() {
    let root = CommandProfilingRoot::new();
    root.receipt()
        .begin_physical_submission()
        .expect("copy-only submission")
        .complete(PhysicalSubmissionTerminal::Completed {
            gpu_execution: None,
            kernel_count: Some(0),
        });
    let snapshot = root.seal();
    assert_eq!(snapshot.physical_submissions.expect("valid").incomplete, 1);
    assert_eq!(
        snapshot.physical_metric_coverage,
        Some(PhysicalMetricCoverage {
            gpu_execution: false,
            kernel_count: true,
        })
    );
    assert_eq!(
        snapshot.physical_attributed_metrics,
        Some(PhysicalAttributedMetrics {
            gpu_execution: None,
            kernel_count: Some(0),
        })
    );
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );
}

#[test]
fn checked_physical_sums_are_independent_across_multiple_submissions() {
    let root = CommandProfilingRoot::new();
    let receipt = root.receipt();
    receipt
        .begin_physical_submission()
        .expect("first encoder")
        .complete(PhysicalSubmissionTerminal::Completed {
            gpu_execution: Some(Duration::from_micros(3)),
            kernel_count: Some(2),
        });
    receipt
        .begin_physical_submission()
        .expect("second encoder")
        .complete(PhysicalSubmissionTerminal::Completed {
            gpu_execution: None,
            kernel_count: Some(4),
        });
    let snapshot = root.seal();
    assert_eq!(snapshot.physical_submissions.expect("valid").completed, 1);
    assert_eq!(snapshot.physical_submissions.expect("valid").incomplete, 1);
    assert_eq!(
        snapshot.physical_metric_coverage,
        Some(PhysicalMetricCoverage {
            gpu_execution: false,
            kernel_count: true,
        })
    );
    assert_eq!(
        snapshot.physical_attributed_metrics,
        Some(PhysicalAttributedMetrics {
            gpu_execution: None,
            kernel_count: Some(6),
        })
    );
    assert_eq!(snapshot.kernel_count, None);
}

#[test]
fn pending_drop_and_late_completion_cannot_mutate_published_snapshot() {
    let root = CommandProfilingRoot::new();
    let borrowed = root.receipt();
    let missing = borrowed.begin_physical_submission().expect("abandoned");
    let late = borrowed.begin_physical_submission().expect("pending");
    drop(missing);
    let published = root.seal();
    assert_eq!(
        published.physical_submissions,
        Some(PhysicalSubmissionCounts {
            pending: 1,
            missing: 1,
            ..PhysicalSubmissionCounts::default()
        })
    );
    assert_eq!(
        published.physical_metric_coverage,
        Some(PhysicalMetricCoverage {
            gpu_execution: false,
            kernel_count: false,
        })
    );
    assert_eq!(
        published.physical_attributed_metrics,
        Some(PhysicalAttributedMetrics {
            gpu_execution: None,
            kernel_count: None,
        })
    );
    late.complete(PhysicalSubmissionTerminal::Completed {
        gpu_execution: Some(Duration::from_micros(9)),
        kernel_count: Some(3),
    });
    assert!(borrowed.begin_physical_submission().is_none());
    assert_eq!(published.route_coverage, CommandRouteCoverage::Incomplete);
    assert_eq!(
        (published.gpu_execution, published.kernel_count),
        (None, None)
    );
}

#[test]
fn interleaved_commands_and_root_cancellation_remain_isolated() {
    let first = CommandProfilingRoot::new();
    let second = CommandProfilingRoot::new();
    let first_receipt = first.receipt();
    let abandoned = first_receipt.begin_physical_submission().expect("first");
    second
        .receipt()
        .begin_physical_submission()
        .expect("second")
        .complete(PhysicalSubmissionTerminal::Failed);
    assert_eq!(second.seal().physical_submissions.expect("valid").failed, 1);
    drop(first);
    drop(abandoned);
    assert!(first_receipt.begin_physical_submission().is_none());
}

#[test]
fn absent_physical_admission_never_certifies_coverage_or_zero_totals() {
    let snapshot = CommandProfilingRoot::new().seal();
    assert_eq!(
        snapshot.physical_submissions,
        Some(PhysicalSubmissionCounts::default())
    );
    assert_eq!(
        snapshot.physical_metric_coverage,
        Some(PhysicalMetricCoverage {
            gpu_execution: false,
            kernel_count: false,
        })
    );
    assert_eq!(
        snapshot.observations,
        Some(ProviderObservationCounts::default())
    );
    assert_eq!(
        snapshot.physical_attributed_metrics,
        Some(PhysicalAttributedMetrics {
            gpu_execution: None,
            kernel_count: None,
        })
    );
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );
}

#[test]
fn physical_admission_and_sum_overflow_fail_closed() {
    let admission = CommandProfilingRoot::new();
    admission
        .receipt
        .0
        .lock()
        .expect("test state")
        .physical
        .admitted = u64::MAX;
    assert!(admission.receipt().begin_physical_submission().is_none());
    assert_eq!(admission.seal().physical_submissions, None);

    let sum = CommandProfilingRoot::new();
    let contribution = sum.receipt().begin_physical_submission().expect("one");
    sum.receipt
        .0
        .lock()
        .expect("test state")
        .physical
        .metrics
        .kernel_count = u64::MAX;
    contribution.complete(PhysicalSubmissionTerminal::Completed {
        gpu_execution: None,
        kernel_count: Some(1),
    });
    let snapshot = sum.seal();
    assert_eq!(snapshot.physical_submissions, None);
    assert_eq!(snapshot.physical_metric_coverage, None);
    assert_eq!(snapshot.physical_attributed_metrics, None);
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn physical_contribution_is_send_safe_and_poison_fails_closed() {
    fn require_send<T: Send>() {}
    require_send::<CommandPhysicalSubmission>();
    let root = CommandProfilingRoot::new();
    let physical = root.receipt().begin_physical_submission().expect("one");
    std::thread::spawn(move || physical.complete(PhysicalSubmissionTerminal::Missing))
        .join()
        .expect("settled on worker");
    assert_eq!(root.seal().physical_submissions.expect("valid").missing, 1);

    let poisoned = CommandProfilingRoot::new();
    let state = Arc::clone(&poisoned.receipt.0);
    let _ = std::thread::spawn(move || {
        let _guard = state.lock().expect("test state");
        panic!("poison the accounting lock");
    })
    .join();
    assert!(poisoned.receipt().begin_physical_submission().is_none());
    assert_eq!(poisoned.seal().physical_submissions, None);
}
