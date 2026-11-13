use super::*;
use std::sync::Arc;

#[test]
fn route_escape_is_monotone_negative_and_bare_roots_never_certify() {
    let first = CommandProfilingRoot::new();
    let other = CommandProfilingRoot::new();
    let selected = first.receipt();
    assert!(selected.same_command(&first.receipt()));
    assert!(!selected.same_command(&other.receipt()));
    selected.mark_unproven_device_route();
    selected.mark_unproven_device_route();
    let escaped = first.seal();
    assert!(escaped.unproven_device_route);
    assert_eq!(escaped.route_coverage, CommandRouteCoverage::Incomplete);
    let bare = other.seal();
    assert!(!bare.unproven_device_route);
    assert!(!bare.unproven_device_time);
    assert_eq!(bare.route_coverage, CommandRouteCoverage::Incomplete);
}

#[test]
fn implicit_device_transfer_invalidates_time_without_erasing_launch_evidence() {
    let root = CommandProfilingRoot::new();
    let receipt = root.receipt();
    receipt.mark_unproven_device_time();
    receipt
        .begin_physical_submission()
        .expect("physical encoder")
        .complete(PhysicalSubmissionTerminal::Completed {
            gpu_execution: Some(Duration::from_micros(4)),
            kernel_count: Some(2),
        });
    let snapshot = root.seal();
    assert!(!snapshot.unproven_device_route);
    assert!(snapshot.unproven_device_time);
    assert_eq!(
        snapshot.physical_attributed_metrics.unwrap().kernel_count,
        Some(2)
    );
    assert_eq!(snapshot.route_coverage, CommandRouteCoverage::Incomplete);
}

#[test]
fn interleaved_commands_keep_linear_terminal_evidence_separate() {
    let first = CommandProfilingRoot::new();
    let second = CommandProfilingRoot::new();
    let first_ready = first
        .receipt()
        .begin_provider_operation()
        .expect("first observation");
    let first_abandoned = first
        .receipt()
        .begin_provider_operation()
        .expect("second observation");
    let second_failed = second
        .receipt()
        .begin_provider_operation()
        .expect("other command");

    first_ready.complete(ProviderObservationTerminal::Completed {
        gpu_execution: None,
        kernel_count: None,
    });
    second_failed.complete(ProviderObservationTerminal::Failed);
    drop(first_abandoned);
    assert_eq!(
        first.seal().observations,
        Some(ProviderObservationCounts {
            incomplete: 1,
            missing: 1,
            ..ProviderObservationCounts::default()
        })
    );
    assert_eq!(
        second.seal().observations,
        Some(ProviderObservationCounts {
            failed: 1,
            ..ProviderObservationCounts::default()
        })
    );
}

#[test]
fn publication_freezes_pending_and_late_completion_without_claiming_coverage() {
    let command = CommandProfilingRoot::new();
    let borrowed = command.receipt();
    let pending = borrowed
        .begin_provider_operation()
        .expect("pending receipt");
    let published = command.seal();
    pending.complete(ProviderObservationTerminal::Completed {
        gpu_execution: Some(Duration::from_micros(7)),
        kernel_count: Some(2),
    });
    assert_eq!(published.observations.expect("valid state").pending, 1);
    assert_eq!(published.route_coverage, CommandRouteCoverage::Incomplete);
    assert_eq!(published.gpu_execution, None);
    assert_eq!(published.kernel_count, None);
    assert_eq!(
        published.observed_metric_coverage,
        Some(ObservedMetricCoverage {
            gpu_execution: false,
            kernel_count: false,
        })
    );
    assert!(borrowed.begin_provider_operation().is_none());
}

#[test]
fn verified_dispatches_and_device_time_have_independent_coverage() {
    let count_only = CommandProfilingRoot::new();
    count_only
        .receipt()
        .begin_provider_operation()
        .expect("observed route")
        .complete(ProviderObservationTerminal::Completed {
            gpu_execution: None,
            kernel_count: Some(2),
        });
    assert_eq!(
        count_only.seal().observed_metric_coverage,
        Some(ObservedMetricCoverage {
            gpu_execution: false,
            kernel_count: true,
        })
    );
    let time_only = CommandProfilingRoot::new();
    time_only
        .receipt()
        .begin_provider_operation()
        .expect("observed route")
        .complete(ProviderObservationTerminal::Completed {
            gpu_execution: Some(Duration::from_micros(7)),
            kernel_count: None,
        });
    let snapshot = time_only.seal();
    assert_eq!(
        snapshot.observed_metric_coverage,
        Some(ObservedMetricCoverage {
            gpu_execution: true,
            kernel_count: false,
        })
    );
    assert_eq!(snapshot.gpu_execution, None);
    assert_eq!(snapshot.kernel_count, None);
}

#[test]
fn metric_overflow_invalidates_the_entire_sample_instead_of_saturating() {
    let command = CommandProfilingRoot::new();
    let contribution = command
        .receipt()
        .begin_provider_operation()
        .expect("observed route");
    command
        .receipt
        .0
        .lock()
        .expect("test state")
        .metrics
        .kernel_count = u64::MAX;
    contribution.complete(ProviderObservationTerminal::Completed {
        gpu_execution: None,
        kernel_count: Some(1),
    });
    let snapshot = command.seal();
    assert_eq!(snapshot.observations, None);
    assert_eq!(snapshot.observed_metric_coverage, None);
    assert_eq!(snapshot.kernel_count, None);
}

#[test]
fn admission_overflow_fails_closed_without_issuing_a_plausible_receipt() {
    let command = CommandProfilingRoot::new();
    command.receipt.0.lock().expect("test state").admitted = u64::MAX;
    assert!(command.receipt().begin_provider_operation().is_none());
    assert_eq!(command.seal().observations, None);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn send_safe_contribution_can_settle_after_origin_command_is_gone() {
    fn require_send<T: Send>() {}
    require_send::<CommandProviderContribution>();
    let command = CommandProfilingRoot::new();
    let contribution = command
        .receipt()
        .begin_provider_operation()
        .expect("owned receipt");
    let worker = std::thread::spawn(move || {
        contribution.complete(ProviderObservationTerminal::Missing);
    });
    worker.join().expect("worker settlement");
    assert_eq!(command.seal().observations.expect("valid").missing, 1);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn poisoned_state_cannot_be_reported_as_a_zero_measurement() {
    let command = CommandProfilingRoot::new();
    let state = Arc::clone(&command.receipt.0);
    let _ = std::thread::spawn(move || {
        let _guard = state.lock().expect("unpoisoned fixture");
        panic!("poison the command accounting lock");
    })
    .join();
    let borrowed = command.receipt();
    assert_eq!(command.seal().observations, None);
    assert!(borrowed.begin_provider_operation().is_none());
}
