use crate::execution::ExecutionProfiling;
use runmat_execution::profiling::{CommandProfilingSnapshot, CommandRouteCoverage};

/// Publish only measurements backed by Core's private canonical-route
/// admission and the sealed command's independent physical evidence.
pub(crate) fn execution_profiling(
    command_wall_ms: u64,
    snapshot: &CommandProfilingSnapshot,
    canonical_device_route: bool,
) -> ExecutionProfiling {
    // The public receipt alone cannot certify a route, even if every token it
    // happened to observe settled. Only this Core-private admission-to-seal
    // join may use its partial physical facts.
    let CommandRouteCoverage::Incomplete = snapshot.route_coverage;
    let physical = snapshot.physical_submissions;
    let observed = physical.is_some_and(|counts| {
        counts.pending == 0
            && counts.failed == 0
            && counts.missing == 0
            && (counts.completed != 0 || counts.incomplete != 0)
    });
    let deferred_settled = snapshot.deferred_work.is_some_and(|counts| {
        counts.pending == 0
            && counts.failed == 0
            && counts.missing == 0
            && counts.admitted == counts.completed
    });
    let closed =
        canonical_device_route && observed && deferred_settled && !snapshot.unproven_device_route;
    let kernel_count = closed
        .then_some(snapshot.physical_metric_coverage)
        .flatten()
        .filter(|coverage| coverage.kernel_count)
        .and(snapshot.physical_attributed_metrics)
        .and_then(|metrics| metrics.kernel_count)
        .and_then(|count| u32::try_from(count).ok());
    let gpu_ms = (closed && !snapshot.unproven_device_time)
        .then_some(snapshot.physical_metric_coverage)
        .flatten()
        .filter(|coverage| coverage.gpu_execution)
        .and(snapshot.physical_attributed_metrics)
        .and_then(|metrics| metrics.gpu_execution)
        .and_then(|duration| u64::try_from(duration.as_millis()).ok());
    ExecutionProfiling {
        command_evidence: Some(crate::execution::CommandExecutionEvidence {
            physical_submissions: snapshot.physical_submissions,
            deferred_work: snapshot.deferred_work,
            route_coverage: snapshot.route_coverage,
            canonical_device_route_admitted: canonical_device_route,
            unproven_device_route: snapshot.unproven_device_route,
            unproven_device_time: snapshot.unproven_device_time,
        }),
        total_ms: command_wall_ms,
        gpu_ms,
        kernel_count,
        ..ExecutionProfiling::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::profiling::{
        CommandProfilingRoot, PhysicalSubmissionTerminal, ProviderObservationTerminal,
    };
    use std::time::Duration;

    #[test]
    fn unmeasured_transfer_gates_device_time_without_erasing_exact_kernel_count() {
        let command = CommandProfilingRoot::new();
        let receipt = command.receipt();
        receipt
            .begin_physical_submission()
            .expect("actual physical submission")
            .complete(PhysicalSubmissionTerminal::Completed {
                gpu_execution: Some(Duration::from_millis(7)),
                kernel_count: Some(2),
            });
        receipt.mark_unproven_device_time();
        let snapshot = command.seal();
        let summary = execution_profiling(13, &snapshot, true);
        assert_eq!(summary.total_ms, 13);
        assert_eq!(summary.kernel_count, Some(2));
        assert_eq!(summary.gpu_ms, None);
        let evidence = summary.command_evidence.expect("original sealed evidence");
        assert_eq!(evidence.route_coverage, CommandRouteCoverage::Incomplete);
        assert!(evidence.canonical_device_route_admitted);
        assert!(evidence.unproven_device_time);
        assert_eq!(evidence.physical_submissions.unwrap().completed, 1);
    }

    #[test]
    fn summaries_preserve_independent_totals_without_inventing_device_measurements() {
        let command = CommandProfilingRoot::new();
        let snapshot = command.seal();
        for total_ms in [0, 17, u64::MAX] {
            let first = execution_profiling(total_ms, &snapshot, false);
            let other = execution_profiling(23, &snapshot, false);
            assert_eq!(first.total_ms, total_ms);
            assert_eq!(other.total_ms, 23);
            for summary in [first, other] {
                assert_eq!(summary.cpu_ms, None);
                assert_eq!(summary.gpu_ms, None);
                assert_eq!(summary.kernel_count, None);
            }
        }
    }

    #[test]
    fn command_root_seals_before_late_provider_completion() {
        let command = CommandProfilingRoot::new();
        let borrowed = command.receipt();
        let pending = borrowed
            .begin_provider_operation()
            .expect("selected receipt");
        let snapshot = command.seal();
        pending.complete(ProviderObservationTerminal::Failed);
        assert!(borrowed.begin_provider_operation().is_none());
        assert_eq!(snapshot.observations.expect("valid").pending, 1);
    }
}
