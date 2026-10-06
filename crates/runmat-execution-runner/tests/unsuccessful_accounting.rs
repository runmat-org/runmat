mod common;

use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, AttemptFailureKind, AttemptReport,
    AttemptResourceAccounting, DriverAction, DriverCommand, TaskExecutionEvidence,
};

fn failed() -> AttemptReport {
    AttemptReport::Failed {
        kind: AttemptFailureKind::Infrastructure,
        message: "original provider failure".into(),
    }
}

// Explicit fixture accounting tests transport and aggregation, not actual provider metering.
#[test]
fn every_unsuccessful_disposition_retains_original_accounting_after_cold_import() {
    for report in [
        failed(),
        AttemptReport::Lost {
            message: "lost".into(),
        },
        AttemptReport::Cancelled,
    ] {
        let mut fixture = common::fixture(1, 1);
        let task = common::task(
            "accounted-terminal",
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        );
        let request = common::submit(&mut fixture.driver, task);
        let accounting = AttemptResourceAccounting::for_attempt(&request, 3, 17, 19, 23).unwrap();
        let mut backend = BackendReport::for_request(&request, report);
        backend.unsuccessful_resources = Some(Box::new(accounting.clone()));
        fixture
            .driver
            .handle(DriverCommand::BackendReport(backend))
            .unwrap();
        let original = fixture
            .driver
            .snapshot()
            .task_execution_evidence(request.task_id)
            .unwrap();
        let cold: TaskExecutionEvidence =
            serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
        cold.validate().unwrap();
        assert_eq!(cold, original);
        assert_eq!(
            cold.resource_aggregate(5).unwrap().attempts,
            vec![accounting]
        );
    }
}

#[test]
fn failed_retry_charges_both_original_intervals_and_preserves_diagnostic() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "accounted-retry",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let first = common::submit(&mut fixture.driver, task);
    let accounting = AttemptResourceAccounting::for_attempt(&first, 3, 17, 19, 23).unwrap();
    let mut backend = BackendReport::for_request(&first, failed());
    backend.unsuccessful_resources = Some(Box::new(accounting.clone()));
    let second = fixture
        .driver
        .handle(DriverCommand::BackendReport(backend))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &second,
            common::success(&second),
        )))
        .unwrap();
    let evidence = fixture
        .driver
        .snapshot()
        .task_execution_evidence(first.task_id)
        .unwrap();
    let aggregate = evidence.resource_aggregate(5).unwrap();
    assert_eq!(aggregate.attempts.len(), 2);
    assert!(aggregate.attempts.contains(&accounting));
    assert_eq!(aggregate.summary.artifact_bytes, 17);
    assert_eq!(aggregate.summary.usage.egress_bytes, 19);
    assert_eq!(aggregate.summary.usage.relay_bytes, 23);
    assert_eq!(first.task.resources.cpu_millicores, 1000);
    assert_eq!(aggregate.summary.usage.cpu_core_millis, 4);
    assert_eq!(
        evidence.attempts[0]
            .terminal
            .as_ref()
            .unwrap()
            .diagnostic
            .as_deref(),
        Some("original provider failure")
    );
}

#[test]
fn contradictory_unsuccessful_accounting_is_discarded_before_terminal_mutation() {
    for corruption in 0..5 {
        let mut fixture = common::fixture(1, 1);
        let task = common::task(
            "invalid-accounting",
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        );
        let request = common::submit(&mut fixture.driver, task);
        let mut accounting =
            AttemptResourceAccounting::for_attempt(&request, 3, 17, 19, 23).unwrap();
        let mut backend = BackendReport::for_request(&request, failed());
        match corruption {
            0 => {
                accounting.attempt_id =
                    runmat_execution::identity::AttemptId::derive(&[b"foreign-attempt"])
            }
            1 => accounting.resources.allocation.memory_bytes += 1,
            2 => backend.report = AttemptReport::Started,
            3 => backend.report = common::success(&request),
            _ => {
                backend.execution = Some(Box::new(AttemptExecutionObservation {
                    attempt_id: request.id,
                    worker_id: request.worker_id,
                    wall_millis: 4,
                    termination: AttemptExecutionTermination::ProcessExited {
                        code: Some(17),
                        success: false,
                    },
                    local_ipc_sent_bytes: Some(1),
                    local_ipc_received_bytes: Some(2),
                }))
            }
        }
        backend.unsuccessful_resources = Some(Box::new(accounting));
        let before = fixture.driver.snapshot();
        fixture
            .driver
            .handle(DriverCommand::BackendReport(backend))
            .unwrap();
        let after = fixture.driver.snapshot();
        assert_eq!(after.attempts, before.attempts, "corruption {corruption}");
        assert_eq!(after.tasks, before.tasks, "corruption {corruption}");
        after.validate().unwrap();
    }
}
