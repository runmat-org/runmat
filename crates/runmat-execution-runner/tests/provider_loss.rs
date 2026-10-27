mod common;

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, DriverAction, DriverCommand};

#[test]
fn provider_loss_fences_and_retains_each_original_unresolved_history() {
    let mut f = common::fixture(1, 2);
    let first = common::submit(
        &mut f.driver,
        common::task(
            "first-provider-loss",
            f.scope,
            f.pool,
            RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
        ),
    );
    let second = common::submit(
        &mut f.driver,
        common::task(
            "other-active-provider-work",
            f.scope,
            f.pool,
            RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
        ),
    );
    assert_eq!(first.worker_id, second.worker_id);
    let prior_event_count = f.driver.events().len();
    let actions = f
        .driver
        .handle(DriverCommand::WorkerLostWithReport {
            worker_id: first.worker_id,
            report: Box::new(BackendReport::for_request(
                &first,
                AttemptReport::Lost {
                    message: "original producer connection failed".into(),
                },
            )),
        })
        .unwrap();
    assert!(!actions
        .iter()
        .any(|action| matches!(action, DriverAction::Launch(_))));
    assert!(matches!(f.driver.events()[prior_event_count].kind,
        runmat_execution_runner::driver::DriverEventKind::WorkerLost { worker_id } if worker_id == first.worker_id));
    let first_evidence = f.driver.task_execution_evidence(first.task_id).unwrap();
    let second_evidence = f.driver.task_execution_evidence(second.task_id).unwrap();
    assert_eq!(first_evidence.task.state, TaskState::Indeterminate);
    assert_eq!(second_evidence.task.state, TaskState::Indeterminate);
    assert_eq!(
        first_evidence.attempts[0]
            .unverified_retirement
            .as_ref()
            .unwrap()
            .diagnostic
            .as_str(),
        "original producer connection failed"
    );
    assert_eq!(
        second_evidence.attempts[0]
            .unverified_retirement
            .as_ref()
            .unwrap()
            .diagnostic
            .as_str(),
        "worker was lost"
    );
    assert!(first_evidence.resource_aggregate(1).is_err());
    assert!(second_evidence.resource_aggregate(1).is_err());
    let snapshot = f.driver.snapshot();
    snapshot.validate().unwrap();
    assert_eq!(snapshot.pools[&f.pool].active_attempts, 2);
    assert_ne!(snapshot.pools[&f.pool].allocated, Default::default());
    let cold: runmat_execution_runner::DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(
        cold.task_execution_evidence(first.task_id).unwrap(),
        first_evidence
    );
}

#[test]
fn non_loss_report_cannot_fence_a_worker_or_rewrite_its_active_task() {
    let mut f = common::fixture(1, 1);
    let request = common::submit(
        &mut f.driver,
        common::task("reject-non-loss", f.scope, f.pool, RetryPolicy::Never),
    );
    let original = f.driver.snapshot();
    assert!(f
        .driver
        .handle(DriverCommand::WorkerLostWithReport {
            worker_id: request.worker_id,
            report: Box::new(BackendReport::for_request(
                &request,
                common::success(&request)
            )),
        })
        .is_err());
    assert_eq!(f.driver.snapshot(), original);
}

#[test]
fn stale_foreign_and_malformed_loss_cannot_fence_live_worker_capacity() {
    for substitution in 0..5 {
        let mut f = common::fixture(1, 2);
        let request = common::submit(
            &mut f.driver,
            common::task("invalid-provider-loss", f.scope, f.pool, RetryPolicy::Never),
        );
        let original = f.driver.snapshot();
        let mut report = BackendReport::for_request(
            &request,
            AttemptReport::Lost {
                message: "untrusted loss".into(),
            },
        );
        match substitution {
            0 => report.driver_fence += 1,
            1 => report.task_id = runmat_execution::TaskId::derive(&[b"foreign-task"]),
            2 => report.attempt_id = AttemptId::derive(&[b"foreign-attempt"]),
            3 => {
                let mut resources =
                    runmat_execution_runner::AttemptResourceAccounting::for_attempt(
                        &request, 1, 0, 0, 0,
                    )
                    .unwrap();
                resources.worker_id = WorkerId::derive(&[b"foreign-worker"]);
                report.unsuccessful_resources = Some(Box::new(resources));
            }
            4 => {
                report.execution = Some(Box::new(
                    runmat_execution_runner::AttemptExecutionObservation {
                        attempt_id: request.id,
                        worker_id: WorkerId::derive(&[b"foreign-observation"]),
                        wall_millis: 1,
                        termination:
                            runmat_execution_runner::AttemptExecutionTermination::NotStarted,
                        local_ipc_sent_bytes: Some(0),
                        local_ipc_received_bytes: Some(0),
                    },
                ))
            }
            _ => unreachable!(),
        }
        assert!(!f.driver.accepts_backend_report(&report));
        f.driver
            .handle(DriverCommand::WorkerLostWithReport {
                worker_id: request.worker_id,
                report: Box::new(report),
            })
            .unwrap();
        let rejected = f.driver.snapshot();
        assert_eq!(rejected.pools, original.pools);
        assert_eq!(rejected.tasks, original.tasks);
        assert_eq!(rejected.attempts, original.attempts);
        assert_eq!(rejected.events.len(), original.events.len() + 1);
        assert!(matches!(
            rejected.events.last().unwrap().kind,
            runmat_execution_runner::driver::DriverEventKind::ReportDiscarded { .. }
        ));
        rejected.validate().unwrap();
    }
}

#[test]
fn late_loss_for_a_completed_attempt_cannot_cancel_other_work_on_same_worker() {
    let mut f = common::fixture(1, 2);
    let first = common::submit(
        &mut f.driver,
        common::task("completed-before-loss", f.scope, f.pool, RetryPolicy::Never),
    );
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &first,
            common::success(&first),
        )))
        .unwrap();
    let active = common::submit(
        &mut f.driver,
        common::task(
            "active-after-completion",
            f.scope,
            f.pool,
            RetryPolicy::Never,
        ),
    );
    assert_eq!(active.worker_id, first.worker_id);
    let original = f.driver.snapshot();
    let loss = BackendReport::for_request(
        &first,
        AttemptReport::Lost {
            message: "late loss for an already completed attempt".into(),
        },
    );
    assert!(!f.driver.accepts_backend_report(&loss));
    f.driver
        .handle(DriverCommand::WorkerLostWithReport {
            worker_id: first.worker_id,
            report: Box::new(loss),
        })
        .unwrap();
    let rejected = f.driver.snapshot();
    assert_eq!(rejected.pools, original.pools);
    assert_eq!(rejected.tasks, original.tasks);
    assert_eq!(rejected.attempts, original.attempts);
    assert_eq!(rejected.events.len(), original.events.len() + 1);
    rejected.validate().unwrap();
}
