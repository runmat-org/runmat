mod common;

use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::recovery::reconcile_snapshot;
use runmat_execution_runner::{
    AttemptDisposition, AttemptReport, DriverAction, DriverCommand, DriverSnapshot,
};

#[test]
fn lost_retry_retains_unknown_consumption_and_cannot_claim_winner_only_usage() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "lost-history",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let first = common::submit(&mut fixture.driver, task);
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 7 })
        .unwrap();
    let second = fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &first,
            AttemptReport::Lost {
                message: "producer disappeared".into(),
            },
        )))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 11 })
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &second,
            common::success(&second),
        )))
        .unwrap();
    let original = fixture.driver.snapshot();
    let wire = serde_json::to_vec(&original).unwrap();
    let restored = reconcile_snapshot(serde_json::from_slice(&wire).unwrap())
        .unwrap()
        .snapshot();
    let history = restored.task_attempt_history(first.task_id).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(
        history[0].terminal.as_ref().unwrap().disposition,
        AttemptDisposition::Lost
    );
    assert_eq!(
        history[0]
            .terminal
            .as_ref()
            .unwrap()
            .closed_at_driver_millis,
        7
    );
    assert!(history[0].terminal.as_ref().unwrap().resources.is_none());
    assert_eq!(
        history[1].terminal.as_ref().unwrap().resources,
        original.tasks[&first.task_id]
            .committed
            .as_ref()
            .unwrap()
            .resources
    );
    assert!(restored.task_resource_aggregate(first.task_id, 11).is_err());
    // A late report cannot retroactively replace the missing original evidence.
    let prior = fixture.driver.snapshot();
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &first,
            common::success(&first),
        )))
        .unwrap();
    assert_eq!(fixture.driver.snapshot().attempts, prior.attempts);
}

#[test]
fn original_success_aggregates_but_cold_evidence_substitution_rejects() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "success-history",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let request = common::submit(&mut fixture.driver, task);
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 10 })
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            common::success(&request),
        )))
        .unwrap();
    let original = fixture.driver.snapshot();
    assert_eq!(
        original
            .task_resource_aggregate(request.task_id, 10)
            .unwrap()
            .attempts
            .len(),
        1
    );
    for corruption in 0..5 {
        let mut invalid = original.clone();
        match corruption {
            0 => invalid.attempts.get_mut(&request.id).unwrap().terminal = None,
            1 => {
                invalid
                    .attempts
                    .get_mut(&request.id)
                    .unwrap()
                    .terminal
                    .as_mut()
                    .unwrap()
                    .resources = None
            }
            2 => {
                invalid
                    .attempts
                    .get_mut(&request.id)
                    .unwrap()
                    .terminal
                    .as_mut()
                    .unwrap()
                    .closed_at_driver_millis = 11
            }
            3 => {
                invalid
                    .attempts
                    .get_mut(&request.id)
                    .unwrap()
                    .terminal
                    .as_mut()
                    .unwrap()
                    .disposition = AttemptDisposition::Lost
            }
            _ => invalid.schema_version = 1,
        }
        assert!(invalid.validate().is_err());
    }
    let before = fixture.driver.snapshot();
    assert!(fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 9 })
        .is_err());
    assert_eq!(fixture.driver.snapshot(), before);
    let cold: DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    assert_eq!(cold, original);
}

#[test]
fn recovery_loss_keeps_attempt_history_without_inventing_worker_usage() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "recovery-history",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let request = common::submit(&mut fixture.driver, task);
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 5 })
        .unwrap();
    let restored = reconcile_snapshot(fixture.driver.snapshot())
        .unwrap()
        .snapshot();
    let history = restored.task_attempt_history(request.task_id).unwrap();
    assert_eq!(
        history[0].terminal.as_ref().unwrap().disposition,
        AttemptDisposition::Lost
    );
    assert_eq!(
        history[0]
            .terminal
            .as_ref()
            .unwrap()
            .closed_at_driver_millis,
        5
    );
    assert!(restored
        .task_resource_aggregate(request.task_id, 5)
        .is_err());
}

#[test]
fn portable_history_preserves_lost_attempt_and_rejects_lineage_corruption() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "portable-history",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let first = common::submit(&mut fixture.driver, task);
    let second = fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &first,
            AttemptReport::Lost {
                message: "original producer lost".into(),
            },
        )))
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
    let snapshot = fixture.driver.snapshot();
    let original = snapshot.task_execution_evidence(first.task_id).unwrap();
    original.validate().unwrap();
    let cold: runmat_execution_runner::TaskExecutionEvidence =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    assert_eq!(cold, original);
    assert_eq!(
        cold.attempts[0]
            .terminal
            .as_ref()
            .unwrap()
            .diagnostic
            .as_deref(),
        Some("original producer lost")
    );
    assert!(cold.resource_aggregate(1).is_err());
    for corruption in 0..9 {
        let mut invalid = cold.clone();
        match corruption {
            0 => {
                invalid.attempts.remove(0);
            }
            1 => {
                invalid.attempts[1] = invalid.attempts[0].clone();
            }
            2 => invalid.attempts[0].request.driver_fence += 1,
            3 => invalid.task.submission.request.resources.max_wall_millis += 1,
            4 => invalid.task.committed.as_mut().unwrap().attempt_id = first.id,
            5 => invalid.driver_fence = 0,
            6 => invalid.task.attempt_count = 1,
            7 => {
                invalid.task.state = runmat_execution::state::TaskState::Running;
                invalid.task.committed = None;
            }
            _ => {
                invalid.task.state = runmat_execution::state::TaskState::Ready;
                invalid.task.committed = None;
                invalid.task.active_attempt = Some(first.id);
            }
        }
        assert!(invalid.validate().is_err(), "corruption {corruption}");
    }
}
