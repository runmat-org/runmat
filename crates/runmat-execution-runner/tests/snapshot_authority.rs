mod common;

use runmat_execution::identity::{AttemptId, ResultCommitId, WorkerId};
use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::task::RetryPolicy;
use runmat_execution::TaskId;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::recovery::reconcile_snapshot;
use runmat_execution_runner::{AttemptReport, DriverAction, DriverCommand, DriverSnapshot};

fn completed() -> (DriverSnapshot, TaskId, AttemptId) {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "snapshot-authority",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let request = common::submit(&mut fixture.driver, task);
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            common::success(&request),
        )))
        .unwrap();
    (fixture.driver.snapshot(), request.task_id, request.id)
}

#[test]
fn valid_commit_retains_original_accounting_across_repeated_fence_changes() {
    let (snapshot, task, _) = completed();
    let commit = snapshot.tasks[&task].committed.clone();
    let original_fence = snapshot.driver_fence;
    let mut restored = snapshot;
    for increment in 1..=3 {
        restored = reconcile_snapshot(restored).unwrap().snapshot();
        assert_eq!(restored.driver_fence, original_fence + increment);
        assert_eq!(restored.tasks[&task].committed, commit);
        assert_eq!(restored.tasks[&task].state, TaskState::Succeeded);
    }
}

#[test]
fn restored_success_rejects_missing_or_substituted_commit_accounting() {
    let (snapshot, task, attempt) = completed();
    for corruption in 0..7 {
        let mut invalid = snapshot.clone();
        let commit = invalid
            .tasks
            .get_mut(&task)
            .unwrap()
            .committed
            .as_mut()
            .unwrap();
        match corruption {
            0..=2 => {
                let runmat_execution_runner::task::AttemptSuccess::Values { resources, .. } =
                    &mut commit.result
                else {
                    panic!("fixture commits a metered value result");
                };
                match corruption {
                    0 => *resources = None,
                    1 => resources.as_mut().unwrap().worker_id = WorkerId::derive(&[b"foreign"]),
                    _ => {
                        resources
                            .as_mut()
                            .unwrap()
                            .resources
                            .allocation
                            .memory_bytes += 1
                    }
                }
            }
            3 => commit.id = ResultCommitId::derive(&[b"foreign"]),
            4 => commit.driver_fence += 1,
            5 => commit.attempt_id = AttemptId::derive(&[b"foreign"]),
            _ => invalid.attempts.get_mut(&attempt).unwrap().state = AttemptState::Lost,
        }
        assert!(
            reconcile_snapshot(invalid).is_err(),
            "corruption {corruption}"
        );
    }
}

#[test]
fn restored_task_state_must_agree_with_its_commit() {
    let (snapshot, task, attempt) = completed();
    let mut missing = snapshot.clone();
    missing.tasks.get_mut(&task).unwrap().committed = None;
    assert!(reconcile_snapshot(missing).is_err());
    let mut active = snapshot.clone();
    active.tasks.get_mut(&task).unwrap().active_attempt = Some(attempt);
    assert!(reconcile_snapshot(active).is_err());
    for state in [
        TaskState::Failed,
        TaskState::Cancelled,
        TaskState::Indeterminate,
    ] {
        let mut invalid = snapshot.clone();
        invalid.tasks.get_mut(&task).unwrap().state = state;
        assert!(reconcile_snapshot(invalid).is_err());
    }
}

#[test]
fn restored_attempt_and_dependencies_must_match_admission() {
    let (snapshot, task, attempt) = completed();
    let mut dependency = snapshot.clone();
    dependency
        .tasks
        .get_mut(&task)
        .unwrap()
        .submission
        .dependencies
        .insert(task);
    assert!(reconcile_snapshot(dependency).is_err());
    let mut count = snapshot.clone();
    count.tasks.get_mut(&task).unwrap().attempt_count += 1;
    assert!(reconcile_snapshot(count).is_err());
    for corruption in 0..4 {
        let mut invalid = snapshot.clone();
        let request = &mut invalid.attempts.get_mut(&attempt).unwrap().request;
        match corruption {
            0 => {
                request.task.callable.program = runmat_execution::ProgramCallable::semantic(
                    runmat_execution::ProgramFunctionId(1),
                    Some("substituted".into()),
                );
            }
            1 => request.ordinal += 1,
            2 => request.driver_fence = 0,
            _ => request.scope_id = runmat_execution::ExecutionScopeId::derive(&[b"foreign"]),
        }
        assert!(
            reconcile_snapshot(invalid).is_err(),
            "corruption {corruption}"
        );
    }
}

#[test]
fn restored_reverse_graph_index_cannot_be_substituted() {
    let (snapshot, task, _) = completed();
    let mut wire = serde_json::to_value(&snapshot).unwrap();
    wire["graph"]["dependents"][task.to_string()] = serde_json::json!([task]);
    let corrupted: DriverSnapshot = serde_json::from_value(wire).unwrap();
    assert!(reconcile_snapshot(corrupted).is_err());
}

#[test]
fn cancelled_live_attempt_retains_reservation_and_original_cause_after_recovery() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "cancelled-recovery",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let request = common::submit(&mut fixture.driver, task);
    fixture
        .driver
        .handle(DriverCommand::CancelScope {
            scope_id: fixture.scope,
            reason: runmat_execution::CancellationReason::User,
            now_millis: 1,
        })
        .unwrap();
    let snapshot = fixture.driver.snapshot();
    assert_eq!(snapshot.tasks[&request.task_id].state, TaskState::Assigned);
    assert_eq!(snapshot.attempts[&request.id].state, AttemptState::Assigned);
    let restored = reconcile_snapshot(snapshot).unwrap().snapshot();
    assert_eq!(
        restored.tasks[&request.task_id].state,
        TaskState::Indeterminate
    );
    assert_eq!(restored.attempts[&request.id].state, AttemptState::Assigned);
    assert!(restored.tasks[&request.task_id].committed.is_none());
}

#[test]
fn completed_retry_preserves_both_attempts_without_resetting_budget() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "retried-snapshot",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let first = common::submit(&mut fixture.driver, task);
    let second = fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &first,
            AttemptReport::Failed {
                kind: runmat_execution_runner::AttemptFailureKind::Infrastructure,
                message: "retired first attempt failed".into(),
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
    let restored = reconcile_snapshot(fixture.driver.snapshot())
        .unwrap()
        .snapshot();
    assert_eq!(restored.tasks[&first.task_id].attempt_count, 2);
    assert_eq!(restored.attempts.len(), 2);
    assert_eq!(
        restored.tasks[&first.task_id]
            .committed
            .as_ref()
            .unwrap()
            .attempt_id,
        second.id
    );
    assert_eq!(restored.attempts[&first.id].state, AttemptState::Completed);
}

#[test]
fn internally_consistent_cycle_and_duplicate_ordinals_are_rejected() {
    let (snapshot, task, attempt) = completed();
    let mut cycle = snapshot.clone();
    cycle
        .tasks
        .get_mut(&task)
        .unwrap()
        .submission
        .dependencies
        .insert(task);
    let mut wire = serde_json::to_value(cycle).unwrap();
    wire["graph"]["dependencies"][task.to_string()] = serde_json::json!([task]);
    wire["graph"]["dependents"][task.to_string()] = serde_json::json!([task]);
    assert!(reconcile_snapshot(serde_json::from_value(wire).unwrap()).is_err());

    let mut duplicate = snapshot.clone();
    let mut old = duplicate.attempts[&attempt].clone();
    old.request.driver_fence -= 1;
    old.request.id = AttemptId::derive(&[
        task.bytes().as_slice(),
        old.request.ordinal.to_be_bytes().as_slice(),
        old.request.driver_fence.to_be_bytes().as_slice(),
    ]);
    old.state = AttemptState::Lost;
    duplicate.attempts.insert(old.request.id, old);
    duplicate.tasks.get_mut(&task).unwrap().attempt_count = 2;
    assert!(reconcile_snapshot(duplicate).is_err());
}
