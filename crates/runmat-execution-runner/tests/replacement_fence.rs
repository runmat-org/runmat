mod common;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::recovery::{reconcile_snapshot_at_fence, DriverCheckpoint};
use runmat_execution_runner::DriverCommand;

#[test]
fn replacement_requires_a_strictly_newer_nonzero_fence() {
    let snapshot = common::fixture(1, 1).driver.snapshot();
    for fence in [0, snapshot.driver_fence - 1, snapshot.driver_fence] {
        assert!(reconcile_snapshot_at_fence(snapshot.clone(), fence).is_err());
        assert!(DriverCheckpoint::new(3, snapshot.clone())
            .restore_at_fence(fence)
            .is_err());
    }
    let checkpoint = DriverCheckpoint::new(3, snapshot.clone());
    assert_eq!(
        checkpoint
            .clone()
            .restore()
            .unwrap()
            .snapshot()
            .driver_fence,
        snapshot.driver_fence + 1
    );
    assert_eq!(
        checkpoint
            .restore_at_fence(100)
            .unwrap()
            .snapshot()
            .driver_fence,
        100
    );
}

#[test]
fn checkpoint_schema_and_exhausted_local_fence_fail_closed() {
    let snapshot = common::fixture(1, 1).driver.snapshot();
    for schema_version in [1, 3] {
        let mut invalid = DriverCheckpoint::new(1, snapshot.clone());
        invalid.schema_version = schema_version;
        assert!(invalid.clone().restore().is_err());
        assert!(invalid.restore_at_fence(100).is_err());
    }
    let highest = reconcile_snapshot_at_fence(snapshot, u64::MAX)
        .unwrap()
        .snapshot();
    assert!(DriverCheckpoint::new(1, highest.clone()).restore().is_err());
    assert!(DriverCheckpoint::new(1, highest)
        .restore_at_fence(u64::MAX)
        .is_err());
}

#[test]
fn jumping_fence_preserves_committed_authority_and_reconciles_only_live_attempts() {
    let mut fixture = common::fixture(1, 1);
    let committed = common::task("committed", fixture.scope, fixture.pool, RetryPolicy::Never);
    let completed_request = common::submit(&mut fixture.driver, committed);
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &completed_request,
            common::success(&completed_request),
        )))
        .unwrap();
    let mut pending = common::task(
        "pending",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    pending.request.deadline_unix_millis = Some(50_000);
    pending.priority = 9;
    let active_request = common::submit(&mut fixture.driver, pending);
    let before = fixture.driver.snapshot();
    let after = DriverCheckpoint::new(8, before.clone())
        .restore_at_fence(100)
        .unwrap()
        .snapshot();
    assert_eq!(after.driver_fence, 100);
    assert_eq!(
        after.tasks[&completed_request.task_id],
        before.tasks[&completed_request.task_id]
    );
    assert_eq!(
        after.attempts[&completed_request.id],
        before.attempts[&completed_request.id]
    );
    let task = &after.tasks[&active_request.task_id];
    assert_eq!(task.state, TaskState::Indeterminate);
    assert_eq!(
        task.submission,
        before.tasks[&active_request.task_id].submission
    );
    assert_eq!(task.attempt_count, 1);
    assert_eq!(task.active_attempt, Some(active_request.id));
    assert_eq!(after.attempts[&active_request.id].request, active_request);
    assert_eq!(
        after.attempts[&active_request.id].state,
        AttemptState::Assigned
    );
    assert_ne!(after.deadlines, before.deadlines);
    assert_eq!(
        &after.events[..before.events.len()],
        before.events.as_slice()
    );
    assert!(after.events[before.events.len()..]
        .iter()
        .all(|event| event.driver_fence == 100));
}

#[test]
fn higher_authority_does_not_replenish_retry_budget_or_replay_unknown_effects() {
    for (retry, terminal) in [
        (
            RetryPolicy::ExplicitlyIdempotent { max_attempts: 1 },
            TaskState::Indeterminate,
        ),
        (RetryPolicy::Never, TaskState::Indeterminate),
    ] {
        let mut fixture = common::fixture(1, 1);
        let mut submission = common::task("bounded", fixture.scope, fixture.pool, retry);
        submission.request.deadline_unix_millis = Some(50_000);
        let request = common::submit(&mut fixture.driver, submission);
        let before = fixture.driver.snapshot();
        let after = reconcile_snapshot_at_fence(before.clone(), 100)
            .unwrap()
            .snapshot();
        let task = &after.tasks[&request.task_id];
        assert_eq!(task.state, terminal);
        assert_eq!(task.attempt_count, 1);
        assert_eq!(task.submission, before.tasks[&request.task_id].submission);
        assert_eq!(after.attempts[&request.id].request, request);
        assert_eq!(after.attempts[&request.id].state, AttemptState::Assigned);
    }
}
