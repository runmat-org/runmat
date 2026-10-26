mod common;

use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, ExecutionScopeId};
use runmat_execution_runner::driver::{DriverAction, DriverCommand};

fn retired(
    driver: &mut runmat_execution_runner::Driver,
    request: &runmat_execution_runner::AttemptRequest,
) {
    driver
        .handle(DriverCommand::BackendReport(
            runmat_execution_runner::port::BackendReport::for_request(
                request,
                runmat_execution_runner::AttemptReport::Retired,
            ),
        ))
        .unwrap();
}

#[test]
fn parent_cancellation_cascades_and_requests_active_attempt_cancellation() {
    let mut fixture = common::fixture(1, 1);
    let child = ExecutionScopeId::derive(&[b"child"]);
    fixture
        .driver
        .handle(DriverCommand::RegisterScope {
            scope_id: child,
            parent: Some(fixture.scope),
        })
        .unwrap();
    let submission = common::task("child-task", child, fixture.pool, RetryPolicy::Never);
    let task_id = submission.request.id;
    let request = common::submit(&mut fixture.driver, submission);
    let actions = fixture
        .driver
        .handle(DriverCommand::CancelScope {
            scope_id: fixture.scope,
            reason: CancellationReason::User,
            now_millis: 100,
        })
        .unwrap();
    assert_ne!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Cancelled
    );
    retired(&mut fixture.driver, &request);
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Cancelled
    );
    assert_eq!(
        actions
            .iter()
            .filter(|action| matches!(action, DriverAction::Cancel(_)))
            .count(),
        1
    );
}

#[test]
fn task_cancellation_retire_only_the_selected_active_or_queued_task() {
    let mut fixture = common::fixture(1, 1);
    let active = common::task("active", fixture.scope, fixture.pool, RetryPolicy::Never);
    let active_id = active.request.id;
    let request = common::submit(&mut fixture.driver, active);
    let queued = common::task("queued", fixture.scope, fixture.pool, RetryPolicy::Never);
    let queued_id = queued.request.id;
    fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(queued)))
        .unwrap();

    let queued_actions = fixture
        .driver
        .handle(DriverCommand::CancelTask {
            task_id: queued_id,
            reason: CancellationReason::User,
            now_millis: 10,
        })
        .unwrap();
    assert!(queued_actions.is_empty());
    assert_eq!(
        fixture.driver.snapshot().tasks[&queued_id].state,
        TaskState::Cancelled
    );
    assert_ne!(
        fixture.driver.snapshot().tasks[&active_id].state,
        TaskState::Cancelled
    );

    let active_actions = fixture
        .driver
        .handle(DriverCommand::CancelTask {
            task_id: active_id,
            reason: CancellationReason::User,
            now_millis: 11,
        })
        .unwrap();
    assert_eq!(
        active_actions
            .iter()
            .filter(|action| matches!(action, DriverAction::Cancel(_)))
            .count(),
        1
    );
    assert_ne!(
        fixture.driver.snapshot().tasks[&active_id].state,
        TaskState::Cancelled
    );
    retired(&mut fixture.driver, &request);
    assert_eq!(
        fixture.driver.snapshot().tasks[&active_id].state,
        TaskState::Cancelled
    );
}

#[test]
fn expired_deadline_cancels_only_the_expired_task() {
    let mut fixture = common::fixture(1, 1);
    let mut first = common::task("deadline", fixture.scope, fixture.pool, RetryPolicy::Never);
    first.request.deadline_unix_millis = Some(50);
    let first_id = first.request.id;
    let request = common::submit(&mut fixture.driver, first);
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 50 })
        .unwrap();
    assert_ne!(
        fixture.driver.snapshot().tasks[&first_id].state,
        TaskState::Cancelled
    );
    retired(&mut fixture.driver, &request);
    assert_eq!(
        fixture.driver.snapshot().tasks[&first_id].state,
        TaskState::Cancelled
    );
}

#[test]
fn maximum_wall_time_and_cancellation_escalation_are_deterministic() {
    let mut fixture = common::fixture(1, 1);
    let submission = common::task("wall", fixture.scope, fixture.pool, RetryPolicy::Never);
    let task_id = submission.request.id;
    let request = common::submit(&mut fixture.driver, submission);

    let cancel = fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 10_000 })
        .unwrap();
    assert!(cancel
        .iter()
        .any(|action| matches!(action, DriverAction::Cancel(_))));
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Assigned
    );

    let terminate = fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 12_000 })
        .unwrap();
    assert!(terminate
        .iter()
        .any(|action| matches!(action, DriverAction::Terminate(_))));

    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 20_000 })
        .unwrap();
    assert_eq!(
        fixture.driver.snapshot().pools[&fixture.pool].active_attempts,
        1
    );
    retired(&mut fixture.driver, &request);
    assert_eq!(
        fixture.driver.snapshot().pools[&fixture.pool].active_attempts,
        0
    );
}

#[test]
fn direct_task_cancellation_preserves_reason_isolates_siblings_and_validates_cold_lineage() {
    let mut fixture = common::fixture(2, 2);
    let first = common::task("direct", fixture.scope, fixture.pool, RetryPolicy::Never);
    let first_id = first.request.id;
    let _first = common::submit(&mut fixture.driver, first);
    let second = common::task("sibling", fixture.scope, fixture.pool, RetryPolicy::Never);
    let second_id = second.request.id;
    let _second = common::submit(&mut fixture.driver, second);
    let actions = fixture
        .driver
        .handle(DriverCommand::CancelTask {
            task_id: first_id,
            reason: CancellationReason::Quota,
            now_millis: 100,
        })
        .unwrap();
    assert_eq!(actions.iter().filter(|action| matches!(action, DriverAction::Cancel(request) if request.task_id == first_id)).count(), 1);
    let snapshot = fixture.driver.snapshot();
    assert_eq!(snapshot.tasks[&first_id].state, TaskState::Assigned);
    assert_ne!(snapshot.tasks[&second_id].state, TaskState::Cancelled);
    assert!(
        matches!(snapshot.tasks[&first_id].cancellation.unwrap().cause, runmat_execution_runner::TaskCancellationCause::Task { task_id, reason: CancellationReason::Quota } if task_id == first_id)
    );
    let cold: runmat_execution_runner::DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
    cold.validate().unwrap();
    let mut foreign = cold.clone();
    foreign
        .tasks
        .get_mut(&first_id)
        .unwrap()
        .cancellation
        .as_mut()
        .unwrap()
        .cause = runmat_execution_runner::TaskCancellationCause::Task {
        task_id: second_id,
        reason: CancellationReason::Quota,
    };
    assert!(foreign.validate().is_err());
    let mut wrong_reason = cold;
    wrong_reason
        .tasks
        .get_mut(&first_id)
        .unwrap()
        .cancellation
        .as_mut()
        .unwrap()
        .cause = runmat_execution_runner::TaskCancellationCause::Task {
        task_id: first_id,
        reason: CancellationReason::User,
    };
    assert!(wrong_reason.validate().is_err());
    fixture
        .driver
        .handle(DriverCommand::CancelTask {
            task_id: first_id,
            reason: CancellationReason::User,
            now_millis: 101,
        })
        .unwrap();
    assert_eq!(
        fixture.driver.snapshot().tasks[&first_id].cancellation,
        snapshot.tasks[&first_id].cancellation
    );
}
