mod common;

use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, ExecutionScopeId};
use runmat_execution_runner::driver::{DriverAction, DriverCommand};

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
    let _request = common::submit(&mut fixture.driver, submission);
    let actions = fixture
        .driver
        .handle(DriverCommand::CancelScope {
            scope_id: fixture.scope,
            reason: CancellationReason::User,
            now_millis: 100,
        })
        .unwrap();
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
fn expired_deadline_cancels_only_the_expired_task() {
    let mut fixture = common::fixture(1, 1);
    let mut first = common::task("deadline", fixture.scope, fixture.pool, RetryPolicy::Never);
    first.request.deadline_unix_millis = Some(50);
    let first_id = first.request.id;
    let _request = common::submit(&mut fixture.driver, first);
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 50 })
        .unwrap();
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
    let _request = common::submit(&mut fixture.driver, submission);

    let cancel = fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 10_000 })
        .unwrap();
    assert!(cancel
        .iter()
        .any(|action| matches!(action, DriverAction::Cancel(_))));
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Cancelled
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
        0
    );
}
