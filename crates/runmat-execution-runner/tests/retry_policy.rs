mod common;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, Digest, ProgramCallable};
use runmat_execution_runner::driver::{DriverAction, DriverCommand};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::AttemptReport;

fn lose_once(policy: RetryPolicy) -> (TaskState, usize) {
    let mut fixture = common::fixture(1, 1);
    let submission = common::task("retry", fixture.scope, fixture.pool, policy);
    let task_id = submission.request.id;
    let request = common::submit(&mut fixture.driver, submission);
    let actions = fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Lost {
                message: "lost".into(),
            },
        )))
        .unwrap();
    (
        fixture.driver.snapshot().tasks[&task_id].state,
        actions
            .iter()
            .filter(|action| matches!(action, DriverAction::Launch(_)))
            .count(),
    )
}

#[test]
fn unknown_effect_loss_is_never_silently_replayed() {
    assert_eq!(lose_once(RetryPolicy::Never), (TaskState::Indeterminate, 0));
}

#[test]
fn reviewed_idempotency_and_test_policy_allow_bounded_replay() {
    assert_eq!(
        lose_once(RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 }),
        (TaskState::Assigned, 1)
    );
    assert_eq!(
        lose_once(RetryPolicy::TestPolicy { max_attempts: 2 }),
        (TaskState::Assigned, 1)
    );
    assert_eq!(
        lose_once(RetryPolicy::IdempotentInfrastructure),
        (TaskState::Assigned, 1)
    );
}

#[test]
fn ordinary_execution_failures_are_only_retried_by_test_policy() {
    for (policy, expected_launches) in [
        (RetryPolicy::Never, 0),
        (RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 }, 0),
        (RetryPolicy::TestPolicy { max_attempts: 2 }, 1),
    ] {
        let mut fixture = common::fixture(1, 1);
        let submission = common::task("failure", fixture.scope, fixture.pool, policy);
        let request = common::submit(&mut fixture.driver, submission);
        let actions = fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                AttemptReport::Failed {
                    kind: runmat_execution_runner::AttemptFailureKind::Execution,
                    message: "user code failed".into(),
                },
            )))
            .unwrap();
        assert_eq!(
            actions
                .iter()
                .filter(|action| matches!(action, DriverAction::Launch(_)))
                .count(),
            expected_launches
        );
    }
}

#[test]
fn unverified_worker_retirement_is_indeterminate_even_for_idempotent_work() {
    let mut fixture = common::fixture(1, 1);
    let submission = common::task(
        "retirement-unverified",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let task_id = submission.request.id;
    let request = common::submit(&mut fixture.driver, submission);
    let actions = fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::RetirementUnverified {
                message: "process tree could not be reaped".into(),
            },
        )))
        .unwrap();
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Indeterminate
    );
    assert_eq!(
        fixture.driver.snapshot().attempts[&request.id].state,
        AttemptState::Lost
    );
    assert!(!actions
        .iter()
        .any(|action| matches!(action, DriverAction::Launch(_))));
}

#[test]
fn active_graph_cancellation_waits_for_verified_worker_retirement() {
    let mut fixture = common::fixture(1, 1);
    let mut submission = common::task(
        "graph-cancel",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    submission.request.callable.program = ProgramCallable::graph_target(Digest::sha256(b"root"));
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
    assert!(actions
        .iter()
        .any(|action| matches!(action, DriverAction::Cancel(_))));
    assert!(!matches!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Cancelled | TaskState::Indeterminate
    ));
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].active_attempt,
        Some(request.id)
    );
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 20_000 })
        .unwrap();
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].active_attempt,
        Some(request.id),
        "scheduler escalation must not retire the Graph attempt before its worker"
    );
    assert!(!matches!(
        fixture.driver.snapshot().attempts[&request.id].state,
        AttemptState::Cancelled | AttemptState::Lost
    ));
    let actions = fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::RetirementUnverified {
                message: "tree still live".into(),
            },
        )))
        .unwrap();
    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Indeterminate
    );
    assert_eq!(
        fixture.driver.snapshot().attempts[&request.id].state,
        AttemptState::Lost
    );
    assert!(!actions
        .iter()
        .any(|action| matches!(action, DriverAction::Launch(_))));
}

#[test]
fn only_an_explicit_graph_retirement_report_certifies_cancellation() {
    for (name, report, task_state, attempt_state) in [
        (
            "graph-cancel-lost",
            AttemptReport::Lost {
                message: "worker disappeared".into(),
            },
            TaskState::Indeterminate,
            AttemptState::Lost,
        ),
        (
            "graph-cancel-retired",
            AttemptReport::GraphRetiredCancelled,
            TaskState::Cancelled,
            AttemptState::Cancelled,
        ),
    ] {
        let mut fixture = common::fixture(1, 1);
        let mut submission = common::task(name, fixture.scope, fixture.pool, RetryPolicy::Never);
        submission.request.callable.program =
            ProgramCallable::graph_target(Digest::sha256(name.as_bytes()));
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        fixture
            .driver
            .handle(DriverCommand::CancelScope {
                scope_id: fixture.scope,
                reason: CancellationReason::User,
                now_millis: 100,
            })
            .unwrap();
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request, report,
            )))
            .unwrap();
        assert_eq!(fixture.driver.snapshot().tasks[&task_id].state, task_state);
        assert_eq!(
            fixture.driver.snapshot().attempts[&request.id].state,
            attempt_state
        );
    }
}
