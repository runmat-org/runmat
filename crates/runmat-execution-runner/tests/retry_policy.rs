mod common;

use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
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
