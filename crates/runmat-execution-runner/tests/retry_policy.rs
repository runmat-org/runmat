mod common;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, Digest, ProgramCallable};
use runmat_execution_runner::driver::{DriverAction, DriverCommand};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::AttemptReport;
use runmat_execution_runner::{AttemptDisposition, DriverSnapshot};

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
fn idempotency_cannot_replay_a_possibly_live_lost_attempt() {
    assert_eq!(
        lose_once(RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 }),
        (TaskState::Indeterminate, 0)
    );
    assert_eq!(
        lose_once(RetryPolicy::TestPolicy { max_attempts: 2 }),
        (TaskState::Indeterminate, 0)
    );
    assert_eq!(
        lose_once(RetryPolicy::IdempotentInfrastructure),
        (TaskState::Indeterminate, 0)
    );
}

#[test]
fn certified_never_started_original_retries_and_reopens_only_its_dependency() {
    let mut fixture = common::fixture(2, 1);
    let parent = common::task(
        "certified-original",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let parent_id = parent.request.id;
    let mut child = common::task("dependent", fixture.scope, fixture.pool, RetryPolicy::Never);
    child.dependencies.insert(parent_id);
    let child_id = child.request.id;
    let original = common::submit(&mut fixture.driver, parent);
    fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(child)))
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &original,
            AttemptReport::RetirementUnverified {
                message: "original allocation lost before start proof".into(),
            },
        )))
        .unwrap();
    let saved = fixture.driver.snapshot();
    assert_eq!(saved.tasks[&child_id].state, TaskState::Indeterminate);
    assert_eq!(
        saved.tasks[&child_id].indeterminate_dependency,
        Some(parent_id)
    );
    let cold: DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&saved).unwrap()).unwrap();
    cold.validate().unwrap();
    let mut successor = runmat_execution_runner::recovery::reconcile_snapshot(cold).unwrap();
    let certified = BackendReport::for_request(&original, AttemptReport::CertifiedNotStarted {});
    let mut extra_report_field = serde_json::to_value(&certified).unwrap();
    extra_report_field["report"]["resources"] = serde_json::json!({});
    assert!(serde_json::from_value::<BackendReport>(extra_report_field).is_err());
    assert!(successor.accepts_backend_report(&certified));
    let mut with_accounting = certified.clone();
    if let AttemptReport::Succeeded { result } = common::success(&original) {
        with_accounting.unsuccessful_resources = result.resources().cloned().map(Box::new);
    }
    assert!(!successor.accepts_backend_report(&with_accounting));
    successor
        .handle(DriverCommand::BackendReport(certified.clone()))
        .unwrap();
    let recovered = successor.snapshot();
    recovered.validate().unwrap();
    assert_eq!(recovered.tasks[&parent_id].state, TaskState::Ready);
    assert_eq!(recovered.tasks[&child_id].state, TaskState::Deferred);
    assert_eq!(recovered.tasks[&child_id].indeterminate_dependency, None);
    assert_eq!(
        recovered.attempts[&original.id]
            .terminal
            .as_ref()
            .unwrap()
            .disposition,
        AttemptDisposition::CertifiedNotStarted
    );
    let mut forged_cold = recovered.clone();
    if let AttemptReport::Succeeded { result } = common::success(&original) {
        forged_cold
            .attempts
            .get_mut(&original.id)
            .unwrap()
            .terminal
            .as_mut()
            .unwrap()
            .resources = result.resources().cloned().map(Box::new);
    }
    assert!(forged_cold.validate().is_err());
    assert!(recovered.task_resource_aggregate(parent_id, 1).is_err());

    let retry = fixture
        .driver
        .handle(DriverCommand::BackendReport(certified))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) if request.task_id == parent_id => Some(request),
            _ => None,
        })
        .unwrap();
    assert_eq!(retry.ordinal, 2);
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &retry,
            common::success(&retry),
        )))
        .unwrap();
    let completed = fixture.driver.snapshot();
    completed.validate().unwrap();
    assert_eq!(completed.tasks[&child_id].state, TaskState::Assigned);
    let aggregate = completed.task_resource_aggregate(parent_id, 1).unwrap();
    assert_eq!(aggregate.attempts.len(), 1);
    assert_eq!(aggregate.attempts[0].attempt_id, retry.id);
}

#[test]
fn certified_parent_reassigns_another_unknown_dependency_and_honors_expired_child() {
    let mut fixture = common::fixture(3, 2);
    let left = common::task(
        "left-original",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let right = common::task(
        "right-original",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let left_id = left.request.id;
    let right_id = right.request.id;
    let left = common::submit(&mut fixture.driver, left);
    let right = common::submit(&mut fixture.driver, right);
    let mut child = common::task(
        "both-dependent",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    child.dependencies.extend([left_id, right_id]);
    child.request.deadline_unix_millis = Some(2);
    let child_id = child.request.id;
    fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(child)))
        .unwrap();
    for request in [&left, &right] {
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                request,
                AttemptReport::RetirementUnverified {
                    message: "original admission is uncertain".into(),
                },
            )))
            .unwrap();
    }
    fixture
        .driver
        .handle(DriverCommand::Tick { now_millis: 2 })
        .unwrap();
    let cause = fixture.driver.snapshot().tasks[&child_id]
        .indeterminate_dependency
        .unwrap();
    let first = if cause == left_id { &left } else { &right };
    let second = if cause == left_id { &right } else { &left };
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            first,
            AttemptReport::CertifiedNotStarted {},
        )))
        .unwrap();
    let one_recovered = fixture.driver.snapshot();
    one_recovered.validate().unwrap();
    assert_eq!(
        one_recovered.tasks[&child_id].state,
        TaskState::Indeterminate
    );
    assert_eq!(
        one_recovered.tasks[&child_id].indeterminate_dependency,
        Some(second.task_id)
    );
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            second,
            AttemptReport::CertifiedNotStarted {},
        )))
        .unwrap();
    let both_recovered = fixture.driver.snapshot();
    both_recovered.validate().unwrap();
    assert_eq!(both_recovered.tasks[&child_id].state, TaskState::Cancelled);
    assert_eq!(
        both_recovered.tasks[&child_id].indeterminate_dependency,
        None
    );
    assert!(matches!(
        both_recovered.tasks[&child_id].cancellation.unwrap().cause,
        runmat_execution_runner::TaskCancellationCause::Deadline
    ));
}

#[test]
fn physically_retired_begun_attempt_retries_only_with_original_usage_and_retryable_cause() {
    for (kind, retries) in [
        (
            runmat_execution_runner::AttemptFailureKind::Infrastructure,
            true,
        ),
        (
            runmat_execution_runner::AttemptFailureKind::Execution,
            false,
        ),
    ] {
        let mut fixture = common::fixture(2, 1);
        let request = common::submit(
            &mut fixture.driver,
            common::task(
                "begun-original",
                fixture.scope,
                fixture.pool,
                RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            ),
        );
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                AttemptReport::Started,
            )))
            .unwrap();
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                AttemptReport::RetirementUnverified {
                    message: "physical completion was not yet proved".into(),
                },
            )))
            .unwrap();
        let mut measured = BackendReport::for_request(
            &request,
            AttemptReport::Failed {
                kind,
                message: "original worker outcome".into(),
            },
        );
        measured.unsuccessful_resources = Some(Box::new(
            runmat_execution_runner::AttemptResourceAccounting::for_attempt(&request, 1, 0, 0, 0)
                .unwrap(),
        ));
        fixture
            .driver
            .handle(DriverCommand::BackendReport(measured))
            .unwrap();
        let actions = fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                AttemptReport::Retired,
            )))
            .unwrap();
        let state = fixture.driver.snapshot();
        state.validate().unwrap();
        assert_eq!(
            state.attempts[&request.id]
                .terminal
                .as_ref()
                .unwrap()
                .disposition,
            AttemptDisposition::Failed { kind }
        );
        let retry = actions.into_iter().find_map(|action| match action {
            DriverAction::Launch(next) if next.task_id == request.task_id => Some(next),
            _ => None,
        });
        assert_eq!(retry.is_some(), retries);
        if let Some(retry) = retry {
            assert_eq!(retry.ordinal, 2);
            fixture
                .driver
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    &retry,
                    common::success(&retry),
                )))
                .unwrap();
            let aggregate = fixture
                .driver
                .snapshot()
                .task_resource_aggregate(request.task_id, 2)
                .unwrap();
            assert_eq!(aggregate.attempts.len(), 2);
        } else {
            assert_eq!(state.tasks[&request.task_id].state, TaskState::Failed);
        }
    }
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
        AttemptState::Assigned
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
        AttemptState::Assigned
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
            AttemptState::Assigned,
        ),
        (
            "graph-cancel-retired",
            AttemptReport::Retired,
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
