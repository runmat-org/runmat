mod common;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::recovery::reconcile_snapshot;

#[test]
fn recovery_fences_old_attempts_and_marks_unknown_effects_indeterminate() {
    let mut fixture = common::fixture(1, 1);
    let submission = common::task(
        "recover-never",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let task_id = submission.request.id;
    let request = common::submit(&mut fixture.driver, submission);
    let old_fence = fixture.driver.snapshot().driver_fence;
    let recovered = reconcile_snapshot(fixture.driver.snapshot()).unwrap();
    let snapshot = recovered.snapshot();
    assert_eq!(snapshot.driver_fence, old_fence + 1);
    assert_eq!(snapshot.tasks[&task_id].state, TaskState::Indeterminate);
    assert_eq!(snapshot.attempts[&request.id].state, AttemptState::Assigned);
    assert_eq!(snapshot.pools[&fixture.pool].active_attempts, 1);
}

#[test]
fn recovery_retains_even_explicitly_replayable_work_until_retirement() {
    let mut fixture = common::fixture(1, 1);
    let submission = common::task(
        "recover-safe",
        fixture.scope,
        fixture.pool,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
    );
    let task_id = submission.request.id;
    let _request = common::submit(&mut fixture.driver, submission);
    let recovered = reconcile_snapshot(fixture.driver.snapshot()).unwrap();
    assert_eq!(
        recovered.snapshot().tasks[&task_id].state,
        TaskState::Indeterminate
    );
}

#[test]
fn exclusive_continuation_preserves_original_assignment_while_replacement_retains_it() {
    use runmat_execution_runner::port::BackendReport;
    use runmat_execution_runner::recovery::DriverCheckpoint;
    use runmat_execution_runner::{AttemptReport, DriverCommand};
    let mut fixture = common::fixture(1, 1);
    let submission = common::task(
        "exclusive-continuation",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let request = common::submit(&mut fixture.driver, submission);
    let original = fixture.driver.snapshot();
    let checkpoint = DriverCheckpoint::new(1, original.clone());
    let wire = serde_json::to_vec(&checkpoint).unwrap();
    let cold: DriverCheckpoint = serde_json::from_slice(&wire).unwrap();
    let mut continued = cold.continue_under_exclusive_ownership().unwrap();
    assert_eq!(continued.snapshot(), original);
    let report = BackendReport::for_request(&request, AttemptReport::Started);
    assert!(continued.accepts_backend_report(&report));
    continued
        .handle(DriverCommand::BackendReport(report))
        .unwrap();
    assert_eq!(continued.snapshot().attempts[&request.id].request, request);
    let replaced = checkpoint.restore().unwrap();
    assert_eq!(replaced.snapshot().driver_fence, original.driver_fence + 1);
    assert_eq!(
        replaced.snapshot().attempts[&request.id].state,
        AttemptState::Assigned
    );
}
