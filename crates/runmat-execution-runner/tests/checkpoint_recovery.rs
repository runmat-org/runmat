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
    assert_eq!(snapshot.attempts[&request.id].state, AttemptState::Lost);
    assert_eq!(snapshot.pools[&fixture.pool].active_attempts, 0);
}

#[test]
fn recovery_requeues_only_explicitly_replayable_work() {
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
    assert_eq!(recovered.snapshot().tasks[&task_id].state, TaskState::Ready);
}
