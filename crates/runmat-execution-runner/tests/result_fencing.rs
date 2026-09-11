mod common;

use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverEventKind};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::AttemptReport;

#[test]
fn duplicate_reordered_and_stale_reports_commit_at_most_once() {
    for permutation in 0..32_u64 {
        let mut fixture = common::fixture(1, 1);
        let submission = common::task("fenced", fixture.scope, fixture.pool, RetryPolicy::Never);
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        let success = BackendReport::for_request(&request, common::success(&request));
        let started =
            BackendReport::for_request(&request, runmat_execution_runner::AttemptReport::Started);
        let order = if permutation & 1 == 0 {
            vec![started, success.clone(), success.clone()]
        } else {
            vec![success.clone(), started, success.clone()]
        };
        for report in order {
            fixture
                .driver
                .handle(DriverCommand::BackendReport(report))
                .unwrap();
        }
        let mut stale = success;
        stale.driver_fence = request.driver_fence.saturating_sub(1);
        let actions = fixture
            .driver
            .handle(DriverCommand::BackendReport(stale))
            .unwrap();
        assert!(actions
            .iter()
            .all(|action| !matches!(action, DriverAction::Launch(_))));
        let snapshot = fixture.driver.snapshot();
        assert_eq!(snapshot.tasks[&task_id].state, TaskState::Succeeded);
        assert_eq!(
            snapshot
                .events
                .iter()
                .filter(|event| matches!(event.kind, DriverEventKind::ResultCommitted { .. }))
                .count(),
            1
        );
    }
}

#[test]
fn scheduler_rejects_missing_or_substituted_attempt_resource_accounting() {
    for substitute in [false, true] {
        let mut fixture = common::fixture(1, 1);
        let submission = common::task("metered", fixture.scope, fixture.pool, RetryPolicy::Never);
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        let mut report = common::success(&request);
        let AttemptReport::Succeeded { result } = &mut report else {
            unreachable!()
        };
        if substitute {
            result
                .resources
                .as_mut()
                .unwrap()
                .resources
                .allocation
                .memory_bytes += 1;
        } else {
            result.resources = None;
        }
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request, report,
            )))
            .unwrap();
        let snapshot = fixture.driver.snapshot();
        assert_eq!(snapshot.tasks[&task_id].state, TaskState::Failed);
        assert!(snapshot.tasks[&task_id].committed.is_none());
    }
}
