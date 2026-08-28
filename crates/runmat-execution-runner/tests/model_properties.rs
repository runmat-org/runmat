mod common;

use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::driver::{DriverCommand, DriverEventKind};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::testing::ReferenceModel;
use runmat_execution_runner::AttemptReport;

#[test]
fn randomized_report_order_preserves_model_and_commit_invariants() {
    for seed in 1..=256_u64 {
        let mut fixture = common::fixture(1, 1);
        let submission = common::task(
            &format!("property-{seed}"),
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        );
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        let mut state = seed;
        for _ in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut report = match state % 3 {
                0 => BackendReport::for_request(&request, AttemptReport::Started),
                _ => BackendReport::for_request(&request, common::success()),
            };
            if state & 8 != 0 {
                report.driver_fence = report.driver_fence.saturating_sub(1);
            }
            fixture
                .driver
                .handle(DriverCommand::BackendReport(report))
                .unwrap();
        }
        let snapshot = fixture.driver.snapshot();
        assert!(snapshot
            .events
            .windows(2)
            .all(|pair| pair[0].sequence + 1 == pair[1].sequence));
        assert!(
            snapshot
                .events
                .iter()
                .filter(|event| matches!(event.kind, DriverEventKind::ResultCommitted { .. }))
                .count()
                <= 1
        );
        let mut model = ReferenceModel::default();
        for event in &snapshot.events {
            model.apply(event);
        }
        assert_eq!(
            model.tasks.get(&task_id),
            Some(&snapshot.tasks[&task_id].state)
        );
        assert!(matches!(
            snapshot.tasks[&task_id].state,
            TaskState::Assigned | TaskState::Running | TaskState::Succeeded
        ));
    }
}

#[test]
fn fairness_forces_a_lower_priority_opportunity_after_each_burst() {
    let policy = runmat_execution_runner::scheduler::FairnessPolicy {
        max_priority_burst: 3,
    };
    let mut state = runmat_execution_runner::scheduler::FairnessState::default();
    let selections = (0..8)
        .map(|_| state.select_priority([10, 10, 0], policy).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(selections, vec![10, 10, 10, 0, 10, 10, 10, 0]);
}

#[test]
fn randomized_loss_policy_never_replays_unknown_effects() {
    for seed in 0..256_u64 {
        let policy = match seed % 4 {
            0 => RetryPolicy::Never,
            1 => RetryPolicy::IdempotentInfrastructure,
            2 => RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            _ => RetryPolicy::TestPolicy { max_attempts: 2 },
        };
        let mut fixture = common::fixture(1, 1);
        let submission = common::task(&format!("loss-{seed}"), fixture.scope, fixture.pool, policy);
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        let actions = fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                AttemptReport::Lost {
                    message: "randomized loss".into(),
                },
            )))
            .unwrap();
        let replayed = actions.iter().any(|action| {
            matches!(
                action,
                runmat_execution_runner::driver::DriverAction::Launch(_)
            )
        });
        if policy == RetryPolicy::Never {
            assert!(!replayed);
            assert_eq!(
                fixture.driver.snapshot().tasks[&task_id].state,
                TaskState::Indeterminate
            );
        } else {
            assert!(replayed);
        }
    }
}
