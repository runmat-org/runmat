mod common;

use runmat_execution::task::RetryPolicy;
use runmat_execution::{
    CancellationReason, GangHandle, GangId, PoolHandle, ProgramCallable, ProgramInvocationContext,
    SpmdTaskContext, TaskId,
};
use runmat_execution_runner::{DriverAction, DriverCommand};
use runmat_types::{LabCount, LabRank, ParallelRegionId, ProgramFunctionId, RegionId};

fn gang(fixture: &common::Fixture, generation: u64) -> (GangHandle, ParallelRegionId) {
    let region = ParallelRegionId(RegionId {
        function: ProgramFunctionId(1),
        ordinal: 1,
    });
    (
        GangHandle {
            id: GangId::derive(&[b"driver-gang", &generation.to_be_bytes()]),
            scope_id: fixture.scope,
            generation,
            pool: PoolHandle {
                id: fixture.pool,
                scope_id: fixture.scope,
                generation: 1,
            },
            labs: LabCount(2),
        },
        region,
    )
}

fn rank_task(
    fixture: &common::Fixture,
    gang: &GangHandle,
    region: ParallelRegionId,
    rank: LabRank,
) -> runmat_execution_runner::TaskSubmission {
    let mut task = common::task(
        &format!("gang-rank-{}", rank.0),
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let callable = ProgramCallable::spmd_region(region);
    task.request.callable = runmat_execution::task::Callable::for_program("test", &callable);
    task.request.invocation_context = ProgramInvocationContext::SpmdTask {
        task: SpmdTaskContext {
            gang: gang.clone(),
            region,
            rank,
        },
    };
    task
}

#[test]
fn batch_admission_is_atomic_and_schedules_only_after_every_rank_exists() {
    let mut fixture = common::fixture(2, 2);
    let (gang, region) = gang(&fixture, 1);
    let first = rank_task(&fixture, &gang, region, LabRank(1));
    let mut invalid = rank_task(&fixture, &gang, region, LabRank(2));
    invalid.request.id = TaskId::derive(&[b"missing-pool"]);
    invalid.request.pool_id = runmat_execution::PoolId::derive(&[b"unknown"]);

    assert!(fixture
        .driver
        .handle(DriverCommand::SubmitBatch(vec![first.clone(), invalid]))
        .is_err());
    assert!(fixture.driver.snapshot().tasks.is_empty());

    let second = rank_task(&fixture, &gang, region, LabRank(2));
    let actions = fixture
        .driver
        .handle(DriverCommand::SubmitBatch(vec![first, second]))
        .unwrap();
    assert_eq!(fixture.driver.snapshot().tasks.len(), 2);
    assert_eq!(
        actions
            .iter()
            .filter(|action| matches!(action, DriverAction::Launch(_)))
            .count(),
        2
    );
}

#[test]
fn cancelling_one_gang_does_not_cancel_unrelated_tasks() {
    let mut fixture = common::fixture(2, 2);
    let (gang, region) = gang(&fixture, 1);
    let launches = fixture
        .driver
        .handle(DriverCommand::SubmitBatch(vec![
            rank_task(&fixture, &gang, region, LabRank(1)),
            rank_task(&fixture, &gang, region, LabRank(2)),
        ]))
        .unwrap()
        .into_iter()
        .filter_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(launches.len(), 2);
    let unrelated = common::task("unrelated", fixture.scope, fixture.pool, RetryPolicy::Never);
    let unrelated_id = unrelated.request.id;
    fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(unrelated)))
        .unwrap();

    let cancellation = fixture
        .driver
        .handle(DriverCommand::CancelGang {
            gang,
            reason: CancellationReason::DependencyFailed,
            now_millis: 1,
        })
        .unwrap();
    let snapshot = fixture.driver.snapshot();
    snapshot.validate().unwrap();
    let cold: runmat_execution_runner::DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold.tasks, snapshot.tasks);
    for request in &launches {
        let cause = snapshot.tasks[&request.task_id].cancellation.unwrap().cause;
        assert_eq!(
            cause.reason_for_attempt(request).unwrap(),
            CancellationReason::DependencyFailed
        );
        let mut foreign = request.clone();
        if let ProgramInvocationContext::SpmdTask { task } = &mut foreign.task.invocation_context {
            task.gang.generation += 1;
        }
        assert!(cause.reason_for_attempt(&foreign).is_err());
    }
    assert_eq!(
        snapshot.tasks[&unrelated_id].state,
        runmat_execution::state::TaskState::Ready
    );
    assert_eq!(
        snapshot
            .tasks
            .values()
            .filter(|task| task.cancellation.is_some() && task.active_attempt.is_some())
            .count(),
        2
    );
    assert_eq!(
        cancellation
            .iter()
            .filter(|action| matches!(action, DriverAction::Cancel(_)))
            .count(),
        2
    );
    for request in launches {
        fixture
            .driver
            .handle(DriverCommand::BackendReport(
                runmat_execution_runner::port::BackendReport::for_request(
                    &request,
                    runmat_execution_runner::AttemptReport::Retired,
                ),
            ))
            .unwrap();
    }
    let retired = fixture.driver.snapshot();
    retired.validate().unwrap();
    assert_eq!(
        retired
            .tasks
            .values()
            .filter(|task| task.state == runmat_execution::state::TaskState::Cancelled)
            .count(),
        2
    );
    assert!(retired.tasks[&unrelated_id].cancellation.is_none());
}
