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
    fixture
        .driver
        .handle(DriverCommand::SubmitBatch(vec![
            rank_task(&fixture, &gang, region, LabRank(1)),
            rank_task(&fixture, &gang, region, LabRank(2)),
        ]))
        .unwrap();
    let unrelated = common::task("unrelated", fixture.scope, fixture.pool, RetryPolicy::Never);
    let unrelated_id = unrelated.request.id;
    fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(unrelated)))
        .unwrap();

    fixture
        .driver
        .handle(DriverCommand::CancelGang {
            gang,
            reason: CancellationReason::DependencyFailed,
            now_millis: 1,
        })
        .unwrap();
    let snapshot = fixture.driver.snapshot();
    assert_eq!(
        snapshot.tasks[&unrelated_id].state,
        runmat_execution::state::TaskState::Ready
    );
    assert_eq!(
        snapshot
            .tasks
            .values()
            .filter(|task| task.state == runmat_execution::state::TaskState::Cancelled)
            .count(),
        2
    );
}
