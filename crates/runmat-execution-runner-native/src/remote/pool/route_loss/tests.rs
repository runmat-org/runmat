use super::*;
use crate::remote::pool_tests::{executable_bundle, inventory, submission};
use runmat_execution::identity::WorkerId;
use runmat_execution::{ExecutionScopeId, PoolId, TaskId};
use runmat_execution_runner::{AttemptReport, DriverAction, PoolSpec, WorkerSpec};

#[tokio::test]
async fn stale_missing_route_report_does_not_fence_the_current_worker_or_attempt() {
    let scope = ExecutionScopeId::derive(&[b"stale-route-loss"]);
    let pool_id = PoolId::derive(&[b"stale-route-loss"]);
    let worker_id = WorkerId::derive(&[b"current-route-worker"]);
    let (_, artifact_id, bundle) = executable_bundle().await;
    let pool = RemotePoolDriver::new(
        scope,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        37,
        bundle,
    )
    .unwrap();
    let request = {
        let mut driver = pool.driver.lock().unwrap();
        driver
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                host: crate::test_support::host_inventory(),
                id: worker_id,
                pool_id,
                resources: inventory(1_000),
            }))
            .unwrap();
        driver
            .handle(DriverCommand::SetPoolState {
                pool_id,
                state: runmat_execution::state::PoolState::Ready,
            })
            .unwrap();
        driver
            .handle(DriverCommand::Submit(Box::new(submission(
                scope,
                pool_id,
                TaskId::derive(&[b"current-route-work"]),
                artifact_id,
            ))))
            .unwrap()
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(request) => Some(request),
                _ => None,
            })
            .unwrap()
    };
    let before = pool.snapshot();
    let mut stale = BackendReport::for_request(
        &request,
        AttemptReport::Lost {
            message: "old route failed".into(),
        },
    );
    stale.driver_fence += 1;
    assert!(!pool.fence_failed_route(None, &stale).await.unwrap());
    assert_eq!(pool.snapshot(), before);
    pool.apply_report(stale);
    let after = pool.snapshot();
    assert_eq!(after.pools, before.pools);
    assert_eq!(after.attempts, before.attempts);
    assert_eq!(after.tasks, before.tasks);
    after.validate().unwrap();
}
