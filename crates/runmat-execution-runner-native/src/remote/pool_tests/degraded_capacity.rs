use super::*;
use runmat_execution::state::PoolState;

#[tokio::test]
async fn loss_below_provisioning_floor_retries_on_healthy_capacity_and_retains_provider_diagnostic()
{
    let scope_id = ExecutionScopeId::derive(&[b"degraded-capacity-scope"]);
    let pool_id = PoolId::derive(&[b"degraded-capacity-pool"]);
    let bundle = Arc::new(executable_bundle().await.2);
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 2,
            max_workers: 2,
            max_in_flight: 2,
            resource_limit: inventory(2_000),
        },
        31,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let worker_ids = [
        WorkerId::derive(&[b"degraded-a"]),
        WorkerId::derive(&[b"degraded-b"]),
    ];
    let failed_id = *worker_ids.iter().min().unwrap(); // Deterministic shared best-fit tie.
    for id in worker_ids {
        pool.add_worker(Arc::new(FailingRouteWorker {
            spec: WorkerSpec {
                host: crate::test_support::host_inventory(),
                id,
                pool_id,
                resources: inventory(1_000),
            },
            bundle: Arc::clone(&bundle),
            fail_execute: id == failed_id,
            fail_drain: false,
        }))
        .await
        .unwrap();
    }
    let (program, artifact_id) = program().await;
    let task_id = TaskId::derive(&[b"degraded-retry"]);
    let mut task = submission(scope_id, pool_id, task_id, artifact_id);
    task.request.retry = RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 };
    let completion = pool.submit(task, program).unwrap();
    let success = tokio::time::timeout(Duration::from_secs(1), completion.wait())
        .await
        .unwrap()
        .unwrap();
    let evidence = pool.task_execution_evidence(task_id).unwrap();
    evidence.validate().unwrap();
    assert_eq!(evidence.attempts.len(), 2);
    assert_eq!(evidence.attempts[0].request.worker_id, failed_id);
    assert_eq!(
        evidence.attempts[0]
            .terminal
            .as_ref()
            .unwrap()
            .diagnostic
            .as_deref(),
        Some("native execution protocol failed: fixture route failed")
    );
    assert_ne!(evidence.attempts[1].request.worker_id, failed_id);
    assert_eq!(
        evidence.task.committed.as_ref().unwrap().result.resources(),
        success.resources()
    );
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.pools[&pool_id].state, PoolState::Ready);
    assert_eq!(
        snapshot.pools[&pool_id].workers[&failed_id].lifecycle,
        runmat_execution_runner::pool::WorkerLifecycle::Lost
    );
    // Lost execution consumption stays unknown even though its retry succeeds.
    assert!(evidence.resource_aggregate(1).is_err());
}
