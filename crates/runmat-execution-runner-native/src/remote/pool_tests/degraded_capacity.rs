use super::*;
use runmat_execution::state::PoolState;

#[tokio::test]
async fn loss_below_provisioning_floor_retains_unknown_work_and_admits_distinct_healthy_work() {
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
    let completion = pool.submit(task, program.clone()).unwrap();
    let failure = tokio::time::timeout(Duration::from_secs(1), completion.wait())
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(&failure, NativeProgramFailure::WorkerLost(message)
        if message == "native execution protocol failed: fixture route failed"));
    let evidence = pool.task_execution_evidence(task_id).unwrap();
    evidence.validate().unwrap();
    assert_eq!(evidence.attempts.len(), 1);
    let original = &evidence.attempts[0];
    assert_eq!(original.request.worker_id, failed_id);
    assert!(original.terminal.is_none());
    assert_eq!(
        original.unverified_retirement.as_ref().unwrap().diagnostic,
        "native execution protocol failed: fixture route failed"
    );
    assert!(original
        .unverified_retirement
        .as_ref()
        .unwrap()
        .retired_at_driver_millis
        .is_none());
    assert_eq!(
        evidence.task.state,
        runmat_execution::state::TaskState::Indeterminate
    );
    assert_eq!(evidence.task.active_attempt, Some(original.request.id));
    assert!(evidence.task.committed.is_none());
    assert!(evidence.resource_aggregate(1).is_err());
    let before = pool.snapshot();
    let retained = before.pools[&pool_id].workers[&failed_id].allocated.clone();
    assert_eq!(before.pools[&pool_id].active_attempts, 1);
    assert_eq!(retained.cpu_millicores, 1_000);
    assert_eq!(
        before.pools[&pool_id].workers[&failed_id].lifecycle,
        runmat_execution_runner::pool::WorkerLifecycle::Lost
    );
    assert_eq!(before.pools[&pool_id].state, PoolState::Ready);

    // Healthy capacity may execute separately admitted work, never replay the
    // unknown operation even when its retry budget would otherwise allow it.
    let new_id = TaskId::derive(&[b"degraded-independent-work"]);
    let new_completion = pool
        .submit(submission(scope_id, pool_id, new_id, artifact_id), program)
        .unwrap();
    let success = tokio::time::timeout(Duration::from_secs(1), new_completion.wait())
        .await
        .unwrap()
        .unwrap();
    let new_evidence = pool.task_execution_evidence(new_id).unwrap();
    assert_eq!(new_evidence.attempts.len(), 1);
    assert_ne!(new_evidence.attempts[0].request.worker_id, failed_id);
    assert_eq!(
        new_evidence
            .task
            .committed
            .as_ref()
            .unwrap()
            .result
            .resources(),
        success.resources()
    );
    let after = pool.snapshot();
    assert_eq!(after.tasks[&task_id], before.tasks[&task_id]);
    assert_eq!(
        after.attempts[&original.request.id],
        before.attempts[&original.request.id]
    );
    assert_eq!(
        after.pools[&pool_id].workers[&failed_id].allocated,
        retained
    );
    assert_eq!(after.pools[&pool_id].active_attempts, 1);
}
