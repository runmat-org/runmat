//! Awaiting a live original job must yield so the same bridge can process cancellation.
use super::*;
use runmat_execution::CancellationReason;
use runmat_execution_runner::TaskCancellationCause;
use runmat_runtime::execution::ExecutionServiceError;

pub(super) async fn verify(service: &NativeExecutionService, config: &NativeExecutionConfig) {
    let original = call_source(1,
        "function [first, second] = original_output_values(); while true; first = 13; end; second = 29; end").await;
    let job = service
        .submit_job(
            original.call.clone(),
            DurableJobOptions {
                idempotency_key: Some("original-pending-cancellation".into()),
                retention_millis: runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
                resources: ResourceRequest {
                    cpu_millicores: config.worker_cpu_millicores,
                    memory_bytes: config.worker_memory_bytes,
                    scratch_bytes: config.worker_scratch_bytes,
                    max_wall_millis: 30_000,
                    max_artifact_bytes: 1024 * 1024,
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: vec![],
                    required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
                },
            },
        )
        .unwrap();
    let started = std::time::Instant::now();
    let AwaitAction::Pending(Value::Job(retained)) = service
        .begin_await_in_context(Value::Job(job.clone().into()), &original.runtime)
        .unwrap()
    else {
        panic!("live original job did not yield pending");
    };
    assert_eq!(retained.as_ref(), &job);
    assert!(started.elapsed() < Duration::from_secs(5));
    service
        .cancel(&Value::Job(job.clone().into()), CancellationReason::Quota)
        .unwrap();
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match service.begin_await_in_context(Value::Job(job.clone().into()), &original.runtime)
            {
                Ok(AwaitAction::Pending(Value::Job(retained))) => {
                    assert_eq!(retained.as_ref(), &job);
                    if let Some(wait) = service
                        .wait_pending(&Value::Job(retained), &original.runtime)
                        .unwrap()
                    {
                        match wait.await {
                            Ok(()) => {}
                            Err(ExecutionServiceError::Cancelled) => break,
                            Err(error) => panic!("original pending job wait returned {error:?}"),
                        }
                    } else {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
                Err(ExecutionServiceError::Cancelled) => break,
                other => panic!("original pending job cancellation returned {other:?}"),
            }
        }
    })
    .await
    .expect("original pending cancellation did not close");
    let client = LocalSupervisorClient::for_current_executable().unwrap();
    let cold = client.attach(job.id, 0, 0).await.unwrap();
    assert_eq!(cold.record.handle, job);
    let evidence = cold.execution.unwrap();
    evidence.validate().unwrap();
    assert_eq!(
        evidence.task.cancellation.unwrap().cause,
        TaskCancellationCause::Task {
            task_id: evidence.task.id(),
            reason: CancellationReason::Quota,
        }
    );
    assert!(evidence
        .attempts
        .iter()
        .all(|attempt| attempt.terminal.is_some()));
    service
        .cancel(&Value::Job(job.clone().into()), CancellationReason::User)
        .unwrap();
    assert_eq!(
        client
            .attach(job.id, 0, 0)
            .await
            .unwrap()
            .execution
            .unwrap(),
        evidence
    );
    println!("original durable runtime await yields pending and preserves quota cancellation through cold attachment");
}
