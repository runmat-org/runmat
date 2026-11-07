//! Awaiting a live original job must yield so the same bridge can process cancellation.
use super::*;
use runmat_execution::console::{ConsoleCursor, ConsoleLimits, ConsoleReadBudget};
use runmat_execution::CancellationReason;
use runmat_runtime::execution::ExecutionServiceError;

pub(super) async fn verify(service: &NativeExecutionService, config: &NativeExecutionConfig) {
    let mut original = call_source(1,
        "function [first, second] = original_output_values(); fprintf(1,'original before termination'); while true; first = 13; end; second = 29; end").await;
    original.console = Some(ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    });
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
                    // Reserve the canonical console subtree, as ordinary durable tasks do.
                    max_artifact_bytes: u64::from(config.max_message_bytes),
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: vec![],
                    required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
                },
            },
        )
        .unwrap();
    let client = LocalSupervisorClient::for_current_executable().unwrap();
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let observation = client.attach(job.id, 0, 0).await.unwrap();
            if let Some(terminal) = observation
                .execution
                .as_ref()
                .and_then(|history| history.attempts.last())
                .and_then(|attempt| attempt.terminal.as_ref())
            {
                panic!("original worker retired before prefix: {terminal:?}");
            }
            if let Some(attempt) = observation
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
            {
                if client
                    .console_page(
                        job.clone(),
                        attempt.request.id,
                        ConsoleCursor::default(),
                        ConsoleReadBudget {
                            max_fragments: 1,
                            max_text_bytes: 100,
                        },
                    )
                    .await
                    .unwrap()
                    .is_some_and(|page| page.observed_records == 1)
                {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original worker did not publish its prefix before cancellation");
    let started = std::time::Instant::now();
    let AwaitAction::Pending(Value::Job(retained)) = service
        .begin_await_in_context(Value::Job(job.clone().into()), &original.runtime)
        .unwrap()
    else {
        panic!("live original job did not yield pending");
    };
    assert_eq!(retained.as_ref(), &job);
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "live durable await took {elapsed:?} before yielding pending"
    );
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
    let cold = client.attach(job.id, 0, 0).await.unwrap();
    assert_eq!(cold.record.handle, job);
    let evidence = cold.execution.unwrap();
    super::cancellation::verify_evidence(&evidence);
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
