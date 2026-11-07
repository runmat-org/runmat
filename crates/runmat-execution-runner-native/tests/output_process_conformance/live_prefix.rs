//! Read original bounded writes while live, then kill their publisher and reopen cold.
use super::*;
use runmat_execution::console::*;

pub(super) async fn verify(
    service: &NativeExecutionService,
    config: &NativeExecutionConfig,
    root: &std::path::Path,
) {
    let mut original = call_source(1,
        "function [first, second] = original_output_values(); fprintf(1,'λ🙂original prefix'); fprintf(2,'second'); clc; while true; first=13; end; second=29; end").await;
    original.console = Some(ConsoleLimits {
        max_records: 8,
        max_text_bytes: 1000,
    });
    let job = service
        .submit_job(
            original.call.clone(),
            DurableJobOptions {
                idempotency_key: Some("original-live-prefix".into()),
                // Preserve the record across the independent observation windows,
                // paged reads and cold supervisor restart below. This witness
                // checks retained output, not minimum-retention garbage collection.
                retention_millis: 5 * 60_000,
                resources: ResourceRequest {
                    cpu_millicores: config.worker_cpu_millicores,
                    memory_bytes: config.worker_memory_bytes,
                    scratch_bytes: config.worker_scratch_bytes,
                    // Allow the observation and paged-read windows to finish before
                    // this witness deliberately terminates its publisher.
                    max_wall_millis: 5 * 60_000,
                    // The durable source retains a bounded console subtree.
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
    let attempt_id = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let attached = client.attach(job.id, 0, 0).await.unwrap();
            if let Some(attempt) = attached.execution.as_ref().and_then(|h| h.attempts.last()) {
                break attempt.request.id;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let budget = ConsoleReadBudget {
        max_fragments: 1,
        max_text_bytes: 4,
    };
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if client
                .console_page(job.clone(), attempt_id, ConsoleCursor::default(), budget)
                .await
                .unwrap()
                .is_some_and(|page| page.observed_records == 3)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original writes were not published while source was live");
    let live = collect(&client, &job, attempt_id, budget).await;
    assert_eq!(
        live,
        vec![
            (ConsoleStream::Stdout, "λ🙂original prefix".into()),
            (ConsoleStream::Stderr, "second".into()),
            (ConsoleStream::ClearScreen, "".into())
        ]
    );
    let mut forged = job.clone();
    forged.generation += 1;
    assert!(client
        .console_page(forged, attempt_id, ConsoleCursor::default(), budget)
        .await
        .is_err());
    let before_loss = client
        .attach(job.id, 0, 0)
        .await
        .unwrap()
        .execution
        .unwrap();
    assert!(
        before_loss.attempts[0].terminal.is_none(),
        "publisher retired before the loss trigger: {:?}",
        before_loss.attempts[0].terminal
    );
    assert!(before_loss.attempts[0].unverified_retirement.is_none());
    let directory = root.join("jobs").join(job.id.to_string());
    std::fs::write(
        directory.join("test-provider-exit"),
        b"exit original publisher",
    )
    .unwrap();
    let lost = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let attached = client.attach(job.id, 0, 0).await.unwrap();
            let history = attached.execution.unwrap();
            assert!(
                history.attempts[0].terminal.is_none(),
                "publisher loss produced an unexpected terminal result: {:?}",
                history.attempts[0].terminal
            );
            if history.attempts[0].unverified_retirement.is_some() {
                break history;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    lost.validate().unwrap();
    assert_eq!(lost.attempts.len(), 1);
    assert_eq!(
        lost.task.state,
        runmat_execution::state::TaskState::Indeterminate
    );
    let attempt = &lost.attempts[0];
    assert_eq!(attempt.request.id, attempt_id);
    // Publisher loss is not proof that its complete process tree retired.
    // Preserve uncertainty and original observations instead of certifying a
    // terminal attempt or fabricating zero consumption.
    assert!(attempt.terminal.is_none());
    let uncertainty = attempt.unverified_retirement.as_ref().unwrap();
    assert!(!uncertainty.diagnostic.is_empty());
    assert!(uncertainty.retired_at_driver_millis.is_none());
    assert!(uncertainty.resources.is_none());
    assert!(uncertainty.execution.is_none());
    assert!(lost.resource_aggregate(1).is_err());
    let prefix = uncertainty.console.as_ref().unwrap();
    assert_eq!(prefix.completion(), ConsoleCompletion::Interrupted);
    assert!(prefix.source_receipt().is_none());
    assert_eq!(prefix.records().len(), 3);
    assert_eq!(collect(&client, &job, attempt_id, budget).await, live);
    // Reopen the scheduler under a new supervisor after the real publisher ended.
    std::fs::write(root.join("test-stop"), b"stop").unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let stopped = root.join("test-stopped").exists()
                && runmat_filesystem::ExclusiveFileLock::try_acquire(&root.join("supervisor.lock"))
                    .is_ok_and(|guard| guard.is_some());
            if stopped {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original supervisor did not release its authority");
    std::fs::remove_file(root.join("test-stop")).unwrap();
    std::fs::remove_file(root.join("test-stopped")).unwrap();
    let cold = LocalSupervisorClient::for_current_executable().unwrap();
    cold.connect_or_start().await.unwrap();
    assert_eq!(collect(&cold, &job, attempt_id, budget).await, live);
    let reopened = cold.attach(job.id, 0, 0).await.unwrap().execution.unwrap();
    reopened.validate().unwrap();
    assert!(reopened.resource_aggregate(1).is_err());
    let runmat_execution_runner::TaskExecutionEvidence {
        task,
        attempts,
        driver_fence,
        observed_at_driver_millis,
    } = reopened;
    assert_eq!(task, lost.task);
    assert_eq!(attempts, lost.attempts);
    assert_eq!(driver_fence, lost.driver_fence);
    // The unretired attempt keeps the durable clock active across reopening;
    // inspection time advances, but original attempt evidence cannot change.
    assert!(observed_at_driver_millis >= lost.observed_at_driver_millis);
    println!("original live UTF-8/control prefix survives abrupt publisher exit and cold supervisor restart");
}

async fn collect(
    client: &LocalSupervisorClient,
    job: &runmat_execution::JobHandle,
    attempt: runmat_execution::identity::AttemptId,
    budget: ConsoleReadBudget,
) -> Vec<(ConsoleStream, String)> {
    let mut cursor = ConsoleCursor::default();
    let mut writes: Vec<(ConsoleStream, String)> = vec![];
    loop {
        let page = client
            .console_page(job.clone(), attempt, cursor, budget)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(page.completion, ConsoleCompletion::Interrupted);
        assert!(page.source_receipt.is_none());
        for f in &page.fragments {
            if f.text_offset == 0 {
                writes.push((f.stream, String::new()));
            }
            writes.last_mut().unwrap().1.push_str(&f.text);
        }
        assert!(page.fragments.len() <= 1);
        assert!(page.fragments.iter().map(|f| f.text.len()).sum::<usize>() <= 4);
        if page.next_cursor.sequence == page.observed_records + 1 {
            break;
        }
        assert_ne!(page.next_cursor, cursor);
        cursor = page.next_cursor;
    }
    writes
}
