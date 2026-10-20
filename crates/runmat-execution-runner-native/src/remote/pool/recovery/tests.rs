use super::*;
use crate::remote::pool_tests::{executable_bundle, inventory, object_reference, submission};
use runmat_execution::identity::WorkerId;
use runmat_execution::state::PoolState;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, DriverAction, DriverCommand, WorkerSpec,
};

fn authority(checkpoint: &RemotePoolCheckpoint) -> RemotePoolRestoreAuthority {
    RemotePoolRestoreAuthority {
        scope_id: checkpoint.scope_id,
        pool: checkpoint.scheduler.snapshot.pools[&checkpoint.pool_id]
            .spec
            .clone(),
        value_scope: checkpoint.value_scope.clone(),
        driver_fence: 51,
    }
}

async fn fixture() -> (
    Arc<RemotePoolDriver>,
    ProgramExecutionRequest,
    runmat_execution_runner::TaskSubmission,
) {
    let (pool, program, task) = unsubmitted_fixture().await;
    let _waiter = pool.submit(task.clone(), program.clone()).unwrap();
    (pool, program, task)
}

async fn unsubmitted_fixture() -> (
    Arc<RemotePoolDriver>,
    ProgramExecutionRequest,
    runmat_execution_runner::TaskSubmission,
) {
    let (program, artifact, bundle) = executable_bundle().await;
    let scope = ExecutionScopeId::derive(&[b"checkpoint-scope"]);
    let pool_id = PoolId::derive(&[b"checkpoint-pool"]);
    let pool = RemotePoolDriver::new(
        scope,
        PoolSpec {
            id: pool_id,
            min_workers: 0,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1000),
        },
        7,
        bundle,
    )
    .unwrap();
    let task = submission(
        scope,
        pool_id,
        TaskId::derive(&[b"checkpoint-task"]),
        artifact,
    );
    (pool, program, task)
}

fn launch(pool: &RemotePoolDriver) -> runmat_execution_runner::AttemptRequest {
    let mut driver = pool.driver.lock().unwrap();
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool.pool_id,
            state: PoolState::Ready,
        })
        .unwrap();
    driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            host: crate::test_fixtures::host_inventory(),
            id: WorkerId::derive(&[b"checkpoint-worker"]),
            pool_id: pool.pool_id,
            resources: inventory(1000),
        }))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(attempt) => Some(attempt),
            _ => None,
        })
        .unwrap()
}

#[tokio::test]
async fn committed_download_survives_cold_restore_and_gc_but_pending_download_is_excluded() {
    use crate::remote::pool_tests::ResultObjectWorker;
    let (pool, _, _) = fixture().await;
    let attempt = launch(&pool);
    let reference = object_reference(&pool.value_scope, b"committed-only");
    let worker = ResultObjectWorker {
        spec: WorkerSpec {
            host: crate::test_fixtures::host_inventory(),
            id: attempt.worker_id,
            pool_id: pool.pool_id,
            resources: inventory(1000),
        },
        bundle: Arc::new(pool.test_bundle().to_vec()),
        reference: reference.clone(),
        bytes: Arc::new(b"committed-only".to_vec()),
        corrupt_download: false,
    };
    let lease = pool
        .execution_objects
        .receive_results(&worker, std::slice::from_ref(&reference), &pool.value_scope)
        .await
        .unwrap();
    let pending = pool.capture_checkpoint(1).unwrap();
    assert!(pending.object_references().unwrap().is_empty());
    let accepted = AttemptSuccess::new(
        &attempt,
        vec![ValuePayload::Object(Box::new(reference.clone()))],
        vec![reference.clone()],
        1,
        reference.encoded_length,
        0,
    )
    .unwrap();
    pool.apply_report(BackendReport::for_request(
        &attempt,
        AttemptReport::Succeeded { result: accepted },
    ));
    drop(lease);
    let checkpoint = pool.capture_checkpoint(2).unwrap();
    let committed_watermark = checkpoint
        .scheduler
        .snapshot
        .events
        .iter()
        .find_map(|event| {
            matches!(
                &event.kind,
                runmat_execution_runner::driver::DriverEventKind::ResultCommitted { .. }
            )
            .then_some(event.sequence + 1)
        })
        .expect("the accepted result has a scheduler commit event");
    assert_eq!(
        pool.committed_event_watermark
            .load(std::sync::atomic::Ordering::Acquire),
        committed_watermark
    );
    // Consume the ephemeral wake first. The event watermark must still let a
    // restarted publication waiter discover the accepted commit immediately.
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        pool.committed_task.notified(),
    )
    .await
    .unwrap();
    assert_eq!(
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            pool.committed_task_after(0),
        )
        .await
        .unwrap(),
        committed_watermark
    );
    assert_eq!(
        checkpoint.scheduler_event_watermark(),
        checkpoint.scheduler.snapshot.next_event_sequence
    );
    assert!(checkpoint.execution_inputs.is_empty());
    assert_eq!(
        checkpoint.object_references().unwrap(),
        vec![reference.clone()]
    );
    let bytes = pool.checkpoint_object(&reference).unwrap();
    let bundle = pool.test_bundle();
    drop(pool);
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        bundle,
        |_| Ok(Arc::clone(&bytes)),
    )
    .unwrap();
    restored
        .execution_objects
        .discard_results(std::slice::from_ref(&reference));
    assert_eq!(
        restored.checkpoint_object(&reference).unwrap().as_ref(),
        b"committed-only"
    );
    let recaptured = restored.capture_checkpoint(3).unwrap();
    assert!(recaptured.execution_inputs.is_empty());
    assert_eq!(recaptured.object_references().unwrap(), vec![reference]);
    assert_eq!(
        recaptured.scheduler.snapshot.tasks,
        checkpoint.scheduler.snapshot.tasks
    );
}

#[tokio::test]
async fn scoped_descendant_inventory_preserves_typed_aliases_across_restore() {
    let (pool, program, task) = unsubmitted_fixture().await;
    let root = object_reference(&pool.value_scope, b"root");
    let descendant = object_reference(&pool.value_scope, b"descendant");
    for (reference, bytes) in [(&root, &b"root"[..]), (&descendant, &b"descendant"[..])] {
        pool.register_execution_object(reference.clone(), Arc::<[u8]>::from(bytes))
            .unwrap();
    }
    let mut driver_root = root.clone();
    driver_root.kind = ValueRefKind::DriverObject;
    let mut driver_descendant = descendant.clone();
    driver_descendant.kind = ValueRefKind::DriverObject;
    let scoped = vec![driver_root, driver_descendant];
    pool.register_program_input_inventory(&program, scoped.clone())
        .unwrap();
    let _waiter = pool.submit(task.clone(), program.clone()).unwrap();
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    assert_eq!(checkpoint.object_references().unwrap().len(), 2);
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        pool.test_bundle(),
        |reference| pool.checkpoint_object(reference),
    )
    .unwrap();
    let mut expected = scoped;
    expected.sort_by_key(|reference| reference.id);
    assert_eq!(
        restored
            .input_inventories
            .lock()
            .unwrap()
            .get(task.request.id),
        Some(expected.clone())
    );
    let recaptured = restored.capture_checkpoint(2).unwrap();
    assert_eq!(recaptured.tasks[&task.request.id].inventory, Some(expected));
    for reference in recaptured.tasks[&task.request.id]
        .inventory
        .as_ref()
        .unwrap()
    {
        assert!(restored.execution_object(reference).unwrap().is_some());
    }
}

#[tokio::test]
async fn missing_inventory_and_oversized_references_fail_before_object_io() {
    let (pool, program, task) = unsubmitted_fixture().await;
    let reference = object_reference(&pool.value_scope, b"scoped");
    pool.register_execution_object(reference.clone(), Arc::<[u8]>::from(&b"scoped"[..]))
        .unwrap();
    pool.register_program_input_inventory(&program, vec![reference])
        .unwrap();
    let _waiter = pool.submit(task, program).unwrap();
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    for case in 0..3 {
        let mut damaged = checkpoint.clone();
        match case {
            0 => damaged.execution_inputs.clear(),
            1 => {
                damaged.execution_inputs[0].encoded_length =
                    runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_BYTES + 1
            }
            _ => {
                damaged.execution_inputs[0].resident_fence =
                    Some(runmat_execution::value::ResidentFence {
                        worker_id: WorkerId::derive(&[b"resident-worker"]),
                        node_lease_id: runmat_execution::identity::NodeLeaseId::derive(&[
                            b"resident-node-lease",
                        ]),
                        process_generation: 1,
                        accelerator: None,
                    });
            }
        }
        assert!(RemotePoolDriver::restore_checkpoint(
            damaged,
            authority(&checkpoint),
            pool.test_bundle(),
            |_| panic!("invalid inventory must fail before IO")
        )
        .is_err());
    }
}

#[tokio::test]
async fn elapsed_deadline_is_terminal_before_replacement_worker_registration() {
    let (pool, program, mut task) = unsubmitted_fixture().await;
    let saved_now = pool.snapshot().now_millis;
    task.request.deadline_unix_millis = Some(saved_now + 60_000);
    let task_id = task.request.id;
    let _waiter = pool.submit(task, program).unwrap();
    let mut checkpoint = pool.capture_checkpoint(1).unwrap();
    // Translate the clock origin without waiting: the saved task was unexpired,
    // but both saved time and its unchanged absolute deadline are now in the past.
    checkpoint.scheduler.snapshot.now_millis = 1;
    let request = &mut checkpoint
        .scheduler
        .snapshot
        .tasks
        .get_mut(&task_id)
        .unwrap()
        .submission
        .request;
    let previous = request.deadline_unix_millis;
    request.deadline_unix_millis = Some(2);
    checkpoint
        .scheduler
        .snapshot
        .deadlines
        .remove(task_id, previous);
    checkpoint
        .scheduler
        .snapshot
        .deadlines
        .insert(task_id, Some(2));
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        pool.test_bundle(),
        |_| panic!("no objects"),
    )
    .unwrap();
    let snapshot = restored.snapshot();
    assert!(matches!(
        snapshot.tasks[&task_id].state,
        runmat_execution::state::TaskState::Cancelled | runmat_execution::state::TaskState::Failed
    ));
    let actions = restored
        .driver
        .lock()
        .unwrap()
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            host: crate::test_fixtures::host_inventory(),
            id: WorkerId::derive(&[b"late-worker"]),
            pool_id: restored.pool_id,
            resources: inventory(1000),
        }))
        .unwrap();
    assert!(!actions
        .iter()
        .any(|action| matches!(action, DriverAction::Launch(_))));
    assert!(restored.snapshot().attempts.is_empty());
}

#[tokio::test]
async fn cold_capture_roundtrip_preserves_ready_tasks_and_streams_verified_objects() {
    let (pool, _, task) = fixture().await;
    let reference = object_reference(&pool.value_scope, b"retained input");
    pool.register_execution_object(reference.clone(), Arc::<[u8]>::from(&b"retained input"[..]))
        .unwrap();
    let checkpoint = pool.capture_checkpoint(12).unwrap();
    assert_eq!(
        checkpoint.object_references().unwrap(),
        vec![reference.clone()]
    );
    let encoded = checkpoint.encode().unwrap();
    let checkpoint = RemotePoolCheckpoint::decode(&encoded).unwrap();
    let bytes = pool.checkpoint_object(&reference).unwrap();
    let bundle = pool.test_bundle();
    let before = pool.snapshot().tasks[&task.request.id].clone();
    drop(pool);
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        bundle,
        |requested| {
            assert_eq!(requested, &reference);
            Ok(Arc::clone(&bytes))
        },
    )
    .unwrap();
    assert_eq!(restored.snapshot().driver_fence, 51);
    assert_eq!(restored.snapshot().tasks[&task.request.id], before);
    assert!(restored.channels.read().await.is_empty());
    assert_eq!(restored.completions.lock().unwrap().len(), 1);
    assert_eq!(
        restored
            .capture_checkpoint(13)
            .unwrap()
            .object_references()
            .unwrap(),
        vec![reference]
    );
}

#[tokio::test]
async fn completed_program_and_original_commit_survive_cold_restore_without_dispatch() {
    let (pool, program, task) = fixture().await;
    let attempt = {
        let mut driver = pool.driver.lock().unwrap();
        driver
            .handle(DriverCommand::SetPoolState {
                pool_id: pool.pool_id,
                state: PoolState::Ready,
            })
            .unwrap();
        driver
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                host: crate::test_fixtures::host_inventory(),
                id: WorkerId::derive(&[b"checkpoint-worker"]),
                pool_id: pool.pool_id,
                resources: inventory(1000),
            }))
            .unwrap()
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(attempt) => Some(attempt),
                _ => None,
            })
            .unwrap()
    };
    let accepted = AttemptSuccess::new(
        &attempt,
        vec![ValuePayload::Inline(Box::new(InlineValue::String(
            "retained".into(),
        )))],
        Vec::new(),
        29,
        0,
        0,
    )
    .unwrap();
    pool.apply_report(BackendReport::for_request(
        &attempt,
        AttemptReport::Succeeded {
            result: accepted.clone(),
        },
    ));
    let checkpoint = pool.capture_checkpoint(2).unwrap();
    let committed = pool.snapshot().tasks[&task.request.id].committed.clone();
    let bundle = pool.test_bundle();
    drop(pool);
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        bundle,
        |_| panic!("inline result needs no object reads"),
    )
    .unwrap();
    assert_eq!(
        restored
            .submit(task.clone(), program)
            .unwrap()
            .wait()
            .await
            .unwrap(),
        accepted
    );
    assert_eq!(
        restored.snapshot().tasks[&task.request.id].committed,
        committed
    );
    assert_eq!(restored.snapshot().attempts.len(), 1);
    assert!(restored.channels.read().await.is_empty());
}

#[tokio::test]
async fn authority_and_metadata_substitutions_fail_before_object_reads() {
    let (pool, _, task) = fixture().await;
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    for case in 0..10 {
        let mut damaged = checkpoint.clone();
        let mut replacement = authority(&checkpoint);
        match case {
            0 => damaged.tasks.clear(),
            1 => damaged.bundle_digest = Digest::sha256(b"wrong bundle"),
            2 => replacement.scope_id = ExecutionScopeId::derive(&[b"foreign scope"]),
            3 => replacement.value_scope = "foreign run".into(),
            4 => replacement.driver_fence = 7,
            5 => {
                damaged
                    .tasks
                    .get_mut(&task.request.id)
                    .unwrap()
                    .program
                    .requested_outputs += 1
            }
            6 => damaged.schema_version = 1,
            7 => damaged.scheduler.schema_version = 1,
            8 => damaged.scheduler.snapshot.schema_version = 2,
            9 => damaged.schema_version = 3,
            _ => unreachable!(),
        }
        assert!(RemotePoolDriver::restore_checkpoint(
            damaged,
            replacement,
            pool.test_bundle(),
            |_| panic!("invalid authority must fail before reading objects")
        )
        .is_err());
    }
}

#[tokio::test]
async fn missing_or_corrupt_checkpoint_objects_never_expose_a_restored_pool() {
    let (pool, _, _) = fixture().await;
    let reference = object_reference(&pool.value_scope, b"required");
    pool.register_execution_object(reference, Arc::<[u8]>::from(&b"required"[..]))
        .unwrap();
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    for missing in [true, false] {
        assert!(RemotePoolDriver::restore_checkpoint(
            checkpoint.clone(),
            authority(&checkpoint),
            pool.test_bundle(),
            |_| if missing {
                Err(invalid("unavailable"))
            } else {
                Ok(Arc::from(&b"corrupted"[..]))
            }
        )
        .is_err());
    }
}

#[tokio::test]
async fn restored_waiter_can_attach_before_or_after_authoritative_completion() {
    for complete_first in [false, true] {
        let (pool, program, task) = fixture().await;
        let checkpoint = pool.capture_checkpoint(1).unwrap();
        let restored = RemotePoolDriver::restore_checkpoint(
            checkpoint.clone(),
            authority(&checkpoint),
            pool.test_bundle(),
            |_| panic!("no objects"),
        )
        .unwrap();
        let attempt = launch(&restored);
        let accepted = AttemptSuccess::new(
            &attempt,
            vec![ValuePayload::Inline(Box::new(InlineValue::String(
                "reattached".into(),
            )))],
            Vec::new(),
            3,
            0,
            0,
        )
        .unwrap();
        let report = BackendReport::for_request(
            &attempt,
            AttemptReport::Succeeded {
                result: accepted.clone(),
            },
        );
        let completion = if complete_first {
            restored.apply_report(report);
            restored.submit(task.clone(), program.clone()).unwrap()
        } else {
            let completion = restored.submit(task.clone(), program.clone()).unwrap();
            assert!(restored.submit(task.clone(), program.clone()).is_err());
            restored.apply_report(report);
            completion
        };
        assert_eq!(completion.wait().await.unwrap(), accepted);
        assert_eq!(restored.snapshot().attempts.len(), 1);
        assert!(restored.restored_completions.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn rejected_mixed_graph_does_not_consume_restored_waiter() {
    let (pool, program, task) = fixture().await;
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        pool.test_bundle(),
        |_| panic!("no objects"),
    )
    .unwrap();
    let mut invalid = task.clone();
    invalid.request.id = TaskId::derive(&[b"new-invalid-task"]);
    invalid
        .dependencies
        .insert(TaskId::derive(&[b"missing-dependency"]));
    assert!(restored
        .submit_graph(vec![
            (program.clone(), task.clone()),
            (program.clone(), invalid)
        ])
        .is_err());
    assert_eq!(restored.restored_completions.lock().unwrap().len(), 1);
    assert_eq!(restored.snapshot().tasks.len(), 1);
    let completion = restored.submit(task, program).unwrap();
    let attempt = launch(&restored);
    let accepted = AttemptSuccess::new(
        &attempt,
        vec![ValuePayload::Inline(Box::new(InlineValue::String(
            "retained waiter".into(),
        )))],
        Vec::new(),
        3,
        0,
        0,
    )
    .unwrap();
    restored.apply_report(BackendReport::for_request(
        &attempt,
        AttemptReport::Succeeded {
            result: accepted.clone(),
        },
    ));
    assert_eq!(completion.wait().await.unwrap(), accepted);
}

#[tokio::test]
async fn cold_active_attempt_retry_preserves_admission_and_fences_stale_success() {
    use runmat_execution::state::{AttemptState, TaskState};
    use runmat_execution::task::RetryPolicy;

    for retry in [
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 1 },
        RetryPolicy::Never,
    ] {
        let (pool, program, mut task) = unsubmitted_fixture().await;
        task.request.retry = retry;
        task.request.deadline_unix_millis = Some(pool.snapshot().now_millis + 3_600_000);
        task.priority = 7;
        let original_submission = task.clone();
        let _original_waiter = pool.submit(task.clone(), program.clone()).unwrap();
        let old_attempt = launch(&pool);
        let checkpoint = pool.capture_checkpoint(4).unwrap();
        let bundle = pool.test_bundle();
        drop(pool);

        let restored = RemotePoolDriver::restore_checkpoint(
            checkpoint.clone(),
            authority(&checkpoint),
            bundle,
            |_| panic!("inline task has no object inventory"),
        )
        .unwrap();
        let snapshot = restored.snapshot();
        assert_eq!(snapshot.driver_fence, 51);
        assert_eq!(
            snapshot.tasks[&task.request.id].submission,
            original_submission
        );
        assert_eq!(snapshot.tasks[&task.request.id].attempt_count, 1);
        assert_eq!(snapshot.attempts[&old_attempt.id].request, old_attempt);
        assert_eq!(snapshot.attempts[&old_attempt.id].state, AttemptState::Lost);

        // Reconstructed host policy must not extend the persisted deadline or budget.
        task.request.deadline_unix_millis =
            Some(original_submission.request.deadline_unix_millis.unwrap() + 3_600_000);
        task.request.retry = RetryPolicy::ExplicitlyIdempotent { max_attempts: 99 };
        task.priority = 0;
        let completion = restored.submit(task.clone(), program).unwrap();
        assert_eq!(
            restored.snapshot().tasks[&task.request.id].submission,
            original_submission
        );

        if retry != (RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 }) {
            let expected = if retry == RetryPolicy::Never {
                TaskState::Indeterminate
            } else {
                TaskState::Failed
            };
            assert_eq!(restored.snapshot().tasks[&task.request.id].state, expected);
            assert!(completion.wait().await.is_err());
            let actions = restored
                .driver
                .lock()
                .unwrap()
                .handle(DriverCommand::RegisterWorker(WorkerSpec {
                    host: crate::test_fixtures::host_inventory(),
                    id: old_attempt.worker_id,
                    pool_id: restored.pool_id,
                    resources: inventory(1000),
                }))
                .unwrap();
            assert!(!actions
                .iter()
                .any(|action| matches!(action, DriverAction::Launch(_))));
            assert_eq!(restored.snapshot().attempts.len(), 1);
            continue;
        }

        assert_eq!(
            restored.snapshot().tasks[&task.request.id].state,
            TaskState::Ready
        );
        let actions = restored
            .driver
            .lock()
            .unwrap()
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                host: crate::test_fixtures::host_inventory(),
                id: old_attempt.worker_id,
                pool_id: restored.pool_id,
                resources: inventory(1000),
            }))
            .unwrap();
        let retries: Vec<_> = actions
            .into_iter()
            .filter_map(|action| match action {
                DriverAction::Launch(attempt) => Some(attempt),
                _ => None,
            })
            .collect();
        assert_eq!(retries.len(), 1);
        let new_attempt = &retries[0];
        assert_eq!(new_attempt.driver_fence, 51);
        assert_ne!(new_attempt.id, old_attempt.id);
        assert_eq!(new_attempt.task, original_submission.request);
        assert_eq!(restored.snapshot().tasks[&task.request.id].attempt_count, 2);
        let success = |attempt: &runmat_execution_runner::AttemptRequest, label: &str| {
            AttemptSuccess::new(
                attempt,
                vec![ValuePayload::Inline(Box::new(InlineValue::String(
                    label.into(),
                )))],
                Vec::new(),
                13,
                0,
                0,
            )
            .unwrap()
        };
        let mut waiting = Box::pin(completion.wait());
        restored.apply_report(BackendReport::for_request(
            &old_attempt,
            AttemptReport::Succeeded {
                result: success(&old_attempt, "stale"),
            },
        ));
        assert!(restored.snapshot().tasks[&task.request.id]
            .committed
            .is_none());
        assert_eq!(
            restored.snapshot().tasks[&task.request.id].active_attempt,
            Some(new_attempt.id)
        );
        tokio::select! {
            biased;
            result = &mut waiting => panic!("stale success resolved the waiter: {result:?}"),
            () = std::future::ready(()) => {}
        }
        let accepted = success(new_attempt, "authoritative retry");
        restored.apply_report(BackendReport::for_request(
            new_attempt,
            AttemptReport::Succeeded {
                result: accepted.clone(),
            },
        ));
        assert_eq!(waiting.await.unwrap(), accepted);
        let snapshot = restored.snapshot();
        let commit = snapshot.tasks[&task.request.id].committed.as_ref().unwrap();
        assert_eq!(commit.attempt_id, new_attempt.id);
        assert_eq!(commit.driver_fence, 51);
        assert_eq!(commit.result.resources(), accepted.resources());
        assert_eq!(snapshot.attempts.len(), 2);
        assert_eq!(snapshot.attempts[&old_attempt.id].state, AttemptState::Lost);
    }
}
