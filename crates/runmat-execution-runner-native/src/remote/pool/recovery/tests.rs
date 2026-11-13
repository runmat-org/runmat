use super::*;
use crate::remote::pool_tests::{executable_bundle, inventory, object_reference, submission};
use runmat_execution::identity::WorkerId;
use runmat_execution::state::PoolState;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, DriverAction, DriverCommand, WorkerSpec,
};

#[test]
fn typed_clock_checkpoint_entries_roundtrip_and_reject_duplicate_authority() {
    #[derive(Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    struct ClockEntries {
        #[serde(
            serialize_with = "super::clock_codec::serialize",
            deserialize_with = "super::clock_codec::deserialize_clock"
        )]
        entries: BTreeMap<[u8; 16], u8>,
    }

    let entries = ClockEntries {
        entries: [([7; 16], 3), ([8; 16], 4)].into_iter().collect(),
    };
    let encoded = serde_json::to_vec(&entries).unwrap();
    assert_eq!(
        serde_json::from_slice::<ClockEntries>(&encoded).unwrap(),
        entries
    );

    let duplicated = serde_json::json!({
        "entries": serde_json::to_value([([7_u8; 16], 3_u8), ([7_u8; 16], 4_u8)]).unwrap()
    });
    assert!(serde_json::from_value::<ClockEntries>(duplicated).is_err());
}

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
    let mut actions = driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: WorkerId::derive(&[b"checkpoint-worker"]),
            pool_id: pool.pool_id,
            resources: inventory(1000),
        }))
        .unwrap();
    // Restore deliberately discards the old pool's software ceiling. Derive
    // it from this newly registered fixture worker, as production membership
    // does, rather than resurrecting historical readiness from a checkpoint.
    actions.extend(
        driver
            .handle(DriverCommand::RefreshPoolCapabilities {
                pool_id: pool.pool_id,
            })
            .unwrap(),
    );
    actions
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(attempt) => Some(attempt),
            _ => None,
        })
        .unwrap()
}

#[tokio::test]
async fn restoration_reproves_software_without_releasing_original_physical_authority() {
    let (pool, _, task) = fixture().await;
    let original = launch(&pool);
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    let saved_pool = &checkpoint.scheduler.snapshot.pools[&checkpoint.pool_id];
    assert!(!saved_pool.spec.resource_limit.capabilities.is_empty());
    let allocation = saved_pool.allocated.clone();
    let mut replacement = authority(&checkpoint);
    // Bootstrap knows physical allocation, not a replacement binary's software proof.
    replacement.pool.resource_limit.capabilities.clear();
    let bundle = Arc::clone(&pool.direct_bundle.as_ref().unwrap().bytes);
    let restored = RemotePoolDriver::restore_checkpoint(checkpoint, replacement, bundle, |_| {
        panic!("inline task must have no object inventory")
    })
    .unwrap();
    let snapshot = restored.snapshot();
    snapshot.validate().unwrap();
    let restored_pool = &snapshot.pools[&pool.pool_id];
    assert!(restored_pool.spec.resource_limit.capabilities.is_empty());
    assert_eq!(restored_pool.allocated, allocation);
    assert_eq!(
        restored_pool.workers[&original.worker_id].lifecycle,
        runmat_execution_runner::pool::WorkerLifecycle::Lost
    );
    let attempt = &snapshot.attempts[&original.id];
    assert_eq!(attempt.request, original);
    assert!(attempt.terminal.is_none());
    assert!(attempt
        .unverified_retirement
        .as_ref()
        .unwrap()
        .retired_at_driver_millis
        .is_none());
    assert_eq!(
        snapshot.tasks[&task.request.id].active_attempt,
        Some(original.id)
    );
    assert_eq!(
        snapshot.tasks[&task.request.id].state,
        runmat_execution::state::TaskState::Indeterminate
    );
    // A directly restored pool has no authenticated original worker owner.
    // It cannot await a Server certificate, so its original waiter fails
    // closed instead of remaining pending indefinitely.
    assert!(!restored
        .completions
        .lock()
        .unwrap()
        .contains_key(&task.request.id));
    let completion = restored
        .restored_completions
        .lock()
        .unwrap()
        .remove(&task.request.id)
        .unwrap();
    assert!(matches!(
        completion.wait().await,
        Err(crate::NativeProgramFailure::WorkerLost(_))
    ));
}

#[tokio::test]
async fn software_reproof_exception_never_changes_physical_pool_capacity_or_policy() {
    let (pool, _, _) = unsubmitted_fixture().await;
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    let bundle = Arc::clone(&pool.direct_bundle.as_ref().unwrap().bytes);
    let mutations: [fn(&mut PoolSpec); 6] = [
        |pool| pool.resource_limit.cpu_millicores += 1,
        |pool| pool.resource_limit.memory_bytes += 1,
        |pool| pool.resource_limit.scratch_bytes += 1,
        |pool| pool.min_workers += 1,
        |pool| pool.max_workers += 1,
        |pool| pool.max_in_flight += 1,
    ];
    for mutate in mutations {
        let mut replacement = authority(&checkpoint);
        replacement.pool.resource_limit.capabilities.clear();
        mutate(&mut replacement.pool);
        let result = RemotePoolDriver::restore_checkpoint(
            checkpoint.clone(),
            replacement,
            Arc::clone(&bundle),
            |_| panic!("authority mismatch must reject before object IO"),
        );
        assert!(
            matches!(result, Err(NativeExecutionError::Protocol(message)) if message == "checkpoint differs from replacement allocation authority")
        );
    }
}

#[tokio::test]
async fn committed_download_survives_cold_restore_and_gc_but_pending_download_is_excluded() {
    use crate::remote::pool_tests::ResultObjectWorker;
    let (pool, _, _) = fixture().await;
    let attempt = launch(&pool);
    let reference = object_reference(&pool.value_scope, b"committed-only");
    let worker = ResultObjectWorker {
        spec: WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: attempt.worker_id,
            pool_id: pool.pool_id,
            resources: inventory(1000),
        },
        bundle: Arc::new(
            pool.direct_bundle
                .as_ref()
                .expect("direct fixture bundle")
                .bytes
                .to_vec(),
        ),
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
    let bundle = Arc::clone(
        &pool
            .direct_bundle
            .as_ref()
            .expect("direct fixture bundle")
            .bytes,
    );
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
        Arc::clone(
            &pool
                .direct_bundle
                .as_ref()
                .expect("direct fixture bundle")
                .bytes,
        ),
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
            Arc::clone(
                &pool
                    .direct_bundle
                    .as_ref()
                    .expect("direct fixture bundle")
                    .bytes
            ),
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
        Arc::clone(
            &pool
                .direct_bundle
                .as_ref()
                .expect("direct fixture bundle")
                .bytes,
        ),
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
            host: crate::test_support::host_inventory(),
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
    let bundle = Arc::clone(
        &pool
            .direct_bundle
            .as_ref()
            .expect("direct fixture bundle")
            .bytes,
    );
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
    // The pool's original host clock began before submission. Measure the fixture's actual
    // post-submission work for the provider interval, rather than inventing elapsed time.
    let observed = std::time::Instant::now();
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
                host: crate::test_support::host_inventory(),
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
    let measured_wall_millis =
        u64::try_from(observed.elapsed().as_nanos().div_ceil(1_000_000)).unwrap();
    assert!(measured_wall_millis > 0);
    let accepted = AttemptSuccess::new(
        &attempt,
        vec![ValuePayload::Inline(Box::new(InlineValue::String(
            "retained".into(),
        )))],
        Vec::new(),
        measured_wall_millis,
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
    let saved_wall_millis = checkpoint.completed_task_wall_millis[&task.request.id];
    assert!(saved_wall_millis >= measured_wall_millis);
    let committed = pool.snapshot().tasks[&task.request.id].committed.clone();
    let bundle = Arc::clone(
        &pool
            .direct_bundle
            .as_ref()
            .expect("direct fixture bundle")
            .bytes,
    );
    drop(pool);
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        bundle,
        |_| panic!("inline result needs no object reads"),
    )
    .unwrap();
    let attempts_before_replay = restored.snapshot().attempts.len();
    let completion = restored.submit(task.clone(), program).unwrap();
    assert_eq!(
        completion.original_graph_wall_millis(),
        Some(saved_wall_millis)
    );
    assert_eq!(completion.wait().await.unwrap(), accepted);
    assert_eq!(
        restored.snapshot().tasks[&task.request.id].committed,
        committed
    );
    assert_eq!(restored.snapshot().attempts.len(), attempts_before_replay);
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
            Arc::clone(
                &pool
                    .direct_bundle
                    .as_ref()
                    .expect("direct fixture bundle")
                    .bytes
            ),
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
            Arc::clone(
                &pool
                    .direct_bundle
                    .as_ref()
                    .expect("direct fixture bundle")
                    .bytes
            ),
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
        use crate::remote::clock_observer::ClockCommand;
        use crate::remote::RemoteObservedWave;
        use runmat_execution::security::{
            EndpointIdentitySigner, EndpointTrustPolicy, NodeClockEndV1, NodeClockOwnerV1,
            NodeClockStartV1,
        };

        let (pool, program, task) = unsubmitted_fixture().await;
        let task_id = task.request.id;
        let run_id = pool.value_scope().to_owned();
        let (clock_sender, mut clock_commands) = tokio::sync::mpsc::channel(4);
        let clock_owner = tokio::spawn(async move {
            let signer = EndpointIdentitySigner::from_secret([41; 32]).unwrap();
            let fingerprint = signer.fingerprint();
            let policy = EndpointTrustPolicy {
                permitted_tiers: Default::default(),
                trusted_identity_fingerprints: [fingerprint.clone()].into_iter().collect(),
                allowed_attestation_classes: Default::default(),
                require_pinned_identity: true,
                now_unix_millis: 0,
                maximum_clock_skew_millis: 0,
            };
            let mut original_start: Option<NodeClockStartV1> = None;
            while let Some(command) = clock_commands.recv().await {
                match command {
                    ClockCommand::Begin(reply) => {
                        assert!(original_start.is_none());
                        let reading = runmat_process_host::clock::observe_monotonic().unwrap();
                        let mut start = NodeClockStartV1 {
                            schema_version: 1,
                            owner: NodeClockOwnerV1 {
                                org_id: "org-restored-waiter".into(),
                                cluster_id: "cluster-restored-waiter".into(),
                                node_id: "node-restored-waiter".into(),
                                run_id: run_id.clone(),
                                driver_allocation_lease_id: "driver-restored-waiter".into(),
                                driver_allocation_fence: 7,
                                driver_process_launch_nonce: [13; 16],
                                observation_id: [15; 16],
                                clock_domain_nonce: [17; 16],
                                ticks_per_second: reading.ticks_per_second,
                                identity_public_key: signer.public_key().to_vec(),
                                identity_fingerprint: fingerprint.clone(),
                            },
                            started_monotonic_ticks: reading.ticks,
                            signature: Vec::new(),
                        };
                        signer.sign_node_clock_start(&mut start).unwrap();
                        policy
                            .verify_node_clock_start(&start, &fingerprint)
                            .unwrap();
                        original_start = Some(start.clone());
                        reply.send(Ok(start)).unwrap();
                    }
                    ClockCommand::Checkpoint(reply) | ClockCommand::Close(reply) => {
                        reply.send(Ok(())).unwrap();
                    }
                    ClockCommand::End { start, respond } => {
                        assert_eq!(original_start.as_ref(), Some(start.as_ref()));
                        let reading = runmat_process_host::clock::observe_monotonic().unwrap();
                        assert_eq!(reading.ticks_per_second, start.owner.ticks_per_second);
                        let mut end = NodeClockEndV1 {
                            schema_version: 1,
                            owner: start.owner.clone(),
                            start_receipt_digest: start.receipt_digest().unwrap(),
                            demand_challenge: [19; 16],
                            ended_monotonic_ticks: reading.ticks,
                            signature: Vec::new(),
                        };
                        signer.sign_node_clock_end(&mut end).unwrap();
                        policy.verify_node_clock_end(&end, &fingerprint).unwrap();
                        assert!(end.measured_wall_millis(&start).unwrap() >= 3);
                        respond.send(Ok(end)).unwrap();
                    }
                }
            }
        });
        pool.install_clock_commands(clock_sender.clone()).unwrap();
        let mut original_wave = RemoteObservedWave::begin(Arc::clone(&pool), &[task_id])
            .await
            .unwrap();
        original_wave.arm(&[task_id]).unwrap();
        let original_owner = *pool
            .task_clocks
            .lock()
            .unwrap()
            .get(&task_id)
            .expect("armed task clock owner");
        assert!(matches!(
            pool.task_wall_intervals.lock().unwrap().get(&task_id),
            Some(super::super::timing::TaskWallInterval::ObservedPending(owner))
                if *owner == original_owner
        ));
        let _original_waiter = pool.submit(task.clone(), program.clone()).unwrap();
        original_wave.checkpoint().await.unwrap();
        let checkpoint = pool.capture_checkpoint(1).unwrap();
        let original_start = checkpoint.clock_starts.values().next().unwrap().clone();
        assert_eq!(
            checkpoint.task_clocks.get(&task_id),
            Some(&original_start.owner.observation_id)
        );
        let bundle = Arc::clone(
            &pool
                .direct_bundle
                .as_ref()
                .expect("direct fixture bundle")
                .bytes,
        );
        drop(original_wave);
        if !complete_first {
            let missing = RemotePoolDriver::restore_checkpoint(
                checkpoint.clone(),
                authority(&checkpoint),
                Arc::clone(&bundle),
                |_| panic!("no objects"),
            )
            .unwrap();
            assert!(matches!(
                missing.task_wall_intervals.lock().unwrap().remove(&task_id),
                Some(super::super::timing::TaskWallInterval::ObservedPending(owner))
                    if owner == original_owner
            ));
            let (missing_sender, mut missing_commands) = tokio::sync::mpsc::channel(1);
            missing.install_clock_commands(missing_sender).unwrap();
            let missing_wave = RemoteWaveIdentity::new(missing.scope_id, &[task_id]).unwrap();
            assert!(matches!(
                missing.begin_clock_wave(missing_wave, &[task_id]).await,
                Err(NativeExecutionError::Protocol(_))
            ));
            assert!(matches!(
                missing_commands.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ));
        }
        drop(pool);
        let restored = RemotePoolDriver::restore_checkpoint(
            checkpoint.clone(),
            authority(&checkpoint),
            bundle,
            |_| panic!("no objects"),
        )
        .unwrap();
        restored.install_clock_commands(clock_sender).unwrap();
        assert!(matches!(
            restored.task_wall_intervals.lock().unwrap().get(&task_id),
            Some(super::super::timing::TaskWallInterval::ObservedPending(owner))
                if *owner == original_owner
        ));
        let mut resumed_wave = RemoteObservedWave::begin(Arc::clone(&restored), &[task_id])
            .await
            .unwrap();
        resumed_wave.arm(&[task_id]).unwrap();
        resumed_wave.checkpoint().await.unwrap();
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
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            let wall = resumed_wave.close().await.unwrap().unwrap();
            assert!(wall >= 3);
            restored.submit(task.clone(), program.clone()).unwrap()
        } else {
            let completion = restored.submit(task.clone(), program.clone()).unwrap();
            assert!(restored.submit(task.clone(), program.clone()).is_err());
            restored.apply_report(report);
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            let wall = resumed_wave.close().await.unwrap().unwrap();
            assert!(wall >= 3);
            completion
        };
        assert_eq!(completion.wait().await.unwrap(), accepted);
        let saved_wall = restored
            .task_wall_intervals
            .lock()
            .unwrap()
            .get(&task_id)
            .and_then(|interval| match interval {
                super::super::timing::TaskWallInterval::Completed(wall) => Some(*wall),
                _ => None,
            })
            .unwrap();
        assert!(saved_wall >= 3);
        let retained = restored.capture_checkpoint(2).unwrap();
        assert_eq!(
            retained.completed_task_wall_millis.get(&task_id),
            Some(&saved_wall)
        );
        assert_eq!(
            retained
                .clock_starts
                .get(&original_start.owner.observation_id),
            Some(&original_start)
        );
        assert_eq!(
            retained.clock_ends[&original_start.owner.observation_id]
                .measured_wall_millis(&original_start)
                .unwrap(),
            saved_wall
        );
        assert_eq!(restored.snapshot().attempts.len(), 1);
        assert!(restored.restored_completions.lock().unwrap().is_empty());
        drop(restored);
        clock_owner.await.unwrap();
    }
}

#[tokio::test]
async fn rejected_mixed_graph_does_not_consume_restored_waiter() {
    let (pool, program, task) = fixture().await;
    let checkpoint = pool.capture_checkpoint(1).unwrap();
    let restored = RemotePoolDriver::restore_checkpoint(
        checkpoint.clone(),
        authority(&checkpoint),
        Arc::clone(
            &pool
                .direct_bundle
                .as_ref()
                .expect("direct fixture bundle")
                .bytes,
        ),
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
async fn cold_active_attempt_preserves_admission_and_reservations_without_replay() {
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
        // A direct pool snapshot without an authenticated original node owner
        // cannot issue negative admission authority from an absent journal.
        assert!(checkpoint.authoritative_unstarted_inventory().is_err());
        let saved_pool = &checkpoint.scheduler.snapshot.pools[&pool.pool_id];
        let allocation = saved_pool.allocated.clone();
        let worker_allocation = saved_pool.workers[&old_attempt.worker_id].allocated.clone();
        let bundle = Arc::clone(&pool.direct_bundle.as_ref().unwrap().bytes);
        drop(pool);
        let restored = RemotePoolDriver::restore_checkpoint(
            checkpoint.clone(),
            authority(&checkpoint),
            bundle,
            |_| panic!("inline task has no object inventory"),
        )
        .unwrap();

        // Reattachment cannot extend the original admission, including retry budget.
        task.request.deadline_unix_millis =
            Some(original_submission.request.deadline_unix_millis.unwrap() + 3_600_000);
        task.request.retry = RetryPolicy::ExplicitlyIdempotent { max_attempts: 99 };
        task.priority = 0;
        let completion = restored.submit(task.clone(), program).unwrap();
        assert!(matches!(
            completion.wait().await,
            Err(crate::NativeProgramFailure::WorkerLost(_))
        ));

        let stale = AttemptSuccess::new(
            &old_attempt,
            vec![ValuePayload::Inline(Box::new(InlineValue::String(
                "stale".into(),
            )))],
            Vec::new(),
            13,
            0,
            0,
        )
        .unwrap();
        restored.apply_report(BackendReport::for_request(
            &old_attempt,
            AttemptReport::Succeeded { result: stale },
        ));
        let snapshot = restored.snapshot();
        assert_eq!(snapshot.driver_fence, 51);
        let task_record = &snapshot.tasks[&task.request.id];
        assert_eq!(task_record.submission, original_submission);
        assert_eq!(task_record.state, TaskState::Indeterminate);
        assert_eq!(task_record.attempt_count, 1);
        assert_eq!(task_record.active_attempt, Some(old_attempt.id));
        assert!(task_record.committed.is_none());
        let attempt = &snapshot.attempts[&old_attempt.id];
        assert_eq!(attempt.request, old_attempt);
        assert_eq!(attempt.state, AttemptState::Assigned);
        assert!(attempt.terminal.is_none());
        let uncertainty = attempt.unverified_retirement.as_ref().unwrap();
        assert_eq!(
            uncertainty.diagnostic,
            "driver authority replaced before a retirement certificate"
        );
        assert!(uncertainty.retired_at_driver_millis.is_none());
        assert_eq!(snapshot.pools[&restored.pool_id].allocated, allocation);
        assert_eq!(
            snapshot.pools[&restored.pool_id].workers[&old_attempt.worker_id].allocated,
            worker_allocation
        );

        let mut driver = restored.driver.lock().unwrap();
        assert!(driver
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                host: crate::test_support::host_inventory(),
                id: old_attempt.worker_id,
                pool_id: restored.pool_id,
                resources: inventory(1000),
            }))
            .is_err());
        let actions = driver
            .handle(DriverCommand::Tick {
                now_millis: snapshot.now_millis,
            })
            .unwrap();
        assert!(!actions
            .iter()
            .any(|action| matches!(action, DriverAction::Launch(_))));
        assert_eq!(driver.snapshot().attempts.len(), 1);
        // Original-owner retirement releases capacity but not replay authority;
        // that trusted boundary is covered by Runner's retirement_authority.rs
        // and checkpoint_recovery.rs, not a fabricated remote certificate here.
    }
}
