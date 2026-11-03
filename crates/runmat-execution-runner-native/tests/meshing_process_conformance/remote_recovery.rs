use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use runmat_execution::identity::{AttemptId, ValueId};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_runner::{AttemptReport, AttemptRequest, WorkerSpec};
use runmat_execution_runner_native::{
    AuthenticatedPhysicalRetirement, NativeExecutionResult, QuicRemoteWorkerChannel, RemoteAttempt,
    RemoteBundleReceipt, RemoteObjectReceipt, RemoteValueReceipt, RemoteWorkerChannel,
};

use super::*;

struct ObservedRemoteChannel {
    inner: Arc<QuicRemoteWorkerChannel>,
    hold_report: bool,
    report_ready: Arc<tokio::sync::Notify>,
    release_report: Arc<tokio::sync::Notify>,
    retirement_available: Arc<AtomicBool>,
    transferred_objects: AtomicUsize,
    executions: AtomicUsize,
    attempt_ids: Mutex<Vec<AttemptId>>,
}

#[async_trait]
impl RemoteWorkerChannel for ObservedRemoteChannel {
    fn node_identity(&self) -> &str {
        self.inner.node_identity()
    }

    fn worker(&self) -> &WorkerSpec {
        self.inner.worker()
    }

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        self.inner.describe_worker().await
    }

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.inner.install_bundle(bundle_digest, bundle).await
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.inner.activate_bundle(bundle_digest).await
    }

    async fn transfer_value(
        &self,
        reference: ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteValueReceipt> {
        self.inner.transfer_value(reference, encoded).await
    }

    async fn transfer_object(
        &self,
        reference: ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        self.transferred_objects.fetch_add(1, Ordering::AcqRel);
        self.inner.transfer_object(reference, encoded).await
    }

    async fn download_object(&self, reference: ValueRef) -> NativeExecutionResult<Vec<u8>> {
        self.inner.download_object(reference).await
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        self.executions.fetch_add(1, Ordering::AcqRel);
        self.attempt_ids
            .lock()
            .expect("observed attempt registry poisoned")
            .push(attempt.scheduling.id);
        let report = self.inner.execute(attempt).await;
        if self.hold_report {
            self.report_ready.notify_one();
            self.release_report.notified().await;
        }
        report
    }

    async fn authenticated_physical_retirement(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<Option<AuthenticatedPhysicalRetirement>> {
        if !self.retirement_available.load(Ordering::Acquire) {
            return Ok(None);
        }
        self.inner.authenticated_physical_retirement(request).await
    }

    async fn acknowledge_physical_retirement(
        &self,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<()> {
        self.inner.acknowledge_physical_retirement(request).await
    }

    fn drain_progress(
        &self,
        attempt_id: AttemptId,
    ) -> Vec<runmat_execution_runner_native::ProgramProgress> {
        self.inner.drain_progress(attempt_id)
    }

    async fn cancel(
        &self,
        request: &AttemptRequest,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        self.inner.cancel(request, reason).await
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        self.inner.drain().await
    }
}

pub(super) async fn run() {
    tokio::task::LocalSet::new()
        .run_until(async {
            run_inner(RetirementDelivery::BeforeReport).await;
            run_inner(RetirementDelivery::WithLateReport).await;
        })
        .await;
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RetirementDelivery {
    BeforeReport,
    WithLateReport,
}

async fn run_inner(delivery: RetirementDelivery) {
    const RUN_ID: &str = "remote-meshing-recovery-run";
    let (host, request, _, _, bundle) = remote_fixture(RUN_ID);
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"remote-recovery-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"remote-recovery-pool"]);
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([29; 32]).unwrap();
    let first_worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-recovery-worker-1"]),
        pool_id,
        resources: worker_inventory(),
        host: worker_host_inventory(),
    };
    let second_worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-recovery-worker-2"]),
        pool_id,
        resources: worker_inventory(),
        host: worker_host_inventory(),
    };
    let (first_server, first_inner) = start_remote_worker(
        RUN_ID,
        "remote-recovery-node-1",
        first_worker.clone(),
        41,
        [31; 16],
        run_key.clone(),
    )
    .await;
    let (second_server, second_inner) = start_remote_worker(
        RUN_ID,
        "remote-recovery-node-2",
        second_worker.clone(),
        41,
        [32; 16],
        run_key,
    )
    .await;
    let report_ready = Arc::new(tokio::sync::Notify::new());
    let release_report = Arc::new(tokio::sync::Notify::new());
    let retirement_available = Arc::new(AtomicBool::new(
        delivery == RetirementDelivery::BeforeReport,
    ));
    let first = Arc::new(ObservedRemoteChannel {
        inner: Arc::clone(&first_inner),
        hold_report: true,
        report_ready: Arc::clone(&report_ready),
        release_report: Arc::clone(&release_report),
        retirement_available: Arc::clone(&retirement_available),
        transferred_objects: AtomicUsize::new(0),
        executions: AtomicUsize::new(0),
        attempt_ids: Mutex::new(Vec::new()),
    });
    let second = Arc::new(ObservedRemoteChannel {
        inner: Arc::clone(&second_inner),
        hold_report: false,
        report_ready: Arc::new(tokio::sync::Notify::new()),
        release_report: Arc::new(tokio::sync::Notify::new()),
        retirement_available: Arc::new(AtomicBool::new(true)),
        transferred_objects: AtomicUsize::new(0),
        executions: AtomicUsize::new(0),
        attempt_ids: Mutex::new(Vec::new()),
    });
    let pool = RemotePoolDriver::new_with_value_scope(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: worker_inventory(),
        },
        41,
        bundle.as_ref().clone(),
        RUN_ID,
    )
    .unwrap();
    pool.add_worker(first.clone()).await.unwrap();

    let prerequisite_bytes = Arc::<[u8]>::from(vec![0x5c; 600 * 1024]);
    let prerequisite = execution_object_reference(RUN_ID, prerequisite_bytes.as_ref());
    pool.register_execution_object(prerequisite.clone(), Arc::clone(&prerequisite_bytes))
        .unwrap();
    register_host_invocation(&pool, &host, &request);
    let retry_submission = submission_for(scope_id, pool_id, &host, &request);
    let task_id = retry_submission.request.id;
    let mut completion = Some(pool.submit(retry_submission, request.clone()).unwrap());

    tokio::time::timeout(Duration::from_secs(10), report_ready.notified())
        .await
        .expect("first remote attempt did not reach its held result");
    pool.remove_worker(first_worker.id, true).await.unwrap();
    if delivery == RetirementDelivery::WithLateReport {
        // A direct pool has no durable original-worker owner. It reports loss
        // to its current waiter while retaining the unresolved scheduler task.
        // Later retirement may recover that task; it cannot change the already
        // delivered caller outcome into success.
        let (outcome, progress) = completion.take().unwrap().wait_with_progress().await;
        assert!(matches!(
            outcome,
            Err(runmat_execution_runner_native::NativeProgramFailure::WorkerLost(_))
        ));
        assert!(!progress.is_empty());
        assert!(progress
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence));
        let unresolved = pool.snapshot();
        assert_eq!(
            unresolved.tasks[&task_id].state,
            runmat_execution::state::TaskState::Indeterminate
        );
        assert!(unresolved.tasks[&task_id].committed.is_none());
        // The old route cannot supply retirement during membership loss. Its
        // original result later arrives with exact authenticated owner proof.
        retirement_available.store(true, Ordering::Release);
        release_report.notify_one();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let retired = pool.snapshot().attempts.values().any(|attempt| {
                    attempt.request.worker_id == first_worker.id
                        && attempt
                            .unverified_retirement
                            .as_ref()
                            .is_some_and(|record| record.retired_at_driver_millis.is_some())
                });
                if retired {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("late original result did not retire with its resource accounting");
    }
    pool.add_worker(second.clone()).await.unwrap();

    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if pool
                .snapshot()
                .tasks
                .get(&task_id)
                .is_some_and(|task| task.state == runmat_execution::state::TaskState::Succeeded)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|error| {
        panic!(
            "replacement remote worker did not commit the retried stage: {error}; snapshot: {:#?}",
            pool.snapshot()
        )
    });
    let success = match completion {
        Some(completion) => {
            let (outcome, progress) = completion.wait_with_progress().await;
            assert!(progress.len() >= 6);
            assert!(progress
                .windows(2)
                .all(|pair| pair[0].sequence < pair[1].sequence));
            outcome.unwrap()
        }
        None => pool.snapshot().tasks[&task_id]
            .committed
            .as_ref()
            .expect("late retirement recovers the retained task")
            .result
            .clone(),
    };
    let (outputs, result_objects) = success.values().expect("ordinary task result");
    let [ValuePayload::Object(remote_root)] = outputs else {
        panic!("retried remote meshing returned a non-object root")
    };
    let mut retry_store = TestStore::default();
    for reference in result_objects {
        retry_store.0.insert(
            reference.logical_digest,
            pool.execution_object(reference)
                .unwrap()
                .expect("committed retry result object")
                .to_vec(),
        );
    }
    import_result_publication(
        &retry_store,
        remote_root,
        host.artifact_access.clone(),
        limits().inventory,
    )
    .unwrap();

    release_report.notify_one();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.tasks.get(&task_id).unwrap().attempt_count, 2);
    for attempt in snapshot.attempts.values() {
        let resources = attempt
            .terminal
            .as_ref()
            .and_then(|terminal| terminal.resources.as_deref())
            .expect("both original and retry retain their exact allocated usage");
        resources.validate_against(&attempt.request).unwrap();
        assert!(resources.resources.usage.wall_millis > 0);
        assert!(resources.resources.usage.cpu_core_millis > 0);
    }
    assert!(snapshot.attempts.values().any(|attempt| {
        attempt.state == runmat_execution::state::AttemptState::Lost
            && attempt
                .unverified_retirement
                .as_ref()
                .is_some_and(|record| record.retired_at_driver_millis.is_some())
    }));
    assert_eq!(
        snapshot
            .attempts
            .values()
            .filter(|attempt| attempt.state == runmat_execution::state::AttemptState::Lost)
            .count(),
        1
    );
    assert_eq!(first.executions.load(Ordering::Acquire), 1);
    assert_eq!(second.executions.load(Ordering::Acquire), 1);
    assert_eq!(first.transferred_objects.load(Ordering::Acquire), 2);
    assert_eq!(second.transferred_objects.load(Ordering::Acquire), 2);
    assert_eq!(
        pool.execution_object(&prerequisite)
            .unwrap()
            .expect("registered prerequisite remains reusable")
            .as_ref(),
        prerequisite_bytes.as_ref()
    );

    first_inner.drain().await.unwrap();
    second_inner.drain().await.unwrap();
    first_server.await.unwrap().unwrap();
    second_server.await.unwrap().unwrap();
}

async fn start_remote_worker(
    run_id: &str,
    node_identity: &str,
    worker: WorkerSpec,
    driver_fence: u64,
    session_id: [u8; 16],
    run_key: runmat_execution_artifact::encryption::RunKeyMaterial,
) -> (
    tokio::task::JoinHandle<NativeExecutionResult<()>>,
    Arc<QuicRemoteWorkerChannel>,
) {
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let authority = listener.local_addr().unwrap();
    let server = tokio::task::spawn_local(run_remote_meshing_worker_quic(
        RemoteMeshingWorkerQuicRequest {
            worker: RemoteWorkerQuicRequest {
                listener,
                run_identity: run_id.to_string(),
                worker: worker.clone(),
                driver_fence,
                session_id,
                run_key: run_key.clone(),
                limits: FrameLimits::default(),
            },
            kernel: Arc::new(AdmissionKernel),
            meshing_limits: limits(),
        },
    ));
    let channel = QuicRemoteWorkerChannel::connect(
        RemoteWorkerChannelConfig {
            run_identity: run_id.into(),
            node_identity: node_identity.into(),
            worker,
            driver_fence,
            session_id,
            run_key,
            limits: FrameLimits::default(),
        },
        &PinnedQuicEndpoint {
            authority,
            server_name: "runmat.execution".into(),
            certificate_der,
        },
    )
    .await
    .unwrap();
    (server, channel)
}

fn execution_object_reference(authorization_scope: &str, bytes: &[u8]) -> ValueRef {
    ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: ValueId::derive(&[b"remote-meshing-prerequisite", bytes]),
        logical_digest: Digest::sha256(bytes),
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.meshing-prerequisite.v2+cbor".into(),
        value_schema: "runmat.meshing-prerequisite.v2".into(),
        encryption_context: Digest::sha256(b"remote-meshing-recovery-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: authorization_scope.into(),
        resident_fence: None,
    }
}
