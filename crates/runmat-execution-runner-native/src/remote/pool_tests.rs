use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{Digest, ExecutionScopeId, OutputContract, PoolId, ProgramRevision, TaskId};
use runmat_execution_artifact::{
    archive::{write_bundle, ArchiveLimits},
    ExecutableForm, ExecutionBundleBuilder, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, PoolSpec, TaskSubmission, WorkerSpec,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};

use super::{run_remote_worker_quic, QuicRemoteWorkerChannel, RemoteWorkerQuicRequest};
use super::{RemoteAttempt, RemoteBundleReceipt, RemotePoolDriver, RemoteWorkerChannel};
use crate::NativeExecutionResult;

struct FakeWorker {
    node: String,
    spec: WorkerSpec,
    installs: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    maximum_active: Arc<AtomicUsize>,
    delay: Duration,
    cancellations: Arc<AtomicUsize>,
    drains: Arc<AtomicUsize>,
    bundle: Arc<Vec<u8>>,
}

struct ResultObjectWorker {
    spec: WorkerSpec,
    bundle: Arc<Vec<u8>>,
    reference: runmat_execution::value::ValueRef,
    bytes: Arc<Vec<u8>>,
    corrupt_download: bool,
}

#[async_trait]
impl RemoteWorkerChannel for ResultObjectWorker {
    fn node_identity(&self) -> &str {
        "node-result-object"
    }

    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        Ok(bundle_receipt(bundle_digest, bundle))
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        Ok(bundle_receipt(bundle_digest, &self.bundle))
    }

    async fn transfer_value(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<super::RemoteValueReceipt> {
        Ok(super::RemoteValueReceipt {
            value_id: reference.id,
            encoded_bytes: encoded.len() as u64,
        })
    }

    async fn execute(&self, _attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        Ok(AttemptReport::Succeeded {
            result: AttemptSuccess {
                outputs: vec![ValuePayload::Object(Box::new(self.reference.clone()))],
                result_objects: vec![self.reference.clone()],
            },
        })
    }

    async fn download_object(
        &self,
        reference: runmat_execution::value::ValueRef,
    ) -> NativeExecutionResult<Vec<u8>> {
        if reference != self.reference {
            return Err(crate::NativeExecutionError::Protocol(
                "unexpected result object reference".into(),
            ));
        }
        if self.corrupt_download {
            Ok(b"substituted result bytes".to_vec())
        } else {
            Ok(self.bytes.as_ref().clone())
        }
    }

    async fn cancel(
        &self,
        _request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        Ok(())
    }
}

#[async_trait]
impl RemoteWorkerChannel for FakeWorker {
    fn node_identity(&self) -> &str {
        &self.node
    }

    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }

    async fn install_bundle(
        &self,
        bundle_digest: Digest,
        bundle: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.installs.fetch_add(1, Ordering::SeqCst);
        Ok(bundle_receipt(bundle_digest, bundle))
    }

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum_active.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(AttemptReport::Succeeded {
            result: AttemptSuccess {
                outputs: vec![ValuePayload::Inline(Box::new(InlineValue::String(
                    attempt.scheduling.worker_id.to_string(),
                )))],
                result_objects: Vec::new(),
            },
        })
    }

    async fn activate_bundle(
        &self,
        bundle_digest: Digest,
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        Ok(bundle_receipt(bundle_digest, &self.bundle))
    }

    async fn transfer_value(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<super::RemoteValueReceipt> {
        Ok(super::RemoteValueReceipt {
            value_id: reference.id,
            encoded_bytes: encoded.len() as u64,
        })
    }

    async fn cancel(
        &self,
        _request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        self.cancellations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        self.drains.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_pool_installs_once_per_node_and_schedules_concurrently() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-scope"]);
    let pool_id = PoolId::derive(&[b"remote-pool"]);
    let resources = inventory(3_000);
    let bundle = Arc::new(executable_bundle().await.2);
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 2,
            max_workers: 3,
            max_in_flight: 3,
            resource_limit: resources,
        },
        7,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let installs = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let maximum_active = Arc::new(AtomicUsize::new(0));
    for (ordinal, node) in [(0_u8, "node-a"), (1, "node-b"), (2, "node-a")] {
        pool.add_worker(Arc::new(FakeWorker {
            node: node.into(),
            spec: WorkerSpec {
                id: WorkerId::derive(&[b"worker", &[ordinal]]),
                pool_id,
                resources: inventory(1_000),
            },
            installs: Arc::clone(&installs),
            active: Arc::clone(&active),
            maximum_active: Arc::clone(&maximum_active),
            delay: Duration::from_millis(25),
            cancellations: Arc::new(AtomicUsize::new(0)),
            drains: Arc::new(AtomicUsize::new(0)),
            bundle: Arc::clone(&bundle),
        }))
        .await
        .unwrap();
    }
    assert_eq!(installs.load(Ordering::SeqCst), 2);

    let (program, artifact_id) = program().await;
    let mut completions = Vec::new();
    for ordinal in 0_u8..3 {
        let task_id = TaskId::derive(&[b"task", &[ordinal]]);
        completions.push(
            pool.submit(
                TaskSubmission {
                    request: TaskRequest {
                        id: task_id,
                        scope_id,
                        pool_id,
                        program_artifact_id: artifact_id,
                        callable: Callable {
                            owner_identity: "remote-test".into(),
                            qualified_name: "answer".into(),
                            entrypoint_digest: Digest::sha256(b"answer"),
                        },
                        inputs: Vec::new(),
                        outputs: OutputContract {
                            requested_outputs: 1,
                        },
                        resources: request(),
                        retry: RetryPolicy::Never,
                        deadline_unix_millis: None,
                    },
                    dependencies: BTreeSet::new(),
                    priority: 0,
                },
                program.clone(),
            )
            .unwrap(),
        );
    }
    for completion in completions {
        assert_eq!(completion.wait().await.unwrap().outputs.len(), 1);
    }
    assert!(maximum_active.load(Ordering::SeqCst) >= 2);
    assert!(pool
        .snapshot()
        .tasks
        .values()
        .all(|task| { task.state == runmat_execution::state::TaskState::Succeeded }));
}

#[tokio::test]
async fn remote_pool_downloads_and_verifies_externalized_objects_before_success() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-object-scope"]);
    let pool_id = PoolId::derive(&[b"remote-object-pool"]);
    let bundle = Arc::new(executable_bundle().await.2);
    let pool = RemotePoolDriver::new_with_value_scope(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        8,
        bundle.as_ref().clone(),
        "run-result-object",
    )
    .unwrap();
    let bytes = Arc::new(b"canonical externalized result".to_vec());
    let reference = object_reference("run-result-object", &bytes);
    pool.add_worker(Arc::new(ResultObjectWorker {
        spec: WorkerSpec {
            id: WorkerId::derive(&[b"result-object-worker"]),
            pool_id,
            resources: inventory(1_000),
        },
        bundle,
        reference: reference.clone(),
        bytes: Arc::clone(&bytes),
        corrupt_download: false,
    }))
    .await
    .unwrap();
    let (result_program, artifact_id) = program().await;
    let success = pool
        .submit(
            submission(
                scope_id,
                pool_id,
                TaskId::derive(&[b"result-object-task"]),
                artifact_id,
            ),
            result_program,
        )
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(success.result_objects, vec![reference.clone()]);
    assert_eq!(
        pool.execution_object(&reference).unwrap().unwrap().as_ref(),
        bytes.as_slice()
    );

    let mut substituted = reference;
    substituted.logical_digest = Digest::sha256(b"substituted");
    assert!(pool.execution_object(&substituted).is_err());

    let corrupt_scope = ExecutionScopeId::derive(&[b"remote-corrupt-object-scope"]);
    let corrupt_pool_id = PoolId::derive(&[b"remote-corrupt-object-pool"]);
    let corrupt_bundle = Arc::new(executable_bundle().await.2);
    let corrupt_pool = RemotePoolDriver::new_with_value_scope(
        corrupt_scope,
        PoolSpec {
            id: corrupt_pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        9,
        corrupt_bundle.as_ref().clone(),
        "run-result-object",
    )
    .unwrap();
    corrupt_pool
        .add_worker(Arc::new(ResultObjectWorker {
            spec: WorkerSpec {
                id: WorkerId::derive(&[b"corrupt-result-object-worker"]),
                pool_id: corrupt_pool_id,
                resources: inventory(1_000),
            },
            bundle: corrupt_bundle,
            reference: substituted.clone(),
            bytes,
            corrupt_download: true,
        }))
        .await
        .unwrap();
    let (program, artifact_id) = program().await;
    assert!(corrupt_pool
        .submit(
            submission(
                corrupt_scope,
                corrupt_pool_id,
                TaskId::derive(&[b"corrupt-result-object-task"]),
                artifact_id,
            ),
            program,
        )
        .unwrap()
        .wait()
        .await
        .is_err());
}

#[tokio::test]
async fn remote_pool_cancels_active_work_and_fences_lost_workers() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-cancel-scope"]);
    let pool_id = PoolId::derive(&[b"remote-cancel-pool"]);
    let bundle = Arc::new(executable_bundle().await.2);
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        11,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let cancellations = Arc::new(AtomicUsize::new(0));
    let drains = Arc::new(AtomicUsize::new(0));
    let worker_id = WorkerId::derive(&[b"cancel-worker"]);
    pool.add_worker(Arc::new(FakeWorker {
        node: "node-cancel".into(),
        spec: WorkerSpec {
            id: worker_id,
            pool_id,
            resources: inventory(1_000),
        },
        installs: Arc::new(AtomicUsize::new(0)),
        active: Arc::clone(&active),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        delay: Duration::from_secs(30),
        cancellations: Arc::clone(&cancellations),
        drains: Arc::clone(&drains),
        bundle: Arc::clone(&bundle),
    }))
    .await
    .unwrap();
    let (cancel_program, artifact_id) = program().await;
    let task_id = TaskId::derive(&[b"cancel-task"]);
    let completion = pool
        .submit(
            submission(scope_id, pool_id, task_id, artifact_id),
            cancel_program,
        )
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while active.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pool.cancel(runmat_execution::CancellationReason::User)
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), completion.wait())
            .await
            .unwrap()
            .is_err()
    );
    assert_eq!(
        pool.snapshot().tasks.get(&task_id).unwrap().state,
        runmat_execution::state::TaskState::Cancelled
    );
    assert_eq!(cancellations.load(Ordering::SeqCst), 1);

    let scope_id = ExecutionScopeId::derive(&[b"remote-loss-scope"]);
    let pool_id = PoolId::derive(&[b"remote-loss-pool"]);
    let bundle = Arc::new(executable_bundle().await.2);
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        12,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let drains = Arc::new(AtomicUsize::new(0));
    let worker_id = WorkerId::derive(&[b"lost-worker"]);
    pool.add_worker(Arc::new(FakeWorker {
        node: "node-loss".into(),
        spec: WorkerSpec {
            id: worker_id,
            pool_id,
            resources: inventory(1_000),
        },
        installs: Arc::new(AtomicUsize::new(0)),
        active: Arc::clone(&active),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        delay: Duration::from_secs(30),
        cancellations: Arc::new(AtomicUsize::new(0)),
        drains: Arc::clone(&drains),
        bundle,
    }))
    .await
    .unwrap();
    let (lost_program, artifact_id) = program().await;
    let lost_task_id = TaskId::derive(&[b"lost-task"]);
    let completion = pool
        .submit(
            submission(scope_id, pool_id, lost_task_id, artifact_id),
            lost_program,
        )
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while active.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pool.remove_worker(worker_id, true).await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), completion.wait())
            .await
            .unwrap()
            .is_err()
    );
    assert_eq!(
        pool.snapshot().tasks.get(&lost_task_id).unwrap().state,
        runmat_execution::state::TaskState::Indeterminate
    );
    assert_eq!(drains.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn pinned_quic_worker_executes_only_the_installed_exact_bundle() {
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
    let scope_id = ExecutionScopeId::derive(&[b"quic-scope"]);
    let pool_id = PoolId::derive(&[b"quic-pool"]);
    let worker = WorkerSpec {
        id: WorkerId::derive(&[b"quic-worker"]),
        pool_id,
        resources: inventory(1_000),
    };
    let task_id = TaskId::derive(&[b"quic-task"]);
    let (mut program, artifact_id, bundle) = executable_bundle_with_input().await;
    let value = ValuePayload::Inline(Box::new(InlineValue::String("transferred".into())));
    let encoded_value = serde_json::to_vec(&value).unwrap();
    let value_reference = runmat_execution::value::ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
        id: runmat_execution::identity::ValueId::derive(&[b"quic-value"]),
        logical_digest: value.logical_digest().unwrap(),
        encoded_length: encoded_value.len() as u64,
        media_type: "application/vnd.runmat.value+json".into(),
        value_schema: "runmat-value-payload-v1".into(),
        encryption_context: Digest::sha256(b"worker-session"),
        kind: runmat_execution::value::ValueRefKind::DriverObject,
        authorization_scope: "run-quic-worker".into(),
        resident_fence: None,
    };
    let object = ValuePayload::Object(Box::new(value_reference.clone()));
    program.arguments = vec![object.clone()];
    let scheduling = runmat_execution_runner::AttemptRequest {
        id: runmat_execution::identity::AttemptId::derive(&[b"quic-attempt"]),
        task_id,
        scope_id,
        worker_id: worker.id,
        ordinal: 1,
        driver_fence: 9,
        task: TaskRequest {
            id: task_id,
            scope_id,
            pool_id,
            program_artifact_id: artifact_id,
            callable: Callable {
                owner_identity: "remote-test".into(),
                qualified_name: "answer".into(),
                entrypoint_digest: Digest::sha256(b"answer"),
            },
            inputs: vec![object],
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources: request(),
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([5; 32]).unwrap();
    let server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener,
        run_identity: "run-quic-worker".into(),
        worker: worker.clone(),
        driver_fence: 9,
        session_id: [7; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let channel = QuicRemoteWorkerChannel::connect(
            super::RemoteWorkerChannelConfig {
                run_identity: "run-quic-worker".into(),
                node_identity: "node-quic".into(),
                worker,
                driver_fence: 9,
                session_id: [7; 16],
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
        let digest = Digest::sha256(&bundle);
        let expected_receipt = bundle_receipt(digest, &bundle);
        assert_eq!(
            channel.install_bundle(digest, &bundle).await.unwrap(),
            expected_receipt
        );
        let object_bytes = vec![0x5a; 600 * 1024];
        let object_reference = object_reference("run-quic-worker", &object_bytes);
        let receipt = channel
            .transfer_object(object_reference.clone(), &object_bytes)
            .await
            .unwrap();
        assert!(receipt.complete);
        assert_eq!(receipt.next_offset, object_bytes.len() as u64);
        assert_eq!(
            channel
                .transfer_object(object_reference.clone(), &object_bytes)
                .await
                .unwrap(),
            receipt
        );
        assert_eq!(
            channel.download_object(object_reference,).await.unwrap(),
            object_bytes
        );
        assert_eq!(
            channel
                .transfer_value(value_reference.clone(), &encoded_value)
                .await
                .unwrap(),
            super::RemoteValueReceipt {
                value_id: value_reference.id,
                encoded_bytes: encoded_value.len() as u64,
            }
        );
        let report = channel
            .execute(RemoteAttempt {
                scheduling,
                program,
            })
            .await
            .unwrap();
        assert!(
            matches!(report, AttemptReport::Succeeded { .. }),
            "unexpected remote attempt report: {report:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        channel.drain().await.unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
}

fn inventory(cpu_millicores: u32) -> ResourceInventory {
    ResourceInventory {
        cpu_millicores,
        memory_bytes: 3 * 1024 * 1024,
        scratch_bytes: 3 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    }
}

fn request() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1024,
        scratch_bytes: 1024,
        max_wall_millis: 60_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 1024 * 1024,
        max_relay_bytes: 1024 * 1024,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    }
}

fn object_reference(authorization_scope: &str, bytes: &[u8]) -> runmat_execution::value::ValueRef {
    runmat_execution::value::ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
        id: runmat_execution::identity::ValueId::derive(&[b"remote-execution-object", bytes]),
        logical_digest: Digest::sha256(bytes),
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.test-object".into(),
        value_schema: "runmat.test-object.v1".into(),
        encryption_context: Digest::sha256(b"remote-object-context"),
        kind: runmat_execution::value::ValueRefKind::ResultObject,
        authorization_scope: authorization_scope.into(),
        resident_fence: None,
    }
}

fn submission(
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    task_id: TaskId,
    artifact_id: ArtifactId,
) -> TaskSubmission {
    TaskSubmission {
        request: TaskRequest {
            id: task_id,
            scope_id,
            pool_id,
            program_artifact_id: artifact_id,
            callable: Callable {
                owner_identity: "remote-test".into(),
                qualified_name: "answer".into(),
                entrypoint_digest: Digest::sha256(b"answer"),
            },
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources: request(),
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: BTreeSet::new(),
        priority: 0,
    }
}

async fn program() -> (ProgramExecutionRequest, ArtifactId) {
    let (program, artifact_id, _) = executable_bundle().await;
    (program, artifact_id)
}

async fn executable_bundle() -> (ProgramExecutionRequest, ArtifactId, Vec<u8>) {
    build_executable_bundle(false).await
}

async fn executable_bundle_with_input() -> (ProgramExecutionRequest, ArtifactId, Vec<u8>) {
    build_executable_bundle(true).await
}

async fn build_executable_bundle(
    accepts_input: bool,
) -> (ProgramExecutionRequest, ArtifactId, Vec<u8>) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-pool\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let source_text = if accepts_input {
        "function y = answer(x); y = x; end\n"
    } else {
        "function y = answer(); y = 42; end\n"
    };
    std::fs::write(project_root.path().join("src/answer.m"), source_text).unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_revision = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("root", "src/answer.m", source_text),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope_for(Some("answer")).unwrap();
    let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision.clone(),
        entrypoint: function.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-pool-test"),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let executable_bytes = envelope.canonical_bytes().unwrap();
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV3, executable_bytes)
            .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, recipe.program_revision.clone())
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            recipe.clone(),
            ExecutableForm::ExecutableUnitV3,
            artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    assert!(!bundle.requires_source_project());
    assert!(bundle.objects.is_empty());
    assert!(bundle.manifest.sources.is_empty());
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();
    let program = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    (program, artifact_id, bundle_bytes)
}

fn bundle_receipt(bundle_digest: Digest, bytes: &[u8]) -> RemoteBundleReceipt {
    let bundle = runmat_execution_artifact::archive::read_bundle(
        bytes,
        runmat_execution_artifact::archive::ArchiveLimits::default(),
    )
    .unwrap();
    RemoteBundleReceipt {
        bundle_digest,
        bundle_identity: bundle.identity().unwrap(),
        project_revision: bundle.manifest.project_revision,
        stored_bytes: bytes.len() as u64,
    }
}
