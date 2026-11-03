use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{
    AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId, AcceleratorFeature,
    AcceleratorRequest, Capability, ResourceInventory, ResourceRequest,
};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{
    Digest, ExecutionScopeId, GangHandle, GangId, OutputContract, ParallelChunk,
    ParallelRandomnessContext, PoolHandle, PoolId, ProgramCallable, ProgramFunctionId,
    ProgramInvocationContext, ProgramRevision, TaskId,
};
use runmat_execution_artifact::{
    archive::{write_bundle, ArchiveLimits},
    ExecutableForm, ExecutionBundleBuilder, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, PoolSpec, TaskSubmission, WorkerSpec,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use tokio::sync::Notify;

use super::{
    run_remote_worker_quic, run_remote_worker_quic_ready_v1, QuicRemoteWorkerChannel,
    RemoteWorkerQuicRequest,
};
use super::{
    RemoteAttempt, RemoteBundleReceipt, RemotePoolDriver, RemoteSpmdGangProgram,
    RemoteWorkerChannel,
};
use crate::{NativeExecutionResult, NativeProgramFailure};

#[path = "pool_tests/source_lifetime.rs"]
mod source_lifetime;

mod environment;
mod input_inventory;
mod program;

struct FakeWorker {
    node: String,
    spec: WorkerSpec,
    installs: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    maximum_active: Arc<AtomicUsize>,
    start_gate: Option<Arc<ConcurrencyGate>>,
    delay: Duration,
    cancellations: Arc<AtomicUsize>,
    cancellation_signal: Arc<Notify>,
    cancel_ack_gate: Option<Arc<Notify>>,
    drains: Arc<AtomicUsize>,
    bundle: Arc<Vec<u8>>,
    described: Option<WorkerSpec>,
}

pub(in crate::remote) struct ResultObjectWorker {
    pub(in crate::remote) spec: WorkerSpec,
    pub(in crate::remote) bundle: Arc<Vec<u8>>,
    pub(in crate::remote) reference: runmat_execution::value::ValueRef,
    pub(in crate::remote) bytes: Arc<Vec<u8>>,
    pub(in crate::remote) corrupt_download: bool,
}

struct ConcurrencyGate {
    target: usize,
    arrivals: AtomicUsize,
    notify: tokio::sync::Notify,
}

impl ConcurrencyGate {
    fn new(target: usize) -> Self {
        Self {
            target,
            arrivals: AtomicUsize::new(0),
            notify: tokio::sync::Notify::new(),
        }
    }

    async fn arrive_and_wait(&self) {
        self.arrivals.fetch_add(1, Ordering::AcqRel);
        self.notify.notify_waiters();
        loop {
            let notified = self.notify.notified();
            if self.arrivals.load(Ordering::Acquire) >= self.target {
                return;
            }
            notified.await;
        }
    }
}

struct FailingRouteWorker {
    spec: WorkerSpec,
    bundle: Arc<Vec<u8>>,
    fail_execute: bool,
    fail_drain: bool,
}

struct ExecutingWorker {
    spec: WorkerSpec,
    bundle: Arc<Vec<u8>>,
    assignments: Arc<std::sync::Mutex<Vec<runmat_execution::ProgramExecutionAssignment>>>,
}

struct CompiledSpmdProgram {
    recipe: ProgramBuildRecipe,
    artifact: ProgramArtifact,
    region: runmat_types::ParallelRegionId,
    bundle: Vec<u8>,
    requested_outputs: u16,
}

#[async_trait]
impl RemoteWorkerChannel for ExecutingWorker {
    async fn authenticated_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<super::AuthenticatedPhysicalRetirement>> {
        Ok(Some(super::AuthenticatedPhysicalRetirement::issue(
            request.clone(),
        )))
    }
    async fn acknowledge_physical_retirement(
        &self,
        _: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    fn node_identity(&self) -> &str {
        "node-executing"
    }

    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        Ok(self.spec.clone())
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

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        let super::RemoteAttemptProgram::Direct(program) = attempt.program else {
            return Err(crate::NativeExecutionError::Protocol(
                "direct pool fixture received a graph attempt".into(),
            ));
        };
        let assignment = program.assignment.clone().ok_or_else(|| {
            crate::NativeExecutionError::Protocol(
                "remote execution request is missing scheduler placement".into(),
            )
        })?;
        self.assignments
            .lock()
            .expect("assignment observations poisoned")
            .push(assignment);
        let response = tokio::task::spawn_blocking(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("isolated worker runtime")
                .block_on(crate::execute_host_program_request(*program))
        })
        .await
        .map_err(|error| {
            crate::NativeExecutionError::Protocol(format!(
                "isolated remote worker did not complete: {error}"
            ))
        })?;
        Ok(super::worker_execution::report(
            &attempt.scheduling,
            response,
            1,
        ))
    }

    async fn cancel(
        &self,
        _request: &runmat_execution_runner::AttemptRequest,
        _reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        Ok(())
    }
}

#[async_trait]
impl RemoteWorkerChannel for FailingRouteWorker {
    async fn authenticated_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<super::AuthenticatedPhysicalRetirement>> {
        if self.fail_execute {
            // This fixture models a permanently failed route, not a live
            // worker whose durable retirement receipt is still pending.
            Err(crate::NativeExecutionError::Protocol(
                "fixture route has no retirement authority".into(),
            ))
        } else {
            Ok(Some(super::AuthenticatedPhysicalRetirement::issue(
                request.clone(),
            )))
        }
    }
    async fn acknowledge_physical_retirement(
        &self,
        _: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    fn node_identity(&self) -> &str {
        "node-failing-route"
    }

    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        Ok(self.spec.clone())
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

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        if self.fail_execute {
            Err(crate::NativeExecutionError::Protocol(
                "fixture route failed".into(),
            ))
        } else {
            Ok(AttemptReport::Succeeded {
                result: AttemptSuccess::new(
                    &attempt.scheduling,
                    vec![ValuePayload::Inline(Box::new(InlineValue::String(
                        attempt.scheduling.worker_id.to_string().into(),
                    )))],
                    Vec::new(),
                    1,
                    0,
                    0,
                )
                .unwrap(),
            })
        }
    }

    async fn cancel(
        &self,
        _request: &runmat_execution_runner::AttemptRequest,
        _reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        if self.fail_drain {
            Err(crate::NativeExecutionError::Protocol(
                "fixture drain failed".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl RemoteWorkerChannel for ResultObjectWorker {
    async fn authenticated_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<super::AuthenticatedPhysicalRetirement>> {
        Ok(Some(super::AuthenticatedPhysicalRetirement::issue(
            request.clone(),
        )))
    }
    async fn acknowledge_physical_retirement(
        &self,
        _: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    fn node_identity(&self) -> &str {
        "node-result-object"
    }

    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        Ok(self.spec.clone())
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

    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        Ok(AttemptReport::Succeeded {
            result: AttemptSuccess::new(
                &attempt.scheduling,
                vec![ValuePayload::Object(Box::new(self.reference.clone()))],
                vec![self.reference.clone()],
                1,
                self.reference.encoded_length,
                0,
            )
            .unwrap(),
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
        _reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        Ok(())
    }
}

#[async_trait]
impl RemoteWorkerChannel for FakeWorker {
    async fn authenticated_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<super::AuthenticatedPhysicalRetirement>> {
        Ok((self.active.load(Ordering::SeqCst) == 0)
            .then(|| super::AuthenticatedPhysicalRetirement::issue(request.clone())))
    }
    async fn acknowledge_physical_retirement(
        &self,
        _: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    fn node_identity(&self) -> &str {
        &self.node
    }

    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }

    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        Ok(self.described.clone().unwrap_or_else(|| self.spec.clone()))
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
        if let Some(gate) = &self.start_gate {
            gate.arrive_and_wait().await;
        }
        let report = tokio::select! {
            () = tokio::time::sleep(self.delay) => AttemptReport::Succeeded {
                result: AttemptSuccess::new(
                    &attempt.scheduling,
                    vec![ValuePayload::Inline(Box::new(InlineValue::String(
                        attempt.scheduling.worker_id.to_string().into(),
                    )))],
                    Vec::new(),
                    1,
                    0,
                    0,
                ).unwrap(),
            },
            () = self.cancellation_signal.notified() => AttemptReport::Cancelled,
        };
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(report)
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
        _reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        self.cancellations.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.cancel_ack_gate {
            gate.notified().await;
        }
        self.cancellation_signal.notify_waiters();
        while self.active.load(Ordering::SeqCst) != 0 {
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    async fn drain(&self) -> NativeExecutionResult<()> {
        self.drains.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn direct_admission_pairs_the_original_task_assignment_and_host_policy() {
    use super::direct_admission::{validate, Admission};
    let (mut program, artifact_id, _) = executable_bundle().await;
    let scope_id = ExecutionScopeId::derive(&[b"remote-submit-rollback-scope"]);
    let pool_id = PoolId::derive(&[b"remote-submit-rollback-pool"]);
    let task_id = TaskId::derive(&[b"remote-submit-rollback-task"]);
    let mut task = submission(scope_id, pool_id, task_id, artifact_id).request;
    task.callable = Callable::for_program("remote-test", &program.callable);
    task.host = program
        .artifact
        .execution_host_requirement(&program.recipe, task.host.permitted_trust_tiers.clone())
        .unwrap();
    let pool_accepts = |program: &ProgramExecutionRequest, task: &TaskRequest| {
        validate(
            program,
            Admission::Pool {
                task,
                scope_id,
                pool_id,
            },
        )
    };
    assert!(pool_accepts(&program, &task).is_ok());
    let worker = WorkerSpec {
        id: WorkerId::derive(&[b"quic-worker"]),
        pool_id,
        resources: inventory(1_000),
        host: host_inventory(),
    };
    let mut scheduling = runmat_execution_runner::AttemptRequest {
        id: runmat_execution::identity::AttemptId::derive(&[b"quic-attempt"]),
        task_id,
        scope_id,
        worker_id: worker.id,
        ordinal: 1,
        driver_fence: 9,
        resource_assignment: Default::default(),
        task,
    };
    let worker_accepts = |program: &ProgramExecutionRequest,
                          scheduling: &runmat_execution_runner::AttemptRequest,
                          worker: &WorkerSpec| {
        validate(program, Admission::Worker { scheduling, worker })
    };
    assert!(worker_accepts(&program, &scheduling, &worker).is_err());
    program.assignment = Some(runmat_execution::ProgramExecutionAssignment {
        scope_id,
        pool_id,
        task_id,
        attempt_id: scheduling.id,
        worker_id: worker.id,
        backend: runmat_execution::PoolBackend::Remote,
        resources: scheduling.resource_assignment.clone(),
    });
    program.job_id = Some(runmat_execution::JobId::derive(&[b"quic-task"]));
    assert!(worker_accepts(&program, &scheduling, &worker).is_ok());
    assert!(pool_accepts(&program, &scheduling.task).is_err());
    let mut foreign = program.clone();
    foreign.assignment.as_mut().unwrap().scope_id = ExecutionScopeId::derive(&[b"quic-scope"]);
    assert!(worker_accepts(&foreign, &scheduling, &worker).is_err());
    let mut foreign_worker = worker.clone();
    foreign_worker.pool_id = PoolId::derive(&[b"quic-pool"]);
    assert!(worker_accepts(&program, &scheduling, &foreign_worker).is_err());
    foreign_worker = worker.clone();
    foreign_worker.host.trust_tier = runmat_execution::security::ExecutionTrustTier::HostedOrdinary;
    assert!(worker_accepts(&program, &scheduling, &foreign_worker).is_err());
    // Display/owner labels are not a second execution identity.
    scheduling.task.callable.owner_identity = "remote-parfor".into();
    assert!(worker_accepts(&program, &scheduling, &worker).is_ok());
    scheduling.task.invocation_context = ProgramInvocationContext::ParallelTask {
        task: runmat_execution::ParallelTaskContext {
            region: runmat_types::ParallelRegionId(runmat_types::RegionId {
                function: program.callable.semantic_function().unwrap(),
                ordinal: 0,
            }),
            chunk: ParallelChunk {
                ordinal: 0,
                start: 0,
                len: 1,
            },
            randomness: ParallelRandomnessContext::Inherit,
        },
    };
    assert!(worker_accepts(&program, &scheduling, &worker).is_err());

    // A recipe grant is checked at both boundaries, not inferred from a host
    // capability or from an otherwise self-consistent portable artifact.
    let (program, artifact_id, _) = program::build(
        "function y = answer(); y = 42; end",
        Vec::new(),
        vec![accelerator_request()],
    )
    .await;
    let mut task = submission(scope_id, pool_id, task_id, artifact_id).request;
    task.callable = Callable::for_program("remote-test", &program.callable);
    task.host = program
        .artifact
        .execution_host_requirement(&program.recipe, task.host.permitted_trust_tiers.clone())
        .unwrap();
    task.resources.accelerators = program.recipe.accelerators.clone();
    assert!(pool_accepts(&program, &task).is_ok());
    task.resources.accelerators.clear();
    assert!(pool_accepts(&program, &task).is_err());
}

#[tokio::test]
async fn rejected_submission_does_not_poison_remote_task_registries() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-submit-rollback-scope"]);
    let pool_id = PoolId::derive(&[b"remote-submit-rollback-pool"]);
    let bundle = executable_bundle().await.2;
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 0,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        6,
        bundle,
    )
    .unwrap();
    let (program, artifact_id) = program().await;
    let task_id = TaskId::derive(&[b"remote-submit-rollback-task"]);
    let mut rejected = submission(scope_id, pool_id, task_id, artifact_id);
    rejected.dependencies = BTreeSet::from([TaskId::derive(&[b"missing-dependency"])]);

    assert!(pool.submit(rejected, program.clone()).is_err());
    assert!(!pool.snapshot().tasks.contains_key(&task_id));

    let _completion = pool
        .submit(submission(scope_id, pool_id, task_id, artifact_id), program)
        .expect("a rejected submission must not reserve its task registries");
    assert!(pool.snapshot().tasks.contains_key(&task_id));
}

#[tokio::test]
async fn remote_task_graph_admission_is_atomic() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-graph-rollback-scope"]);
    let pool_id = PoolId::derive(&[b"remote-graph-rollback-pool"]);
    let bundle = executable_bundle().await.2;
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 0,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        8,
        bundle,
    )
    .unwrap();
    let (program, artifact_id) = program().await;
    let first_id = TaskId::derive(&[b"remote-graph-first"]);
    let second_id = TaskId::derive(&[b"remote-graph-second"]);
    let first = submission(scope_id, pool_id, first_id, artifact_id);
    let mut rejected = submission(scope_id, pool_id, second_id, artifact_id);
    rejected.dependencies = BTreeSet::from([TaskId::derive(&[b"remote-graph-missing"])]);

    assert!(pool
        .submit_graph(vec![
            (program.clone(), first.clone()),
            (program.clone(), rejected),
        ])
        .is_err());
    assert!(pool.snapshot().tasks.is_empty());

    let mut dependent = submission(scope_id, pool_id, second_id, artifact_id);
    dependent.dependencies = BTreeSet::from([first_id]);
    let handles = pool
        .submit_graph(vec![(program.clone(), first), (program, dependent)])
        .expect("a rejected graph must not reserve scheduler or remote registry state");
    assert_eq!(handles.len(), 2);
    let snapshot = pool.snapshot();
    assert!(snapshot.tasks.contains_key(&first_id));
    assert!(snapshot.tasks.contains_key(&second_id));
}

#[tokio::test]
async fn late_dependent_on_failed_remote_task_resolves_without_dispatch() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-late-terminal-scope"]);
    let pool_id = PoolId::derive(&[b"remote-late-terminal-pool"]);
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
        7,
        bundle.as_ref().clone(),
    )
    .unwrap();
    pool.add_worker(Arc::new(FailingRouteWorker {
        spec: WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: WorkerId::derive(&[b"remote-late-terminal-worker"]),
            pool_id,
            resources: inventory(1_000),
        },
        bundle,
        fail_execute: true,
        fail_drain: false,
    }))
    .await
    .unwrap();
    let (program, artifact_id) = program().await;
    let predecessor_id = TaskId::derive(&[b"remote-late-terminal-predecessor"]);
    let predecessor = pool
        .submit(
            submission(scope_id, pool_id, predecessor_id, artifact_id),
            program.clone(),
        )
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), predecessor.wait())
            .await
            .unwrap()
            .is_err()
    );

    let dependent_id = TaskId::derive(&[b"remote-late-terminal-dependent"]);
    let mut dependent = submission(scope_id, pool_id, dependent_id, artifact_id);
    dependent.dependencies = BTreeSet::from([predecessor_id]);
    let dependent = pool.submit(dependent, program).unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(1), dependent.wait())
        .await
        .expect("late terminal dependency must resolve without worker dispatch");
    assert!(outcome.is_err());
    assert!(matches!(
        pool.snapshot().tasks[&dependent_id].state,
        runmat_execution::state::TaskState::Failed
            | runmat_execution::state::TaskState::Indeterminate
            | runmat_execution::state::TaskState::Cancelled
    ));
}

#[tokio::test]
async fn remote_pool_rejects_worker_inventory_that_differs_from_description() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-description-scope"]);
    let pool_id = PoolId::derive(&[b"remote-description-pool"]);
    let bundle = Arc::new(executable_bundle().await.2);
    let advertised = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: WorkerId::derive(&[b"advertised-worker"]),
        pool_id,
        resources: inventory(1_000),
    };
    let mut described = advertised.clone();
    described.resources = inventory(500);
    let installs = Arc::new(AtomicUsize::new(0));
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        6,
        bundle.as_ref().clone(),
    )
    .unwrap();

    let error = pool
        .add_worker(Arc::new(FakeWorker {
            node: "node-description-mismatch".into(),
            spec: advertised,
            installs: Arc::clone(&installs),
            active: Arc::new(AtomicUsize::new(0)),
            maximum_active: Arc::new(AtomicUsize::new(0)),
            start_gate: None,
            delay: Duration::ZERO,
            cancellations: Arc::new(AtomicUsize::new(0)),
            cancellation_signal: Arc::new(Notify::new()),
            cancel_ack_gate: None,
            drains: Arc::new(AtomicUsize::new(0)),
            bundle,
            described: Some(described),
        }))
        .await
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("inventory differs from its encrypted description"));
    assert_eq!(installs.load(Ordering::SeqCst), 0);
    assert!(pool.snapshot().pools[&pool_id].workers.is_empty());
}

#[tokio::test]
async fn duplicate_worker_registration_preserves_the_original_route_and_state() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-duplicate-scope"]);
    let pool_id = PoolId::derive(&[b"remote-duplicate-pool"]);
    let worker_id = WorkerId::derive(&[b"remote-duplicate-worker"]);
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
        6,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let original_installs = Arc::new(AtomicUsize::new(0));
    let original_drains = Arc::new(AtomicUsize::new(0));
    pool.add_worker(Arc::new(FakeWorker {
        node: "node-original".into(),
        spec: WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: worker_id,
            pool_id,
            resources: inventory(1_000),
        },
        installs: Arc::clone(&original_installs),
        active: Arc::new(AtomicUsize::new(0)),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        start_gate: None,
        delay: Duration::ZERO,
        cancellations: Arc::new(AtomicUsize::new(0)),
        cancellation_signal: Arc::new(Notify::new()),
        cancel_ack_gate: None,
        drains: Arc::clone(&original_drains),
        bundle: Arc::clone(&bundle),
        described: None,
    }))
    .await
    .unwrap();
    let snapshot_before_duplicate = pool.snapshot();
    let replacement_installs = Arc::new(AtomicUsize::new(0));
    let replacement_drains = Arc::new(AtomicUsize::new(0));

    let error = pool
        .add_worker(Arc::new(FakeWorker {
            node: "node-replacement".into(),
            spec: WorkerSpec {
                host: crate::test_support::host_inventory(),
                id: worker_id,
                pool_id,
                resources: inventory(1_000),
            },
            installs: Arc::clone(&replacement_installs),
            active: Arc::new(AtomicUsize::new(0)),
            maximum_active: Arc::new(AtomicUsize::new(0)),
            start_gate: None,
            delay: Duration::ZERO,
            cancellations: Arc::new(AtomicUsize::new(0)),
            cancellation_signal: Arc::new(Notify::new()),
            cancel_ack_gate: None,
            drains: Arc::clone(&replacement_drains),
            bundle,
            described: None,
        }))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("already registered"));
    assert_eq!(pool.snapshot(), snapshot_before_duplicate);
    assert_eq!(original_installs.load(Ordering::SeqCst), 1);
    assert_eq!(replacement_installs.load(Ordering::SeqCst), 0);

    pool.remove_worker(worker_id, false).await.unwrap();
    assert_eq!(original_drains.load(Ordering::SeqCst), 1);
    assert_eq!(replacement_drains.load(Ordering::SeqCst), 0);
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
    let start_gate = Arc::new(ConcurrencyGate::new(3));
    for (ordinal, node) in [(0_u8, "node-a"), (1, "node-b"), (2, "node-a")] {
        pool.add_worker(Arc::new(FakeWorker {
            node: node.into(),
            spec: WorkerSpec {
                id: WorkerId::derive(&[b"worker", &[ordinal]]),
                pool_id,
                resources: inventory(1_000),
                host: host_inventory(),
            },
            installs: Arc::clone(&installs),
            active: Arc::clone(&active),
            maximum_active: Arc::clone(&maximum_active),
            start_gate: Some(Arc::clone(&start_gate)),
            delay: Duration::from_millis(25),
            cancellations: Arc::new(AtomicUsize::new(0)),
            cancellation_signal: Arc::new(Notify::new()),
            cancel_ack_gate: None,
            drains: Arc::new(AtomicUsize::new(0)),
            bundle: Arc::clone(&bundle),
            described: None,
        }))
        .await
        .unwrap();
    }
    assert_eq!(installs.load(Ordering::SeqCst), 2);

    let (program, artifact_id) = program().await;
    let tasks = (0_u8..3)
        .map(|ordinal| {
            let task_id = TaskId::derive(&[b"task", &[ordinal]]);
            (
                program.clone(),
                TaskSubmission {
                    request: TaskRequest {
                        console: None,
                        environment: runmat_execution::EnvironmentSnapshot::default(),
                        id: task_id,
                        scope_id,
                        pool_id,
                        schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                        executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                            artifact_id,
                        ),
                        callable: Callable::for_program("remote-test", &program.callable),
                        invocation_context: runmat_execution::ProgramInvocationContext::Direct,
                        inputs: Vec::new(),
                        outputs: OutputContract {
                            requested_outputs: 1,
                            owned_work: None,
                        },
                        resources: request(),
                        host: host_requirement(),
                        retry: RetryPolicy::Never,
                        deadline_unix_millis: None,
                    },
                    dependencies: BTreeSet::new(),
                    priority: 0,
                },
            )
        })
        .collect();
    let completions = pool.submit_graph(tasks).unwrap();
    let scheduled = pool.snapshot();
    assert_eq!(scheduled.pools[&pool_id].active_attempts, 3);
    assert_eq!(
        scheduled.pools[&pool_id]
            .workers
            .values()
            .filter(|worker| worker.active_attempts == 1)
            .count(),
        3
    );
    for completion in completions {
        let success = completion.wait().await.unwrap();
        let (outputs, _) = success.values().expect("ordinary remote task result");
        assert_eq!(outputs.len(), 1);
    }
    assert!(maximum_active.load(Ordering::SeqCst) >= 2);
    assert!(pool
        .snapshot()
        .tasks
        .values()
        .all(|task| { task.state == runmat_execution::state::TaskState::Succeeded }));
}

#[tokio::test]
async fn remote_compiled_program_receives_its_exact_accelerator_lease() {
    let accelerator = accelerator_request();
    let (program, artifact_id, bundle) =
        build_executable_bundle(false, vec![accelerator.clone()]).await;
    let scope_id = ExecutionScopeId::derive(&[b"remote-accelerator-scope"]);
    let pool_id = PoolId::derive(&[b"remote-accelerator-pool"]);
    let mut pool_resources = inventory(1_000);
    pool_resources.accelerators = vec![accelerator_device()];
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: pool_resources.clone(),
        },
        17,
        bundle.clone(),
    )
    .unwrap();
    let assignments = Arc::new(std::sync::Mutex::new(Vec::new()));
    let worker_id = WorkerId::derive(&[b"remote-accelerator-worker"]);
    pool.add_worker(Arc::new(ExecutingWorker {
        spec: WorkerSpec {
            id: worker_id,
            pool_id,
            resources: pool_resources,
            host: host_inventory(),
        },
        bundle: Arc::new(bundle),
        assignments: Arc::clone(&assignments),
    }))
    .await
    .unwrap();
    let task_id = TaskId::derive(&[b"remote-accelerator-task"]);
    let weak = submission(scope_id, pool_id, task_id, artifact_id);
    assert!(pool.submit(weak, program.clone()).is_err());
    let mut task = submission(scope_id, pool_id, task_id, artifact_id);
    task.request.resources.accelerators = vec![accelerator];
    pool.submit(task, program).unwrap().wait().await.unwrap();

    let assignments = assignments.lock().unwrap();
    assert_eq!(assignments.len(), 1);
    let lease = &assignments[0].resources.accelerator_leases[0];
    assert_eq!(lease.device_id, accelerator_device().id);
    assert_eq!(lease.provider, accelerator_device().provider);
    lease
        .validate_for_attempt(assignments[0].attempt_id)
        .unwrap();
}

#[tokio::test]
async fn remote_pool_executes_the_compiler_bound_parallel_region() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-parfor-scope"]);
    let pool_id = PoolId::derive(&[b"remote-parfor-pool"]);
    let worker_id = WorkerId::derive(&[b"remote-parfor-worker"]);
    let (program, artifact_id, bundle, bytecode) = parallel_region_bundle().await;
    let bundle = Arc::new(bundle);
    let assignments = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pool = RemotePoolDriver::new(
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
    )
    .unwrap();
    pool.add_worker(Arc::new(ExecutingWorker {
        spec: WorkerSpec {
            id: worker_id,
            pool_id,
            resources: inventory(1_000),
            host: host_inventory(),
        },
        bundle,
        assignments: Arc::clone(&assignments),
    }))
    .await
    .unwrap();

    let task_id = TaskId::derive(&[b"remote-parfor-task"]);
    let context = program.context.clone();
    let callable = program.callable.clone();
    let inputs = program.arguments.clone();
    let result = pool
        .submit(
            TaskSubmission {
                request: TaskRequest {
                    console: None,
                    environment: runmat_execution::EnvironmentSnapshot::default(),
                    id: task_id,
                    scope_id,
                    pool_id,
                    schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                    executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                        artifact_id,
                    ),
                    callable: Callable::for_program("remote-parfor", &callable),
                    invocation_context: context,
                    inputs,
                    outputs: OutputContract {
                        requested_outputs: 1,
                        owned_work: None,
                    },
                    resources: request(),
                    host: host_requirement(),
                    retry: RetryPolicy::IdempotentInfrastructure,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            },
            program,
        )
        .unwrap()
        .wait()
        .await
        .unwrap();

    let (outputs, _) = result.values().expect("parallel region value result");
    let value = runmat_runtime::execution::value_codec::decode_inline_value(&outputs[0])
        .expect("remote parallel result decodes");
    let runmat_value::Value::Cell(outputs) = value else {
        panic!("parallel region returned a non-cell result envelope");
    };
    let region = bytecode
        .parfor_regions
        .first()
        .expect("compiled parallel region");
    let observed = region
        .output_variables()
        .zip(
            outputs
                .into_owned_data()
                .expect("fresh decoded result envelope")
                .into_iter(),
        )
        .map(|(variable, value)| {
            let name = bytecode
                .var_names
                .get(&variable.slot)
                .expect("parallel output has a source binding")
                .clone();
            (name, value)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for name in ["task_seen", "worker_seen"] {
        let runmat_value::Value::Tensor(values) = &observed[name] else {
            panic!("{name} is not a tensor");
        };
        assert_eq!(values.materialize_f64(), vec![1.0; 4]);
    }
    let runmat_value::Value::Tensor(values) = &observed["values"] else {
        panic!("values is not a tensor");
    };
    assert_eq!(values.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);

    let assignments = assignments
        .lock()
        .expect("assignment observations poisoned");
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].scope_id, scope_id);
    assert_eq!(assignments[0].pool_id, pool_id);
    assert_eq!(assignments[0].task_id, task_id);
    assert_eq!(assignments[0].worker_id, worker_id);
    assert_eq!(
        assignments[0].backend,
        runmat_execution::PoolBackend::Remote
    );
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
            host: host_inventory(),
        },
        bundle,
        reference: reference.clone(),
        bytes: Arc::clone(&bytes),
        corrupt_download: false,
    }))
    .await
    .unwrap();
    let (result_program, artifact_id) = program().await;
    let task_id = TaskId::derive(&[b"result-object-task"]);
    let task_submission = submission(scope_id, pool_id, task_id, artifact_id);
    let success = pool
        .submit(task_submission.clone(), result_program.clone())
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(success.result_objects(), std::slice::from_ref(&reference));
    assert_eq!(
        pool.execution_object(&reference).unwrap().unwrap().as_ref(),
        bytes.as_slice()
    );
    let reused = pool
        .submit(task_submission.clone(), result_program.clone())
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(reused, success);
    let [batch_reused] = pool
        .submit_graph(vec![(result_program.clone(), task_submission.clone())])
        .unwrap()
        .try_into()
        .unwrap_or_else(|_| panic!("one committed graph task must return one completion"));
    assert_eq!(batch_reused.wait().await.unwrap(), success);
    let mut collision = task_submission;
    collision.request.callable.owner_identity = "different-answer".into();
    assert!(pool.submit(collision, result_program).is_err());

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
                host: host_inventory(),
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
    let accelerator = accelerator_request();
    let (cancel_program, artifact_id, bundle) =
        build_executable_bundle(false, vec![accelerator.clone()]).await;
    let bundle = Arc::new(bundle);
    let mut resources = inventory(1_000);
    resources.accelerators = vec![accelerator_device()];
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: resources.clone(),
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
            resources,
            host: host_inventory(),
        },
        installs: Arc::new(AtomicUsize::new(0)),
        active: Arc::clone(&active),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        start_gate: None,
        delay: Duration::from_secs(30),
        cancellations: Arc::clone(&cancellations),
        cancellation_signal: Arc::new(Notify::new()),
        cancel_ack_gate: None,
        drains: Arc::clone(&drains),
        bundle: Arc::clone(&bundle),
        described: None,
    }))
    .await
    .unwrap();
    let task_id = TaskId::derive(&[b"cancel-task"]);
    let mut cancel_submission = submission(scope_id, pool_id, task_id, artifact_id);
    cancel_submission.request.resources.accelerators = vec![accelerator];
    let completion = pool.submit(cancel_submission, cancel_program).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while active.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pool.cancel(runmat_execution::CancellationReason::User)
        .unwrap();
    let cancelled = tokio::time::timeout(Duration::from_secs(1), completion.wait())
        .await
        .unwrap();
    assert!(matches!(
        cancelled,
        Err(crate::NativeProgramFailure::Cancelled)
    ));
    assert_eq!(
        pool.snapshot().tasks.get(&task_id).unwrap().state,
        runmat_execution::state::TaskState::Cancelled
    );
    assert_eq!(cancellations.load(Ordering::SeqCst), 1);
    let snapshot = pool.snapshot();
    // The fixture reports cancellation only after its active execution has
    // stopped, so its exact retirement certificate releases scalar and device
    // reservations before the terminal cancellation is published.
    assert_eq!(snapshot.pools[&pool_id].active_attempts, 0);
    assert_eq!(snapshot.pools[&pool_id].allocated.cpu_millicores, 0);
    assert_eq!(snapshot.pools[&pool_id].allocated.memory_bytes, 0);
    assert_eq!(snapshot.pools[&pool_id].allocated.scratch_bytes, 0);
    assert_eq!(snapshot.tasks[&task_id].active_attempt, None);
    assert!(snapshot.pools[&pool_id]
        .workers
        .values()
        .all(|worker| worker.allocated.leased_accelerator_domains.is_empty()));
    let evidence = pool.task_execution_evidence(task_id).unwrap();
    assert_eq!(
        evidence.attempts[0].terminal.as_ref().unwrap().disposition,
        runmat_execution_runner::task::AttemptDisposition::Cancelled
    );
    assert!(evidence.attempts[0].unverified_retirement.is_none());
    assert!(evidence.resource_aggregate(1).is_err());

    let scope_id = ExecutionScopeId::derive(&[b"remote-loss-scope"]);
    let pool_id = PoolId::derive(&[b"remote-loss-pool"]);
    let accelerator = accelerator_request();
    let (lost_program, artifact_id, bundle) =
        build_executable_bundle(false, vec![accelerator.clone()]).await;
    let bundle = Arc::new(bundle);
    let mut resources = inventory(1_000);
    resources.accelerators = vec![accelerator_device()];
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: resources.clone(),
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
            resources,
            host: host_inventory(),
        },
        installs: Arc::new(AtomicUsize::new(0)),
        active: Arc::clone(&active),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        start_gate: None,
        delay: Duration::from_secs(30),
        cancellations: Arc::new(AtomicUsize::new(0)),
        cancellation_signal: Arc::new(Notify::new()),
        cancel_ack_gate: None,
        drains: Arc::clone(&drains),
        bundle,
        described: None,
    }))
    .await
    .unwrap();
    let lost_task_id = TaskId::derive(&[b"lost-task"]);
    let mut lost_submission = submission(scope_id, pool_id, lost_task_id, artifact_id);
    lost_submission.request.resources.accelerators = vec![accelerator];
    let completion = pool.submit(lost_submission, lost_program).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while active.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pool.remove_worker(worker_id, true).await.unwrap();
    let lost = tokio::time::timeout(Duration::from_secs(1), completion.wait())
        .await
        .unwrap();
    assert!(matches!(
        lost,
        Err(crate::NativeProgramFailure::WorkerLost(_))
    ));
    assert_eq!(
        pool.snapshot().tasks.get(&lost_task_id).unwrap().state,
        runmat_execution::state::TaskState::Indeterminate
    );
    assert_eq!(drains.load(Ordering::SeqCst), 0);
    let snapshot = pool.snapshot();
    assert!(snapshot.pools[&pool_id]
        .allocated
        .leased_accelerator_domains
        .is_empty());
}

#[tokio::test]
async fn pinned_quic_ready_v1_upgrades_only_the_opted_in_worker() {
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
    let worker = WorkerSpec {
        id: WorkerId::derive(&[b"quic-ready-worker"]),
        pool_id: PoolId::derive(&[b"quic-ready-pool"]),
        resources: inventory(1_000),
        host: host_inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([29; 32]).unwrap();
    let server = run_remote_worker_quic_ready_v1(RemoteWorkerQuicRequest {
        listener,
        run_identity: "run-quic-ready".into(),
        worker: worker.clone(),
        driver_fence: 23,
        session_id: [31; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let channel = QuicRemoteWorkerChannel::connect(
            super::RemoteWorkerChannelConfig {
                run_identity: "run-quic-ready".into(),
                node_identity: "node-quic-ready".into(),
                worker,
                driver_fence: 23,
                session_id: [31; 16],
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
        let capabilities = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(capabilities) = channel.verified_software_capabilities() {
                    break capabilities;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(capabilities.contains(&Capability::GraphSelectedValuesV11));
        assert!(capabilities.contains(&Capability::GraphCallableArgumentsV12));
        channel.drain().await.unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
}

#[tokio::test]
async fn lost_route_retains_unknown_attempt_and_blocks_same_allocation_reconnect() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-route-recovery-scope"]);
    let pool_id = PoolId::derive(&[b"remote-route-recovery-pool"]);
    let bundle = Arc::new(executable_bundle().await.2);
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 2,
            max_in_flight: 2,
            resource_limit: inventory(2_000),
        },
        21,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: WorkerId::derive(&[b"route-recovery-worker"]),
        pool_id,
        resources: inventory(1_000),
    };
    pool.add_worker(Arc::new(FailingRouteWorker {
        spec: worker.clone(),
        bundle: Arc::clone(&bundle),
        fail_execute: true,
        fail_drain: false,
    }))
    .await
    .unwrap();
    let (program, artifact_id) = program().await;
    let task_id = TaskId::derive(&[b"route-recovery-task"]);
    let mut retrying = submission(scope_id, pool_id, task_id, artifact_id);
    retrying.request.retry = RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 };
    let completion = pool.submit(retrying, program.clone()).unwrap();

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let snapshot = pool.snapshot();
            let lifecycle = snapshot.pools[&pool_id].workers[&worker.id].lifecycle;
            if lifecycle == runmat_execution_runner::pool::WorkerLifecycle::Lost {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!pool.worker_is_ready_and_routed(worker.id).await);

    let failure = tokio::time::timeout(Duration::from_secs(1), completion.wait())
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(&failure, NativeProgramFailure::WorkerLost(message)
        if message == "native execution protocol failed: fixture route failed"));
    let before = pool.snapshot();
    let attempt = &before.attempts[&before.tasks[&task_id].active_attempt.unwrap()];
    assert!(attempt.terminal.is_none());
    assert!(attempt
        .unverified_retirement
        .as_ref()
        .unwrap()
        .retired_at_driver_millis
        .is_none());
    assert_eq!(
        before.tasks[&task_id].state,
        runmat_execution::state::TaskState::Indeterminate
    );
    assert_eq!(before.tasks[&task_id].attempt_count, 1);
    assert!(before.tasks[&task_id].committed.is_none());
    assert_eq!(
        before.pools[&pool_id].workers[&worker.id].active_attempts,
        1
    );
    assert_eq!(
        before.pools[&pool_id].workers[&worker.id]
            .allocated
            .cpu_millicores,
        1_000
    );
    let reconnect = pool
        .add_worker(Arc::new(FailingRouteWorker {
            spec: worker.clone(),
            bundle: Arc::clone(&bundle),
            fail_execute: false,
            fail_drain: false,
        }))
        .await
        .unwrap_err();
    assert!(
        matches!(reconnect, crate::NativeExecutionError::Configuration(message)
        if message.contains("already registered"))
    );
    assert!(!pool.worker_is_ready_and_routed(worker.id).await);

    // Distinct inventory is separately admitted; it cannot revive the old
    // TaskId or reuse the uncertain worker's reservation.
    let healthy = WorkerSpec {
        id: WorkerId::derive(&[b"route-independent-worker"]),
        ..worker.clone()
    };
    pool.add_worker(Arc::new(FailingRouteWorker {
        spec: healthy.clone(),
        bundle,
        fail_execute: false,
        fail_drain: false,
    }))
    .await
    .unwrap();
    let new_id = TaskId::derive(&[b"route-independent-task"]);
    let success = tokio::time::timeout(
        Duration::from_secs(1),
        pool.submit(submission(scope_id, pool_id, new_id, artifact_id), program)
            .unwrap()
            .wait(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(success.result_objects().is_empty());
    let after = pool.snapshot();
    assert_eq!(after.tasks[&task_id], before.tasks[&task_id]);
    assert_eq!(after.attempts[&attempt.request.id], *attempt);
    assert_eq!(
        after.pools[&pool_id].workers[&worker.id].allocated,
        before.pools[&pool_id].workers[&worker.id].allocated
    );
    assert_eq!(after.tasks[&new_id].attempt_count, 1);
    assert_eq!(
        after.attempts[&after.tasks[&new_id].committed.as_ref().unwrap().attempt_id]
            .request
            .worker_id,
        healthy.id
    );
}

#[tokio::test]
async fn failed_drain_fences_the_worker_and_removes_its_route() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-drain-failure-scope"]);
    let pool_id = PoolId::derive(&[b"remote-drain-failure-pool"]);
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
        22,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: WorkerId::derive(&[b"drain-failure-worker"]),
        pool_id,
        resources: inventory(1_000),
    };
    pool.add_worker(Arc::new(FailingRouteWorker {
        spec: worker.clone(),
        bundle,
        fail_execute: false,
        fail_drain: true,
    }))
    .await
    .unwrap();

    assert!(pool.remove_worker(worker.id, false).await.is_err());
    assert_eq!(
        pool.snapshot().pools[&pool_id].workers[&worker.id].lifecycle,
        runmat_execution_runner::pool::WorkerLifecycle::Lost
    );
    assert!(!pool.worker_is_ready_and_routed(worker.id).await);
}

#[tokio::test]
async fn remote_pool_resolves_cancellation_only_after_worker_quiescence_acknowledgement() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-cancel-ack-scope"]);
    let pool_id = PoolId::derive(&[b"remote-cancel-ack-pool"]);
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
        14,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let cancellations = Arc::new(AtomicUsize::new(0));
    let cancel_ack_gate = Arc::new(Notify::new());
    pool.add_worker(Arc::new(FakeWorker {
        node: "node-cancel-ack".into(),
        spec: WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: WorkerId::derive(&[b"cancel-ack-worker"]),
            pool_id,
            resources: inventory(1_000),
        },
        installs: Arc::new(AtomicUsize::new(0)),
        active: Arc::clone(&active),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        start_gate: None,
        delay: Duration::from_secs(30),
        cancellations: Arc::clone(&cancellations),
        cancellation_signal: Arc::new(Notify::new()),
        cancel_ack_gate: Some(Arc::clone(&cancel_ack_gate)),
        drains: Arc::new(AtomicUsize::new(0)),
        bundle,
        described: None,
    }))
    .await
    .unwrap();
    let (program, artifact_id) = program().await;
    let completion = pool
        .submit(
            submission(
                scope_id,
                pool_id,
                TaskId::derive(&[b"cancel-ack-task"]),
                artifact_id,
            ),
            program,
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
    tokio::time::timeout(Duration::from_secs(1), async {
        while cancellations.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let mut waiter = tokio::spawn(completion.wait());
    assert!(tokio::time::timeout(Duration::from_millis(25), &mut waiter)
        .await
        .is_err());
    assert_eq!(active.load(Ordering::SeqCst), 1);
    let settlement = pool.wait_for_cancellation_settlement();
    tokio::pin!(settlement);
    assert!(
        tokio::time::timeout(Duration::from_millis(25), settlement.as_mut())
            .await
            .is_err()
    );

    cancel_ack_gate.notify_one();
    let result = tokio::time::timeout(Duration::from_secs(1), &mut waiter)
        .await
        .unwrap()
        .unwrap();
    assert!(result.is_err());
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert!(tokio::time::timeout(Duration::from_secs(1), settlement)
        .await
        .unwrap());
}

#[tokio::test]
async fn remote_pool_tick_enforces_task_deadlines() {
    let scope_id = ExecutionScopeId::derive(&[b"remote-deadline-scope"]);
    let pool_id = PoolId::derive(&[b"remote-deadline-pool"]);
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
        13,
        bundle.as_ref().clone(),
    )
    .unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let cancellations = Arc::new(AtomicUsize::new(0));
    pool.add_worker(Arc::new(FakeWorker {
        node: "node-deadline".into(),
        spec: WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: WorkerId::derive(&[b"deadline-worker"]),
            pool_id,
            resources: inventory(1_000),
        },
        installs: Arc::new(AtomicUsize::new(0)),
        active: Arc::clone(&active),
        maximum_active: Arc::new(AtomicUsize::new(0)),
        start_gate: None,
        delay: Duration::from_secs(30),
        cancellations: Arc::clone(&cancellations),
        cancellation_signal: Arc::new(Notify::new()),
        cancel_ack_gate: None,
        drains: Arc::new(AtomicUsize::new(0)),
        bundle,
        described: None,
    }))
    .await
    .unwrap();
    let (program, artifact_id) = program().await;
    let task_id = TaskId::derive(&[b"deadline-task"]);
    let mut deadline_submission = submission(scope_id, pool_id, task_id, artifact_id);
    deadline_submission.request.deadline_unix_millis = Some(1);
    let completion = pool.submit(deadline_submission, program).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while active.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    pool.tick().unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), completion.wait())
            .await
            .unwrap()
            .is_err()
    );
    assert_eq!(cancellations.load(Ordering::SeqCst), 1);
    assert_eq!(
        pool.snapshot().tasks[&task_id].state,
        runmat_execution::state::TaskState::Cancelled
    );
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.pools[&pool_id].active_attempts, 0);
    assert_eq!(snapshot.pools[&pool_id].allocated.cpu_millicores, 0);
    let evidence = pool.task_execution_evidence(task_id).unwrap();
    assert_eq!(
        evidence.attempts[0].terminal.as_ref().unwrap().disposition,
        runmat_execution_runner::task::AttemptDisposition::Cancelled
    );
    assert!(evidence.attempts[0].unverified_retirement.is_none());
}

#[tokio::test]
async fn remote_pool_seeds_the_scheduler_clock_before_assigning_work() {
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let (_, _, bundle) = executable_bundle().await;
    let pool_id = PoolId::derive(&[b"seeded-clock-pool"]);
    let pool = RemotePoolDriver::new(
        ExecutionScopeId::derive(&[b"seeded-clock-scope"]),
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        1,
        bundle,
    )
    .unwrap();
    let after = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let initialized = pool.snapshot().now_millis;
    assert!(initialized >= before);
    assert!(initialized <= after);
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
        host: host_inventory(),
    };
    let task_id = TaskId::derive(&[b"quic-task"]);
    let (mut program, artifact_id, bundle) = executable_bundle_with_input().await;
    let value = ValuePayload::Inline(Box::new(InlineValue::String("transferred".into())));
    let encoded_value = serde_json::to_vec(&value).unwrap();
    let value_reference = runmat_execution::value::ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: runmat_execution::identity::ValueId::derive(&[b"quic-value"]),
        logical_digest: value.logical_digest().unwrap(),
        encoded_length: encoded_value.len() as u64,
        media_type: "application/vnd.runmat.value+json".into(),
        value_schema: "runmat-value-payload-v2".into(),
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
        resource_assignment: Default::default(),
        task: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: task_id,
            scope_id,
            pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
            callable: Callable::for_program("remote-test", &program.callable),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            inputs: vec![object],
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: request(),
            host: host_requirement(),
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
    };
    program.assignment = Some(runmat_execution::ProgramExecutionAssignment {
        scope_id,
        pool_id,
        task_id,
        attempt_id: scheduling.id,
        worker_id: worker.id,
        backend: runmat_execution::PoolBackend::Remote,
        resources: scheduling.resource_assignment.clone(),
    });
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
                worker: worker.clone(),
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
        assert_eq!(channel.describe_worker().await.unwrap(), worker);
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
                program: super::RemoteAttemptProgram::Direct(Box::new(program)),
            })
            .await
            .unwrap();
        assert!(
            matches!(report, AttemptReport::Succeeded { .. }),
            "unexpected remote attempt report: {report:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(channel.verified_software_capabilities().is_none());
        channel.drain().await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.expect("remote worker retired successfully");
}

#[tokio::test]
async fn encrypted_remote_pool_executes_one_compiler_bound_rank_per_worker() {
    let results = execute_encrypted_spmd_program(spmd_region_bundle().await)
        .await
        .unwrap();
    assert_eq!(
        results.iter().map(|result| result.rank).collect::<Vec<_>>(),
        vec![runmat_types::LabRank(1), runmat_types::LabRank(2)]
    );
    let mut distributed_handle = None;
    for result in results {
        let mut saw_reduction = false;
        let mut saw_distributed = false;
        for output in result.outputs.into_iter().flatten() {
            match output {
                runmat_execution::SpmdOutputValue::Value(value) => {
                    assert_eq!(
                        runmat_runtime::execution::value_codec::decode_inline_value(&value)
                            .unwrap(),
                        runmat_value::Value::Int(runmat_value::IntValue::U32(3))
                    );
                    saw_reduction = true;
                }
                runmat_execution::SpmdOutputValue::Distributed(snapshot) => {
                    snapshot.validate().unwrap();
                    assert_eq!(snapshot.owned.layout.rank, result.rank);
                    if let Some(handle) = &distributed_handle {
                        assert_eq!(handle, &snapshot.handle);
                    } else {
                        distributed_handle = Some(snapshot.handle.clone());
                    }
                    let local = runmat_runtime::execution::value_codec::decode_inline_value(
                        &snapshot.owned.value,
                    )
                    .unwrap();
                    let expected = match result.rank.0 {
                        1 => vec![1_u64, 2],
                        2 => vec![3_u64, 4],
                        _ => unreachable!(),
                    };
                    let runmat_value::Value::Tensor(local) = local else {
                        panic!("remote distributed shard must retain its typed local tensor");
                    };
                    let Some(runmat_value::IntegerStorage::U64(values)) = local.integer_storage()
                    else {
                        panic!("remote distributed shard must retain uint64 storage");
                    };
                    assert_eq!(values, expected.as_slice());
                    saw_distributed = true;
                }
            }
        }
        assert!(saw_reduction && saw_distributed);
    }
}

#[tokio::test]
async fn encrypted_remote_ranks_resolve_the_compiler_owned_source_name() {
    let program = compile_spmd_region_bundle(
        r#"
spmd
  origin = mfilename();
end
"#,
        1,
    )
    .await;
    let results = execute_encrypted_spmd_program(program).await.unwrap();
    assert_eq!(results.len(), 2);
    for result in results {
        let [Some(runmat_execution::SpmdOutputValue::Value(value))] = result.outputs.as_slice()
        else {
            panic!("remote rank must return one ordinary compiler-owned source name");
        };
        assert_eq!(
            runmat_runtime::execution::value_codec::decode_inline_value(value).unwrap(),
            runmat_value::Value::String("remote_spmd".into()),
            "remote rank {} resolved a different source after transport",
            result.rank.0
        );
    }
}

// witness: FEA-SHARED-EXECUTION-001/collective_failure_and_epoch_fencing @
//   shared_infrastructure/canonical_mixed_topology/remote_execution; coverage: partial;
//   boundary: A remote rank failure releases a peer blocked in a collective within five seconds and
//   surfaces the primary structured error with its source ordinal; SPMD runtime program, not a
//   numerical solve.
#[tokio::test]
async fn remote_rank_failure_releases_a_peer_blocked_in_a_collective() {
    let program = compile_spmd_region_bundle(
        r#"
spmd
  if spmdIndex() == 1
    error("RunMat:test:RemoteSpmdPrimary", "primary rank failed");
  end
  spmdBarrier();
  reached = spmdIndex();
end
"#,
        1,
    )
    .await;
    let failure = tokio::time::timeout(
        Duration::from_secs(5),
        execute_encrypted_spmd_program(program),
    )
    .await
    .expect("a peer blocked in a remote collective must be released")
    .unwrap_err();
    let NativeProgramFailure::Runtime(failure) = failure else {
        panic!("primary rank failure lost its structured runtime identity: {failure:?}");
    };
    assert_eq!(
        failure.identifier.as_deref(),
        Some("RunMat:test:RemoteSpmdPrimary")
    );
    assert_eq!(failure.message, "primary rank failed");
    assert!(
        failure
            .call_frames
            .iter()
            .any(|frame| { frame.program_source_id == Some(runmat_types::ProgramSourceId(0)) }),
        "the structured remote error must retain the admitted program source ordinal: {failure:?}"
    );
    assert!(
        failure
            .call_stack
            .iter()
            .any(|frame| frame.contains("remote_spmd.m:")),
        "the worker must render its exact source before retiring the lease: {failure:?}"
    );
}

async fn execute_encrypted_spmd_program(
    program: CompiledSpmdProgram,
) -> Result<Vec<runmat_runtime::execution::SpmdRankResult>, NativeProgramFailure> {
    let CertifiedKey {
        cert: first_cert,
        signing_key: first_key,
    } = generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let first_certificate = first_cert.der().to_vec();
    let first_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![first_certificate.clone()],
        first_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let first_authority = first_listener.local_addr().unwrap();
    let CertifiedKey {
        cert: second_cert,
        signing_key: second_key,
    } = generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let second_certificate = second_cert.der().to_vec();
    let second_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![second_certificate.clone()],
        second_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let second_authority = second_listener.local_addr().unwrap();

    let scope_id = ExecutionScopeId::derive(&[b"remote-spmd-scope"]);
    let pool_id = PoolId::derive(&[b"remote-spmd-pool"]);
    let workers = [
        WorkerSpec {
            id: WorkerId::derive(&[b"remote-spmd-worker-1"]),
            pool_id,
            resources: inventory(1_000),
            host: host_inventory(),
        },
        WorkerSpec {
            id: WorkerId::derive(&[b"remote-spmd-worker-2"]),
            pool_id,
            resources: inventory(1_000),
            host: host_inventory(),
        },
    ];
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([19; 32]).unwrap();
    let first_server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener: first_listener,
        run_identity: "run-remote-spmd".into(),
        worker: workers[0].clone(),
        driver_fence: 17,
        session_id: [21; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let second_server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener: second_listener,
        run_identity: "run-remote-spmd".into(),
        worker: workers[1].clone(),
        driver_fence: 17,
        session_id: [22; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let pool = RemotePoolDriver::new(
            scope_id,
            PoolSpec {
                id: pool_id,
                min_workers: 2,
                max_workers: 2,
                max_in_flight: 2,
                resource_limit: inventory(2_000),
            },
            17,
            program.bundle,
        )
        .unwrap();
        let mut owned_channels = Vec::with_capacity(workers.len());
        for (ordinal, (worker, authority, certificate)) in [
            (workers[0].clone(), first_authority, first_certificate),
            (workers[1].clone(), second_authority, second_certificate),
        ]
        .into_iter()
        .enumerate()
        {
            let session_id = [u8::try_from(ordinal + 21).unwrap(); 16];
            let channel = QuicRemoteWorkerChannel::connect(
                super::RemoteWorkerChannelConfig {
                    run_identity: "run-remote-spmd".into(),
                    node_identity: format!("remote-spmd-node-{ordinal}"),
                    worker,
                    driver_fence: 17,
                    session_id,
                    run_key: run_key.clone(),
                    limits: FrameLimits::default(),
                },
                &PinnedQuicEndpoint {
                    authority,
                    server_name: "runmat.execution".into(),
                    certificate_der: certificate,
                },
            )
            .await
            .unwrap();
            pool.add_worker(channel.clone()).await.unwrap();
            owned_channels.push(channel);
        }
        let gang = GangHandle {
            id: GangId::derive(&[b"remote-spmd-gang"]),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: pool_id,
                scope_id,
                generation: 1,
            },
            labs: runmat_types::LabCount(2),
        };
        let completion = pool
            .submit_spmd_gang(RemoteSpmdGangProgram {
                gang,
                region: program.region,
                recipe: program.recipe,
                artifact: program.artifact,
                captures: Vec::new(),
                requested_outputs: program.requested_outputs,
                resources: request(),
                console: None,
            })
            .unwrap();
        let result = completion.wait().await;
        // A failed rank may fence a route without certifying physical
        // retirement. Test-owned transport teardown is independent of that
        // scheduler evidence: drain the actual channels even when membership
        // removal must retain Lost/active reservations.
        for (worker, channel) in workers.into_iter().zip(owned_channels) {
            // A rank fault may already fence its route. Cleanup must not
            // turn Lost back into Draining or release unresolved capacity.
            let lost = pool.snapshot().pools[&pool_id].workers[&worker.id].lifecycle
                == runmat_execution_runner::pool::WorkerLifecycle::Lost;
            let retained = lost.then(|| {
                let snapshot = pool.snapshot();
                let worker = &snapshot.pools[&pool_id].workers[&worker.id];
                (worker.allocated.clone(), worker.active_attempts)
            });
            if lost {
                let Err(crate::NativeExecutionError::RemoteChannel { .. }) = channel.drain().await
                else {
                    panic!("lost rank route unexpectedly accepted a drain request")
                };
            }
            pool.remove_worker(worker.id, lost).await.unwrap();
            if let Some((allocation, attempts)) = retained {
                let snapshot = pool.snapshot();
                let worker = &snapshot.pools[&pool_id].workers[&worker.id];
                assert_eq!(worker.allocated, allocation);
                assert_eq!(worker.active_attempts, attempts);
                assert_eq!(
                    worker.lifecycle,
                    runmat_execution_runner::pool::WorkerLifecycle::Lost
                );
            }
        }
        result
    };
    let (first, second, result) = tokio::join!(first_server, second_server, client);
    first.unwrap();
    second.unwrap();
    result
}

pub(in crate::remote) fn inventory(cpu_millicores: u32) -> ResourceInventory {
    ResourceInventory {
        cpu_millicores,
        memory_bytes: 3 * 1024 * 1024,
        scratch_bytes: 3 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    }
}

fn host_inventory() -> runmat_execution::host::ExecutionHostInventory {
    runmat_core::RunMatSession::with_options(false, false)
        .unwrap()
        .execution_host_inventory(runmat_execution::security::ExecutionTrustTier::CustomerTrusted)
        .unwrap()
}

fn host_requirement() -> runmat_execution::host::ExecutionHostRequirement {
    runmat_execution::host::ExecutionHostRequirement::portable(
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        runmat_types::CapabilitySet(BTreeSet::from([
            runmat_types::CapabilityRequirement::HostRuntime,
        ])),
        runmat_types::InteropManifest::empty(),
        BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
    )
    .unwrap()
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

fn accelerator_request() -> AcceleratorRequest {
    AcceleratorRequest {
        class: AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: 1 << 20,
        provider: Some(
            runmat_accelerate_api::execution_provider_contract(
                runmat_accelerate_api::RUNMAT_CUDA_PROVIDER_ID,
                runmat_accelerate_api::RUNMAT_BUILTIN_PROVIDER_VERSION,
                runmat_accelerate_api::ExecutionProviderContract::CudaNativeV1,
            )
            .unwrap(),
        ),
        required_features: BTreeSet::from([AcceleratorFeature::Compute]),
    }
}

fn accelerator_device() -> AcceleratorDevice {
    AcceleratorDevice {
        id: AcceleratorDeviceId::new("device:remote-accelerator").unwrap(),
        allocation_domain: runmat_execution::resource::AcceleratorAllocationDomainId::new(
            "allocation:remote-accelerator",
        )
        .unwrap(),
        class: AcceleratorClass::new("gpu").unwrap(),
        provider: accelerator_request().provider.unwrap(),
        max_allocation_bytes: 1 << 30,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 9,
    }
}

pub(in crate::remote) fn object_reference(
    authorization_scope: &str,
    bytes: &[u8],
) -> runmat_execution::value::ValueRef {
    runmat_execution::value::ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
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

pub(in crate::remote) fn submission(
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    task_id: TaskId,
    artifact_id: ArtifactId,
) -> TaskSubmission {
    TaskSubmission {
        request: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: task_id,
            scope_id,
            pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
            callable: Callable::for_program(
                "remote-test",
                &runmat_execution::ProgramCallable::semantic(ProgramFunctionId(0), None),
            ),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: request(),
            host: host_requirement(),
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

pub(in crate::remote) async fn executable_bundle() -> (ProgramExecutionRequest, ArtifactId, Vec<u8>)
{
    build_executable_bundle(false, Vec::new()).await
}

async fn executable_bundle_with_input() -> (ProgramExecutionRequest, ArtifactId, Vec<u8>) {
    build_executable_bundle(true, Vec::new()).await
}

async fn parallel_region_bundle() -> (
    ProgramExecutionRequest,
    ArtifactId,
    Vec<u8>,
    runmat_vm::Bytecode,
) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-parfor\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let source_text = r#"
values = zeros(1, 4);
task_seen = zeros(1, 4);
worker_seen = zeros(1, 4);
parfor index = 1:4
  values(index) = index * 2;
  task = getCurrentTask();
  worker = getCurrentWorker();
  task_seen(index) = task.ID == task.ID;
  worker_seen(index) = worker.ID == worker.ID;
end
"#;
    std::fs::write(
        project_root.path().join("src/parallel_region.m"),
        source_text,
    )
    .unwrap();
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
        .await
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new(
                project.graph.root.to_string(),
                "src/parallel_region.m",
                source_text,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    let bytecode_payload = envelope
        .component(runmat_execution::ExecutableComponentKind::Bytecode)
        .expect("portable parallel product contains bytecode");
    let bytecode: runmat_vm::Bytecode = serde_json::from_slice(&bytecode_payload.bytes).unwrap();
    let region = bytecode
        .parfor_regions
        .first()
        .expect("source compiles one parallel region");
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: ProgramCallable::parallel_region(region.contract.id).recipe_entrypoint(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-parfor-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let source_map = runmat_execution::ExecutableSourceMapV2::decode(
        &envelope
            .component(runmat_execution::ExecutableComponentKind::SourceMap)
            .expect("portable parallel product contains source map")
            .bytes,
    )
    .unwrap();
    let executable_bytes = runmat_vm::encode_portable_region_sources_v1(
        &bytecode,
        &source_map,
        &recipe.program_revision,
    )
    .unwrap();
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterRegionSourcesV1,
        executable_bytes,
    )
    .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, recipe.program_revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
            &recipe.program_revision,
        ))
        .unwrap()
        .with_materialized_program(
            recipe.clone(),
            ExecutableForm::InterpreterRegionSourcesV1,
            artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();

    let mut arguments = Vec::with_capacity(region.input_variables().count() + 1);
    let iterations = (1..=4)
        .map(|value| runmat_value::Value::Num(f64::from(value)))
        .collect::<Vec<_>>();
    arguments.push(runmat_value::Value::Cell(
        runmat_value::CellArray::new(iterations, 1, 4).unwrap(),
    ));
    arguments.extend(region.input_variables().map(|variable| {
        let name = bytecode
            .var_names
            .get(&variable.slot)
            .expect("parallel input has a source binding");
        assert!(matches!(
            name.as_str(),
            "values" | "task_seen" | "worker_seen"
        ));
        runmat_value::Value::Tensor(runmat_value::Tensor::zeros2(1, 4))
    }));
    let arguments = arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::encode_inline_value)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let callable = ProgramCallable::parallel_region(region.contract.id);
    let program = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable,
        context: ProgramInvocationContext::ParallelTask {
            task: runmat_execution::ParallelTaskContext {
                region: region.contract.id,
                chunk: ParallelChunk {
                    ordinal: 0,
                    start: 0,
                    len: 4,
                },
                randomness: ParallelRandomnessContext::Inherit,
            },
        },
        assignment: None,
        job_id: None,
        arguments,
        requested_outputs: 1,
    };
    program.validate_for_portable_host().unwrap();
    (program, artifact_id, bundle_bytes, bytecode)
}

async fn spmd_region_bundle() -> CompiledSpmdProgram {
    compile_spmd_region_bundle(
        r#"
spmd
  built = codistributed.build(uint64([2 * spmdIndex() - 1, 2 * spmdIndex()]), codistributor1d(uint32(2), uint64([2, 2]), uint64([1, 4])), "noCommunication");
  total = spmdPlus(uint32(spmdIndex()));
end
"#,
        2,
    )
    .await
}

async fn compile_spmd_region_bundle(
    source_text: &str,
    expected_outputs: usize,
) -> CompiledSpmdProgram {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-spmd\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    std::fs::write(project_root.path().join("src/remote_spmd.m"), source_text).unwrap();
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
        .await
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new(
                project.graph.root.to_string(),
                "src/remote_spmd.m",
                source_text,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    let bytecode_payload = envelope
        .component(runmat_execution::ExecutableComponentKind::Bytecode)
        .expect("portable SPMD product contains bytecode");
    let bytecode: runmat_vm::Bytecode = serde_json::from_slice(&bytecode_payload.bytes).unwrap();
    let region = bytecode.spmd_regions.first().unwrap();
    assert!(
        region.captures.is_empty(),
        "unexpected SPMD captures: {:?}",
        region
            .captures
            .iter()
            .map(|capture| (bytecode.var_names.get(&capture.slot), capture.slot))
            .collect::<Vec<_>>()
    );
    assert_eq!(region.outputs.len(), expected_outputs);
    let region_id = region.contract.id;
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: ProgramCallable::spmd_region(region_id).recipe_entrypoint(),
        outputs: OutputContract {
            requested_outputs: u16::try_from(expected_outputs).unwrap(),
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-spmd-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let source_map = runmat_execution::ExecutableSourceMapV2::decode(
        &envelope
            .component(runmat_execution::ExecutableComponentKind::SourceMap)
            .expect("portable SPMD product contains source map")
            .bytes,
    )
    .unwrap();
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterRegionSourcesV1,
        runmat_vm::encode_portable_region_sources_v1(
            &bytecode,
            &source_map,
            &recipe.program_revision,
        )
        .unwrap(),
    )
    .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, recipe.program_revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
            &recipe.program_revision,
        ))
        .unwrap()
        .with_materialized_program(
            recipe.clone(),
            ExecutableForm::InterpreterRegionSourcesV1,
            artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let mut bytes = Vec::new();
    write_bundle(&bundle, &mut bytes, ArchiveLimits::default()).unwrap();
    CompiledSpmdProgram {
        recipe,
        artifact,
        region: region_id,
        bundle: bytes,
        requested_outputs: u16::try_from(expected_outputs).unwrap(),
    }
}

async fn build_executable_bundle(
    accepts_input: bool,
    accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
) -> (ProgramExecutionRequest, ArtifactId, Vec<u8>) {
    let source = if accepts_input {
        "function y = answer(x); y = x; end\n"
    } else {
        "function y = answer(); y = 42; end\n"
    };
    program::build(source, Vec::new(), accelerators).await
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

#[path = "pool_tests/degraded_capacity.rs"]
mod degraded_capacity;

mod delivery_accounting;
