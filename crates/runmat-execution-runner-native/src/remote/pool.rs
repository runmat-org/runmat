use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::state::PoolState;
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::{
    GraphExecutionRequest, ProgramExecutionRequest, ProjectRevisionRecord, WorkerProgramRequest,
};
use runmat_execution_runner::pool::ResizeRequest;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission,
};
use runmat_execution_transport_native::control::DriverArtifactDownload;
use tokio::sync::{oneshot, Mutex as AsyncMutex, RwLock};

use super::pool_progress::RemoteTaskCompletion;
use super::{RemoteAttempt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult};

type CompletionResult = Result<AttemptSuccess, crate::NativeProgramFailure>;

/// Signed transfer authority for one already-admitted, immutable Graph
/// Program. The scheduler retains the source for retries; attempt frames carry
/// only the digest and length issued by a worker after authentication.
pub(super) struct SignedGraphTransfer {
    pub(super) encrypted_manifest: Arc<[u8]>,
    pub(super) ordered_downloads: Arc<[DriverArtifactDownload]>,
    pub(super) maximum_program_bytes: u64,
    pub(super) program_digest: Digest,
    pub(super) program_length: u64,
}

mod completion;

/// Portable scheduler composition over allocation-scoped remote worker routes.
pub struct RemotePoolDriver {
    pub(super) scope_id: ExecutionScopeId,
    pub(super) pool_id: PoolId,
    direct_bundle: Option<DirectPoolBundle>,
    pub(super) driver: Mutex<Driver>,
    channels: RwLock<HashMap<runmat_execution::identity::WorkerId, Arc<dyn RemoteWorkerChannel>>>,
    applied_software_readiness: Mutex<HashSet<runmat_execution::identity::WorkerId>>,
    installed_nodes: AsyncMutex<BTreeSet<String>>,
    value_scope: String,
    values: super::pool_values::RemoteValueCatalog,
    execution_objects: super::pool_objects::RemoteObjectCatalog,
    pub(super) programs: Mutex<HashMap<TaskId, Arc<WorkerProgramRequest>>>,
    graph_transfers: Mutex<HashMap<TaskId, Arc<SignedGraphTransfer>>>,
    // The exact authenticated parent stage must outlive every active Graph
    // attempt. Spawned attempt tasks retain this pool Arc even if their
    // caller's bounded cancellation drain becomes indeterminate.
    _graph_stage: Mutex<Option<super::program_staging::StagedGraphProgram>>,
    attempt_cancellation: Mutex<HashMap<runmat_execution::identity::AttemptId, CancellationSource>>,
    pub(super) completions: Mutex<HashMap<TaskId, oneshot::Sender<(u64, CompletionResult)>>>,
    completion_sequence: AtomicU64,
    pub(super) progress: Mutex<HashMap<TaskId, Arc<super::pool_progress::RemoteProgressBuffer>>>,
    pub(super) collectives: Arc<crate::driver::collective::ProcessCollectiveBroker>,
}

/// Direct V7 admission is backed by the installed bundle. Graph V10 instead
/// carries exact ordered selected archives in each task request.
struct DirectPoolBundle {
    digest: Digest,
    identity: Digest,
    project_revision: ProjectRevisionRecord,
    bytes: Arc<[u8]>,
}

impl RemotePoolDriver {
    pub fn new(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        bundle: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<Arc<Self>> {
        Self::new_with_value_scope(scope_id, pool, driver_fence, bundle, scope_id.to_string())
    }

    pub fn new_with_value_scope(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        bundle: impl Into<Arc<[u8]>>,
        value_scope: impl Into<String>,
    ) -> NativeExecutionResult<Arc<Self>> {
        let bundle = bundle.into();
        let decoded = read_bundle(bundle.as_ref(), ArchiveLimits::default()).map_err(|error| {
            NativeExecutionError::Protocol(format!("remote pool bundle is invalid: {error}"))
        })?;
        let bundle_identity = decoded.identity().map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "remote pool bundle identity is invalid: {error}"
            ))
        })?;
        let direct_bundle = DirectPoolBundle {
            digest: Digest::sha256(bundle.as_ref()),
            identity: bundle_identity,
            project_revision: decoded.manifest.project_revision,
            bytes: bundle,
        };
        Self::new_with_authority(
            scope_id,
            pool,
            driver_fence,
            Some(direct_bundle),
            value_scope,
        )
    }

    pub fn new_graph(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        value_scope: impl Into<String>,
    ) -> NativeExecutionResult<Arc<Self>> {
        Self::new_with_authority(scope_id, pool, driver_fence, None, value_scope)
    }

    fn new_with_authority(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        direct_bundle: Option<DirectPoolBundle>,
        value_scope: impl Into<String>,
    ) -> NativeExecutionResult<Arc<Self>> {
        let pool_id = pool.id;
        let mut driver = Driver::new(DriverConfig::default(), driver_fence)?;
        driver.handle(DriverCommand::RegisterScope {
            scope_id,
            parent: None,
        })?;
        driver.handle(DriverCommand::CreatePool(pool))?;
        Ok(Arc::new(Self {
            scope_id,
            pool_id,
            direct_bundle,
            driver: Mutex::new(driver),
            channels: RwLock::new(HashMap::new()),
            applied_software_readiness: Mutex::new(HashSet::new()),
            installed_nodes: AsyncMutex::new(BTreeSet::new()),
            value_scope: value_scope.into(),
            values: super::pool_values::RemoteValueCatalog::default(),
            execution_objects: super::pool_objects::RemoteObjectCatalog::default(),
            programs: Mutex::new(HashMap::new()),
            graph_transfers: Mutex::new(HashMap::new()),
            _graph_stage: Mutex::new(None),
            attempt_cancellation: Mutex::new(HashMap::new()),
            completions: Mutex::new(HashMap::new()),
            completion_sequence: AtomicU64::new(0),
            progress: Mutex::new(HashMap::new()),
            collectives: Arc::new(crate::driver::collective::ProcessCollectiveBroker::default()),
        }))
    }

    pub fn register_value_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<()> {
        self.values
            .register(reference, encoded.into(), &self.value_scope)
    }

    pub fn register_execution_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<()> {
        let encoded = encoded.into();
        self.execution_objects
            .register(reference, encoded, &self.value_scope)
    }

    pub fn execution_object(
        &self,
        reference: &runmat_execution::value::ValueRef,
    ) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        self.execution_objects.get(reference)
    }

    pub async fn add_worker(
        self: &Arc<Self>,
        channel: Arc<dyn RemoteWorkerChannel>,
    ) -> NativeExecutionResult<()> {
        let worker = channel.worker().clone();
        if worker.pool_id != self.pool_id {
            return Err(NativeExecutionError::Configuration(
                "remote worker belongs to a different pool".into(),
            ));
        }
        let node = channel.node_identity().to_string();
        if let Some(bundle) = &self.direct_bundle {
            let mut installed = self.installed_nodes.lock().await;
            let receipt = if installed.contains(&node) {
                channel.activate_bundle(bundle.digest).await?
            } else {
                channel.install_bundle(bundle.digest, &bundle.bytes).await?
            };
            if receipt.bundle_digest != bundle.digest
                || receipt.bundle_identity != bundle.identity
                || receipt.project_revision != bundle.project_revision
                || receipt.stored_bytes != bundle.bytes.len() as u64
            {
                return Err(NativeExecutionError::Protocol(
                    "remote worker did not acknowledge the exact bundle".into(),
                ));
            }
            if !installed.contains(&node) {
                installed.insert(node);
            }
        }
        self.channels.write().await.insert(worker.id, channel);
        let actions = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let mut actions = driver.handle(DriverCommand::RegisterWorker(worker))?;
            actions.extend(driver.handle(DriverCommand::RefreshPoolCapabilities {
                pool_id: self.pool_id,
            })?);
            actions
        };
        self.dispatch(actions);
        self.refresh_pool_state()?;
        self.refresh_worker_software_capabilities().await?;
        Ok(())
    }

    /// Pull only readiness already authenticated by each encrypted route.
    /// This does not query or wait for a worker and is safe on the ordinary
    /// V10 reconciliation path.
    pub async fn refresh_worker_software_capabilities(
        self: &Arc<Self>,
    ) -> NativeExecutionResult<()> {
        let observed = self
            .channels
            .read()
            .await
            .iter()
            .filter_map(|(id, channel)| {
                channel
                    .verified_software_capabilities()
                    .map(|capabilities| (*id, capabilities))
            })
            .collect::<Vec<_>>();
        if observed.is_empty() {
            return Ok(());
        }
        let actions = {
            let mut applied = self
                .applied_software_readiness
                .lock()
                .expect("worker readiness registry poisoned");
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let mut actions = Vec::new();
            for (worker_id, capabilities) in observed {
                if applied.contains(&worker_id) {
                    continue;
                }
                actions.extend(
                    driver.handle(DriverCommand::AdmitWorkerSoftwareCapabilities {
                        worker_id,
                        capabilities,
                    })?,
                );
                actions.extend(driver.handle(DriverCommand::RefreshPoolCapabilities {
                    pool_id: self.pool_id,
                })?);
                applied.insert(worker_id);
            }
            actions
        };
        self.dispatch(actions);
        Ok(())
    }

    pub fn submit(
        self: &Arc<Self>,
        submission: TaskSubmission,
        program: ProgramExecutionRequest,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        if self.direct_bundle.is_none() {
            return Err(NativeExecutionError::Configuration(
                "direct request cannot use a graph-only remote pool".into(),
            ));
        }
        program.validate().map_err(|error| {
            NativeExecutionError::Protocol(format!("remote program request is invalid: {error}"))
        })?;
        program
            .recipe
            .validate_resource_request(&submission.request.resources)
            .map_err(|error| {
                NativeExecutionError::Protocol(format!(
                    "remote scheduler resources do not satisfy the program: {error}"
                ))
            })?;
        self.submit_selected(submission, WorkerProgramRequest::Direct(program), None)
    }

    pub(super) fn submit_graph(
        self: &Arc<Self>,
        submission: TaskSubmission,
        graph: GraphExecutionRequest,
        encoded_bytes: usize,
        transfer: Option<SignedGraphTransfer>,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        if self.direct_bundle.is_some() {
            return Err(NativeExecutionError::Configuration(
                "graph request cannot use a direct-bundle remote pool".into(),
            ));
        }
        graph.validate().map_err(|error| {
            NativeExecutionError::Protocol(format!("remote graph request is invalid: {error}"))
        })?;
        graph
            .recipe
            .validate_resource_request(&submission.request.resources)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let demand = graph
            .input_allocation_floor_bytes(encoded_bytes)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        if submission.request.resources.memory_bytes < demand
            || !submission.request.inputs.is_empty()
            || submission.request.outputs.requested_outputs != graph.requested_outputs
            || submission.request.invocation_context != graph.context
        {
            return Err(NativeExecutionError::Protocol(
                "remote graph workload does not match its resource or invocation admission".into(),
            ));
        }
        let transfer = transfer.ok_or_else(|| {
            NativeExecutionError::Protocol(
                "remote Graph execution has no authenticated staged transfer".into(),
            )
        })?;
        if transfer.program_length != encoded_bytes as u64 {
            return Err(NativeExecutionError::Protocol(
                "remote Graph allocation length differs from its staged Program".into(),
            ));
        }
        self.submit_selected(
            submission,
            WorkerProgramRequest::Graph(graph),
            Some(transfer),
        )
    }

    fn submit_selected(
        self: &Arc<Self>,
        submission: TaskSubmission,
        program: WorkerProgramRequest,
        graph_transfer: Option<SignedGraphTransfer>,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        if submission.request.scope_id != self.scope_id
            || submission.request.pool_id != self.pool_id
        {
            return Err(NativeExecutionError::Configuration(
                "remote task is outside this driver authority".into(),
            ));
        }
        let task_id = submission.request.id;
        let (sender, receiver) = oneshot::channel();
        let progress = Arc::new(super::pool_progress::RemoteProgressBuffer::default());
        let mut driver = self.driver.lock().expect("remote driver poisoned");
        let actions = driver.handle(DriverCommand::Submit(Box::new(submission)))?;
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .insert(task_id, Arc::new(program));
        if let Some(transfer) = graph_transfer {
            self.graph_transfers
                .lock()
                .expect("remote Graph transfer registry poisoned")
                .insert(task_id, Arc::new(transfer));
        }
        self.completions
            .lock()
            .expect("remote completion registry poisoned")
            .insert(task_id, sender);
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .insert(task_id, Arc::clone(&progress));
        drop(driver);
        self.dispatch(actions);
        Ok(RemoteTaskCompletion::new(receiver, progress))
    }

    pub fn resize(self: &Arc<Self>, desired_workers: u32) -> NativeExecutionResult<()> {
        let actions = self.driver.lock().expect("remote driver poisoned").handle(
            DriverCommand::ResizePool {
                pool_id: self.pool_id,
                request: ResizeRequest { desired_workers },
            },
        )?;
        self.dispatch(actions);
        Ok(())
    }

    pub async fn remove_worker(
        self: &Arc<Self>,
        worker_id: runmat_execution::identity::WorkerId,
        lost: bool,
    ) -> NativeExecutionResult<()> {
        let channel = self.channels.write().await.remove(&worker_id);
        self.applied_software_readiness
            .lock()
            .expect("worker readiness registry poisoned")
            .remove(&worker_id);
        let actions = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let mut actions = if lost {
                driver.handle(DriverCommand::WorkerLost(worker_id))?
            } else {
                driver.handle(DriverCommand::DrainWorker(worker_id))?
            };
            actions.extend(driver.handle(DriverCommand::RefreshPoolCapabilities {
                pool_id: self.pool_id,
            })?);
            actions
        };
        self.dispatch(actions);
        self.resolve_non_success_terminals();
        if !lost {
            if let Some(channel) = channel {
                channel.drain().await?;
            }
        }
        self.refresh_pool_state()?;
        Ok(())
    }

    pub fn cancel(self: &Arc<Self>, reason: CancellationReason) -> NativeExecutionResult<()> {
        let actions = self.driver.lock().expect("remote driver poisoned").handle(
            DriverCommand::CancelScope {
                scope_id: self.scope_id,
                reason,
                now_millis: now_millis(),
            },
        )?;
        self.dispatch(actions);
        Ok(())
    }

    pub(super) fn retain_graph_stage(
        &self,
        stage: Option<super::program_staging::StagedGraphProgram>,
    ) {
        *self
            ._graph_stage
            .lock()
            .expect("remote Graph stage owner poisoned") = stage;
    }

    pub fn snapshot(&self) -> runmat_execution_runner::DriverSnapshot {
        self.driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot()
    }

    fn refresh_pool_state(self: &Arc<Self>) -> NativeExecutionResult<()> {
        let mut driver = self.driver.lock().expect("remote driver poisoned");
        let snapshot = driver.snapshot();
        let pool = snapshot
            .pools
            .get(&self.pool_id)
            .ok_or_else(|| NativeExecutionError::Protocol("remote pool disappeared".into()))?;
        let ready = pool
            .workers
            .values()
            .filter(|worker| worker.accepts_work())
            .count() as u32
            >= pool.spec.min_workers;
        let target = if ready {
            PoolState::Ready
        } else {
            PoolState::Creating
        };
        if pool.state != target {
            let actions = driver.handle(DriverCommand::SetPoolState {
                pool_id: self.pool_id,
                state: target,
            })?;
            drop(driver);
            self.dispatch(actions);
        }
        Ok(())
    }

    pub(super) fn dispatch(self: &Arc<Self>, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => self.launch(request),
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    self.cancel_attempt(request)
                }
                DriverAction::Checkpoint => {}
                DriverAction::GarbageCollectResults { objects, .. } => {
                    self.execution_objects.discard_results(&objects);
                }
                DriverAction::ResizePool { .. } => {
                    // Coarse allocation is owned by the Server/client adapter;
                    // workers are registered only after fenced routes exist.
                }
            }
        }
    }

    fn launch(self: &Arc<Self>, request: AttemptRequest) {
        let this = Arc::clone(self);
        let cancellation = CancellationSource::new();
        self.attempt_cancellation
            .lock()
            .expect("remote attempt cancellation registry poisoned")
            .insert(request.id, cancellation.clone());
        tokio::spawn(async move {
            let channel = this.channels.read().await.get(&request.worker_id).cloned();
            let program = this
                .programs
                .lock()
                .expect("remote program catalog poisoned")
                .get(&request.task_id)
                .map(Arc::clone);
            let progress = this
                .progress
                .lock()
                .expect("remote progress registry poisoned")
                .get(&request.task_id)
                .cloned();
            let report = match (channel, program) {
                (Some(channel), Some(program)) => {
                    // Keep the admitted Graph in the scheduler catalog. A
                    // retry stages its signed chunks and sends only the
                    // worker-issued digest/length in the attempt frame.
                    let program = program.as_ref();
                    let collective_context = match program {
                        WorkerProgramRequest::Direct(direct)
                            if matches!(
                                &direct.context,
                                runmat_execution::ProgramInvocationContext::SpmdTask { .. }
                            ) =>
                        {
                            let runmat_execution::ProgramInvocationContext::SpmdTask { task } =
                                &direct.context
                            else {
                                unreachable!("SPMD route checked above")
                            };
                            Some(super::pool_progress::RemoteCollectiveAssignment::new(
                                Arc::clone(&this.collectives),
                                task.clone(),
                            ))
                        }
                        _ => None,
                    };
                    let assignment = runmat_execution::ProgramExecutionAssignment {
                        scope_id: request.scope_id,
                        pool_id: request.task.pool_id,
                        task_id: request.task_id,
                        attempt_id: request.id,
                        worker_id: request.worker_id,
                        backend: runmat_execution::PoolBackend::Remote,
                        resources: request.resource_assignment.clone(),
                    };
                    let transfer = match program {
                        WorkerProgramRequest::Direct(direct) => {
                            let mut direct = direct.clone();
                            direct.arguments = request.task.inputs.clone();
                            direct.assignment = Some(assignment);
                            let objects = this
                                .execution_objects
                                .transfer_all(channel.as_ref(), request.worker_id)
                                .await;
                            let transferred = match objects {
                                Ok(()) => {
                                    this.values
                                        .transfer(
                                            channel.as_ref(),
                                            request.worker_id,
                                            &direct.arguments,
                                            &this.execution_objects,
                                        )
                                        .await
                                }
                                Err(error) => Err(error),
                            };
                            transferred
                                .map(|()| super::RemoteAttemptProgram::Direct(Box::new(direct)))
                        }
                        WorkerProgramRequest::Graph(_) => {
                            let source = this
                                .graph_transfers
                                .lock()
                                .expect("remote Graph transfer registry poisoned")
                                .get(&request.task_id)
                                .cloned();
                            match source {
                                Some(source) => {
                                    let token = cancellation.token();
                                    let receipt = tokio::select! {
                                        result = channel.stage_graph_program(
                                            request.id,
                                            &source.encrypted_manifest,
                                            &source.ordered_downloads,
                                            source.maximum_program_bytes,
                                        ) => result,
                                        _ = token.cancelled() => {
                                            Err(NativeExecutionError::Protocol(
                                                "remote Graph attempt was cancelled during staging".into(),
                                            ))
                                        }
                                    };
                                    match receipt {
                                        Ok(receipt)
                                            if receipt.program_digest == source.program_digest
                                                && receipt.program_length
                                                    == source.program_length =>
                                        {
                                            Ok(super::RemoteAttemptProgram::GraphStaged(
                                                super::RemoteStagedGraphReference::StagedGraphV10 {
                                                    program_digest: receipt.program_digest,
                                                    program_length: receipt.program_length,
                                                },
                                            ))
                                        }
                                        Ok(_) => Err(NativeExecutionError::Protocol(
                                            "remote worker acknowledged a different Graph Program"
                                                .into(),
                                        )),
                                        Err(error) => Err(error),
                                    }
                                }
                                None => Err(NativeExecutionError::Protocol(
                                    "remote Graph task lost its staged transfer authority".into(),
                                )),
                            }
                        }
                    };
                    match transfer {
                        Err(_) if cancellation.token().is_cancelled() => AttemptReport::Cancelled,
                        Err(error) => AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message: error.to_string(),
                        },
                        Ok(_) if cancellation.token().is_cancelled() => AttemptReport::Cancelled,
                        Ok(program) => match super::pool_progress::execute(
                            channel.as_ref(),
                            RemoteAttempt {
                                scheduling: request.clone(),
                                program,
                            },
                            progress.as_deref(),
                            collective_context,
                        )
                        .await
                        {
                            Ok(AttemptReport::Succeeded { result }) => {
                                match this
                                    .execution_objects
                                    .receive_results(
                                        channel.as_ref(),
                                        result.result_objects(),
                                        &this.value_scope,
                                    )
                                    .await
                                {
                                    Ok(()) => AttemptReport::Succeeded { result },
                                    Err(error) => AttemptReport::Failed {
                                        kind: AttemptFailureKind::Rejected,
                                        message: error.to_string(),
                                    },
                                }
                            }
                            Ok(report) => report,
                            Err(error) => AttemptReport::Lost {
                                message: error.to_string(),
                            },
                        },
                    }
                }
                _ => AttemptReport::Failed {
                    kind: AttemptFailureKind::Infrastructure,
                    message: "remote worker route or exact program is unavailable".into(),
                },
            };
            this.attempt_cancellation
                .lock()
                .expect("remote attempt cancellation registry poisoned")
                .remove(&request.id);
            this.apply_report(BackendReport::for_request(&request, report));
        });
    }

    fn cancel_attempt(self: &Arc<Self>, request: AttemptRequest) {
        let graph_attempt = matches!(
            &request.task.callable.program,
            runmat_execution::ProgramCallable::GraphTarget { .. }
        );
        if let Some(source) = self
            .attempt_cancellation
            .lock()
            .expect("remote attempt cancellation registry poisoned")
            .get(&request.id)
            .cloned()
        {
            source.cancel(CancellationReason::User);
        }
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Some(channel) = this.channels.read().await.get(&request.worker_id).cloned() {
                let _ = channel.cancel(&request).await;
            }
            // A remote Cancel acknowledgement proves delivery of a request,
            // not that an active Graph worker has retired its foreign host or
            // private staging. Only its eventual attempt report may close the
            // Graph attempt; Direct keeps its existing immediate completion.
            if !graph_attempt {
                this.apply_report(BackendReport::for_request(
                    &request,
                    AttemptReport::Cancelled,
                ));
            }
        });
    }
}

pub(super) fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or(0)
}
