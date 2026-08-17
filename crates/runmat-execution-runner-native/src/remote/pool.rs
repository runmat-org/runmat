use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use runmat_execution::state::PoolState;
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::{ProgramExecutionRequest, ProjectRevisionRecord};
use runmat_execution_runner::pool::ResizeRequest;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission,
};
use tokio::sync::{oneshot, Mutex as AsyncMutex, RwLock};

use super::pool_progress::RemoteTaskCompletion;
use super::{RemoteAttempt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult};

type CompletionResult = Result<AttemptSuccess, String>;

mod completion;

/// Portable scheduler composition over allocation-scoped remote worker routes.
pub struct RemotePoolDriver {
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    bundle_digest: Digest,
    bundle_identity: Digest,
    project_revision: ProjectRevisionRecord,
    bundle: Arc<[u8]>,
    driver: Mutex<Driver>,
    channels: RwLock<HashMap<runmat_execution::identity::WorkerId, Arc<dyn RemoteWorkerChannel>>>,
    installed_nodes: AsyncMutex<BTreeSet<String>>,
    value_scope: String,
    values: super::pool_values::RemoteValueCatalog,
    execution_objects: super::pool_objects::RemoteObjectCatalog,
    programs: Mutex<HashMap<TaskId, ProgramExecutionRequest>>,
    completions: Mutex<HashMap<TaskId, oneshot::Sender<CompletionResult>>>,
    progress: Mutex<HashMap<TaskId, Arc<super::pool_progress::RemoteProgressBuffer>>>,
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
        let pool_id = pool.id;
        let bundle = bundle.into();
        let decoded = read_bundle(bundle.as_ref(), ArchiveLimits::default()).map_err(|error| {
            NativeExecutionError::Protocol(format!("remote pool bundle is invalid: {error}"))
        })?;
        let bundle_identity = decoded.identity().map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "remote pool bundle identity is invalid: {error}"
            ))
        })?;
        let project_revision = decoded.manifest.project_revision;
        let mut driver = Driver::new(DriverConfig::default(), driver_fence)?;
        driver.handle(DriverCommand::RegisterScope {
            scope_id,
            parent: None,
        })?;
        driver.handle(DriverCommand::CreatePool(pool))?;
        Ok(Arc::new(Self {
            scope_id,
            pool_id,
            bundle_digest: Digest::sha256(bundle.as_ref()),
            bundle_identity,
            project_revision,
            bundle,
            driver: Mutex::new(driver),
            channels: RwLock::new(HashMap::new()),
            installed_nodes: AsyncMutex::new(BTreeSet::new()),
            value_scope: value_scope.into(),
            values: super::pool_values::RemoteValueCatalog::default(),
            execution_objects: super::pool_objects::RemoteObjectCatalog::default(),
            programs: Mutex::new(HashMap::new()),
            completions: Mutex::new(HashMap::new()),
            progress: Mutex::new(HashMap::new()),
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
        {
            let mut installed = self.installed_nodes.lock().await;
            let receipt = if installed.contains(&node) {
                channel.activate_bundle(self.bundle_digest).await?
            } else {
                channel
                    .install_bundle(self.bundle_digest, &self.bundle)
                    .await?
            };
            if receipt.bundle_digest != self.bundle_digest
                || receipt.bundle_identity != self.bundle_identity
                || receipt.project_revision != self.project_revision
                || receipt.stored_bytes != self.bundle.len() as u64
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
            driver.handle(DriverCommand::RegisterWorker(worker))?
        };
        self.dispatch(actions);
        self.refresh_pool_state()?;
        Ok(())
    }

    pub fn submit(
        self: &Arc<Self>,
        submission: TaskSubmission,
        program: ProgramExecutionRequest,
    ) -> NativeExecutionResult<RemoteTaskCompletion> {
        if submission.request.scope_id != self.scope_id
            || submission.request.pool_id != self.pool_id
        {
            return Err(NativeExecutionError::Configuration(
                "remote task is outside this driver authority".into(),
            ));
        }
        program.validate().map_err(|error| {
            NativeExecutionError::Protocol(format!("remote program request is invalid: {error}"))
        })?;
        let task_id = submission.request.id;
        let (sender, receiver) = oneshot::channel();
        let progress = Arc::new(super::pool_progress::RemoteProgressBuffer::default());
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .insert(task_id, program);
        self.completions
            .lock()
            .expect("remote completion registry poisoned")
            .insert(task_id, sender);
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .insert(task_id, Arc::clone(&progress));
        let actions = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .handle(DriverCommand::Submit(Box::new(submission)))?;
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
        let actions = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            if lost {
                driver.handle(DriverCommand::WorkerLost(worker_id))?
            } else {
                driver.handle(DriverCommand::DrainWorker(worker_id))?
            }
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
        let ready = pool.workers.len() as u32 >= pool.spec.min_workers;
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

    fn dispatch(self: &Arc<Self>, actions: Vec<DriverAction>) {
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
        tokio::spawn(async move {
            let channel = this.channels.read().await.get(&request.worker_id).cloned();
            let program = this
                .programs
                .lock()
                .expect("remote program catalog poisoned")
                .get(&request.task_id)
                .cloned();
            let progress = this
                .progress
                .lock()
                .expect("remote progress registry poisoned")
                .get(&request.task_id)
                .cloned();
            let report = match (channel, program) {
                (Some(channel), Some(mut program)) => {
                    program.arguments = request.task.inputs.clone();
                    let transfer = this
                        .execution_objects
                        .transfer_all(channel.as_ref(), request.worker_id)
                        .await;
                    let transfer = match transfer {
                        Ok(()) => {
                            this.values
                                .transfer(
                                    channel.as_ref(),
                                    request.worker_id,
                                    &program.arguments,
                                    &this.execution_objects,
                                )
                                .await
                        }
                        Err(error) => Err(error),
                    };
                    match transfer {
                        Err(error) => AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message: error.to_string(),
                        },
                        Ok(()) => match super::pool_progress::execute(
                            channel.as_ref(),
                            RemoteAttempt {
                                scheduling: request.clone(),
                                program,
                            },
                            progress.as_deref(),
                        )
                        .await
                        {
                            Ok(AttemptReport::Succeeded { result }) => {
                                match this
                                    .execution_objects
                                    .receive_results(
                                        channel.as_ref(),
                                        &result.result_objects,
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
            this.apply_report(BackendReport::for_request(&request, report));
        });
    }

    fn cancel_attempt(self: &Arc<Self>, request: AttemptRequest) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Some(channel) = this.channels.read().await.get(&request.worker_id).cloned() {
                let _ = channel.cancel(&request).await;
            }
            this.apply_report(BackendReport::for_request(
                &request,
                AttemptReport::Cancelled,
            ));
        });
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or(0)
}
