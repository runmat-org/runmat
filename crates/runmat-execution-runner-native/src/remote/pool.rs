use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use runmat_execution::state::{PoolState, TaskState};
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::pool::ResizeRequest;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission,
};
use tokio::sync::{oneshot, Mutex as AsyncMutex, RwLock};

use super::{RemoteAttempt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult};

type CompletionResult = Result<AttemptSuccess, String>;
type ValueObject = (runmat_execution::value::ValueRef, Arc<[u8]>);
type ValueObjectCatalog = HashMap<runmat_execution::identity::ValueId, ValueObject>;

pub struct RemoteTaskCompletion {
    receiver: oneshot::Receiver<CompletionResult>,
}

impl RemoteTaskCompletion {
    pub async fn wait(self) -> CompletionResult {
        self.receiver
            .await
            .unwrap_or_else(|_| Err("remote task completion channel closed".into()))
    }
}

/// Native composition of the portable scheduler with allocation-scoped remote
/// worker routes. Server remains unaware of tasks and attempts.
pub struct RemotePoolDriver {
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    bundle_digest: Digest,
    bundle: Arc<[u8]>,
    driver: Mutex<Driver>,
    channels: RwLock<HashMap<runmat_execution::identity::WorkerId, Arc<dyn RemoteWorkerChannel>>>,
    installed_nodes: AsyncMutex<BTreeSet<String>>,
    value_scope: String,
    objects: Mutex<ValueObjectCatalog>,
    transferred: AsyncMutex<
        HashSet<(
            runmat_execution::identity::WorkerId,
            runmat_execution::identity::ValueId,
        )>,
    >,
    programs: Mutex<HashMap<TaskId, ProgramExecutionRequest>>,
    completions: Mutex<HashMap<TaskId, oneshot::Sender<CompletionResult>>>,
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
            bundle,
            driver: Mutex::new(driver),
            channels: RwLock::new(HashMap::new()),
            installed_nodes: AsyncMutex::new(BTreeSet::new()),
            value_scope: value_scope.into(),
            objects: Mutex::new(HashMap::new()),
            transferred: AsyncMutex::new(HashSet::new()),
            programs: Mutex::new(HashMap::new()),
            completions: Mutex::new(HashMap::new()),
        }))
    }

    pub fn register_value_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<()> {
        let encoded = encoded.into();
        super::value_transfer::decode_value(&reference, &encoded, &self.value_scope)?;
        let mut objects = self.objects.lock().expect("remote value catalog poisoned");
        if let Some((existing_reference, existing_bytes)) = objects.get(&reference.id) {
            if existing_reference == &reference && existing_bytes.as_ref() == encoded.as_ref() {
                return Ok(());
            }
            return Err(NativeExecutionError::Protocol(
                "remote value object id was reused for different content".into(),
            ));
        }
        objects.insert(reference.id, (reference, encoded));
        Ok(())
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
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .insert(task_id, program);
        self.completions
            .lock()
            .expect("remote completion registry poisoned")
            .insert(task_id, sender);
        let actions = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .handle(DriverCommand::Submit(Box::new(submission)))?;
        self.dispatch(actions);
        Ok(RemoteTaskCompletion { receiver })
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
                DriverAction::Checkpoint | DriverAction::GarbageCollectResults { .. } => {}
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
            let report = match (channel, program) {
                (Some(channel), Some(mut program)) => {
                    program.arguments = request.task.inputs.clone();
                    let transfer = this
                        .transfer_values(channel.as_ref(), request.worker_id, &program.arguments)
                        .await;
                    match transfer {
                        Err(error) => AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message: error.to_string(),
                        },
                        Ok(()) => match channel
                            .execute(RemoteAttempt {
                                scheduling: request.clone(),
                                program,
                            })
                            .await
                        {
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

    async fn transfer_values(
        &self,
        channel: &dyn RemoteWorkerChannel,
        worker_id: runmat_execution::identity::WorkerId,
        values: &[runmat_execution::value::ValuePayload],
    ) -> NativeExecutionResult<()> {
        let references = super::value_transfer::collect_references(values);
        for reference in references {
            let key = (worker_id, reference.id);
            if self.transferred.lock().await.contains(&key) {
                continue;
            }
            let (stored_reference, encoded) = self
                .objects
                .lock()
                .expect("remote value catalog poisoned")
                .get(&reference.id)
                .cloned()
                .ok_or_else(|| {
                    NativeExecutionError::Protocol(format!(
                        "remote value object {} is not registered",
                        reference.id
                    ))
                })?;
            if stored_reference != reference {
                return Err(NativeExecutionError::Protocol(
                    "remote value reference differs from its registered object".into(),
                ));
            }
            let receipt = channel.transfer_value(reference.clone(), &encoded).await?;
            if receipt.value_id != reference.id || receipt.encoded_bytes != encoded.len() as u64 {
                return Err(NativeExecutionError::Protocol(
                    "remote worker acknowledged a different value object".into(),
                ));
            }
            self.transferred.lock().await.insert(key);
        }
        Ok(())
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

    fn apply_report(self: &Arc<Self>, report: BackendReport) {
        let task_id = report.task_id;
        let (actions, terminal) = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let actions = match driver.handle(DriverCommand::BackendReport(report.clone())) {
                Ok(actions) => actions,
                Err(_) => return,
            };
            let snapshot = driver.snapshot();
            let terminal = snapshot.tasks.get(&task_id).and_then(|task| {
                matches!(
                    task.state,
                    TaskState::Succeeded
                        | TaskState::Failed
                        | TaskState::Cancelled
                        | TaskState::Indeterminate
                )
                .then_some(task.state)
            });
            (actions, terminal)
        };
        self.dispatch(actions);
        if let Some(state) = terminal {
            let outcome = match report.report {
                AttemptReport::Succeeded { result } => Ok(result),
                AttemptReport::Failed { message, .. } | AttemptReport::Lost { message } => {
                    Err(message)
                }
                AttemptReport::Cancelled => Err("remote task was cancelled".into()),
                AttemptReport::Started => Err(format!(
                    "remote task reached terminal state {state:?} without a terminal report"
                )),
            };
            if let Some(sender) = self
                .completions
                .lock()
                .expect("remote completion registry poisoned")
                .remove(&task_id)
            {
                let _ = sender.send(outcome);
            }
            self.programs
                .lock()
                .expect("remote program catalog poisoned")
                .remove(&task_id);
        }
    }

    fn resolve_non_success_terminals(&self) {
        let terminal = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot()
            .tasks
            .iter()
            .filter_map(|(task_id, task)| {
                let message = match task.state {
                    TaskState::Failed => "remote task failed",
                    TaskState::Cancelled => "remote task was cancelled",
                    TaskState::Indeterminate => "remote worker was lost",
                    _ => return None,
                };
                Some((*task_id, message.to_string()))
            })
            .collect::<Vec<_>>();
        let mut completions = self
            .completions
            .lock()
            .expect("remote completion registry poisoned");
        let mut programs = self
            .programs
            .lock()
            .expect("remote program catalog poisoned");
        for (task_id, message) in terminal {
            if let Some(sender) = completions.remove(&task_id) {
                let _ = sender.send(Err(message));
                programs.remove(&task_id);
            }
        }
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or(0)
}
