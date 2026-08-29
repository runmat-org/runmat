use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::state::{PoolState, TaskState};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::ValuePayload;
use runmat_execution::{
    CancellationReason, ExecutionScopeId, OutputContract, PoolId, ProgramCallable, TaskId,
};
use runmat_execution_artifact::{
    ProgramArtifact, ProgramBuildRecipe, ProgramExecutionDescriptor,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};

pub(crate) mod collective;
mod process;

use crate::local_store::{prepare_session_root, ArtifactStore, CheckpointStore};
use crate::protocol::StoredProgram;
use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeObjectStore,
};

pub const NATIVE_OBJECT_STORE_ROOT_ENV: &str = "RUNMAT_EXECUTION_OBJECT_STORE_ROOT";
const MAX_BUFFERED_PROGRESS: usize = 256;
const NO_COMPLETION_ORDER: u64 = u64::MAX;
static NEXT_TASK_COMPLETION_ORDER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn next_task_completion_order() -> u64 {
    NEXT_TASK_COMPLETION_ORDER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TransferFailure {
    Execution(String),
    Infrastructure(String),
    WorkerLost(String),
    Cancelled,
    Runtime(Box<runmat_execution::ProgramRuntimeFailure>),
}

impl std::fmt::Display for TransferFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(message)
            | Self::Infrastructure(message)
            | Self::WorkerLost(message) => formatter.write_str(message),
            Self::Cancelled => formatter.write_str("execution was cancelled"),
            Self::Runtime(failure) => formatter.write_str(&failure.message),
        }
    }
}

pub(crate) type TransferResult = Result<AttemptSuccess, TransferFailure>;

pub(crate) struct LocalProgramSubmission {
    pub(crate) task_id: TaskId,
    pub(crate) callable: ProgramCallable,
    pub(crate) invocation_context: runmat_execution::ProgramInvocationContext,
    pub(crate) recipe: ProgramBuildRecipe,
    pub(crate) artifact: ProgramArtifact,
    pub(crate) inputs: Vec<ValuePayload>,
    pub(crate) outputs: OutputContract,
    pub(crate) retry: RetryPolicy,
}

pub(crate) struct TaskCompletion {
    value: Mutex<Option<TransferResult>>,
    progress: Mutex<VecDeque<crate::protocol::ProgramProgress>>,
    cancelled: AtomicBool,
    completion_order: AtomicU64,
}

impl TaskCompletion {
    fn new() -> Self {
        Self {
            value: Mutex::new(None),
            progress: Mutex::new(VecDeque::new()),
            cancelled: AtomicBool::new(false),
            completion_order: AtomicU64::new(NO_COMPLETION_ORDER),
        }
    }

    pub(crate) fn try_value(&self) -> Option<TransferResult> {
        self.value.lock().expect("task completion poisoned").clone()
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.mark_completed();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn completion_order(&self) -> Option<u64> {
        let order = self.completion_order.load(Ordering::Acquire);
        (order != NO_COMPLETION_ORDER).then_some(order)
    }

    pub(crate) fn record_progress(&self, progress: crate::protocol::ProgramProgress) {
        let mut buffered = self.progress.lock().expect("task progress poisoned");
        if buffered.len() == MAX_BUFFERED_PROGRESS {
            buffered.pop_front();
        }
        buffered.push_back(progress);
    }

    pub(crate) fn drain_progress(&self) -> Vec<crate::protocol::ProgramProgress> {
        self.progress
            .lock()
            .expect("task progress poisoned")
            .drain(..)
            .collect()
    }

    fn complete(&self, value: TransferResult) {
        let mut result = self.value.lock().expect("task completion poisoned");
        if result.is_none() {
            *result = Some(value);
            self.mark_completed();
        }
    }

    fn mark_completed(&self) {
        let _ = self.completion_order.compare_exchange(
            NO_COMPLETION_ORDER,
            next_task_completion_order(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

pub(crate) struct LocalDriver {
    config: NativeExecutionConfig,
    store_root: std::path::PathBuf,
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    driver: Mutex<Driver>,
    artifacts: ArtifactStore,
    objects: NativeObjectStore,
    checkpoints: CheckpointStore,
    completions: Mutex<HashMap<TaskId, Arc<TaskCompletion>>>,
    collectives: collective::ProcessCollectiveBroker,
}

impl LocalDriver {
    pub(crate) fn new(
        config: NativeExecutionConfig,
        scope_id: ExecutionScopeId,
    ) -> NativeExecutionResult<Arc<Self>> {
        config
            .validate()
            .map_err(NativeExecutionError::Configuration)?;
        let pool_id = PoolId::derive(&[scope_id.bytes(), b"local"]);
        let cpu = config.max_workers.saturating_mul(1000);
        let memory = u64::from(config.max_workers).saturating_mul(1024 * 1024 * 1024);
        let resources = ResourceInventory {
            cpu_millicores: cpu,
            memory_bytes: memory,
            scratch_bytes: memory,
            accelerators: Vec::new(),
            capabilities: config.worker_capabilities.clone(),
        };
        let mut driver = Driver::new(DriverConfig::default(), 1)?;
        driver.handle(DriverCommand::RegisterScope {
            scope_id,
            parent: None,
        })?;
        driver.handle(DriverCommand::CreatePool(PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: config.max_workers,
            max_in_flight: config.max_workers,
            resource_limit: resources.clone(),
        }))?;
        driver.handle(DriverCommand::SetPoolState {
            pool_id,
            state: PoolState::Ready,
        })?;
        for index in 0..config.max_workers {
            let index_bytes = index.to_be_bytes();
            let worker_id = WorkerId::derive(&[pool_id.bytes().as_slice(), &index_bytes]);
            driver.handle(DriverCommand::RegisterWorker(WorkerSpec {
                id: worker_id,
                pool_id,
                resources: ResourceInventory {
                    cpu_millicores: 1000,
                    memory_bytes: 1024 * 1024 * 1024,
                    scratch_bytes: 1024 * 1024 * 1024,
                    accelerators: Vec::new(),
                    capabilities: config.worker_capabilities.clone(),
                },
            }))?;
        }
        prepare_session_root(&config.store_root)?;
        let artifacts = ArtifactStore::new(config.store_root.join("artifacts"))?;
        let objects =
            NativeObjectStore::open(config.store_root.join("objects"), config.max_object_bytes)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let checkpoints = CheckpointStore::new(config.store_root.join("checkpoints"))?;
        let local = Arc::new(Self {
            store_root: config.store_root.clone(),
            config,
            scope_id,
            pool_id,
            driver: Mutex::new(driver),
            artifacts,
            objects,
            checkpoints,
            completions: Mutex::new(HashMap::new()),
            collectives: collective::ProcessCollectiveBroker::default(),
        });
        local.checkpoint()?;
        Ok(local)
    }

    pub(crate) fn object_store(&self) -> NativeObjectStore {
        self.objects.clone()
    }

    pub(crate) const fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    pub(crate) const fn pool_id(&self) -> PoolId {
        self.pool_id
    }

    pub(crate) const fn max_workers(&self) -> u32 {
        self.config.max_workers
    }

    pub(crate) fn active_workers(&self) -> NativeExecutionResult<u32> {
        let driver = self.driver.lock().expect("local driver poisoned");
        let pool = driver
            .snapshot()
            .pools
            .get(&self.pool_id)
            .cloned()
            .ok_or_else(|| NativeExecutionError::Protocol("local pool disappeared".into()))?;
        u32::try_from(
            pool.workers
                .values()
                .filter(|worker| worker.accepts_work())
                .count(),
        )
        .map_err(|_| NativeExecutionError::Protocol("local worker count exceeds u32".into()))
    }

    pub(crate) fn resize_pool(self: &Arc<Self>, desired_workers: u32) -> NativeExecutionResult<()> {
        let actions = self
            .driver
            .lock()
            .expect("local driver poisoned")
            .resize_registered_pool(self.pool_id, desired_workers)?;
        self.checkpoint()?;
        Self::dispatch(Arc::clone(self), actions);
        Ok(())
    }

    pub(crate) fn submit(
        self: &Arc<Self>,
        submission: LocalProgramSubmission,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        let LocalProgramSubmission {
            task_id,
            callable,
            invocation_context,
            recipe,
            artifact,
            inputs,
            outputs,
            retry,
        } = submission;
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let request = TaskRequest {
            id: task_id,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            program_artifact_id: artifact_id,
            callable: Callable::for_program("local-session", &callable),
            invocation_context,
            inputs,
            outputs,
            resources: ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: 1024 * 1024,
                scratch_bytes: 1024 * 1024,
                max_wall_millis: 24 * 60 * 60 * 1000,
                max_artifact_bytes: u64::from(self.config.max_message_bytes),
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            retry,
            deadline_unix_millis: None,
        };
        self.submit_task(
            TaskSubmission {
                request,
                dependencies: BTreeSet::new(),
                priority: 0,
            },
            recipe,
            artifact,
        )
    }

    pub(crate) fn submit_task(
        self: &Arc<Self>,
        submission: TaskSubmission,
        recipe: ProgramBuildRecipe,
        artifact: ProgramArtifact,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        let callable = &submission.request.callable.program;
        ProgramExecutionDescriptor {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
            recipe: recipe.clone(),
            artifact: artifact.clone(),
            callable: callable.clone(),
            requested_outputs: submission.request.outputs.requested_outputs,
        }
        .validate_for_portable_host()
        .map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "local program descriptor failed validation: {error}"
            ))
        })?;
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        if submission.request.scope_id != self.scope_id
            || submission.request.pool_id != self.pool_id
            || submission.request.program_artifact_id != artifact_id
            || submission
                .request
                .invocation_context
                .validate_for(callable)
                .is_err()
        {
            return Err(NativeExecutionError::Protocol(
                "local task submission differs from its session or program artifact".into(),
            ));
        }
        let task_id = submission.request.id;
        callable
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let stored = serde_json::to_vec(&StoredProgram { recipe, artifact })
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        self.artifacts.put(artifact_id, &stored)?;
        let completion = Arc::new(TaskCompletion::new());
        self.completions
            .lock()
            .expect("completion registry poisoned")
            .insert(task_id, Arc::clone(&completion));
        let actions = self
            .driver
            .lock()
            .expect("local driver poisoned")
            .handle(DriverCommand::Submit(Box::new(submission)))?;
        self.checkpoint()?;
        Self::dispatch(Arc::clone(self), actions);
        Ok(completion)
    }

    pub(crate) fn cancel_all(self: &Arc<Self>, reason: CancellationReason) {
        let actions = self
            .driver
            .lock()
            .expect("local driver poisoned")
            .handle(DriverCommand::CancelScope {
                scope_id: self.scope_id,
                reason,
                now_millis: runmat_time_millis(),
            })
            .unwrap_or_default();
        Self::dispatch(Arc::clone(self), actions);
        let _ = self.checkpoint();
    }

    pub(crate) fn fail_collective_gang(
        &self,
        gang: &runmat_execution::GangHandle,
        reason: impl Into<String>,
    ) {
        self.collectives.fail_gang(gang, reason);
    }

    pub(crate) fn terminate_collective_rank(
        &self,
        gang: &runmat_execution::GangHandle,
        rank: runmat_types::LabRank,
    ) {
        self.collectives.rank_terminated(gang, rank);
    }

    fn dispatch(this: Arc<Self>, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => Self::launch(Arc::clone(&this), request),
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    if let Some(completion) = this
                        .completions
                        .lock()
                        .expect("completion registry poisoned")
                        .get(&request.task_id)
                    {
                        completion.cancel();
                    }
                }
                DriverAction::Checkpoint => {
                    let _ = this.checkpoint();
                }
                DriverAction::ResizePool { .. } | DriverAction::GarbageCollectResults { .. } => {}
            }
        }
    }

    fn launch(this: Arc<Self>, request: AttemptRequest) {
        let started = BackendReport::for_request(&request, AttemptReport::Started);
        let actions = this
            .driver
            .lock()
            .expect("local driver poisoned")
            .handle(DriverCommand::BackendReport(started))
            .unwrap_or_default();
        Self::dispatch(Arc::clone(&this), actions);
        std::thread::spawn(move || {
            let spmd_gang = match &request.task.invocation_context {
                runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
                    Some(task.gang.clone())
                }
                _ => None,
            };
            let completion = this
                .completions
                .lock()
                .expect("completion registry poisoned")
                .get(&request.task_id)
                .cloned()
                .expect("scheduled task has completion");
            let result = process::execute_attempt(&this, &request, &completion);
            let report = match &result {
                Ok(success) => AttemptReport::Succeeded {
                    result: success.clone(),
                },
                Err(TransferFailure::Execution(message)) => AttemptReport::Failed {
                    kind: AttemptFailureKind::Execution,
                    message: message.clone(),
                },
                Err(TransferFailure::Infrastructure(message)) => AttemptReport::Failed {
                    kind: AttemptFailureKind::Infrastructure,
                    message: message.clone(),
                },
                Err(TransferFailure::WorkerLost(message)) => AttemptReport::Lost {
                    message: message.clone(),
                },
                Err(TransferFailure::Cancelled) => AttemptReport::Cancelled,
                Err(TransferFailure::Runtime(failure)) => AttemptReport::RuntimeFailed {
                    failure: *failure.clone(),
                },
            };
            let actions = this
                .driver
                .lock()
                .expect("local driver poisoned")
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    &request, report,
                )))
                .unwrap_or_default();
            let terminal_state = this
                .driver
                .lock()
                .expect("local driver poisoned")
                .snapshot()
                .tasks
                .get(&request.task_id)
                .map(|task| task.state);
            if let Some(result) = terminal_completion(terminal_state, result) {
                completion.complete(result);
                if let Some(gang) = spmd_gang {
                    if completion.try_value().is_some_and(|result| result.is_err()) {
                        // Publish the originating task result before releasing
                        // peers blocked in this gang's collectives.
                        this.collectives.fail_gang(&gang, "an SPMD peer failed");
                    }
                }
            }
            let _ = this.checkpoint();
            Self::dispatch(Arc::clone(&this), actions);
        });
    }

    fn checkpoint(&self) -> NativeExecutionResult<()> {
        self.checkpoints.write(
            &self
                .driver
                .lock()
                .expect("local driver poisoned")
                .snapshot(),
        )
    }
}

fn terminal_completion(state: Option<TaskState>, result: TransferResult) -> Option<TransferResult> {
    match state? {
        TaskState::Succeeded | TaskState::Failed | TaskState::Indeterminate => Some(result),
        TaskState::Cancelled => Some(Err(TransferFailure::Cancelled)),
        TaskState::Deferred
        | TaskState::Ready
        | TaskState::Assigned
        | TaskState::Running
        | TaskState::Committing => None,
    }
}

impl Drop for LocalDriver {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.store_root);
    }
}

fn runmat_time_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

#[cfg(test)]
mod task_completion_tests {
    use std::collections::BTreeSet;

    use runmat_execution::resource::Capability;

    use super::{
        terminal_completion, LocalDriver, TaskCompletion, TransferFailure, TransferResult,
    };

    #[test]
    fn completion_is_pollable_without_blocking_the_await_caller() {
        let completion = TaskCompletion::new();
        assert_eq!(completion.try_value(), None);

        let result: TransferResult = Err(TransferFailure::Execution("completed".into()));
        completion.complete(result.clone());
        assert_eq!(completion.try_value(), Some(result));
    }

    #[test]
    fn scheduler_terminal_state_is_authoritative_for_cancellation_and_loss() {
        let late_success = Ok(runmat_execution_runner::AttemptSuccess::Values {
            outputs: Vec::new(),
            result_objects: Vec::new(),
        });
        assert_eq!(
            terminal_completion(
                Some(runmat_execution::state::TaskState::Cancelled),
                late_success
            ),
            Some(Err(TransferFailure::Cancelled))
        );

        let worker_loss = Err(TransferFailure::WorkerLost("worker exited".into()));
        assert_eq!(
            terminal_completion(
                Some(runmat_execution::state::TaskState::Indeterminate),
                worker_loss.clone(),
            ),
            Some(worker_loss)
        );
        assert_eq!(
            terminal_completion(
                Some(runmat_execution::state::TaskState::Running),
                Err(TransferFailure::Execution("not terminal".into())),
            ),
            None
        );
    }

    #[test]
    fn local_pool_resizes_its_schedulable_worker_inventory() {
        let temporary = tempfile::tempdir().unwrap();
        let scope = crate::config::fresh_scope_id(b"resize-test", 1);
        let driver = LocalDriver::new(
            crate::NativeExecutionConfig {
                executable: std::env::current_exe().unwrap(),
                worker_arguments: vec!["--execution-worker".into()],
                max_workers: 3,
                max_message_bytes: 1024,
                max_object_bytes: 1024,
                max_stderr_bytes: 1024,
                store_root: temporary.path().join("session"),
                worker_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            scope,
        )
        .unwrap();
        assert_eq!(driver.active_workers().unwrap(), 3);
        driver.resize_pool(1).unwrap();
        assert_eq!(driver.active_workers().unwrap(), 1);
        driver.resize_pool(3).unwrap();
        assert_eq!(driver.active_workers().unwrap(), 3);
    }
}
