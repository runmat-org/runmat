use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::state::{PoolState, TaskState};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::ValuePayload;
use runmat_execution::{
    CancellationReason, Digest, ExecutionScopeId, OutputContract, PoolId, TaskId,
};
use runmat_execution_artifact::{ProgramArtifact, ProgramBuildRecipe};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction,
    DriverCommand, DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};

mod process;

use crate::local_store::{prepare_session_root, ArtifactStore, CheckpointStore};
use crate::protocol::StoredProgram;
use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeObjectStore,
};

pub const NATIVE_OBJECT_STORE_ROOT_ENV: &str = "RUNMAT_EXECUTION_OBJECT_STORE_ROOT";
const MAX_BUFFERED_PROGRESS: usize = 256;

pub(crate) type TransferResult = Result<AttemptSuccess, String>;

pub(crate) struct TaskCompletion {
    value: Mutex<Option<TransferResult>>,
    progress: Mutex<VecDeque<crate::protocol::ProgramProgress>>,
    cancelled: AtomicBool,
}

impl TaskCompletion {
    fn new() -> Self {
        Self {
            value: Mutex::new(None),
            progress: Mutex::new(VecDeque::new()),
            cancelled: AtomicBool::new(false),
        }
    }

    pub(crate) fn try_value(&self) -> Option<TransferResult> {
        self.value.lock().expect("task completion poisoned").clone()
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
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
        }
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
            min_workers: config.max_workers,
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

    pub(crate) fn submit(
        self: &Arc<Self>,
        task_id: TaskId,
        function: usize,
        recipe: ProgramBuildRecipe,
        artifact: ProgramArtifact,
        inputs: Vec<ValuePayload>,
        outputs: OutputContract,
    ) -> NativeExecutionResult<Arc<TaskCompletion>> {
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let request = TaskRequest {
            id: task_id,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            program_artifact_id: artifact_id,
            callable: Callable {
                owner_identity: "local-session".into(),
                qualified_name: function.to_string(),
                entrypoint_digest: Digest::sha256(function.to_be_bytes()),
            },
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
            retry: RetryPolicy::Never,
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
        artifact.validate_against(&recipe).map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "local program artifact failed validation: {error}"
            ))
        })?;
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        if submission.request.scope_id != self.scope_id
            || submission.request.pool_id != self.pool_id
            || submission.request.program_artifact_id != artifact_id
        {
            return Err(NativeExecutionError::Protocol(
                "local task submission differs from its session or program artifact".into(),
            ));
        }
        let task_id = submission.request.id;
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
                Err(_) if completion.cancelled.load(Ordering::Acquire) => AttemptReport::Cancelled,
                Err(message) => AttemptReport::Failed {
                    kind: AttemptFailureKind::Execution,
                    message: message.clone(),
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
            let accepted = this
                .driver
                .lock()
                .expect("local driver poisoned")
                .snapshot()
                .tasks
                .get(&request.task_id)
                .is_some_and(|task| {
                    matches!(
                        task.state,
                        TaskState::Succeeded | TaskState::Failed | TaskState::Cancelled
                    )
                });
            if accepted {
                completion.complete(result);
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
    use super::{TaskCompletion, TransferResult};

    #[test]
    fn completion_is_pollable_without_blocking_the_await_caller() {
        let completion = TaskCompletion::new();
        assert_eq!(completion.try_value(), None);

        let result: TransferResult = Err("completed".into());
        completion.complete(result.clone());
        assert_eq!(completion.try_value(), Some(result));
    }
}
