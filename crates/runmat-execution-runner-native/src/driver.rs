use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::HostCommand;
use tokio::io::BufReader;

use crate::local_store::{ArtifactStore, CheckpointStore};
use crate::protocol::{
    StoredProgram, WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use crate::{NativeExecutionConfig, NativeExecutionError, NativeExecutionResult};

pub(crate) type TransferResult = Result<AttemptSuccess, String>;

pub(crate) struct TaskCompletion {
    value: Mutex<Option<TransferResult>>,
    cancelled: AtomicBool,
}

impl TaskCompletion {
    fn new() -> Self {
        Self {
            value: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        }
    }

    pub(crate) fn try_value(&self) -> Option<TransferResult> {
        self.value.lock().expect("task completion poisoned").clone()
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
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
            capabilities: BTreeSet::from([Capability::ProcessIsolation]),
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
                    capabilities: resources.capabilities.clone(),
                },
            }))?;
        }
        let artifacts = ArtifactStore::new(config.store_root.join("artifacts"))?;
        let checkpoints = CheckpointStore::new(config.store_root.join("checkpoints"))?;
        let local = Arc::new(Self {
            store_root: config.store_root.clone(),
            config,
            scope_id,
            pool_id,
            driver: Mutex::new(driver),
            artifacts,
            checkpoints,
            completions: Mutex::new(HashMap::new()),
        });
        local.checkpoint()?;
        Ok(local)
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
        artifact.validate_against(&recipe).map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "local program artifact failed validation: {error}"
            ))
        })?;
        let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
        let stored = serde_json::to_vec(&StoredProgram { recipe, artifact })
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        self.artifacts.put(artifact_id, &stored)?;
        let completion = Arc::new(TaskCompletion::new());
        self.completions
            .lock()
            .expect("completion registry poisoned")
            .insert(task_id, Arc::clone(&completion));
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
        let actions =
            self.driver
                .lock()
                .expect("local driver poisoned")
                .handle(DriverCommand::Submit(Box::new(TaskSubmission {
                    request,
                    dependencies: BTreeSet::new(),
                    priority: 0,
                })))?;
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
            let result = execute_attempt(&this, &request, &completion);
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

fn execute_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    let stored = driver
        .artifacts
        .get(request.task.program_artifact_id)
        .map_err(|error| error.to_string())?;
    let function = request
        .task
        .callable
        .qualified_name
        .parse::<usize>()
        .map_err(|error| format!("invalid callable identity: {error}"))?;
    let stored: StoredProgram =
        serde_json::from_slice(&stored).map_err(|error| error.to_string())?;
    let worker_request = WorkerRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: stored.recipe,
        artifact: stored.artifact,
        function,
        arguments: request.task.inputs.clone(),
        requested_outputs: request.task.outputs.requested_outputs,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(run_process(driver, worker_request, completion))
}

async fn run_process(
    driver: &LocalDriver,
    request: WorkerRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    let mut command = HostCommand::new(&driver.config.executable);
    command.arguments = driver.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    command.max_stderr_bytes = driver.config.max_stderr_bytes;
    let mut child = command.spawn().await.map_err(|error| error.to_string())?;
    let stderr = child.captured_stderr();
    let stdio = child.take_stdio().map_err(|error| error.to_string())?;
    let mut reader = BufReader::new(stdio.stdout);
    let mut writer = stdio.stdin;
    let limits = FrameLimits {
        max_message_bytes: driver.config.max_message_bytes,
    };
    let payload = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
    write_payload(&mut writer, &payload, limits)
        .await
        .map_err(|error| error.to_string())?;
    let payload = loop {
        tokio::select! {
            response = read_payload(&mut reader, limits) => {
                break response.map_err(|error| {
                    let stderr = stderr.text();
                    if stderr.is_empty() { error.to_string() } else { format!("{error}; worker stderr: {stderr}") }
                })?;
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if completion.cancelled.load(Ordering::Acquire) {
                    let _ = child.terminate_tree().await;
                    return Err("execution was cancelled".into());
                }
            }
        }
    };
    let response: WorkerResponse =
        serde_json::from_slice(&payload).map_err(|error| error.to_string())?;
    let _ = child.wait().await;
    response
        .validate_against(&request)
        .map_err(|error| error.to_string())?;
    match response {
        WorkerResponse::Success { value } => Ok(AttemptSuccess {
            outputs: vec![value],
            result_objects: Vec::new(),
        }),
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => Ok(AttemptSuccess {
            outputs,
            result_objects,
        }),
        WorkerResponse::Failure { message } => Err(message),
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
