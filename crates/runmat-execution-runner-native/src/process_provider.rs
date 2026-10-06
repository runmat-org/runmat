//! Execute an original admitted attempt; scheduling and persistence belong to its caller.
use crate::config::NativeExecutionConfig;

use crate::local_store::ArtifactStore;
use crate::FilesystemObjectStore;
use std::sync::atomic::AtomicBool;

pub const NATIVE_OBJECT_STORE_ROOT_ENV: &str = "RUNMAT_EXECUTION_OBJECT_STORE_ROOT";
pub const NATIVE_ACCELERATOR_REQUIRED_ENV: &str = "RUNMAT_EXECUTION_ACCELERATOR_REQUIRED";
pub const NATIVE_ACCELERATOR_REQUESTS_ENV: &str = "RUNMAT_EXECUTION_ACCELERATOR_REQUESTS";
pub(crate) type TransferResult = Result<AttemptSuccess, String>;

pub(crate) trait ProcessControl: Send + Sync {
    fn cancellation(&self) -> &AtomicBool;
    fn record_progress(&self, progress: crate::protocol::ProgramProgress);
    /// A failed control channel stops execution but is not a user cancellation.
    fn failure(&self) -> Option<String> {
        None
    }
}

pub(crate) struct NativeProcessProvider<'a> {
    pub config: &'a NativeExecutionConfig,
    pub artifacts: &'a ArtifactStore,
    pub objects: &'a FilesystemObjectStore,
}
use crate::protocol::{
    StoredProgram, WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, AttemptFailureKind, AttemptReport,
    AttemptRequest, AttemptResourceAccounting, AttemptSuccess,
};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::IpcByteCounter;
use runmat_process_host::{ChildProcess, HostCommand, ProcessExit};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
pub(crate) mod exchange;

#[derive(Default)]
struct TransportMeters {
    sent: IpcByteCounter,
    received: IpcByteCounter,
}

pub(crate) struct ObservedAttempt {
    pub result: TransferResult,
    pub response: Option<WorkerResponse>,
    pub execution: AttemptExecutionObservation,
    pub unsuccessful_resources: Option<Box<AttemptResourceAccounting>>,
}

pub(crate) async fn execute_attempt(
    provider: &NativeProcessProvider<'_>,
    request: &AttemptRequest,
    control: &dyn ProcessControl,
    started: Instant,
) -> ObservedAttempt {
    let meters = TransportMeters::default();
    let mut termination = AttemptExecutionTermination::NotStarted;
    let response = if control.cancellation().load(Ordering::Acquire) {
        Err("execution was cancelled before process launch".into())
    } else {
        match prepare_program(provider, request) {
            Ok(program) => run_process(provider, program, control, &meters, &mut termination).await,
            Err(error) => Err(error),
        }
    };
    observe(request, started, response, meters, termination)
}

impl ObservedAttempt {
    pub(crate) fn into_report(
        self,
        request: &AttemptRequest,
        control: &dyn ProcessControl,
    ) -> (TransferResult, runmat_execution_runner::port::BackendReport) {
        let report = if let Some(message) = control.failure() {
            AttemptReport::Failed {
                kind: AttemptFailureKind::Execution,
                message,
            }
        } else {
            match &self.result {
                Ok(result) => AttemptReport::Succeeded {
                    result: result.clone(),
                },
                Err(_) if control.cancellation().load(Ordering::Acquire) => {
                    AttemptReport::Cancelled
                }
                Err(message) => AttemptReport::Failed {
                    kind: AttemptFailureKind::Execution,
                    message: message.clone(),
                },
            }
        };
        let mut report = runmat_execution_runner::port::BackendReport::for_request(request, report);
        report.execution = Some(Box::new(self.execution));
        report.unsuccessful_resources = self.unsuccessful_resources;
        (self.result, report)
    }

    pub(crate) fn not_started(request: &AttemptRequest, started: Instant, message: String) -> Self {
        observe(
            request,
            started,
            Err(message),
            TransportMeters::default(),
            AttemptExecutionTermination::NotStarted,
        )
    }
}

fn observe(
    request: &AttemptRequest,
    started: Instant,
    response: Result<WorkerResponse, String>,
    meters: TransportMeters,
    termination: AttemptExecutionTermination,
) -> ObservedAttempt {
    let wall_millis = u64::try_from(started.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let execution = AttemptExecutionObservation {
        attempt_id: request.id,
        worker_id: request.worker_id,
        wall_millis,
        termination,
        local_ipc_sent_bytes: meters.sent.bytes(),
        local_ipc_received_bytes: meters.received.bytes(),
    };
    let original_response = response.as_ref().ok().cloned();
    let mut result = response.and_then(|response| match response {
        WorkerResponse::Success { value } => {
            AttemptSuccess::new(request, vec![value], Vec::new(), wall_millis, 0, 0)
        }
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => AttemptSuccess::new(request, outputs, result_objects, wall_millis, 0, 0),
        WorkerResponse::Failure { message } => Err(message),
    });
    // Preflight only reads the admitted program. No process or result producer has
    // been invoked, so output/egress/relay consumption is known zero. Once launch
    // is attempted, no such claim can be inferred from its eventual error.
    let unsuccessful_resources =
        if result.is_err() && termination == AttemptExecutionTermination::NotStarted {
            match AttemptResourceAccounting::for_attempt(request, wall_millis, 0, 0, 0) {
                Ok(resources) => Some(Box::new(resources)),
                Err(error) => {
                    result = Err(format!(
                        "{}; preflight accounting unavailable: {error}",
                        result.as_ref().expect_err("unsuccessful preflight")
                    ));
                    None
                }
            }
        } else {
            None
        };
    ObservedAttempt {
        result,
        response: original_response,
        execution,
        unsuccessful_resources,
    }
}

fn prepare_program(
    provider: &NativeProcessProvider<'_>,
    request: &AttemptRequest,
) -> Result<WorkerRequest, String> {
    let stored = provider
        .artifacts
        .get(
            request.task.program_artifact_id,
            u64::from(provider.config.max_message_bytes),
        )
        .map_err(|error| error.to_string())?;
    let stored: StoredProgram =
        serde_json::from_slice(&stored).map_err(|error| error.to_string())?;
    let worker_request = WorkerRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: stored.recipe,
        artifact: stored.artifact,
        function: stored.function,
        arguments: request.task.inputs.clone(),
        requested_outputs: request.task.outputs.requested_outputs,
    };
    // Never repair a malformed stored entrypoint or invoke a foreign artifact.
    worker_request
        .validate_for_portable_host()
        .map_err(|error| error.to_string())?;
    if runmat_execution::identity::ArtifactId::derive(&[worker_request.artifact.id.0.bytes()])
        != request.task.program_artifact_id
    {
        return Err("native stored program differs from original admitted artifact".into());
    }
    Ok(worker_request)
}

async fn run_process(
    provider: &NativeProcessProvider<'_>,
    request: WorkerRequest,
    control: &dyn ProcessControl,
    meters: &TransportMeters,
    termination: &mut AttemptExecutionTermination,
) -> Result<WorkerResponse, String> {
    if control.cancellation().load(Ordering::Acquire) {
        return Err("execution was cancelled before process launch".into());
    }
    let mut command = HostCommand::new(&provider.config.executable);
    command.arguments = provider.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    command.environment.insert(
        NATIVE_OBJECT_STORE_ROOT_ENV.into(),
        provider.objects.root().to_string_lossy().into_owned(),
    );
    if !provider.config.worker_accelerators.is_empty() {
        command
            .environment
            .insert(NATIVE_ACCELERATOR_REQUIRED_ENV.into(), "1".into());
        command.environment.insert(
            NATIVE_ACCELERATOR_REQUESTS_ENV.into(),
            serde_json::to_string(&provider.config.worker_accelerators)
                .map_err(|error| error.to_string())?,
        );
    }
    command.max_stderr_bytes = provider.config.max_stderr_bytes;
    // Spawn can fail after creating a child (for example during containment).
    // Until root exit is observed, even an error leaves launch unconfirmed.
    *termination = AttemptExecutionTermination::Unconfirmed;
    let mut child = command.spawn().await.map_err(|error| error.to_string())?;
    let stderr = child.captured_stderr();
    let response = exchange::exchange(provider, &mut child, &request, control, meters).await;
    let exit = if response.is_err() {
        stop_and_wait(&mut child).await
    } else {
        wait_or_cancel(&mut child, control).await
    };
    let exit = match exit {
        Ok(exit) => exit,
        Err(error) => match stop_and_wait(&mut child).await {
            Ok(exit) => {
                *termination = AttemptExecutionTermination::ProcessExited {
                    code: exit.code,
                    success: exit.success,
                };
                return Err(format!("{}; process exit observation failed: {error}; subsequent cleanup confirmed root exit",
                    response.as_ref().err().map(String::as_str).unwrap_or("worker returned a terminal response")));
            }
            Err(cleanup) => {
                return Err(format!(
                    "{}; process exit unconfirmed: {error}; cleanup failed: {cleanup}",
                    response
                        .as_ref()
                        .err()
                        .map(String::as_str)
                        .unwrap_or("worker returned a terminal response")
                ))
            }
        },
    };
    *termination = AttemptExecutionTermination::ProcessExited {
        code: exit.code,
        success: exit.success,
    };
    let response = response.map_err(|error| {
        let stderr = stderr.text();
        if stderr.is_empty() {
            error
        } else {
            format!("{error}; worker stderr: {stderr}")
        }
    })?;
    if control.cancellation().load(Ordering::Acquire) {
        return Err("execution was cancelled".into());
    }
    if !exit.success {
        return Err(format!(
            "worker returned a response but exited unsuccessfully: {:?}",
            exit.code
        ));
    }
    Ok(response)
}

async fn stop_and_wait(child: &mut ChildProcess) -> Result<ProcessExit, String> {
    child
        .terminate_tree()
        .await
        .map_err(|error| error.to_string())?;
    child.wait().await.map_err(|error| error.to_string())
}

async fn wait_or_cancel(
    child: &mut ChildProcess,
    control: &dyn ProcessControl,
) -> Result<ProcessExit, String> {
    let exit = {
        let wait = child.wait();
        tokio::pin!(wait);
        loop {
            tokio::select! {
                exit = &mut wait => break Some(exit.map_err(|error| error.to_string())?),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if control.cancellation().load(Ordering::Acquire) { break None; }
                }
            }
        }
    };
    match exit {
        Some(exit) => Ok(exit),
        None => stop_and_wait(child).await,
    }
}
