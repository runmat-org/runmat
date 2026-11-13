//! Execute an original admitted attempt; scheduling and persistence belong to its caller.
use crate::config::NativeExecutionConfig;
use crate::driver::process::LocalRetirementEvidence;

use crate::local_store::ArtifactStore;
use crate::FilesystemObjectStore;
use std::sync::atomic::AtomicBool;

pub const NATIVE_OBJECT_STORE_ROOT_ENV: &str = "RUNMAT_EXECUTION_OBJECT_STORE_ROOT";
pub(crate) const NATIVE_MESHING_SUBTASK_CACHE_ROOT_ENV: &str =
    "RUNMAT_EXECUTION_MESHING_SUBTASK_CACHE_ROOT";
pub(crate) type TransferResult = Result<AttemptSuccess, String>;

pub(crate) trait ProcessControl:
    crate::process_output::OutputDrainControl + Send + Sync
{
    fn begin_raw_output(
        &self,
        attempt: &AttemptRequest,
        materialized: u64,
    ) -> Result<
        std::sync::Mutex<crate::process_output::RawOutputJournal>,
        std::sync::Arc<crate::RawOutputError>,
    >;
    fn cancellation(&self) -> &AtomicBool;
    /// Present only for an authenticated scheduler/user cancellation, not a
    /// failed control channel. Existing non-Graph providers need no wire change.
    fn cancellation_reason(&self) -> Option<runmat_execution::CancellationReason> {
        None
    }
    fn record_progress(&self, progress: crate::protocol::ProgramProgress);
    fn record_console(
        &self,
        _record: &runmat_execution::console::ConsoleRecord,
    ) -> Result<(), String> {
        Ok(())
    }
    fn finish_console(
        &self,
        _receipt: &runmat_execution::console::ConsoleReceipt,
    ) -> Result<(), String> {
        Ok(())
    }
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
    StoredProgram, WorkerLaunchRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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
pub(crate) mod cancellation;

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
    pub console: Option<runmat_execution::console::ConsoleTranscript>,
    pub console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    retirement: LocalRetirementEvidence,
}

/// Mutably borrowed evidence stays with its original attempt; no snapshot is copied.
struct ProcessOutcome<'a> {
    termination: &'a mut AttemptExecutionTermination,
    console: &'a mut Option<runmat_execution::console::ConsoleTranscript>,
    console_scopes: &'a mut Vec<runmat_execution::console::ScopedConsoleTranscript>,
    retirement: &'a mut LocalRetirementEvidence,
}

pub(crate) async fn execute_attempt(
    provider: &NativeProcessProvider<'_>,
    request: &AttemptRequest,
    control: &dyn ProcessControl,
    started: Instant,
) -> ObservedAttempt {
    let meters = TransportMeters::default();
    let mut termination = AttemptExecutionTermination::NotStarted;
    let mut console = None;
    let mut console_scopes = Vec::new();
    let mut retirement = LocalRetirementEvidence::NotStarted;
    let response = if control.cancellation().load(Ordering::Acquire) {
        Err("execution was cancelled before process launch".into())
    } else {
        match prepare_program(provider, request) {
            Ok(program) => {
                run_process(
                    provider,
                    request,
                    program,
                    control,
                    &meters,
                    ProcessOutcome {
                        termination: &mut termination,
                        console: &mut console,
                        console_scopes: &mut console_scopes,
                        retirement: &mut retirement,
                    },
                )
                .await
            }
            Err(error) => Err(error),
        }
    };
    let mut observed = observe(request, started, response, meters, termination, retirement);
    observed.console = console;
    observed.console_scopes = console_scopes;
    observed
}

impl ObservedAttempt {
    pub(crate) fn into_report(
        self,
        request: &AttemptRequest,
        control: &dyn ProcessControl,
    ) -> (TransferResult, runmat_execution_runner::port::BackendReport) {
        let control_failure = control.failure();
        let report = if self.retirement == LocalRetirementEvidence::Unconfirmed {
            let mut message = self.result.as_ref().err().cloned().unwrap_or_else(|| {
                "durable direct worker has no complete process-tree retirement proof".into()
            });
            if let Some(original) = control_failure {
                message.push_str("; original control failure: ");
                message.push_str(&original);
            }
            AttemptReport::RetirementUnverified { message }
        } else if let Some(message) = control_failure {
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
                    // A stop request is not retirement. This branch is reached
                    // only before launch or after the physical owner retired
                    // the complete tree while retaining the original leader.
                    AttemptReport::Retired
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
        report.console = self.console.map(Box::new);
        report.console_scopes = self.console_scopes;
        (self.result, report)
    }

    #[cfg(all(test, unix))]
    pub(crate) fn not_started(request: &AttemptRequest, started: Instant, message: String) -> Self {
        observe(
            request,
            started,
            Err(message),
            TransportMeters::default(),
            AttemptExecutionTermination::NotStarted,
            LocalRetirementEvidence::NotStarted,
        )
    }
}

fn observe(
    request: &AttemptRequest,
    started: Instant,
    response: Result<WorkerResponse, String>,
    meters: TransportMeters,
    termination: AttemptExecutionTermination,
    retirement: LocalRetirementEvidence,
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
        WorkerResponse::Success { outputs } => {
            AttemptSuccess::new(request, outputs, Vec::new(), wall_millis, 0, 0)
        }
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => AttemptSuccess::new(request, outputs, result_objects, wall_millis, 0, 0),
        WorkerResponse::Failure { message } => Err(message),
        WorkerResponse::RuntimeFailure { failure } => Err(failure.message),
        WorkerResponse::GraphSuccess { .. } | WorkerResponse::SpmdSuccess { .. } => {
            Err("durable direct program returned a graph or SPMD response".into())
        }
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
        retirement,
        console: None,
        console_scopes: Vec::new(),
        result,
        response: original_response,
        execution,
        unsuccessful_resources,
    }
}

fn prepare_program(
    provider: &NativeProcessProvider<'_>,
    request: &AttemptRequest,
) -> Result<WorkerLaunchRequest, String> {
    request
        .task
        .validate_invocation_reservation()
        .map_err(|error| error.to_string())?;
    if let Some(limits) = request.task.console {
        limits
            .validate_transport(u64::from(provider.config.max_message_bytes))
            .map_err(|error| error.to_string())?;
    }
    let artifact_id = request
        .task
        .program_artifact_id()
        .ok_or("durable program attempt has no program artifact")?;
    let stored = provider
        .artifacts
        .get(artifact_id, u64::from(provider.config.max_message_bytes))
        .map_err(|error| error.to_string())?;
    let stored: StoredProgram =
        serde_json::from_slice(&stored).map_err(|error| error.to_string())?;
    let StoredProgram::Direct(stored) = stored else {
        return Err("durable program attempts execute direct programs only".into());
    };
    let assignment = runmat_execution::ProgramExecutionAssignment {
        scope_id: request.scope_id,
        pool_id: request.task.pool_id,
        task_id: request.task_id,
        attempt_id: request.id,
        worker_id: request.worker_id,
        backend: runmat_execution::PoolBackend::LocalProcesses,
        resources: request.resource_assignment.clone(),
    };
    let worker_request = runmat_execution_artifact::ProgramExecutionRequest {
        console: request.task.console,
        environment: request.task.environment.clone(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: stored.recipe,
        artifact: stored.artifact,
        callable: request.task.callable.program.clone(),
        context: request.task.invocation_context.clone(),
        assignment: Some(assignment),
        job_id: None,
        arguments: request.task.inputs.clone(),
        requested_outputs: request.task.outputs.requested_outputs,
    };
    // Never repair a malformed stored entrypoint or invoke a foreign artifact.
    worker_request
        .validate_for_portable_host()
        .map_err(|error| error.to_string())?;
    if runmat_execution::identity::ArtifactId::derive(&[worker_request.artifact.id.0.bytes()])
        != artifact_id
    {
        return Err("native stored program differs from original admitted artifact".into());
    }
    let invocation = runmat_execution::console::invocation_memory_bytes(
        &request.task.environment,
        request.task.console,
    )
    .map_err(|error| error.to_string())?;
    let minimum = invocation
        .checked_add(runmat_process_host::ipc::WORKER_PARENT_MONITOR_STACK_BYTES)
        .ok_or("native invocation storage overflow")?;
    if minimum > request.task.resources.memory_bytes {
        return Err("native invocation and parent monitor exceed task memory reservation".into());
    }
    let reserved = Box::new(
        runmat_execution_artifact::ReservedProgramExecutionRequest::new(
            worker_request,
            request.task.resources.clone(),
        )
        .map_err(|error| error.to_string())?,
    );
    match request.task.console {
        Some(controls) => Ok(WorkerLaunchRequest::ReservedConsole {
            request: reserved,
            console_budget: crate::console_source_subgrant(&request.task.resources, controls)
                .map_err(|error| error.to_string())?,
        }),
        None => Ok(WorkerLaunchRequest::Reserved(reserved)),
    }
}

async fn run_process(
    provider: &NativeProcessProvider<'_>,
    attempt: &AttemptRequest,
    request: WorkerLaunchRequest,
    control: &dyn ProcessControl,
    meters: &TransportMeters,
    outcome: ProcessOutcome<'_>,
) -> Result<WorkerResponse, String> {
    let ProcessOutcome {
        termination,
        console,
        console_scopes,
        retirement,
    } = outcome;
    if control.cancellation().load(Ordering::Acquire) {
        return Err("execution was cancelled before process launch".into());
    }
    let mut command = HostCommand::new(&provider.config.executable);
    let listener = runmat_process_host::ipc::private_control::PrivateControlListener::bind()
        .map_err(|error| error.to_string())?;
    command.stdio = runmat_process_host::StdioPolicy::PrivateControl(listener.endpoint().clone());
    command.arguments = provider.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    command.environment.insert(
        NATIVE_OBJECT_STORE_ROOT_ENV.into(),
        provider.objects.root().to_string_lossy().into_owned(),
    );
    command.environment.insert(
        NATIVE_MESHING_SUBTASK_CACHE_ROOT_ENV.into(),
        provider
            .config
            .meshing_subtask_cache_root
            .clone()
            .unwrap_or_else(|| provider.config.store_root.join("meshing-subtask-cache/v1"))
            .to_string_lossy()
            .into_owned(),
    );
    command.max_stderr_bytes = provider.config.max_stderr_bytes;
    // Spawn can fail after creating a child (for example during containment).
    // Until the complete tree retires, even an error leaves launch unconfirmed.
    *termination = AttemptExecutionTermination::Unconfirmed;
    *retirement = LocalRetirementEvidence::Unconfirmed;
    let mut child = command.spawn().await.map_err(|error| error.to_string())?;
    let journal = match control.begin_raw_output(attempt, 0) {
        Ok(journal) => journal,
        Err(error) => {
            control.stop_for_raw_output(std::sync::Arc::clone(&error));
            return retire_output_setup_failure(
                &mut child,
                termination,
                retirement,
                error.to_string(),
            )
            .await;
        }
    };
    let pipes = match child.take_application_output() {
        Ok(pipes) => pipes,
        Err(error) => {
            let mut failure =
                std::sync::Arc::new(crate::RawOutputError::Io(std::io::Error::other(error)));
            {
                let mut journal = journal.lock().expect("raw output journal poisoned");
                if let Err(cleanup) = journal.fail_open_streams() {
                    failure = std::sync::Arc::new(crate::RawOutputError::Cleanup {
                        primary: failure,
                        cleanup: Box::new(cleanup),
                    });
                }
            }
            control.stop_for_raw_output(std::sync::Arc::clone(&failure));
            control.raw_output_changed();
            let result = retire_output_setup_failure(
                &mut child,
                termination,
                retirement,
                failure.to_string(),
            )
            .await;
            if *retirement == LocalRetirementEvidence::FullTreeRetired {
                if let Err(cleanup) = journal
                    .lock()
                    .expect("raw output journal poisoned")
                    .mark_retired()
                {
                    let combined = crate::RawOutputError::Cleanup {
                        primary: failure,
                        cleanup: Box::new(cleanup),
                    };
                    let Err(primary) = result.as_ref() else {
                        unreachable!("setup failure remains an error");
                    };
                    return Err(format!("{primary}; {combined}"));
                }
            }
            return result;
        }
    };
    let output_lifetime = crate::process_output::OutputDrainLifetime::default();
    let owner = async {
        let result = async {
    let stderr = child.captured_stderr();
    let response = exchange::exchange(
        provider, listener, &request, control, meters, console, console_scopes,
    )
    .await;
    let exit = if response.is_err() {
        stop_and_wait(&mut child).await
    } else {
        wait_or_cancel(&mut child, control).await
    };
    let exit = match exit {
        Ok(exit) => exit,
        Err(error) => match stop_and_wait(&mut child).await {
            Ok(exit) => {
                *retirement = LocalRetirementEvidence::FullTreeRetired;
                *termination = AttemptExecutionTermination::ProcessExited {
                    code: exit.code,
                    success: exit.success,
                };
                return Err(format!("{}; process exit observation failed: {error}; subsequent cleanup retired the process tree",
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
    *retirement = LocalRetirementEvidence::FullTreeRetired;
    let response = response.map_err(|error| {
        let stderr = stderr.text();
        if stderr.is_empty() {
            error
        } else {
            format!("{error}; worker stderr: {stderr}")
        }
    })?;
    if let Some(error) = control.raw_output_failure() {
        if !matches!(&response, WorkerResponse::Failure { .. } | WorkerResponse::RuntimeFailure { .. }) {
            return Err(error.to_string());
        }
    }
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
        .await;
        output_lifetime.finish();
        result
    };
    let drain =
        crate::process_output::drain_pipes(Some(pipes), &journal, control, &output_lifetime);
    let (result, drain_result) = tokio::join!(owner, drain);
    let retirement_result = if *retirement == LocalRetirementEvidence::FullTreeRetired {
        Some(
            journal
                .lock()
                .expect("raw output journal poisoned")
                .mark_retired(),
        )
    } else {
        None
    };
    let retirement_error = retirement_result.and_then(Result::err);
    let has_retirement_cleanup_error = retirement_error.is_some();
    let raw_error = match (drain_result.err(), retirement_error) {
        (Some(primary), Some(cleanup)) => {
            Some(std::sync::Arc::new(crate::RawOutputError::Cleanup {
                primary,
                cleanup: Box::new(cleanup),
            }))
        }
        (None, Some(cleanup)) => Some(match control.raw_output_failure() {
            Some(primary) => std::sync::Arc::new(crate::RawOutputError::Cleanup {
                primary,
                cleanup: Box::new(cleanup),
            }),
            None => std::sync::Arc::new(cleanup),
        }),
        (Some(primary), None) => Some(primary),
        (None, None) => None,
    };
    if let Some(error) = raw_error {
        control.stop_for_raw_output(std::sync::Arc::clone(&error));
        if result.as_ref().is_ok_and(|response| {
            !matches!(
                response,
                WorkerResponse::Failure { .. } | WorkerResponse::RuntimeFailure { .. }
            )
        }) || result
            .as_ref()
            .err()
            .is_some_and(|message| message == "execution was cancelled")
        {
            return Err(error.to_string());
        }
        if has_retirement_cleanup_error {
            return result
                .map_err(|message| format!("{message}; output cleanup also failed: {error}"));
        }
    }
    result
}

async fn retire_output_setup_failure(
    child: &mut ChildProcess,
    termination: &mut AttemptExecutionTermination,
    retirement: &mut LocalRetirementEvidence,
    error: String,
) -> Result<WorkerResponse, String> {
    match stop_and_wait(child).await {
        Ok(exit) => {
            *retirement = LocalRetirementEvidence::FullTreeRetired;
            *termination = AttemptExecutionTermination::ProcessExited {
                code: exit.code,
                success: exit.success,
            };
            Err(error)
        }
        Err(retirement_error) => Err(format!(
            "{error}; output setup process-tree retirement unverified: {retirement_error}"
        )),
    }
}

async fn stop_and_wait(child: &mut ChildProcess) -> Result<ProcessExit, String> {
    child
        .terminate_tree()
        .await
        .map_err(|error| error.to_string())?;
    child
        .wait_and_retire_tree()
        .await
        .map_err(|error| error.to_string())
}

async fn wait_or_cancel(
    child: &mut ChildProcess,
    control: &dyn ProcessControl,
) -> Result<ProcessExit, String> {
    let exit = {
        let wait = child.wait_and_retire_tree();
        tokio::pin!(wait);
        loop {
            tokio::select! {
                exit = &mut wait => break Some(exit.map_err(|error| error.to_string())?),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if control.cancellation().load(Ordering::Acquire)
                        || control.raw_output_failure().is_some() { break None; }
                }
            }
        }
    };
    match exit {
        Some(exit) => Ok(exit),
        None => stop_and_wait(child).await,
    }
}
