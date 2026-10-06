use super::{LocalDriver, TaskCompletion, TransferResult, NATIVE_OBJECT_STORE_ROOT_ENV};
use crate::protocol::{
    StoredProgram, WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, AttemptRequest,
    AttemptResourceAccounting, AttemptSuccess,
};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::IpcByteCounter;
use runmat_process_host::{ChildProcess, HostCommand, ProcessExit};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
mod exchange;

#[derive(Default)]
struct TransportMeters {
    sent: IpcByteCounter,
    received: IpcByteCounter,
}

pub(super) struct ObservedAttempt {
    pub result: TransferResult,
    pub execution: AttemptExecutionObservation,
    pub unsuccessful_resources: Option<Box<AttemptResourceAccounting>>,
}

pub(super) fn execute_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
) -> ObservedAttempt {
    let started = Instant::now();
    let meters = TransportMeters::default();
    let mut termination = AttemptExecutionTermination::NotStarted;
    let response = prepare_and_run(driver, request, completion, &meters, &mut termination);
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
        execution,
        unsuccessful_resources,
    }
}

fn prepare_and_run(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
    meters: &TransportMeters,
    termination: &mut AttemptExecutionTermination,
) -> Result<WorkerResponse, String> {
    let stored = driver
        .artifacts
        .get(request.task.program_artifact_id)
        .map_err(|error| error.to_string())?;
    let stored: StoredProgram =
        serde_json::from_slice(&stored).map_err(|error| error.to_string())?;
    let function = match stored.artifact.form {
        runmat_execution_artifact::ExecutableForm::InterpreterScriptV1
        | runmat_execution_artifact::ExecutableForm::TestAttemptV1
        | runmat_execution_artifact::ExecutableForm::MeshingWorkload
        | runmat_execution_artifact::ExecutableForm::AnalysisWorkload
        | runmat_execution_artifact::ExecutableForm::GeometryWorkload => 0,
        _ => stored.function,
    };
    let worker_request = WorkerRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: stored.recipe,
        artifact: stored.artifact,
        function,
        arguments: request.task.inputs.clone(),
        requested_outputs: request.task.outputs.requested_outputs,
    };
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?
        .block_on(run_process(
            driver,
            worker_request,
            completion,
            meters,
            termination,
        ))
}

async fn run_process(
    driver: &LocalDriver,
    request: WorkerRequest,
    completion: &TaskCompletion,
    meters: &TransportMeters,
    termination: &mut AttemptExecutionTermination,
) -> Result<WorkerResponse, String> {
    let mut command = HostCommand::new(&driver.config.executable);
    command.arguments = driver.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    command.environment.insert(
        NATIVE_OBJECT_STORE_ROOT_ENV.into(),
        driver.objects.root().to_string_lossy().into_owned(),
    );
    if !driver.config.worker_accelerators.is_empty() {
        command.environment.insert(
            crate::driver::NATIVE_ACCELERATOR_REQUIRED_ENV.into(),
            "1".into(),
        );
        command.environment.insert(
            crate::driver::NATIVE_ACCELERATOR_REQUESTS_ENV.into(),
            serde_json::to_string(&driver.config.worker_accelerators)
                .map_err(|error| error.to_string())?,
        );
    }
    command.max_stderr_bytes = driver.config.max_stderr_bytes;
    // Spawn can fail after creating a child (for example during containment).
    // Until root exit is observed, even an error leaves launch unconfirmed.
    *termination = AttemptExecutionTermination::Unconfirmed;
    let mut child = command.spawn().await.map_err(|error| error.to_string())?;
    let stderr = child.captured_stderr();
    let response = exchange::exchange(driver, &mut child, &request, completion, meters).await;
    let exit = if response.is_err() {
        stop_and_wait(&mut child).await
    } else {
        wait_or_cancel(&mut child, completion).await
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
    if completion.cancelled.load(Ordering::Acquire) {
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
    completion: &TaskCompletion,
) -> Result<ProcessExit, String> {
    let exit = {
        let wait = child.wait();
        tokio::pin!(wait);
        loop {
            tokio::select! {
                exit = &mut wait => break Some(exit.map_err(|error| error.to_string())?),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if completion.cancelled.load(Ordering::Acquire) { break None; }
                }
            }
        }
    };
    match exit {
        Some(exit) => Ok(exit),
        None => stop_and_wait(child).await,
    }
}

#[cfg(test)]
mod tests;
