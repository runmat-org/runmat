use std::sync::atomic::Ordering;
use std::time::Duration;

use runmat_execution_runner::{AttemptRequest, AttemptSuccess};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::HostCommand;
use tokio::io::BufReader;

use super::{LocalDriver, TaskCompletion, TransferResult, NATIVE_OBJECT_STORE_ROOT_ENV};
use crate::protocol::{
    StoredProgram, WorkerProcessMessage, WorkerRequest, WorkerResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};

pub(super) fn execute_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    let stored = driver
        .artifacts
        .get(request.task.program_artifact_id)
        .map_err(|error| error.to_string())?;
    let stored: StoredProgram =
        serde_json::from_slice(&stored).map_err(|error| error.to_string())?;
    let function = match stored.artifact.form {
        runmat_execution_artifact::ExecutableForm::InterpreterScriptV1
        | runmat_execution_artifact::ExecutableForm::TestAttemptV1
        | runmat_execution_artifact::ExecutableForm::MeshingWorkload => 0,
        _ => request
            .task
            .callable
            .qualified_name
            .parse::<usize>()
            .map_err(|error| format!("invalid callable identity: {error}"))?,
    };
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
    command.environment.insert(
        NATIVE_OBJECT_STORE_ROOT_ENV.into(),
        driver.objects.root().to_string_lossy().into_owned(),
    );
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
    let mut last_progress_sequence = 0;
    let response = loop {
        let payload = tokio::select! {
            response = read_payload(&mut reader, limits) => {
                response.map_err(|error| {
                    let stderr = stderr.text();
                    if stderr.is_empty() { error.to_string() } else { format!("{error}; worker stderr: {stderr}") }
                })?
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if completion.cancelled.load(Ordering::Acquire) {
                    let _ = child.terminate_tree().await;
                    return Err("execution was cancelled".into());
                }
                continue;
            }
        };
        if let Ok(message) = serde_json::from_slice::<WorkerProcessMessage>(&payload) {
            match message {
                WorkerProcessMessage::Progress { progress } => {
                    progress.validate()?;
                    if progress.sequence <= last_progress_sequence {
                        return Err("native worker progress is not strictly monotone".into());
                    }
                    last_progress_sequence = progress.sequence;
                    completion.record_progress(progress);
                }
                WorkerProcessMessage::Completed { response } => break response,
            }
        } else {
            break serde_json::from_slice::<WorkerResponse>(&payload)
                .map_err(|error| error.to_string())?;
        }
    };
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
