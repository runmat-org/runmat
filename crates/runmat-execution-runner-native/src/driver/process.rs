use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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
    let started = Instant::now();
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
        | runmat_execution_artifact::ExecutableForm::AnalysisWorkload => 0,
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
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(run_process(
        driver,
        request,
        worker_request,
        completion,
        started,
    ))
}

async fn run_process(
    driver: &LocalDriver,
    attempt: &AttemptRequest,
    request: WorkerRequest,
    completion: &TaskCompletion,
    started: Instant,
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
        let payload =
            match read_payload_until_cancelled(&mut reader, limits, &completion.cancelled).await {
                Ok(Some(payload)) => payload,
                Ok(None) => {
                    let _ = child.terminate_tree().await;
                    return Err("execution was cancelled".into());
                }
                Err(error) => {
                    let stderr = stderr.text();
                    return Err(if stderr.is_empty() {
                        error.to_string()
                    } else {
                        format!("{error}; worker stderr: {stderr}")
                    });
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
    // The worker treats EOF on stdin as parent death. Keep the request pipe open until a terminal
    // response arrives; without this explicit lifetime, non-lexical drop can cancel healthy work
    // immediately after the one request frame is written.
    drop(writer);
    let _ = child.wait().await;
    response
        .validate_against(&request)
        .map_err(|error| error.to_string())?;
    match response {
        WorkerResponse::Success { value } => AttemptSuccess::new(
            attempt,
            vec![value],
            Vec::new(),
            elapsed_millis(started),
            0,
            0,
        ),
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => AttemptSuccess::new(
            attempt,
            outputs,
            result_objects,
            elapsed_millis(started),
            0,
            0,
        ),
        WorkerResponse::Failure { message } => Err(message),
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1)
}

async fn read_payload_until_cancelled(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    limits: FrameLimits,
    cancelled: &AtomicBool,
) -> runmat_process_host::ProcessHostResult<Option<Vec<u8>>> {
    // A framed read may span several pipe reads. Keep the same future alive
    // while polling cancellation: recreating it after a timer tick discards
    // bytes already consumed from the length-prefixed stream.
    let read = read_payload(reader, limits);
    tokio::pin!(read);
    loop {
        tokio::select! {
            response = &mut read => return response.map(Some),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(None);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use tokio::io::AsyncWriteExt;

    use super::*;

    #[tokio::test]
    async fn cancellation_polling_preserves_a_fragmented_frame() {
        let (mut writer, mut reader) = tokio::io::duplex(64);
        let payload = br#"{"message":"fragmented"}"#.to_vec();
        let length = u32::try_from(payload.len()).unwrap().to_be_bytes();
        let frame = [length.as_slice(), payload.as_slice()].concat();
        let sender = tokio::spawn(async move {
            for byte in frame {
                writer.write_all(&[byte]).await.unwrap();
                tokio::time::sleep(Duration::from_millis(12)).await;
            }
        });

        let received = read_payload_until_cancelled(
            &mut reader,
            FrameLimits {
                max_message_bytes: 1024,
            },
            &AtomicBool::new(false),
        )
        .await
        .unwrap()
        .unwrap();

        sender.await.unwrap();
        assert_eq!(received, payload);
    }
}
