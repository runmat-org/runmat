//! Bootstrap the owned worker's control and application streams separately.
use std::time::Duration;

use runmat_process_host::ipc::private_control::PrivateControlListener;
use runmat_process_host::{CapturedProcessOutput, HostCommand, StdioPolicy};

use super::*;

pub(super) struct WorkerControl {
    pub child: ChildProcess,
    pub reader: BufReader<ControlReader>,
    pub writer: ControlWriter,
    pub stdout: CapturedProcessOutput,
    pub stderr: CapturedProcessOutput,
}

pub(super) async fn spawn(config: &ProcessBackendConfig) -> Result<WorkerControl, BackendError> {
    let listener = PrivateControlListener::bind()
        .map_err(|error| transport_error(ProcessHostError::Io(error)))?;
    let mut command = HostCommand::new(&config.executable);
    command.arguments = config.worker_arguments.clone();
    command.environment = config.environment.clone();
    command.environment_policy = config.environment_policy.clone();
    command.working_directory = config
        .project_handoff
        .as_ref()
        .map(|handoff| handoff.project.workspace_root.clone());
    command.max_stderr_bytes = config
        .max_stderr_bytes
        .min(runmat_test::event::MAX_WORKER_PROCESS_STREAM_BYTES);
    command.stdio = StdioPolicy::PrivateControl(listener.endpoint().clone());
    let mut child = command.spawn().await.map_err(spawn_error)?;
    let application = child.take_application_output().map_err(transport_error)?;
    let stdout = CapturedProcessOutput::new(runmat_test::event::MAX_WORKER_PROCESS_STREAM_BYTES);
    stdout.drain(application.stdout).map_err(transport_error)?;
    let stderr = child.captured_stderr();
    application.stderr_diagnostics.drain(application.stderr);
    let connected = tokio::time::timeout(Duration::from_secs(30), listener.accept()).await;
    let connection = match connected {
        Ok(Ok(connection)) => connection,
        outcome => {
            let error = match outcome {
                Ok(Err(error)) => BackendError::new(BackendErrorKind::Transport, error.to_string()),
                Err(_) => BackendError::new(
                    BackendErrorKind::Transport,
                    "native worker control bootstrap exceeded its deadline",
                ),
                Ok(Ok(_)) => unreachable!("successful control connection already selected"),
            };
            let mut error = failed_spawn(&mut child, &stderr, error).await;
            let text = stdout.text();
            if !text.is_empty() {
                error.message.push_str("\nworker process stdout:\n");
                error.message.push_str(&text);
            }
            return Err(error);
        }
    };
    Ok(WorkerControl {
        child,
        reader: BufReader::new(connection.reader),
        writer: connection.writer,
        stdout,
        stderr,
    })
}
