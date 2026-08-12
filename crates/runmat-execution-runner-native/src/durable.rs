use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use runmat_execution::value::ValuePayload;
use runmat_execution::JobHandle;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::Value;

use crate::supervisor::{LocalJobState, LocalSupervisorClient, ProgramBatchSubmission};

enum Command {
    Submit {
        submission: Box<ProgramBatchSubmission>,
        response: mpsc::Sender<Result<JobHandle, ExecutionServiceError>>,
    },
    Await {
        handle: JobHandle,
        response: mpsc::Sender<Result<ValuePayload, ExecutionServiceError>>,
    },
    Cancel {
        handle: JobHandle,
        response: mpsc::Sender<Result<(), ExecutionServiceError>>,
    },
}

pub(crate) struct DurableJobBridge {
    commands: mpsc::Sender<Command>,
}

impl DurableJobBridge {
    pub(crate) fn start() -> Result<Self, ExecutionServiceError> {
        let (commands, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("runmat-durable-job-client".into())
            .spawn(move || bridge_main(receiver))
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        Ok(Self { commands })
    }

    pub(crate) fn submit(
        &self,
        submission: ProgramBatchSubmission,
    ) -> Result<JobHandle, ExecutionServiceError> {
        self.request(|response| Command::Submit {
            submission: Box::new(submission),
            response,
        })
    }

    pub(crate) fn await_job(&self, handle: JobHandle) -> Result<Value, ExecutionServiceError> {
        let payload = self.request(|response| Command::Await { handle, response })?;
        runmat_runtime::execution::value_codec::decode_inline_value(&payload)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
    }

    pub(crate) fn cancel(&self, handle: JobHandle) -> Result<(), ExecutionServiceError> {
        self.request(|response| Command::Cancel { handle, response })
    }

    fn request<T>(
        &self,
        command: impl FnOnce(mpsc::Sender<Result<T, ExecutionServiceError>>) -> Command,
    ) -> Result<T, ExecutionServiceError> {
        let (response, receiver) = mpsc::channel();
        self.commands
            .send(command(response))
            .map_err(|_| ExecutionServiceError::Failed("durable job bridge stopped".into()))?;
        receiver
            .recv()
            .map_err(|_| ExecutionServiceError::Failed("durable job bridge stopped".into()))?
    }
}

fn bridge_main(commands: mpsc::Receiver<Command>) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => return,
    };
    let client = match LocalSupervisorClient::for_current_executable() {
        Ok(client) => client,
        Err(_) => return,
    };
    while let Ok(command) = commands.recv() {
        match command {
            Command::Submit {
                submission,
                response,
            } => {
                let result = runtime
                    .block_on(client.submit_program(*submission))
                    .map(|(record, _)| record.handle)
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::Await { handle, response } => {
                let result = runtime.block_on(await_job(&client, handle));
                let _ = response.send(result);
            }
            Command::Cancel { handle, response } => {
                let result = runtime
                    .block_on(client.cancel(handle.id))
                    .map(|_| ())
                    .map_err(service_error);
                let _ = response.send(result);
            }
        }
    }
}

async fn await_job(
    client: &LocalSupervisorClient,
    handle: JobHandle,
) -> Result<ValuePayload, ExecutionServiceError> {
    loop {
        let attachment = client
            .attach(handle.id, 0, 0)
            .await
            .map_err(service_error)?;
        if attachment.record.handle != handle {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        match attachment.record.state {
            LocalJobState::Succeeded => {
                let payload = attachment.value.ok_or_else(|| {
                    ExecutionServiceError::Failed(
                        "durable program completed without a result".into(),
                    )
                })?;
                return Ok(payload);
            }
            LocalJobState::Failed | LocalJobState::Indeterminate => {
                return Err(ExecutionServiceError::Failed(
                    attachment
                        .record
                        .message
                        .unwrap_or_else(|| "durable program failed".into()),
                ))
            }
            LocalJobState::Cancelled => return Err(ExecutionServiceError::Cancelled),
            LocalJobState::Queued
            | LocalJobState::Starting
            | LocalJobState::Running
            | LocalJobState::Cancelling => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

fn service_error(error: crate::NativeExecutionError) -> ExecutionServiceError {
    ExecutionServiceError::Failed(error.to_string())
}
