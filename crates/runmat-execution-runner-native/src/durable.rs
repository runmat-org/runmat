use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use runmat_execution::JobHandle;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::Value;

use crate::supervisor::{LocalJobState, LocalSupervisorClient, ProgramBatchSubmission};

enum Command {
    Submit {
        program: Box<runmat_execution_artifact::ProgramExecutionRequest>,
        options: runmat_runtime::execution::DurableJobOptions,
        response: mpsc::Sender<Result<JobHandle, ExecutionServiceError>>,
    },
    Inspect {
        handle: JobHandle,
        response: mpsc::Sender<Result<Option<ProgramExecutionResponse>, ExecutionServiceError>>,
    },
    Cancel {
        handle: JobHandle,
        reason: runmat_execution::CancellationReason,
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
        program: runmat_execution_artifact::ProgramExecutionRequest,
        options: runmat_runtime::execution::DurableJobOptions,
    ) -> Result<JobHandle, ExecutionServiceError> {
        self.request(|response| Command::Submit {
            program: Box::new(program),
            options,
            response,
        })
    }

    pub(crate) fn await_job(&self, handle: JobHandle) -> Result<Value, ExecutionServiceError> {
        loop {
            if let Some(value) = self.try_await_job(&handle)? {
                return Ok(value);
            }
            // Blocking convenience API; the runtime await path uses one inspection
            // and yields Pending instead. No bridge command waits for task lifetime.
            thread::sleep(Duration::from_millis(100));
        }
    }

    pub(crate) fn try_await_job(
        &self,
        handle: &JobHandle,
    ) -> Result<Option<Value>, ExecutionServiceError> {
        let requested = usize::from(handle.outputs.requested_outputs);
        let response = self.request(|response| Command::Inspect {
            handle: handle.clone(),
            response,
        })?;
        response
            .map(|response| decode_outputs(response, requested))
            .transpose()
    }

    pub(crate) fn cancel(
        &self,
        handle: JobHandle,
        reason: runmat_execution::CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        self.request(|response| Command::Cancel {
            handle,
            reason,
            response,
        })
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
                program,
                options,
                response,
            } => {
                let result = runtime
                    .block_on(submit_original(&client, *program, options))
                    .map(|(record, _)| record.handle)
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::Inspect { handle, response } => {
                let result = runtime.block_on(inspect_job(&client, handle));
                let _ = response.send(result);
            }
            Command::Cancel {
                handle,
                reason,
                response,
            } => {
                let result = runtime
                    .block_on(client.cancel_handle(handle, reason))
                    .map(|_| ())
                    .map_err(service_error);
                let _ = response.send(result);
            }
        }
    }
}

async fn inspect_job(
    client: &LocalSupervisorClient,
    handle: JobHandle,
) -> Result<Option<ProgramExecutionResponse>, ExecutionServiceError> {
    let attachment = client
        .attach(handle.id, 0, 0)
        .await
        .map_err(service_error)?;
    if attachment.record.handle != handle {
        return Err(ExecutionServiceError::UnknownHandle);
    }
    match attachment.record.state {
        LocalJobState::Succeeded => attachment.response.map(Some).ok_or_else(|| {
            ExecutionServiceError::Failed("durable program completed without a result".into())
        }),
        LocalJobState::Failed | LocalJobState::Indeterminate => Err(ExecutionServiceError::Failed(
            attachment
                .record
                .message
                .unwrap_or_else(|| "durable program failed".into()),
        )),
        LocalJobState::Cancelled => Err(ExecutionServiceError::Cancelled),
        LocalJobState::Queued
        | LocalJobState::Starting
        | LocalJobState::Running
        | LocalJobState::Cancelling => Ok(None),
    }
}

fn decode_outputs(
    response: ProgramExecutionResponse,
    requested: usize,
) -> Result<Value, ExecutionServiceError> {
    let outputs = match response {
        ProgramExecutionResponse::Success { value } => vec![value],
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } if result_objects.is_empty() => outputs,
        ProgramExecutionResponse::ExternalizedSuccess { .. } => {
            return Err(ExecutionServiceError::Failed(
                "durable runtime value requires an artifact consumer for externalized objects"
                    .into(),
            ))
        }
        ProgramExecutionResponse::Failure { message } => {
            return Err(ExecutionServiceError::Failed(message))
        }
    };
    runmat_runtime::execution::value_codec::decode_inline_outputs(&outputs, requested)
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
}

fn service_error(error: crate::NativeExecutionError) -> ExecutionServiceError {
    ExecutionServiceError::Failed(error.to_string())
}

async fn submit_original(
    client: &LocalSupervisorClient,
    program: runmat_execution_artifact::ProgramExecutionRequest,
    options: runmat_runtime::execution::DurableJobOptions,
) -> crate::NativeExecutionResult<(crate::supervisor::LocalJobRecord, bool)> {
    use runmat_execution::{
        task::{Callable, RetryPolicy, TaskRequest},
        Digest, OutputContract, TaskId,
    };
    let context = client
        .execution_context(options.idempotency_key.as_deref())
        .await?;
    let task = runmat_execution_runner::TaskSubmission {
        request: TaskRequest {
            id: TaskId::derive(&[context.scope_id.bytes(), program.artifact.id.0.bytes()]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[program
                .artifact
                .id
                .0
                .bytes()]),
            callable: Callable {
                owner_identity: "runmat.runtime".into(),
                qualified_name: program.function.to_string(),
                entrypoint_digest: Digest::sha256(program.function.to_be_bytes()),
            },
            inputs: program.arguments.clone(),
            outputs: OutputContract {
                requested_outputs: program.requested_outputs,
            },
            resources: options.resources,
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    client
        .submit_program(ProgramBatchSubmission::from_request(
            program,
            task,
            options.idempotency_key,
            options.retention_millis,
        ))
        .await
}
