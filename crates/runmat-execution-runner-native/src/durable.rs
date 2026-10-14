use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use runmat_execution::{JobHandle, JobId};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::Value;

use crate::supervisor::{
    GraphBatchSubmission, LocalJobState, LocalJobWorkloadKind, LocalSupervisorClient,
    ProgramBatchSubmission,
};

enum Command {
    Submit {
        submission: Box<ProgramBatchSubmission>,
        response: mpsc::Sender<Result<JobHandle, ExecutionServiceError>>,
    },
    SubmitGraph {
        submission: Box<GraphBatchSubmission>,
        response: mpsc::Sender<Result<SubmittedGraphJob, ExecutionServiceError>>,
    },
    Await {
        handle: JobHandle,
        response: mpsc::Sender<Result<ProgramExecutionResponse, ExecutionServiceError>>,
    },
    InspectKind {
        handle: JobHandle,
        response: mpsc::Sender<Result<LocalJobWorkloadKind, ExecutionServiceError>>,
    },
    Cancel {
        handle: JobHandle,
        response: mpsc::Sender<Result<(), ExecutionServiceError>>,
    },
    ExpiredGraphReceipts {
        candidates: Vec<JobId>,
        response: mpsc::Sender<Result<Vec<JobId>, ExecutionServiceError>>,
    },
}

pub(crate) struct DurableJobBridge {
    commands: mpsc::Sender<Command>,
}

pub(crate) struct SubmittedGraphJob {
    pub handle: JobHandle,
    pub retain_until_unix_millis: u64,
    pub created_new: bool,
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

    pub(crate) fn submit_graph(
        &self,
        submission: GraphBatchSubmission,
    ) -> Result<SubmittedGraphJob, ExecutionServiceError> {
        self.request(|response| Command::SubmitGraph {
            submission: Box::new(submission),
            response,
        })
    }

    pub(crate) fn await_response(
        &self,
        handle: JobHandle,
    ) -> Result<ProgramExecutionResponse, ExecutionServiceError> {
        self.request(|response| Command::Await { handle, response })
    }

    pub(crate) fn workload_kind(
        &self,
        handle: JobHandle,
    ) -> Result<LocalJobWorkloadKind, ExecutionServiceError> {
        self.request(|response| Command::InspectKind { handle, response })
    }

    pub(crate) fn await_job(&self, handle: JobHandle) -> Result<Value, ExecutionServiceError> {
        if matches!(
            self.workload_kind(handle.clone())?,
            LocalJobWorkloadKind::Graph | LocalJobWorkloadKind::Legacy
        ) {
            // A durable V8 response requires the submitting session's exact
            // selected receipt, even if a restart retained the job record.
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let response = self.await_response(handle)?;
        if matches!(&response, ProgramExecutionResponse::GraphSuccess { .. }) {
            // Graph values require their local selected result receipt; a
            // restarted or expired service cannot reinterpret them as V7.
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let ProgramExecutionResponse::Success { outputs } = response else {
            return Err(ExecutionServiceError::Failed(
                "durable runtime value cannot consume an externalized program response".into(),
            ));
        };
        let [payload] = outputs.as_slice() else {
            return Err(ExecutionServiceError::Failed(
                "durable runtime value requires exactly one program output".into(),
            ));
        };
        runmat_runtime::execution::value_codec::decode_inline_value(payload)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
    }

    pub(crate) fn cancel(&self, handle: JobHandle) -> Result<(), ExecutionServiceError> {
        self.request(|response| Command::Cancel { handle, response })
    }

    /// Prune only after the supervisor says the job is terminal and past its
    /// exact stored retention deadline (or has already been GC'd).
    pub(crate) fn expired_graph_receipts(
        &self,
        candidates: Vec<JobId>,
    ) -> Result<Vec<JobId>, ExecutionServiceError> {
        self.request(|response| Command::ExpiredGraphReceipts {
            candidates,
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
                submission,
                response,
            } => {
                let result = runtime
                    .block_on(client.submit_program(*submission))
                    .map(|(record, _)| record.handle)
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::SubmitGraph {
                submission,
                response,
            } => {
                let result = runtime
                    .block_on(client.submit_graph(*submission))
                    .map(|(record, created_new)| SubmittedGraphJob {
                        handle: record.handle,
                        retain_until_unix_millis: record.retain_until_unix_millis,
                        created_new,
                    })
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::Await { handle, response } => {
                let result = runtime.block_on(await_job(&client, handle));
                let _ = response.send(result);
            }
            Command::InspectKind { handle, response } => {
                let result = runtime
                    .block_on(client.show(handle.id))
                    .map_err(|error| match error {
                        crate::NativeExecutionError::Protocol(_) => {
                            ExecutionServiceError::UnknownHandle
                        }
                        other => service_error(other),
                    })
                    .and_then(|record| {
                        if record.handle != handle {
                            Err(ExecutionServiceError::UnknownHandle)
                        } else {
                            Ok(record.workload_kind)
                        }
                    });
                let _ = response.send(result);
            }
            Command::Cancel { handle, response } => {
                let result = runtime
                    .block_on(client.cancel(handle.id))
                    .map(|_| ())
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::ExpiredGraphReceipts {
                candidates,
                response,
            } => {
                let result =
                    runtime
                        .block_on(client.list())
                        .map_err(service_error)
                        .map(|records| {
                            let now = crate::supervisor::unix_millis();
                            candidates
                                .into_iter()
                                .filter(|candidate| {
                                    records
                                        .iter()
                                        .find(|record| record.handle.id == *candidate)
                                        .is_none_or(|record| {
                                            record.state.is_terminal()
                                                && record.retain_until_unix_millis <= now
                                        })
                                })
                                .collect()
                        });
                let _ = response.send(result);
            }
        }
    }
}

async fn await_job(
    client: &LocalSupervisorClient,
    handle: JobHandle,
) -> Result<ProgramExecutionResponse, ExecutionServiceError> {
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
                let response = attachment.response.ok_or_else(|| {
                    ExecutionServiceError::Failed(
                        "durable program completed without a result".into(),
                    )
                })?;
                return Ok(response);
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
