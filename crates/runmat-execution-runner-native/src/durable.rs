use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use runmat_execution::{JobHandle, JobId};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::{SharedJobHandle, Value};

use crate::supervisor::{
    GraphBatchSubmission, JobSubmissionDisposition, LocalJobWorkloadKind, LocalSupervisorClient,
    ProgramBatchSubmission,
};

mod startup;
mod response;

enum Command {
    Submit {
        program: Box<runmat_execution_artifact::ProgramExecutionRequest>,
        options: runmat_runtime::execution::DurableJobOptions,
        response: mpsc::Sender<Result<SubmittedJob, ExecutionServiceError>>,
    },
    SubmitGraph {
        program: Box<runmat_execution_artifact::GraphExecutionRequest>,
        options: runmat_runtime::execution::DurableJobOptions,
        response: mpsc::Sender<Result<SubmittedJob, ExecutionServiceError>>,
    },
    /// One non-blocking inspection. No bridge command waits for task lifetime,
    /// so a long-running job never stalls submissions or cancellation.
    Inspect {
        handle: SharedJobHandle,
        response: mpsc::Sender<Result<Option<ProgramExecutionResponse>, ExecutionServiceError>>,
    },
    InspectKind {
        handle: SharedJobHandle,
        response: mpsc::Sender<Result<LocalJobWorkloadKind, ExecutionServiceError>>,
    },
    RawOutputAttempt {
        handle: SharedJobHandle,
        ordinal: u32,
        response:
            mpsc::Sender<Result<crate::supervisor::RawProcessOutputLookup, ExecutionServiceError>>,
    },
    RawOutputPage {
        handle: SharedJobHandle,
        attempt_id: runmat_execution::identity::AttemptId,
        cursor: crate::supervisor::RawProcessOutputCursor,
        response: mpsc::Sender<
            Result<Option<crate::supervisor::RawProcessOutputPage>, ExecutionServiceError>,
        >,
    },
    Cancel {
        handle: SharedJobHandle,
        reason: runmat_execution::CancellationReason,
        response: mpsc::Sender<Result<(), ExecutionServiceError>>,
    },
    ExpiredJobReceipts {
        candidates: Vec<JobId>,
        response: mpsc::Sender<Result<Vec<JobId>, ExecutionServiceError>>,
    },
}

pub(crate) struct DurableJobBridge {
    commands: mpsc::Sender<Command>,
}

pub(crate) struct SubmittedJob {
    pub handle: JobHandle,
    pub retain_until_unix_millis: u64,
    pub disposition: JobSubmissionDisposition,
}

pub(crate) const AWAIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

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
    ) -> Result<SubmittedJob, ExecutionServiceError> {
        self.request(|response| Command::Submit {
            program: Box::new(program),
            options,
            response,
        })
    }

    pub(crate) fn submit_graph(
        &self,
        program: runmat_execution_artifact::GraphExecutionRequest,
        options: runmat_runtime::execution::DurableJobOptions,
    ) -> Result<SubmittedJob, ExecutionServiceError> {
        self.request(|response| Command::SubmitGraph {
            program: Box::new(program),
            options,
            response,
        })
    }

    /// The terminal response, or None while the job is still active.
    pub(crate) fn try_response(
        &self,
        handle: &SharedJobHandle,
    ) -> Result<Option<ProgramExecutionResponse>, ExecutionServiceError> {
        self.request(|response| Command::Inspect {
            handle: handle.clone(),
            response,
        })
    }

    pub(crate) fn raw_output_attempt(
        &self,
        handle: SharedJobHandle,
        ordinal: u32,
    ) -> Result<crate::supervisor::RawProcessOutputLookup, ExecutionServiceError> {
        self.request(|response| Command::RawOutputAttempt {
            handle,
            ordinal,
            response,
        })
    }

    pub(crate) fn raw_output_page(
        &self,
        handle: SharedJobHandle,
        attempt_id: runmat_execution::identity::AttemptId,
        cursor: crate::supervisor::RawProcessOutputCursor,
    ) -> Result<Option<crate::supervisor::RawProcessOutputPage>, ExecutionServiceError> {
        self.request(|response| Command::RawOutputPage {
            handle,
            attempt_id,
            cursor,
            response,
        })
    }

    pub(crate) fn workload_kind(
        &self,
        handle: SharedJobHandle,
    ) -> Result<LocalJobWorkloadKind, ExecutionServiceError> {
        self.request(|response| Command::InspectKind { handle, response })
    }

    pub(crate) fn await_job(
        &self,
        handle: SharedJobHandle,
    ) -> Result<Value, ExecutionServiceError> {
        loop {
            if let Some(value) = self.try_await_job(&handle)? {
                return Ok(value);
            }
            thread::sleep(AWAIT_POLL_INTERVAL);
        }
    }

    pub(crate) fn try_await_job(
        &self,
        handle: &SharedJobHandle,
    ) -> Result<Option<Value>, ExecutionServiceError> {
        if matches!(
            self.workload_kind(handle.clone())?,
            LocalJobWorkloadKind::Graph | LocalJobWorkloadKind::Legacy
        ) {
            // A durable graph response requires the submitting session's exact
            // selected receipt, even if a restart retained the job record.
            return Err(ExecutionServiceError::UnknownHandle);
        }
        let requested = usize::from(handle.outputs.requested_outputs);
        self.try_response(handle)?
            .map(|response| decode_outputs(response, requested))
            .transpose()
    }

    pub(crate) fn cancel(
        &self,
        handle: SharedJobHandle,
        reason: runmat_execution::CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        self.request(|response| Command::Cancel {
            handle,
            reason,
            response,
        })
    }

    /// Prune only after the supervisor says the job is terminal and past its
    /// exact stored retention deadline (or has already been GC'd).
    pub(crate) fn expired_job_receipts(
        &self,
        candidates: Vec<JobId>,
    ) -> Result<Vec<JobId>, ExecutionServiceError> {
        self.request(|response| Command::ExpiredJobReceipts {
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
        Err(error) => {
            return startup::reject_until_closed(
                commands,
                ExecutionServiceError::Infrastructure(error.to_string()),
            );
        }
    };
    let client = match LocalSupervisorClient::for_current_executable() {
        Ok(client) => client,
        Err(error) => {
            drop(runtime);
            return startup::reject_until_closed(commands, service_error(error));
        }
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
                    .map(|(record, disposition)| SubmittedJob {
                        handle: record.handle,
                        retain_until_unix_millis: record.retain_until_unix_millis,
                        disposition,
                    })
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::SubmitGraph {
                program,
                options,
                response,
            } => {
                let result = runtime
                    .block_on(submit_original_graph(&client, *program, options))
                    .map(|(record, disposition)| SubmittedJob {
                        handle: record.handle,
                        retain_until_unix_millis: record.retain_until_unix_millis,
                        disposition,
                    })
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::Inspect { handle, response } => {
                let result = runtime.block_on(inspect_job(&client, handle));
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
                        if &record.handle != handle.as_ref() {
                            Err(ExecutionServiceError::UnknownHandle)
                        } else {
                            Ok(record.workload_kind)
                        }
                    });
                let _ = response.send(result);
            }
            Command::RawOutputAttempt {
                handle,
                ordinal,
                response,
            } => {
                let result = runtime
                    .block_on(client.raw_output_attempt(handle.into_wire_unbounded(), ordinal))
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::RawOutputPage {
                handle,
                attempt_id,
                cursor,
                response,
            } => {
                let result = runtime
                    .block_on(client.raw_output_page(
                        handle.into_wire_unbounded(),
                        attempt_id,
                        cursor,
                    ))
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::Cancel {
                handle,
                reason,
                response,
            } => {
                // The supervisor transport owns its wire command. This is an
                // explicit serialization boundary, not a Value retain or a
                // polling clone; transport admission remains independent.
                let result = runtime
                    .block_on(client.cancel_handle(handle.into_wire_unbounded(), reason))
                    .map(|_| ())
                    .map_err(service_error);
                let _ = response.send(result);
            }
            Command::ExpiredJobReceipts {
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

async fn inspect_job(
    client: &LocalSupervisorClient,
    handle: SharedJobHandle,
) -> Result<Option<ProgramExecutionResponse>, ExecutionServiceError> {
    let attachment = client
        .attach(handle.id, 0, 0)
        .await
        .map_err(service_error)?;
    if &attachment.record.handle != handle.as_ref() {
        return Err(ExecutionServiceError::UnknownHandle);
    }
    response::terminal_response(
        attachment.record.state,
        attachment.record.message,
        attachment.response,
    )
}

pub(crate) fn decode_outputs(
    response: ProgramExecutionResponse,
    requested: usize,
) -> Result<Value, ExecutionServiceError> {
    let outputs = match response {
        ProgramExecutionResponse::Success { outputs } => outputs,
        ProgramExecutionResponse::GraphSuccess { .. } => {
            // Graph values require their local selected result receipt; a
            // restarted or expired service cannot reinterpret them as Direct.
            return Err(ExecutionServiceError::UnknownHandle);
        }
        ProgramExecutionResponse::ExternalizedSuccess { .. } => {
            return Err(ExecutionServiceError::Failed(
                "durable runtime value requires an artifact consumer for externalized objects"
                    .into(),
            ))
        }
        ProgramExecutionResponse::SpmdSuccess { .. } => {
            return Err(ExecutionServiceError::Failed(
                "durable runtime value cannot consume an SPMD gang response".into(),
            ))
        }
        ProgramExecutionResponse::Failure { message } => {
            return Err(ExecutionServiceError::Failed(message))
        }
        ProgramExecutionResponse::RuntimeFailure { failure } => {
            return Err(ExecutionServiceError::RuntimeFailure(Box::new(failure)))
        }
    };
    runmat_runtime::execution::value_codec::decode_inline_outputs(&outputs, requested)
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
}

fn service_error(error: crate::NativeExecutionError) -> ExecutionServiceError {
    if matches!(&error, crate::NativeExecutionError::Cancelled) {
        return ExecutionServiceError::Cancelled;
    }
    runmat_runtime::runtime_error::semantic_error_with_source(
        "RunMat:parallel:InfrastructureFailure",
        error.to_string(),
        error,
    )
    .into()
}

/// Admit the original program into the supervisor's scope; the bridge thread
/// owns the async client, so placement identity is resolved here.
async fn submit_original_graph(
    client: &LocalSupervisorClient,
    program: runmat_execution_artifact::GraphExecutionRequest,
    options: runmat_runtime::execution::DurableJobOptions,
) -> crate::NativeExecutionResult<(crate::supervisor::LocalJobRecord, JobSubmissionDisposition)> {
    use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
    let context = client
        .execution_context(options.idempotency_key.as_deref())
        .await?;
    let host = program
        .execution_host_requirement(std::collections::BTreeSet::from([
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        ]))
        .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))?;
    let callable = program
        .scheduler_callable()
        .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))?;
    let invocation = program.invocation_controls.clone().unwrap_or_default();
    // The high-level caller supplies capacity and additional worker constraints.
    // Derive decoder requirements from the materialized program before the
    // supervisor admits this original task; do not change any capacity grant.
    let mut resources = options.resources;
    resources
        .required_capabilities
        .extend(crate::graph_submission::required_capabilities(
            program.schema_version,
        )?);
    let task = runmat_execution_runner::TaskSubmission {
        request: TaskRequest {
            id: runmat_execution::TaskId::derive(&[
                context.scope_id.bytes(),
                program.artifact.id.0.bytes(),
            ]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                crate::graph_submission::primary_artifact(&program),
            ),
            callable: Callable::for_program("runmat.runtime", &callable),
            host,
            invocation_context: program.context.clone(),
            inputs: Vec::new(),
            environment: invocation.environment,
            console: invocation.console,
            outputs: program.recipe.outputs.clone(),
            resources,
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    client
        .submit_graph(GraphBatchSubmission {
            request: program,
            task: Some(task),
            idempotency_key: options.idempotency_key,
            retention_millis: options.retention_millis,
        })
        .await
}

async fn submit_original(
    client: &LocalSupervisorClient,
    program: runmat_execution_artifact::ProgramExecutionRequest,
    options: runmat_runtime::execution::DurableJobOptions,
) -> crate::NativeExecutionResult<(crate::supervisor::LocalJobRecord, JobSubmissionDisposition)> {
    use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
    use runmat_execution::{OutputContract, TaskId};
    let context = client
        .execution_context(options.idempotency_key.as_deref())
        .await?;
    let host = program
        .artifact
        .execution_host_requirement(
            &program.recipe,
            std::collections::BTreeSet::from([
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            ]),
        )
        .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))?;
    let task = runmat_execution_runner::TaskSubmission {
        request: TaskRequest {
            id: TaskId::derive(&[context.scope_id.bytes(), program.artifact.id.0.bytes()]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                runmat_execution::identity::ArtifactId::derive(&[program.artifact.id.0.bytes()]),
            ),
            callable: Callable::for_program("runmat.runtime", &program.callable),
            host,
            invocation_context: program.context.clone(),
            inputs: program.arguments.clone(),
            environment: program.environment.clone(),
            console: program.console,
            outputs: OutputContract {
                requested_outputs: program.requested_outputs,
                owned_work: None,
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
