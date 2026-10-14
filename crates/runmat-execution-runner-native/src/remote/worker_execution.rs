use std::sync::{Arc, Mutex};

use runmat_execution::identity::AttemptId;
use runmat_execution::resource::ResourceInventory;
use runmat_execution_artifact::{ExecutableForm, ProgramExecutionResponse, WorkerProgramRequest};
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptSuccess};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingProgress};
use runmat_meshing_execution::{MeshingProgressSink, MeshingStageKernel};

use super::object_transfer::RemoteObjectStore;

#[derive(Default)]
pub(super) struct AttemptCancellation(runmat_execution::cancellation::CancellationSource);

impl AttemptCancellation {
    pub(super) fn cancel(&self) {
        self.0.cancel(runmat_execution::CancellationReason::User);
    }

    pub(super) fn token(&self) -> runmat_execution::cancellation::CancellationToken {
        self.0.token()
    }
}

impl MeshingCancellationSignal for AttemptCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.token().is_cancelled()
    }
}

pub(super) struct GraphAttemptContext {
    pub(super) run_identity: String,
    pub(super) attempt_id: AttemptId,
    pub(super) resources: ResourceInventory,
    pub(super) task_resources: runmat_execution::resource::ResourceRequest,
    pub(super) host: runmat_execution::host::ExecutionHostInventory,
}

/// The admitted program and its exact materialized/assigned authority move
/// together into one worker execution.
pub(super) struct WorkerExecutionInput {
    pub program: WorkerProgramRequest,
    pub graph_attempt: Option<GraphAttemptContext>,
    pub project: Option<Arc<crate::materialized_project::MaterializedProject>>,
    pub bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
}

/// Request-local effect ports and cancellation, separate from program proof.
pub(super) struct WorkerExecutionPorts {
    pub collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    pub meshing_host: Option<RemoteMeshingHost>,
    pub objects: RemoteObjectStore,
    pub cancellation: Arc<AttemptCancellation>,
    pub progress_sender: tokio::sync::mpsc::Sender<crate::ProgramProgress>,
}

pub(super) enum WorkerExecutionOutcome {
    Response(ProgramExecutionResponse),
    RetirementUnverified(String),
}

#[derive(Clone)]
pub(super) struct RemoteMeshingHost {
    kernel: Arc<dyn MeshingStageKernel>,
    limits: crate::NativeMeshingHostLimits,
}

impl RemoteMeshingHost {
    pub(super) fn new(
        kernel: Arc<dyn MeshingStageKernel>,
        limits: crate::NativeMeshingHostLimits,
    ) -> Self {
        Self { kernel, limits }
    }
}

pub(super) async fn execute(
    input: WorkerExecutionInput,
    ports: WorkerExecutionPorts,
) -> WorkerExecutionOutcome {
    let WorkerExecutionInput {
        program,
        graph_attempt,
        project,
        bundle,
    } = input;
    let WorkerExecutionPorts {
        collective,
        meshing_host,
        mut objects,
        cancellation,
        progress_sender,
    } = ports;
    let program = match program {
        WorkerProgramRequest::Direct(program) => program,
        WorkerProgramRequest::Graph(graph) => {
            drop(progress_sender);
            let Some(graph_attempt) = graph_attempt else {
                return WorkerExecutionOutcome::RetirementUnverified(
                    "remote Graph child lacks authenticated attempt context".into(),
                );
            };
            return match super::driver::local_graph::execute_remote_attempt(
                &graph_attempt.run_identity,
                graph_attempt.attempt_id,
                graph,
                graph_attempt.resources,
                graph_attempt.task_resources,
                graph_attempt.host,
                cancellation.token(),
            )
            .await
            {
                Ok(super::pool_execution::RemotePoolExecutionOutcome::Completed(response)) => {
                    WorkerExecutionOutcome::Response(response)
                }
                Ok(super::pool_execution::RemotePoolExecutionOutcome::Cancelled) => {
                    WorkerExecutionOutcome::Response(failure("remote Graph attempt was cancelled"))
                }
                Ok(super::pool_execution::RemotePoolExecutionOutcome::Indeterminate(message)) => {
                    WorkerExecutionOutcome::RetirementUnverified(message)
                }
                Err(error) => WorkerExecutionOutcome::RetirementUnverified(error.to_string()),
            };
        }
    };
    if program.artifact.form == ExecutableForm::MeshingWorkload {
        let Some(host) = meshing_host else {
            return WorkerExecutionOutcome::Response(failure(
                "remote worker has no meshing host capability",
            ));
        };
        return WorkerExecutionOutcome::Response(
            tokio::task::spawn_blocking(move || {
                let progress_error = Arc::new(Mutex::new(None));
                let mut progress = ChannelProgress {
                    sender: progress_sender,
                    error: Arc::clone(&progress_error),
                };
                let response = crate::execute_meshing_program_request(
                    &program,
                    &mut objects,
                    host.kernel.as_ref(),
                    cancellation.as_ref(),
                    &mut progress,
                    host.limits,
                );
                let progress_error = progress_error
                    .lock()
                    .expect("remote progress error poisoned")
                    .take();
                match progress_error {
                    Some(error) => failure(&error),
                    None => response,
                }
            })
            .await
            .unwrap_or_else(|error| failure(&error.to_string())),
        );
    }
    drop(progress_sender);
    WorkerExecutionOutcome::Response(match collective {
        Some(collective) => {
            crate::test_workload::execute_host_program_request_with_validated_bundle(
                program,
                project.as_deref(),
                Some(collective),
                bundle,
            )
            .await
        }
        None => {
            crate::test_workload::execute_host_program_request_with_validated_bundle(
                program,
                project.as_deref(),
                None,
                bundle,
            )
            .await
        }
    })
}

pub(super) fn report(response: ProgramExecutionResponse) -> AttemptReport {
    match response {
        ProgramExecutionResponse::Success { outputs } => AttemptReport::Succeeded {
            result: AttemptSuccess::Values {
                outputs,
                result_objects: Vec::new(),
            },
        },
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => AttemptReport::Succeeded {
            result: AttemptSuccess::Values {
                outputs,
                result_objects,
            },
        },
        ProgramExecutionResponse::GraphSuccess { graph } => AttemptReport::Succeeded {
            result: AttemptSuccess::Graph { graph },
        },
        ProgramExecutionResponse::SpmdSuccess { outputs } => AttemptReport::Succeeded {
            result: AttemptSuccess::Spmd { outputs },
        },
        ProgramExecutionResponse::Failure { message } => AttemptReport::Failed {
            kind: AttemptFailureKind::Execution,
            message,
        },
        ProgramExecutionResponse::RuntimeFailure { failure } => {
            AttemptReport::RuntimeFailed { failure }
        }
    }
}

struct ChannelProgress {
    sender: tokio::sync::mpsc::Sender<crate::ProgramProgress>,
    error: Arc<Mutex<Option<String>>>,
}

impl MeshingProgressSink for ChannelProgress {
    fn record(&mut self, progress: &MeshingProgress) {
        let result = crate::meshing_host::encode_meshing_progress(progress).and_then(|progress| {
            self.sender
                .blocking_send(progress)
                .map_err(|_| "remote progress receiver closed".to_string())
        });
        if let Err(error) = result {
            *self.error.lock().expect("remote progress error poisoned") = Some(error);
        }
    }
}

fn failure(message: &str) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: message.into(),
    }
}
