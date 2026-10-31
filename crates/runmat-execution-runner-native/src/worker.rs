use runmat_process_host::ipc::FrameLimits;

use crate::protocol::{
    ReservedProgramExecutionRequest, StagedGraphWorkerReference, StoredGraphProgram,
    WorkerDriverMessage, WorkerLaunchRequest, WorkerRequest, WorkerResponse,
};
use crate::{NativeExecutionError, NativeExecutionResult};

mod collective;
mod console;

/// One bounded stdio frame between a native worker and its parent.
const WORKER_MAX_MESSAGE_BYTES: u32 = 64 * 1024 * 1024;

enum WorkerExecutionOutcome {
    Completed {
        // Ordinary stdio transport control; Value backing keeps its own owners.
        response: Box<WorkerResponse>,
        console: Option<runmat_execution::console::ConsoleReceipt>,
    },
    RetirementUnverified {
        message: String,
        owner: Box<crate::test_workload::RetainedGraphHost>,
    },
}

pub async fn run_worker_stdio() -> NativeExecutionResult<()> {
    // The endpoint also selects binary standard streams; its async reader is replaced by the
    // parent link, which owns the request pipe on its own thread.
    let (_unused_reader, writer) = runmat_process_host::ipc::stdio::endpoint()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let limits = FrameLimits {
        max_message_bytes: WORKER_MAX_MESSAGE_BYTES,
    };
    // Parent loss (the request pipe closing before this worker publishes its terminal frame)
    // ends the worker at once, even while a non-yielding solve holds the executor.
    let (parent, frames) =
        runmat_process_host::ipc::WorkerParentLink::start(std::io::stdin(), limits)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let channel = collective::StdioCollectiveChannel::new(frames, writer, limits);
    let payload = channel.read_payload().await?;
    // The parent assigns this runnable child to its JobObject before writing
    // the first request. No RunMat provider or foreign helper is initialized
    // in the spawn-to-containment gap.
    runmat_accelerate::initialize_acceleration_provider();
    let request: WorkerLaunchRequest = serde_json::from_slice(&payload)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let spmd_task = match &request {
        WorkerLaunchRequest::Direct(direct) => match direct.as_ref() {
            WorkerRequest::Direct(request) => match &request.context {
                runmat_execution::ProgramInvocationContext::SpmdTask { task } => Some(task.clone()),
                _ => None,
            },
            WorkerRequest::Graph(_) => None,
        },
        WorkerLaunchRequest::Reserved(reserved) => match &reserved.request().context {
            runmat_execution::ProgramInvocationContext::SpmdTask { task } => Some(task.clone()),
            _ => None,
        },
        WorkerLaunchRequest::StagedGraph(_) => None,
    };
    let response = if let Some(task) = spmd_task {
        let request = match request {
            WorkerLaunchRequest::Direct(direct) => match *direct {
                WorkerRequest::Direct(request) => request,
                WorkerRequest::Graph(_) => unreachable!("SPMD request has a Direct program"),
            },
            WorkerLaunchRequest::Reserved(reserved) => reserved.into_parts().0,
            WorkerLaunchRequest::StagedGraph(_) => unreachable!("SPMD request uses Direct wire"),
        };
        let service = collective::StdioCollectiveService::new(task, channel.clone());
        Ok(WorkerExecutionOutcome::Completed {
            response: Box::new(
                crate::test_workload::execute_host_program_request_with_collective(
                    request,
                    std::rc::Rc::new(service),
                )
                .await,
            ),
            console: None,
        })
    } else {
        execute_launch(request, channel.clone()).await
    };
    let payload = match response {
        Ok(WorkerExecutionOutcome::Completed { response, console }) => {
            serde_json::to_vec(&crate::protocol::WorkerProcessMessage::Completed {
                response: *response,
                console,
            })
        }
        Ok(WorkerExecutionOutcome::RetirementUnverified { message, owner }) => {
            let _owner = owner;
            parent.disarm();
            let payload = serde_json::to_vec(
                &crate::protocol::WorkerProcessMessage::RetirementUnverified { message },
            );
            // Even serialization or wire failure cannot release a host whose
            // lifecycle callback refused retirement. The parent will observe
            // the missing/fatal frame and kill and reap this child.
            if let Ok(payload) = payload {
                let _ = channel.write_payload(&payload).await;
            }
            return std::future::pending().await;
        }
        Err(error) => serde_json::to_vec(
            &crate::protocol::WorkerProcessMessage::InfrastructureFailure {
                message: error.to_string(),
            },
        ),
    }
    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    // The parent may close the request pipe as soon as it reads the terminal frame.
    parent.disarm();
    channel.write_payload(&payload).await?;
    Ok(())
}

async fn execute_launch(
    request: WorkerLaunchRequest,
    channel: collective::StdioCollectiveChannel,
) -> NativeExecutionResult<WorkerExecutionOutcome> {
    match request {
        WorkerLaunchRequest::Direct(direct) => match *direct {
            WorkerRequest::Direct(request) => Ok(WorkerExecutionOutcome::Completed {
                response: Box::new(execute(WorkerRequest::Direct(request)).await),
                console: None,
            }),
            WorkerRequest::Graph(_) => Err(NativeExecutionError::Protocol(
                "Graph stdio requests require a verified staged artifact reference".into(),
            )),
        },
        WorkerLaunchRequest::Reserved(reserved) => {
            let (response, console) = console::execute_reserved(*reserved).await?;
            Ok(WorkerExecutionOutcome::Completed {
                response: Box::new(response),
                console,
            })
        }
        WorkerLaunchRequest::StagedGraph(staged) => {
            execute_staged_graph(staged.staged_graph_v10, channel).await
        }
    }
}

async fn execute_staged_graph(
    reference: StagedGraphWorkerReference,
    channel: collective::StdioCollectiveChannel,
) -> NativeExecutionResult<WorkerExecutionOutcome> {
    let attempt_id = reference.assignment.attempt_id;
    reference
        .resources
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    reference
        .assignment
        .resources
        .validate_for_request(&reference.resources)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let resources = reference.resources.clone();
    let request = open_staged_graph(reference)?;
    let console_host = console::graph_host(
        request
            .invocation_controls
            .as_ref()
            .and_then(|controls| controls.console),
    )?;
    if request.project_runtime_policy.is_some() {
        let policy = runmat_core::cluster_settings::validate_graph_policy(&request)
            .map_err(NativeExecutionError::Protocol)?;
        crate::remote::runtime_settings::admit_config(
            &policy
                .runtime_config()
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
            &resources,
            request.assignment.as_ref(),
        )
        .map_err(NativeExecutionError::Protocol)?;
    }
    let scratch = std::env::var(crate::driver::NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV).map_err(|_| {
        NativeExecutionError::Protocol("selected Graph attempt scratch is absent".into())
    })?;
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let execution = crate::test_workload::execute_host_graph_request_in_parent_scratch(
        request,
        cancellation.token(),
        std::path::Path::new(&scratch),
        console_host.clone(),
    );
    tokio::pin!(execution);
    let mut control_open = true;
    loop {
        tokio::select! {
            response = &mut execution => return Ok(match response {
                crate::test_workload::GraphHostExecution::Completed(response) => {
                    let transcript = console_host.as_ref()
                        .map(|host| host.finish(!cancellation.token().is_cancelled()))
                        .transpose()
                        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
                    let console = transcript.as_ref().map(|transcript| transcript.receipt())
                        .transpose()
                        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
                    WorkerExecutionOutcome::Completed { response: Box::new(response), console }
                }
                crate::test_workload::GraphHostExecution::RetirementUnverified { message, owner } => {
                    WorkerExecutionOutcome::RetirementUnverified { message, owner }
                }
            }),
            control = channel.read_payload(), if control_open => {
                match control {
                    Ok(payload) => match serde_json::from_slice::<WorkerDriverMessage>(&payload) {
                        Ok(WorkerDriverMessage::CancelAttempt { attempt_id: received, reason })
                            if received == attempt_id => {
                                cancellation.cancel(reason);
                            }
                        _ => {
                            cancellation.cancel(runmat_execution::CancellationReason::Shutdown);
                            control_open = false;
                        }
                    },
                    Err(_) => {
                        cancellation.cancel(runmat_execution::CancellationReason::Shutdown);
                        control_open = false;
                    }
                }
            }
        }
    }
}

fn open_staged_graph(
    reference: StagedGraphWorkerReference,
) -> NativeExecutionResult<runmat_execution_artifact::GraphExecutionRequest> {
    use runmat_execution::identity::ArtifactId;

    let expected_id =
        ArtifactId::derive(&[b"graph-worker-request-v10", reference.stored_digest.bytes()]);
    if reference.artifact_id != expected_id {
        return Err(NativeExecutionError::Protocol(
            "staged Graph artifact ID differs from its exact digest".into(),
        ));
    }
    reference
        .assignment
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let root = std::env::var(crate::driver::NATIVE_ARTIFACT_STORE_ROOT_ENV).map_err(|_| {
        NativeExecutionError::Protocol("staged Graph artifact root is absent".into())
    })?;
    let store = crate::local_store::ArtifactStore::from_existing_root(root.into())?;
    let mut stored: StoredGraphProgram = store.read_verified(
        reference.artifact_id,
        &reference.stored_digest,
        reference.stored_length,
        runmat_execution::resource::MAX_GRAPH_PROGRAM_WIRE_BYTES,
    )?;
    if stored.request.assignment.is_some()
        || stored.request.job_id.is_some()
        || stored.request.context != reference.context
        || stored.request.requested_outputs != reference.requested_outputs
    {
        return Err(NativeExecutionError::Protocol(
            "staged Graph placement differs from its launch descriptor".into(),
        ));
    }
    stored.request.assignment = Some(reference.assignment);
    stored
        .request
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    Ok(stored.request)
}

pub(crate) async fn execute(request: WorkerRequest) -> WorkerResponse {
    match request {
        WorkerRequest::Direct(request) => crate::execute_host_program_request(request).await,
        WorkerRequest::Graph(request) => {
            crate::test_workload::execute_host_graph_request(request, None).await
        }
    }
}

/// Execute a Direct program within its scheduler reservation, so domain
/// workers admit CPU parallelism inside the reserved CPU and memory.
pub(crate) async fn execute_reserved(request: ReservedProgramExecutionRequest) -> WorkerResponse {
    if let Err(error) = request.validate() {
        return WorkerResponse::Failure {
            message: error.to_string(),
        };
    }
    let (program, resources) = request.into_parts();
    crate::test_workload::execute_host_program_request_in_context(
        program,
        crate::test_workload::HostProgramContext {
            resources: Some(&resources),
            ..crate::test_workload::HostProgramContext::default()
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use runmat_execution::{
        Digest, OutputContract, ProgramCallable, ProgramEnvironment, ProgramFunctionId,
        ProgramRevision,
    };
    use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

    use super::*;

    #[tokio::test]
    async fn worker_rejects_a_tampered_materialized_program_before_decoding() {
        let revision = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
                runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "0".into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable(
                "test-interpreter-bytecode-v2",
            ),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let mut artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV2,
            b"not reached".to_vec(),
        )
        .unwrap();
        artifact.executable_bytes.push(0);
        let response = execute(WorkerRequest::Direct(
            runmat_execution_artifact::ProgramExecutionRequest {
                schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe,
                artifact,
                callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
                context: runmat_execution::ProgramInvocationContext::Direct,
                assignment: None,
                job_id: None,
                arguments: Vec::new(),
                requested_outputs: 1,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console: None,
            },
        ))
        .await;
        assert!(
            matches!(response, WorkerResponse::Failure { message } if message.contains("identity mismatch"))
        );
    }
}
