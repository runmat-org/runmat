use runmat_process_host::ipc::FrameLimits;

use crate::protocol::{
    StagedGraphWorkerReference, StoredGraphProgram, WorkerDriverMessage, WorkerLaunchRequest,
    WorkerRequest, WorkerResponse,
};
use crate::{NativeExecutionError, NativeExecutionResult};

mod collective;

enum WorkerExecutionOutcome {
    Completed(WorkerResponse),
    RetirementUnverified {
        message: String,
        owner: Box<crate::test_workload::RetainedGraphHost>,
    },
}

pub async fn run_worker_stdio() -> NativeExecutionResult<()> {
    let (reader, writer) = runmat_process_host::ipc::stdio::endpoint();
    let limits = FrameLimits {
        max_message_bytes: 64 * 1024 * 1024,
    };
    let channel = collective::StdioCollectiveChannel::new(reader, writer, limits);
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
        WorkerLaunchRequest::StagedGraph(_) => None,
    };
    let response = if let Some(task) = spmd_task {
        let WorkerLaunchRequest::Direct(direct) = request else {
            unreachable!("SPMD request uses Direct worker wire")
        };
        let WorkerRequest::Direct(request) = *direct else {
            unreachable!("SPMD request has Direct program")
        };
        let service = collective::StdioCollectiveService::new(task, channel.clone());
        Ok(WorkerExecutionOutcome::Completed(
            crate::test_workload::execute_host_program_request_with_collective(
                request,
                std::rc::Rc::new(service),
            )
            .await,
        ))
    } else {
        execute_launch(request, channel.clone()).await
    };
    let payload = match response {
        Ok(WorkerExecutionOutcome::Completed(response)) => serde_json::to_vec(&response),
        Ok(WorkerExecutionOutcome::RetirementUnverified { message, owner }) => {
            let _owner = owner;
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
    channel.write_payload(&payload).await?;
    Ok(())
}

async fn execute_launch(
    request: WorkerLaunchRequest,
    channel: collective::StdioCollectiveChannel,
) -> NativeExecutionResult<WorkerExecutionOutcome> {
    match request {
        WorkerLaunchRequest::Direct(direct) => match *direct {
            WorkerRequest::Direct(request) => Ok(WorkerExecutionOutcome::Completed(
                execute(WorkerRequest::Direct(request)).await,
            )),
            WorkerRequest::Graph(_) => Err(NativeExecutionError::Protocol(
                "Graph stdio requests require a verified staged artifact reference".into(),
            )),
        },
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
    let request = open_staged_graph(reference)?;
    let scratch = std::env::var(crate::driver::NATIVE_GRAPH_ATTEMPT_SCRATCH_ENV).map_err(|_| {
        NativeExecutionError::Protocol("selected Graph attempt scratch is absent".into())
    })?;
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let execution = crate::test_workload::execute_host_graph_request_in_parent_scratch(
        request,
        cancellation.token(),
        std::path::Path::new(&scratch),
    );
    tokio::pin!(execution);
    let mut control_open = true;
    loop {
        tokio::select! {
            response = &mut execution => return Ok(match response {
                crate::test_workload::GraphHostExecution::Completed(response) => {
                    WorkerExecutionOutcome::Completed(response)
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
            },
        ))
        .await;
        assert!(
            matches!(response, WorkerResponse::Failure { message } if message.contains("identity mismatch"))
        );
    }
}
