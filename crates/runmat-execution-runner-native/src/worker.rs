use runmat_process_host::ipc::FrameLimits;

use crate::protocol::{
    StagedGraphWorkerReference, StoredGraphProgram, WorkerDriverMessage, WorkerLaunchRequest,
    WorkerRequest, WorkerResponse,
};
use crate::{NativeExecutionError, NativeExecutionResult};

mod collective;
mod console;

/// One bounded private control frame between a native worker and its parent.
pub(crate) const WORKER_MAX_MESSAGE_BYTES: u32 = 64 * 1024 * 1024;

enum WorkerExecutionOutcome {
    Completed {
        // Transport metadata only; Value backing keeps its own owners.
        response: Box<WorkerResponse>,
        console: Option<runmat_execution::console::ConsoleReceipt>,
        captured: Option<crate::console_bridge::NativeCapturedConsole>,
    },
    RetirementUnverified {
        message: String,
        owner: Box<crate::test_workload::RetainedGraphHost>,
    },
}

pub async fn run_worker_stdio() -> NativeExecutionResult<()> {
    let original_started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?
        .as_millis();
    let control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .map_err(runmat_process_host::ProcessHostError::Io)?;
    let limits = FrameLimits {
        max_message_bytes: WORKER_MAX_MESSAGE_BYTES,
    };
    // Parent loss (the request pipe closing before this worker publishes its terminal frame)
    // ends the worker at once, even while a non-yielding solve holds the executor.
    let lease_observer = std::sync::Arc::new(crate::fea_delegation::LeaseObserver::default());
    let proxy_replies = std::sync::Arc::new(crate::fea_proxy::Replies::default());
    let replies = proxy_replies.clone();
    let observer = lease_observer.clone();
    let (parent, frames) = runmat_process_host::ipc::WorkerParentLink::start_with_observer(
        control.reader,
        limits,
        Some(std::sync::Arc::new(move |frame| {
            replies.observe(frame) || observer.observe(frame)
        })),
    )?;
    let writer =
        runmat_process_host::ipc::private_control::ControlFrameWriter::new(control.writer, limits)?;
    let channel = collective::WorkerControlChannel::new(frames, writer);
    let payload = channel.read_payload().await?;
    // The parent assigns this runnable child to its JobObject before writing
    // the first request. No RunMat provider or foreign helper is initialized
    // in the spawn-to-containment gap.
    runmat_accelerate::initialize_acceleration_provider();
    let request: WorkerLaunchRequest = serde_json::from_slice(&payload)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let proxy_requested = request.fea_proxy();
    let proxy_host = if proxy_requested {
        if request.fea_grant().is_some() {
            return Err(NativeExecutionError::Protocol(
                "worker received both paid FEA delegation and first-root proxy".into(),
            ));
        }
        let assignment = request
            .assignment()
            .ok_or_else(|| {
                NativeExecutionError::Protocol("FEA proxy lacks original worker assignment".into())
            })?
            .clone();
        let resources = request.resources().ok_or_else(|| {
            NativeExecutionError::Protocol("FEA proxy lacks original worker reservation".into())
        })?;
        if resources.memory_bytes
            < crate::fea_proxy::storage_bytes(0, crate::fea_proxy::StorageRole::Proxy)
                .map_err(NativeExecutionError::Protocol)?
        {
            return Err(NativeExecutionError::Protocol(
                "FEA proxy metadata exceeds worker reservation".into(),
            ));
        }
        let client = crate::fea_proxy::Client::new(
            assignment,
            u64::from(resources.cpu_millicores),
            channel.writer().clone(),
            proxy_replies,
        );
        Some(
            runmat_runtime::analysis::fea_run_admission::FeaRunAdmissionContext::new(
                std::sync::Arc::new(crate::fea_proxy::WorkerAdmission(client)),
            ),
        )
    } else {
        None
    };
    // Only this authenticated private bootstrap accepts a parent FEA grant;
    // the public Program request/JSON executor has no such authorization field.
    let (request, fea_grant) = match request {
        WorkerLaunchRequest::ReservedFeaProxy {
            request,
            console_budget,
            fea_parent_admission: true,
        } => (
            match console_budget {
                Some(console_budget) => WorkerLaunchRequest::ReservedConsole {
                    request,
                    console_budget,
                },
                None => WorkerLaunchRequest::Reserved(request),
            },
            None,
        ),
        WorkerLaunchRequest::ReservedFea {
            request,
            console_budget,
            fea_grant,
        } => (
            match console_budget {
                Some(console_budget) => WorkerLaunchRequest::ReservedConsole {
                    request,
                    console_budget,
                },
                None => WorkerLaunchRequest::Reserved(request),
            },
            Some(fea_grant),
        ),
        WorkerLaunchRequest::StagedGraph(mut graph) => {
            let grant = graph.staged_graph_v10.fea_grant.take();
            (WorkerLaunchRequest::StagedGraph(graph), grant)
        }
        other => (other, None),
    };
    let fea_scope = if let Some(grant) = &fea_grant {
        let assignment = request.assignment().ok_or_else(|| {
            NativeExecutionError::Protocol("FEA grant lacks an assigned child".into())
        })?;
        let resources = request.resources().ok_or_else(|| {
            NativeExecutionError::Protocol("FEA grant lacks reserved child resources".into())
        })?;
        let scope = grant
            .install(assignment, resources)
            .map_err(NativeExecutionError::Protocol)?;
        lease_observer
            .install(grant, &scope)
            .map_err(NativeExecutionError::Protocol)?;
        Some(scope)
    } else {
        None
    };
    let spmd_task = match &request {
        WorkerLaunchRequest::Direct(direct) => match direct.as_ref() {
            WorkerRequest::Direct(request) => match &request.context {
                runmat_execution::ProgramInvocationContext::SpmdTask { task } => Some(task.clone()),
                _ => None,
            },
            WorkerRequest::Graph(_) => None,
        },
        WorkerLaunchRequest::Reserved(reserved)
        | WorkerLaunchRequest::ReservedConsole {
            request: reserved, ..
        } => match &reserved.request().context {
            runmat_execution::ProgramInvocationContext::SpmdTask { task } => Some(task.clone()),
            _ => None,
        },
        WorkerLaunchRequest::StagedGraph(_) => None,
        WorkerLaunchRequest::ReservedFea { .. } => {
            return Err(NativeExecutionError::Protocol(
                "uninstalled FEA private grant".into(),
            ))
        }
        WorkerLaunchRequest::ReservedFeaProxy { .. } => {
            return Err(NativeExecutionError::Protocol(
                "uninstalled FEA admission proxy".into(),
            ))
        }
    };
    let collective = if let Some(task) = spmd_task {
        let service = collective::WorkerCollectiveService::new(task, channel.clone());
        Some(std::rc::Rc::new(service)
            as std::rc::Rc<
                dyn runmat_runtime::context::RuntimeCollectiveService,
            >)
    } else {
        None
    };
    // A first FEA root inside a local callback uses the installed host's lease
    // provider, constrained to this exact private bootstrap reservation. Its
    // original worker wall clock is never restarted by a later FEA call.
    let independent_fea_host = if fea_scope.is_none() {
        request.resources().map(|resources| {
            let deadline = u64::try_from(original_started_at)
                .ok()
                .and_then(|started| started.checked_add(resources.max_wall_millis))
                .ok_or_else(|| NativeExecutionError::Protocol("worker FEA deadline overflow".into()))?;
                let make = || runmat_runtime::analysis::fea_run_admission::FeaRunAdmissionContext::with_current_resource_bounds(u64::from(resources.cpu_millicores), deadline);
                match &proxy_host { Some(host) => host.run_sync(make), None => make() }.map_err(|error| NativeExecutionError::Protocol(error.to_string()))
        }).transpose()?
    } else {
        None
    };
    let response = match &fea_scope {
        Some(scope) => scope
            .run_checked_future(execute_launch(request, channel.clone(), collective))
            .await
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
        None => match independent_fea_host {
            Some(host) => {
                host.run_future(execute_launch(request, channel.clone(), collective))
                    .await
            }
            None => execute_launch(request, channel.clone(), collective).await,
        },
    };
    let payload = match response {
        Ok(WorkerExecutionOutcome::Completed {
            response,
            console,
            captured,
        }) => crate::console_bridge::encode_worker_terminal(
            &response,
            console.as_ref(),
            captured.as_ref(),
        ),
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
                let _ = channel.write_payload(payload).await;
            }
            return std::future::pending().await;
        }
        Err(error) => serde_json::to_vec(
            &crate::protocol::WorkerProcessMessage::InfrastructureFailure {
                message: error.to_string(),
            },
        )
        .map_err(|error| error.to_string()),
    }
    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    // The parent may close the request pipe as soon as it reads the terminal frame.
    if let Some(scope) = &fea_scope {
        scope
            .check()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    }
    parent.disarm();
    channel.write_payload(payload).await?;
    Ok(())
}

async fn execute_launch(
    request: WorkerLaunchRequest,
    channel: collective::WorkerControlChannel,
    collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
) -> NativeExecutionResult<WorkerExecutionOutcome> {
    match request {
        WorkerLaunchRequest::ReservedFeaProxy { .. } => Err(NativeExecutionError::Protocol(
            "uninstalled FEA admission proxy".into(),
        )),
        WorkerLaunchRequest::ReservedFea { .. } => Err(NativeExecutionError::Protocol(
            "FEA grant must be installed at the private worker boundary".into(),
        )),
        WorkerLaunchRequest::Direct(direct) => match *direct {
            WorkerRequest::Direct(request) => Ok(WorkerExecutionOutcome::Completed {
                response: Box::new(match collective {
                    Some(collective) => {
                        crate::test_workload::execute_host_program_request_with_collective(
                            *request, collective,
                        )
                        .await
                    }
                    None => execute(WorkerRequest::Direct(request)).await,
                }),
                console: None,
                captured: None,
            }),
            WorkerRequest::Graph(_) => Err(NativeExecutionError::Protocol(
                "Graph worker requests require a verified staged artifact reference".into(),
            )),
        },
        WorkerLaunchRequest::Reserved(reserved) => {
            let (response, captured) =
                console::execute_reserved(*reserved, None, channel.writer().clone(), collective)
                    .await?;
            let console = captured
                .as_ref()
                .map(|capture| capture.transcript.receipt())
                .transpose()
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            Ok(WorkerExecutionOutcome::Completed {
                response: Box::new(response),
                console,
                captured,
            })
        }
        WorkerLaunchRequest::ReservedConsole {
            request,
            console_budget,
        } => {
            let (response, captured) = console::execute_reserved(
                *request,
                Some(console_budget),
                channel.writer().clone(),
                collective,
            )
            .await?;
            let console = captured
                .as_ref()
                .map(|capture| capture.transcript.receipt())
                .transpose()
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            Ok(WorkerExecutionOutcome::Completed {
                response: Box::new(response),
                console,
                captured,
            })
        }
        WorkerLaunchRequest::StagedGraph(staged) => {
            execute_staged_graph(staged.staged_graph_v10, channel).await
        }
    }
}

async fn execute_staged_graph(
    reference: StagedGraphWorkerReference,
    channel: collective::WorkerControlChannel,
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
    let console_budget = reference.console_budget;
    let wire_bytes = usize::try_from(reference.stored_length).map_err(|_| {
        NativeExecutionError::Protocol("Graph wire length exceeds this worker address space".into())
    })?;
    let assignment = reference.assignment.clone();
    let request = open_staged_graph(reference)?;
    let controls = request
        .invocation_controls
        .as_ref()
        .and_then(|controls| controls.console);
    if let Some(budget) = console_budget {
        let original_floor = request
            .input_allocation_floor_bytes(wire_bytes)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let memory =
            crate::console_retention::allocation_floor(original_floor, controls, Some(budget))?;
        let encoded = crate::console_retention::artifact_floor(0, controls, Some(budget))?;
        if memory > resources.memory_bytes || encoded > resources.max_artifact_bytes {
            return Err(NativeExecutionError::Protocol(
                "Graph console subtree and original inputs exceed the original worker reservation"
                    .into(),
            ));
        }
    }
    let console_host = console::graph_host(
        request
            .invocation_controls
            .as_ref()
            .and_then(|controls| controls.console),
        console_budget,
        &assignment,
        channel.writer().clone(),
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
                    let captured = transcript.map(|transcript| crate::console_bridge::NativeCapturedConsole {
                        transcript: Box::new(transcript),
                        scopes: console_host.as_ref().expect("captured host").retained_console_source(),
                        budget: console_budget.expect("captured grant"),
                    });
                    WorkerExecutionOutcome::Completed { response: Box::new(response), console, captured }
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
        WorkerRequest::Direct(request) => crate::execute_host_program_request(*request).await,
        WorkerRequest::Graph(request) => {
            crate::test_workload::execute_host_graph_request(*request, None).await
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
        let response = execute(WorkerRequest::direct(
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
