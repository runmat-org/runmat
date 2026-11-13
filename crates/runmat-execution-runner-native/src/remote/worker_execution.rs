use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::AttemptId;
use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_execution_artifact::{ExecutableForm, ProgramExecutionResponse, WorkerProgramRequest};
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingProgress};
use runmat_meshing_execution::{MeshingProgressSink, MeshingStageKernel};

use super::object_transfer::RemoteObjectStore;

/// One attempt-scoped cancellation authority. The typed source wakes runtime
/// and Graph listeners; the shared flag reaches synchronous physics kernels.
#[derive(Default)]
pub(super) struct AttemptCancellation {
    source: runmat_execution::cancellation::CancellationSource,
    flag: Arc<AtomicBool>,
}

impl AttemptCancellation {
    pub(super) fn cancel(&self, reason: runmat_execution::CancellationReason) {
        self.flag.store(true, Ordering::Release);
        self.source.cancel(reason);
    }

    pub(super) fn token(&self) -> runmat_execution::cancellation::CancellationToken {
        self.source.token()
    }

    pub(super) fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag)
    }

    pub(super) fn was_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire) || self.source.token().is_cancelled()
    }
}

impl MeshingCancellationSignal for AttemptCancellation {
    fn is_cancelled(&self) -> bool {
        self.was_cancelled()
    }
}

pub(super) struct GraphAttemptContext {
    pub(super) run_identity: String,
    pub(super) resources: ResourceInventory,
    pub(super) host: runmat_execution::host::ExecutionHostInventory,
}

/// The admitted program and its exact materialized/assigned authority move
/// together into one worker execution.
pub(super) struct WorkerExecutionInput {
    pub attempt_id: AttemptId,
    pub program: WorkerProgramRequest,
    pub graph_attempt: Option<GraphAttemptContext>,
    pub project: Option<Arc<crate::materialized_project::MaterializedProject>>,
    pub bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
    /// Frozen portable environment of the installed bundle.
    pub environment: Vec<(String, String)>,
    /// Scheduler reservation this attempt executes under.
    pub resources: ResourceRequest,
    pub console_budget: Option<runmat_execution::console::ConsoleBudget>,
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
    CapturedResponse(
        ProgramExecutionResponse,
        Option<super::attempt_reply::RemoteAttemptConsole>,
    ),
    RetirementUnverified(String),
}

#[derive(Clone)]
pub(super) struct RemoteMeshingHost {
    kernel: Arc<dyn MeshingStageKernel>,
    limits: crate::NativeMeshingHostLimits,
    #[cfg(all(test, feature = "occt-native"))]
    held_attempt: Option<HeldWorkloadAttempt>,
}

#[cfg(all(test, feature = "occt-native"))]
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) enum HeldAttemptUntil {
    UntilWorkerLoss,
    UntilCancellation,
}

#[cfg(all(test, feature = "occt-native"))]
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) enum HeldAttemptTarget {
    Analysis(runmat_analysis_fea::execution::AnalysisWorkKind),
    ExactMeshingAdmission,
}

#[cfg(all(test, feature = "occt-native"))]
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) struct HeldWorkloadAttempt {
    pub(super) target: HeldAttemptTarget,
    pub(super) input_object_id: Option<runmat_execution::identity::ValueId>,
    pub(super) until: HeldAttemptUntil,
}

impl RemoteMeshingHost {
    pub(super) fn new(
        kernel: Arc<dyn MeshingStageKernel>,
        limits: crate::NativeMeshingHostLimits,
    ) -> Self {
        Self {
            kernel,
            limits,
            #[cfg(all(test, feature = "occt-native"))]
            held_attempt: None,
        }
    }

    #[cfg(all(test, feature = "occt-native"))]
    pub(super) fn hold_attempt(mut self, attempt: HeldWorkloadAttempt) -> Self {
        self.held_attempt = Some(attempt);
        self
    }
}

pub(super) async fn execute(
    input: WorkerExecutionInput,
    ports: WorkerExecutionPorts,
) -> WorkerExecutionOutcome {
    let WorkerExecutionInput {
        attempt_id,
        program,
        graph_attempt,
        project,
        bundle,
        environment,
        resources,
        console_budget,
    } = input;
    let WorkerExecutionPorts {
        collective,
        meshing_host,
        objects,
        cancellation,
        progress_sender,
    } = ports;
    let admitted_settings = match &program {
        WorkerProgramRequest::Direct(request) => bundle
            .as_ref()
            .ok_or_else(|| "cluster execution has no admitted project settings bundle".to_owned())
            .and_then(|bundle| {
                super::runtime_settings::admit(bundle, &resources, request.assignment.as_ref())
            }),
        WorkerProgramRequest::Graph(graph) => {
            super::runtime_settings::admit_graph(graph, &resources)
        }
    };
    if let Err(message) = admitted_settings {
        return WorkerExecutionOutcome::Response(failure(&message));
    }
    let program = match program {
        WorkerProgramRequest::Direct(program) => *program,
        WorkerProgramRequest::Graph(graph) => {
            drop(progress_sender);
            let Some(graph_attempt) = graph_attempt else {
                return WorkerExecutionOutcome::RetirementUnverified(
                    "remote Graph child lacks authenticated attempt context".into(),
                );
            };
            return match super::driver::local_graph::execute_remote_attempt(
                graph_attempt,
                attempt_id,
                *graph,
                resources,
                cancellation.token(),
                console_budget,
            )
            .await
            {
                Ok(Ok(result)) => match result.console {
                    Some(super::pool_execution::RemoteTerminalConsole::LocalSources {
                        root,
                        source,
                        budget,
                    }) => WorkerExecutionOutcome::CapturedResponse(
                        result.response,
                        Some(super::attempt_reply::RemoteAttemptConsole::Sources {
                            root,
                            source,
                            budget,
                        }),
                    ),
                    None => WorkerExecutionOutcome::Response(result.response),
                    Some(_) => WorkerExecutionOutcome::RetirementUnverified(
                        "remote Graph child changed its console retention owner".into(),
                    ),
                },
                Ok(Err(crate::remote::pool_execution::RemoteExecutionStop::Cancelled)) => {
                    WorkerExecutionOutcome::Response(failure("remote Graph attempt was cancelled"))
                }
                Ok(Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                    message,
                ))) => WorkerExecutionOutcome::RetirementUnverified(message),
                Err(error) => WorkerExecutionOutcome::RetirementUnverified(error.to_string()),
            };
        }
    };
    if matches!(
        program.artifact.form,
        ExecutableForm::ExecutableUnitV4 | ExecutableForm::InterpreterRegionSourcesV1
    ) {
        drop(progress_sender);
        let console_owner = match (program.console, console_budget, program.assignment.as_ref()) {
            (None, None, _) => None,
            (Some(_), Some(budget), Some(assignment)) => {
                let owner =
                    match crate::console_retention::ConsoleRetentionOwner::from_subgrant(budget) {
                        Ok(owner) => owner,
                        Err(error) => {
                            return WorkerExecutionOutcome::Response(failure(&error.to_string()))
                        }
                    };
                Some(crate::test_workload::HostConsoleOwner {
                    owner,
                    budget,
                    identity: runmat_runtime::context::RuntimeConsoleScopeIdentity {
                        scope_id: assignment.scope_id,
                        pool_id: assignment.pool_id,
                        task_id: assignment.task_id,
                        attempt_id: assignment.attempt_id,
                    },
                })
            }
            _ => {
                return WorkerExecutionOutcome::Response(failure(
                    "remote console has no exact assigned subtree owner",
                ))
            }
        };
        let (response, console) = crate::test_workload::execute_host_program_request_with_console(
            program,
            crate::test_workload::HostProgramContext {
                materialized: project.as_deref(),
                collective,
                bundle,
                environment: Some(&environment),
                cancellation: Some(cancellation.flag()),
                resources: Some(&resources),
                console_owner,
                console_sink: None,
            },
        )
        .await;
        return WorkerExecutionOutcome::CapturedResponse(
            response,
            console.map(super::attempt_reply::RemoteAttemptConsole::Captured),
        );
    }
    WorkerExecutionOutcome::Response(
        execute_direct(
            attempt_id,
            program,
            DirectContext {
                project,
                bundle,
                environment,
                resources,
                collective,
            },
            meshing_host,
            objects,
            cancellation,
            progress_sender,
        )
        .await,
    )
}

struct DirectContext {
    project: Option<Arc<crate::materialized_project::MaterializedProject>>,
    bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
    environment: Vec<(String, String)>,
    resources: ResourceRequest,
    collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
}

async fn execute_direct(
    attempt_id: AttemptId,
    program: runmat_execution_artifact::ProgramExecutionRequest,
    context: DirectContext,
    meshing_host: Option<RemoteMeshingHost>,
    mut objects: RemoteObjectStore,
    cancellation: Arc<AttemptCancellation>,
    progress_sender: tokio::sync::mpsc::Sender<crate::ProgramProgress>,
) -> ProgramExecutionResponse {
    let DirectContext {
        project,
        bundle,
        environment,
        resources,
        collective,
    } = context;
    #[cfg(all(test, feature = "occt-native"))]
    if let Some(held) = meshing_host.as_ref().and_then(|host| host.held_attempt) {
        let matches_target = match held.target {
            HeldAttemptTarget::Analysis(kind) => {
                program.artifact.form == ExecutableForm::AnalysisWorkload
                    && runmat_analysis_execution::AnalysisHostWorkload::from_program_request(
                        &program, &objects,
                    )
                    .is_ok_and(|host| host.workload.kind == kind)
            }
            HeldAttemptTarget::ExactMeshingAdmission => {
                program.artifact.form == ExecutableForm::MeshingWorkload
                    && program.recipe.execution_mode == crate::EXACT_MESHING_ROOT_EXECUTION_MODE
            }
        };
        if matches_target
            && held.input_object_id.is_none_or(|held_id| {
                matches!(
                    program.arguments.first(),
                    Some(runmat_execution::value::ValuePayload::Object(reference))
                        if reference.id == held_id
                )
            })
        {
            match held.until {
                HeldAttemptUntil::UntilWorkerLoss => {
                    while !cancellation.is_cancelled() {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                    return failure("held workload attempt retired with its worker owner");
                }
                HeldAttemptUntil::UntilCancellation => {
                    while !cancellation.is_cancelled() {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                    return failure("held workload attempt observed cancellation");
                }
            }
        }
    }
    if program.artifact.form == ExecutableForm::AnalysisWorkload {
        drop(progress_sender);
        let config = match bundle
            .as_ref()
            .map(|bundle| runmat_core::cluster_settings::runtime_config(bundle))
        {
            Some(Ok(Some(config))) => config,
            _ => return failure("analysis attempt lacks admitted project runtime policy"),
        };
        return tokio::task::spawn_blocking(move || {
            let _guard = install_analysis_cancellation(attempt_id, &cancellation);
            super::runtime_settings::with_domain_policy(&config, program.assignment.clone(), || {
                crate::execute_reserved_analysis_program_request(
                    &program,
                    &mut objects,
                    &crate::NativeAnalysisKernel,
                    crate::NativeAnalysisHostLimits::default(),
                    &resources,
                )
            })
        })
        .await
        .unwrap_or_else(|error| failure(&error.to_string()));
    }
    if program.artifact.form == ExecutableForm::GeometryWorkload {
        drop(progress_sender);
        return tokio::task::spawn_blocking(move || {
            crate::execute_geometry_materialization_program_request(
                &program,
                &mut objects,
                crate::NativeGeometryMaterializationLimits::default(),
            )
        })
        .await
        .unwrap_or_else(|error| failure(&error.to_string()));
    }
    if program.artifact.form == ExecutableForm::MeshingWorkload {
        let Some(host) = meshing_host else {
            return failure("remote worker has no meshing host capability");
        };
        if program.recipe.execution_mode == crate::EXACT_MESHING_ROOT_EXECUTION_MODE {
            drop(progress_sender);
            return tokio::task::spawn_blocking(move || {
                crate::exact_meshing_root::execute_exact_meshing_root_admission(
                    &program,
                    &mut objects,
                    cancellation.flag(),
                    host.limits,
                )
            })
            .await
            .unwrap_or_else(|error| failure(&error.to_string()));
        }
        return tokio::task::spawn_blocking(move || {
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
        .unwrap_or_else(|error| failure(&error.to_string()));
    }
    drop(progress_sender);
    crate::test_workload::execute_host_program_request_in_context(
        program,
        crate::test_workload::HostProgramContext {
            materialized: project.as_deref(),
            collective,
            bundle,
            environment: Some(&environment),
            cancellation: Some(cancellation.flag()),
            resources: Some(&resources),
            console_owner: None,
            console_sink: None,
        },
    )
    .await
}

fn install_analysis_cancellation(
    attempt_id: AttemptId,
    cancellation: &AttemptCancellation,
) -> runmat_analysis_fea::FeaProgressContextGuard {
    crate::analysis_progress::install(
        runmat_analysis_fea::FeaCancellationToken::from_shared(cancellation.flag()),
        Some(attempt_id),
    )
}

pub(super) fn report(
    request: &AttemptRequest,
    response: ProgramExecutionResponse,
    wall_millis: u64,
) -> AttemptReport {
    let success = match response {
        ProgramExecutionResponse::Success { outputs } => {
            AttemptSuccess::new(request, outputs, Vec::new(), wall_millis, 0, 0)
        }
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => {
            let egress_bytes = result_objects.iter().try_fold(0_u64, |total, reference| {
                total.checked_add(reference.encoded_length)
            });
            match egress_bytes {
                Some(egress_bytes) => AttemptSuccess::new(
                    request,
                    outputs,
                    result_objects,
                    wall_millis,
                    egress_bytes,
                    0,
                ),
                None => Err("remote result egress accounting overflowed".into()),
            }
        }
        ProgramExecutionResponse::GraphSuccess { graph } => {
            AttemptSuccess::new_graph_for_remote_transfer(request, graph, wall_millis)
        }
        ProgramExecutionResponse::SpmdSuccess { outputs } => Ok(AttemptSuccess::Spmd { outputs }),
        ProgramExecutionResponse::Failure { message } => {
            return AttemptReport::Failed {
                kind: AttemptFailureKind::Execution,
                message,
            }
        }
        ProgramExecutionResponse::RuntimeFailure { failure } => {
            return AttemptReport::RuntimeFailed { failure }
        }
    };
    match success {
        Ok(result) => AttemptReport::Succeeded { result },
        Err(message) => AttemptReport::Failed {
            kind: AttemptFailureKind::Rejected,
            message,
        },
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_execution::identity::{ArtifactId, AttemptId, WorkerId};
    use runmat_execution::resource::ResourceRequest;
    use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
    use runmat_execution::{ExecutionScopeId, OutputContract, PoolId, ProgramCallable, TaskId};
    use runmat_execution_artifact::ProgramExecutionResponse;
    use runmat_execution_runner::{AttemptReport, AttemptRequest};

    use super::{install_analysis_cancellation, report, AttemptCancellation};

    #[test]
    fn remote_attempt_cancellation_reaches_analysis_solver_context() {
        let cancellation = AttemptCancellation::default();
        let attempt_id = AttemptId::derive(&[b"analysis-cancellation-test"]);
        let _guard = install_analysis_cancellation(attempt_id, &cancellation);
        assert!(!runmat_analysis_fea::is_fea_cancelled());
        cancellation.cancel(runmat_execution::CancellationReason::User);
        assert!(runmat_analysis_fea::is_fea_cancelled());
    }

    #[test]
    fn remote_original_stop_reason_reaches_worker_token_and_first_cause_wins() {
        use runmat_execution::CancellationReason;
        for reason in [CancellationReason::Deadline, CancellationReason::User] {
            let command = super::super::protocol::RemoteWorkerCommand::Cancel {
                attempt_id: AttemptId::derive(&[b"original-stop-reason"]),
                reason,
            };
            let encoded = serde_json::to_vec(&command).unwrap();
            let mut missing_reason = serde_json::to_value(&command).unwrap();
            missing_reason.as_object_mut().unwrap().remove("reason");
            assert!(
                serde_json::from_value::<super::super::protocol::RemoteWorkerCommand>(
                    missing_reason
                )
                .is_err()
            );
            let super::super::protocol::RemoteWorkerCommand::Cancel {
                reason: decoded, ..
            } = serde_json::from_slice(&encoded).unwrap()
            else {
                panic!("cancel command")
            };
            let cancellation = AttemptCancellation::default();
            cancellation.cancel(decoded);
            cancellation.cancel(CancellationReason::Shutdown);
            assert_eq!(cancellation.token().stop_reason(), Some(reason));
            assert!(cancellation.was_cancelled());
            let staged = runmat_execution::cancellation::CancellationSource::new();
            staged.cancel(decoded);
            staged.cancel(CancellationReason::Shutdown);
            assert_eq!(staged.token().stop_reason(), Some(reason));
        }
    }

    #[test]
    fn metered_attempt_report_round_trips_over_the_remote_json_contract() {
        let task_id = TaskId::derive(&[b"task"]);
        let request = AttemptRequest {
            id: AttemptId::derive(&[b"attempt"]),
            task_id,
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            worker_id: WorkerId::derive(&[b"worker"]),
            ordinal: 0,
            driver_fence: 1,
            resource_assignment: Default::default(),
            task: TaskRequest {
                schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                console: None,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                id: task_id,
                scope_id: ExecutionScopeId::derive(&[b"scope"]),
                pool_id: PoolId::derive(&[b"pool"]),
                executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                    ArtifactId::derive(&[b"artifact"]),
                ),
                callable: Callable::for_program(
                    "test",
                    &ProgramCallable::semantic(runmat_types::ProgramFunctionId(0), None),
                ),
                host: runmat_execution::host::ExecutionHostRequirement::portable(
                    runmat_core::program_environment(runmat_core::CompatMode::Matlab),
                    runmat_types::CapabilitySet(BTreeSet::from([
                        runmat_types::CapabilityRequirement::HostRuntime,
                    ])),
                    runmat_types::InteropManifest::empty(),
                    BTreeSet::from([
                        runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                    ]),
                )
                .unwrap(),
                invocation_context: runmat_execution::ProgramInvocationContext::Direct,
                inputs: Vec::new(),
                outputs: OutputContract {
                    requested_outputs: 1,
                    owned_work: None,
                },
                resources: ResourceRequest {
                    cpu_millicores: 1_000,
                    memory_bytes: 16 * 1024 * 1024,
                    scratch_bytes: 8 * 1024 * 1024,
                    max_wall_millis: 1_000,
                    max_artifact_bytes: 1024,
                    max_egress_bytes: 1024,
                    max_relay_bytes: 0,
                    accelerators: Vec::new(),
                    required_capabilities: BTreeSet::new(),
                },
                retry: RetryPolicy::Never,
                deadline_unix_millis: None,
            },
        };
        let report = report(
            &request,
            ProgramExecutionResponse::Success {
                outputs: vec![runmat_execution::value::ValuePayload::Inline(Box::new(
                    runmat_execution::value::InlineValue::Null,
                ))],
            },
            1,
        );
        let encoded = serde_json::to_vec(&report).unwrap();
        let decoded: AttemptReport = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, report);
        let logical_digest = runmat_execution::Digest::sha256(b"remote-graph-result");
        let encryption_context = runmat_execution::Digest::sha256(b"remote-graph-scope");
        let reference = runmat_execution::value::ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: runmat_execution::value::artifact_object_value_id(
                logical_digest,
                encryption_context,
                "remote-graph-scope",
            ),
            logical_digest,
            encryption_context,
            encoded_length: 23,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            kind: runmat_execution::value::ValueRefKind::ResultObject,
            authorization_scope: "remote-graph-scope".into(),
            resident_fence: None,
        };
        for external in [false, true] {
            let value = if external {
                runmat_execution::value::ValuePayload::Object(Box::new(reference.clone()))
            } else {
                runmat_execution::value::ValuePayload::Inline(Box::new(
                    runmat_execution::value::InlineValue::Null,
                ))
            };
            let graph = runmat_execution::value::PayloadGraph {
                roots: vec![runmat_execution::value::GraphNodeRef(0)],
                nodes: vec![
                    runmat_execution::value::GraphNode::Leaf {
                        value: value.clone(),
                    },
                    runmat_execution::value::GraphNode::Leaf { value },
                ],
            };
            let graph_report = super::report(
                &request,
                ProgramExecutionResponse::GraphSuccess { graph },
                3,
            );
            let AttemptReport::Succeeded { result } = &graph_report else {
                panic!("metered Graph success");
            };
            result.validate_against(&request).unwrap();
            let accounting = &result.resources().unwrap().resources;
            let bytes = if external { 23 } else { 0 };
            assert_eq!(accounting.artifact_bytes, bytes);
            assert_eq!(accounting.usage.egress_bytes, bytes);
            assert_eq!(accounting.usage.relay_bytes, 0);
            assert_eq!(accounting.usage.wall_millis, 3);
            assert_eq!(result.result_objects().len(), usize::from(external));
            assert_eq!(
                serde_json::from_slice::<AttemptReport>(
                    &serde_json::to_vec(&graph_report).unwrap()
                )
                .unwrap(),
                graph_report
            );
        }
    }
}
