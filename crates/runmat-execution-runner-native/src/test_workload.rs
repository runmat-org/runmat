use runmat_execution_artifact::{
    ExecutableForm, GraphExecutionRequest, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_test_runner::worker::WorkerExecution;
use runmat_test_runner_execution::TestAttemptWorkload;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
mod graphics;
#[cfg(test)]
#[path = "test_workload/callable_graph_tests.rs"]
mod callable_graph_tests;
pub use graphics::{
    execute_host_graph_request_with_graphics, execute_host_program_request_with_graphics,
};

/// A one-shot child must not release a selected host whose lifecycle callback
/// refused retirement. The parent receives a fatal wire marker and reaps the
/// process tree while this owner remains live in the child.
pub(crate) enum GraphHostExecution {
    Completed(ProgramExecutionResponse),
    RetirementUnverified {
        message: String,
        owner: Box<RetainedGraphHost>,
    },
}

pub(crate) struct RetainedGraphHost {
    _hosts: Vec<Option<runmat_core::ForeignUnitHost>>,
    _materialized: Vec<Option<Rc<runmat_core::foreign_artifacts::SelectedForeignMaterialization>>>,
    _cleanup: runmat_vm::SelectedGraphCleanupContexts,
    _session: runmat_core::RunMatSession,
    _cause: Option<runmat_runtime::RuntimeError>,
}

struct SelectedRetirementFault {
    message: String,
    cause: Option<runmat_runtime::RuntimeError>,
}

pub async fn execute_host_program_request(
    request: ProgramExecutionRequest,
) -> ProgramExecutionResponse {
    execute_host_program_request_with_project_and_collective(request, None, None).await
}

pub(crate) async fn execute_host_program_request_with_collective(
    request: ProgramExecutionRequest,
    collective: std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>,
) -> ProgramExecutionResponse {
    execute_host_program_request_with_project_and_collective(request, None, Some(collective)).await
}

pub(crate) async fn execute_host_program_request_with_project_and_collective(
    request: ProgramExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
    collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
) -> ProgramExecutionResponse {
    execute_host_program_request_with_validated_bundle(request, materialized, collective, None)
        .await
}

pub(crate) async fn execute_host_program_request_with_validated_bundle(
    request: ProgramExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
    collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
) -> ProgramExecutionResponse {
    execute_host_program_request_in_context(
        request,
        HostProgramContext {
            materialized,
            collective,
            bundle,
            ..HostProgramContext::default()
        },
    )
    .await
}

/// Original host-owned context of one admitted program invocation.
#[derive(Default)]
pub(crate) struct HostProgramContext<'a> {
    pub materialized: Option<&'a crate::materialized_project::MaterializedProject>,
    pub collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    pub bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
    /// Bundle-admitted environment; must equal the request's frozen environment.
    pub environment: Option<&'a [(String, String)]>,
    pub cancellation: Option<Arc<AtomicBool>>,
    /// Scheduler reservation within which domain workers admit CPU parallelism.
    pub resources: Option<&'a runmat_execution::resource::ResourceRequest>,
    pub console_owner: Option<HostConsoleOwner>,
    pub console_sink: Option<Rc<dyn runmat_runtime::context::PortableConsoleSink>>,
}

pub(crate) struct HostConsoleOwner {
    pub owner: Arc<dyn runmat_runtime::context::RuntimeConsoleAdmission>,
    pub identity: runmat_runtime::context::RuntimeConsoleScopeIdentity,
    pub budget: runmat_execution::console::ConsoleBudget,
}

pub(crate) async fn execute_host_program_request_in_context(
    request: ProgramExecutionRequest,
    context: HostProgramContext<'_>,
) -> ProgramExecutionResponse {
    execute_host_program_request_inner(request, context, None).await
}

/// One terminal transport owns and seals this invocation's existing capture.
/// Session cleanup completes before the transcript leaves its original owner.
pub(crate) async fn execute_host_program_request_with_console(
    request: ProgramExecutionRequest,
    context: HostProgramContext<'_>,
) -> (
    ProgramExecutionResponse,
    Option<crate::console_bridge::NativeCapturedConsole>,
) {
    let admission =
        runmat_execution::console::invocation_memory_bytes(&request.environment, request.console)
            .map_err(|error| error.to_string())
            .and_then(|bytes| match context.resources {
                Some(resources) if bytes > resources.memory_bytes => {
                    Err("invocation console exceeds its authenticated memory reservation".into())
                }
                None if request.console.is_some() => {
                    Err("captured console requires its authenticated allocation reservation".into())
                }
                _ => Ok(()),
            })
            .and_then(|()| match (request.console, context.resources) {
                (Some(limits), Some(resources))
                    if limits
                        .storage_profile()
                        .map_err(|error| error.to_string())?
                        .max_encoded_bytes
                        > resources.max_artifact_bytes =>
                {
                    Err("invocation console exceeds its authenticated artifact reservation".into())
                }
                _ => Ok(()),
            });
    if let Err(message) = admission {
        return (ProgramExecutionResponse::Failure { message }, None);
    }
    let cancellation = context.cancellation.clone();
    let host = match request
        .console
        .map(|limits| {
            runmat_runtime::context::PortableConsoleHost::with_sink(
                limits,
                context.console_sink.clone(),
            )
        })
        .transpose()
    {
        Ok(host) => host.map(Rc::new),
        Err(error) => {
            return (
                ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                },
                None,
            )
        }
    };
    let mut budget = None;
    if let Some(host) = &host {
        let Some(admission) = &context.console_owner else {
            return (
                ProgramExecutionResponse::Failure {
                    message: "captured console has no original aggregate admission owner".into(),
                },
                None,
            );
        };
        let component = admission.budget.minimum_memory_bytes().and_then(|memory| {
            request
                .environment
                .storage_profile()?
                .minimum_memory_bytes()
                .and_then(|environment| {
                    memory.checked_add(environment).ok_or_else(|| {
                        runmat_execution::ContractError::Invalid {
                            field: "console budget",
                            reason: "invocation storage overflow".into(),
                        }
                    })
                })
        });
        match (component, context.resources) {
            (Ok(bytes), Some(resources))
                if bytes <= resources.memory_bytes
                    && admission.budget.max_encoded_bytes <= resources.max_artifact_bytes => {}
            (Err(error), _) => {
                return (
                    ProgramExecutionResponse::Failure {
                        message: error.to_string(),
                    },
                    None,
                )
            }
            _ => {
                return (
                    ProgramExecutionResponse::Failure {
                        message: "console subtree exceeds its authenticated execution reservation"
                            .into(),
                    },
                    None,
                )
            }
        }
        if let Err(error) =
            host.install_console_admission(Arc::clone(&admission.owner), admission.identity)
        {
            return (
                ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                },
                None,
            );
        }
        budget = Some(admission.budget);
    }
    let response = execute_host_program_request_inner(request, context, host.clone()).await;
    let completed = cancellation
        .as_ref()
        .is_none_or(|flag| !flag.load(std::sync::atomic::Ordering::Acquire));
    match host
        .map(|host| {
            host.finish(completed)
                .map(|transcript| crate::console_bridge::NativeCapturedConsole {
                    transcript: Box::new(transcript),
                    scopes: host.retained_console_source(),
                    budget: budget.expect("capture admission owns its budget"),
                })
        })
        .transpose()
    {
        Ok(console) => (response, console),
        Err(error) => (
            ProgramExecutionResponse::Failure {
                message: error.to_string(),
            },
            None,
        ),
    }
}

async fn execute_host_program_request_inner(
    request: ProgramExecutionRequest,
    context: HostProgramContext<'_>,
    console_host: Option<Rc<runmat_runtime::context::PortableConsoleHost>>,
) -> ProgramExecutionResponse {
    let HostProgramContext {
        materialized,
        collective,
        bundle,
        environment,
        cancellation,
        resources,
        console_owner: _,
        console_sink: _,
    } = context;
    let project = materialized.and_then(|materialized| materialized.handoff());
    if let Err(error) = request.validate_for_portable_host() {
        return ProgramExecutionResponse::Failure {
            message: error.to_string(),
        };
    }
    if environment.is_some_and(|entries| entries != request.environment.entries()) {
        return ProgramExecutionResponse::Failure {
            message: "portable invocation environment differs from admitted bundle context".into(),
        };
    }
    if request.artifact.form == ExecutableForm::AnalysisWorkload {
        // This branch is synchronous: the thread-affine context remains installed
        // throughout the physical solve. Local process cancellation remains owned
        // by the parent guard; callers that supply a token also reach the solver.
        let _progress = crate::analysis_progress::install(
            cancellation
                .as_ref()
                .map(|flag| {
                    runmat_analysis_fea::FeaCancellationToken::from_shared(Arc::clone(flag))
                })
                .unwrap_or_default(),
            None,
        );
        let Some(root) = std::env::var_os(crate::NATIVE_OBJECT_STORE_ROOT_ENV) else {
            return ProgramExecutionResponse::Failure {
                message: "analysis worker object store is not configured".into(),
            };
        };
        let mut store = match crate::FilesystemObjectStore::open(
            root,
            crate::NativeAnalysisHostLimits::default()
                .inventory
                .max_object_bytes,
        ) {
            Ok(store) => store,
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                }
            }
        };
        return match resources {
            Some(resources) => crate::execute_reserved_analysis_program_request(
                &request,
                &mut store,
                &crate::NativeAnalysisKernel,
                crate::NativeAnalysisHostLimits::default(),
                resources,
            ),
            None => crate::execute_analysis_program_request(
                &request,
                &mut store,
                &crate::NativeAnalysisKernel,
                crate::NativeAnalysisHostLimits::default(),
            ),
        };
    }
    if request.artifact.form == ExecutableForm::GeometryWorkload {
        let Some(root) = std::env::var_os(crate::NATIVE_OBJECT_STORE_ROOT_ENV) else {
            return ProgramExecutionResponse::Failure {
                message: "geometry worker object store is not configured".into(),
            };
        };
        let limits = crate::NativeGeometryMaterializationLimits::default();
        let mut store =
            match crate::FilesystemObjectStore::open(root, limits.inventory.max_object_bytes) {
                Ok(store) => store,
                Err(error) => {
                    return ProgramExecutionResponse::Failure {
                        message: error.to_string(),
                    }
                }
            };
        return crate::execute_geometry_materialization_program_request(
            &request, &mut store, limits,
        );
    }
    // Complete V4 units without foreign closures or collectives run in an
    // original session that owns their environment, console and dependencies.
    if request.artifact.form == ExecutableForm::ExecutableUnitV4
        && bundle.is_none()
        && collective.is_none()
        && console_host.is_none()
    {
        return crate::program_workload::execute_unit(&request, project, environment, cancellation)
            .await;
    }
    if request.artifact.form != ExecutableForm::TestAttemptV1 {
        return execute_portable_request(
            request,
            materialized,
            collective,
            bundle,
            console_host,
            cancellation,
        )
        .await;
    }
    match execute_test_attempt(&request, materialized, bundle, cancellation).await {
        Ok(execution) => match runmat_test_runner_execution::encode_execution(&execution) {
            Ok(value) => ProgramExecutionResponse::Success {
                outputs: vec![value],
            },
            Err(message) => ProgramExecutionResponse::Failure { message },
        },
        Err(message) => ProgramExecutionResponse::Failure { message },
    }
}

async fn execute_portable_request(
    request: ProgramExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
    collective: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
    console_host: Option<Rc<runmat_runtime::context::PortableConsoleHost>>,
    cancellation: Option<Arc<AtomicBool>>,
) -> ProgramExecutionResponse {
    program_host::execute_host_unit_request_with_console(
        HostUnitRequest::Direct(Box::new(request)),
        materialized,
        collective,
        bundle,
        console_host,
        cancellation,
    )
    .await
}

pub(crate) async fn execute_host_graph_request(
    request: GraphExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
) -> ProgramExecutionResponse {
    if request.selected_bundle_archives.iter().any(Option::is_some) {
        return ProgramExecutionResponse::Failure {
            message: "archive-bearing Graph interop requires an isolated attempt worker with exact selected-unit materialization".into(),
        };
    }
    execute_host_unit_request(
        HostUnitRequest::Graph(Box::new(request)),
        materialized,
        None,
        None,
    )
    .await
}

pub(crate) async fn execute_host_graph_request_in_parent_scratch(
    request: GraphExecutionRequest,
    cancellation: runmat_execution::cancellation::CancellationToken,
    scratch: &Path,
    console_host: Option<Rc<runmat_runtime::context::PortableConsoleHost>>,
) -> GraphHostExecution {
    execute_host_selected_graph_request(request, Some(cancellation), Some(scratch), console_host)
        .await
}

async fn execute_host_selected_graph_request(
    request: GraphExecutionRequest,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
    scratch: Option<&Path>,
    console_host: Option<Rc<runmat_runtime::context::PortableConsoleHost>>,
) -> GraphHostExecution {
    let admitted = match request.into_validated() {
        Ok(admitted) => admitted,
        Err(error) => {
            return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid graph bundle: {error}"),
            });
        }
    };
    let selected = match admitted.request().selected_unit_host_requirements() {
        Ok(selected) => selected,
        Err(error) => {
            return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid graph unit: {error}"),
            });
        }
    };
    let runtime_config = if admitted.request().project_runtime_policy.is_some() {
        match runmat_core::cluster_settings::validate_graph_policy(admitted.request())
            .and_then(|policy| policy.runtime_config().map_err(|error| error.to_string()))
        {
            Ok(config) => Some(config),
            Err(message) => {
                return GraphHostExecution::Completed(ProgramExecutionResponse::Failure { message })
            }
        }
    } else {
        None
    };
    if scratch.is_none() && admitted.selected_bundles().iter().any(Option::is_some) {
        return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
            message: "selected foreign bundles require parent-owned attempt scratch".into(),
        });
    }
    // All bundle objects are authenticated before any foreign adapter is
    // constructed or installed. The child's typed path views never delete
    // parent-owned attempt bytes before verified process-tree retirement.
    let materialized = match admitted
        .selected_bundles()
        .iter()
        .enumerate()
        .map(|(index, bundle)| {
            bundle
                .as_ref()
                .map(|bundle| {
                    runmat_core::foreign_artifacts::SelectedForeignMaterialization::from_bundle_in_parent_scratch(
                        bundle,
                        scratch.expect("checked selected scratch"),
                        index,
                    ).map(Rc::new)
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, String>>()
    {
        Ok(materialized) => materialized,
        Err(error) => {
            return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
                message: format!("worker could not materialize selected graph bundles: {error}"),
            });
        }
    };
    if let Err(message) = validate_inprocess_python_unit_namespaces(&materialized) {
        return GraphHostExecution::Completed(ProgramExecutionResponse::Failure { message });
    }
    // The shared VM preparation installs the exact invocation snapshot before
    // selected-unit admission. Bootstrap cannot read the worker's environment.
    let (admitted, admitted_bundles) = admitted.into_admitted_with_bundles();
    let mut session = match runmat_core::RunMatSession::with_frozen_environment(
        runtime_config
            .as_ref()
            .is_none_or(|config| config.jit.enabled),
        false,
        Vec::new(),
    ) {
        Ok(session) => session,
        Err(error) => {
            return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
                message: format!("failed to initialize graph execution session: {error}"),
            });
        }
    };
    if let Some(config) = &runtime_config {
        if let Err(error) = session.apply_cluster_runtime_config(config) {
            return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
                message: error.to_string(),
            });
        }
    }
    if let Some(host) = console_host {
        session.install_host_service(host);
    } else {
        session.install_host_service(Rc::new(runmat_runtime::context::NonCapturingPortableHost));
    }
    let runtime =
        session.execution_runtime_context(admitted.request().recipe.program_revision.clone());
    let runtime = if let Some(cancellation) = cancellation {
        let combined = runtime.cancellation().or(&cancellation);
        runtime.with_cancellation_token(combined)
    } else {
        runtime
    };
    let driver = session.local_task_driver();
    let mut unit_hosts = Vec::with_capacity(selected.len());
    let mut foreign_ports = Vec::with_capacity(selected.len());
    // Construct ports without installing any foreign object. The VM can now
    // bind every exact V3 source table once before an installer or its abort
    // cleanup gains the ability to run a MEX lifecycle callback.
    for project in &materialized {
        if let Some(project) = project {
            let host =
                match runmat_core::ForeignUnitHost::new_in_graph_containment(project.as_ref()) {
                    Ok(host) => host,
                    Err(error) => {
                        let _ = session.close_gracefully().await;
                        return GraphHostExecution::Completed(ProgramExecutionResponse::Failure {
                            message: format!(
                                "worker could not initialize selected foreign host: {error}"
                            ),
                        });
                    }
                };
            foreign_ports.push(Some(
                runmat_core::foreign_artifacts::selected_foreign_service(&host, Rc::clone(project)),
            ));
            unit_hosts.push(Some(host));
        } else {
            foreign_ports.push(None);
            unit_hosts.push(None);
        }
    }
    let recorder = runmat_vm::GraphFailureSourceRecorder::new();
    let (prepared, cleanup) =
        match runmat_vm::prepare_graph_request_with_selected_foreign_ports_recording(
            admitted,
            admitted_bundles,
            runtime.clone(),
            runmat_vm::SelectedGraphForeignPorts::new(foreign_ports),
            &recorder,
        ) {
            Ok(prepared) => prepared,
            Err(response) => {
                let _ = session.close_gracefully().await;
                return GraphHostExecution::Completed(response);
            }
        };
    let mut install_error = None;
    for (index, ((unit, project), host)) in selected
        .iter()
        .zip(&materialized)
        .zip(&unit_hosts)
        .enumerate()
    {
        let (Some(project), Some(host)) = (project, host) else {
            continue;
        };
        let result = cleanup
            .with_unit_context(index, |context| async move {
                runmat_core::foreign_artifacts::install_selected_interop(
                    host,
                    &unit.interop,
                    project.as_ref(),
                    Some(&context),
                )
                .await
            })
            .await;
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                install_error = Some(error.into_runtime_error());
                break;
            }
            Err(message) => {
                install_error = Some(
                    runmat_runtime::build_runtime_error(message)
                        .with_identifier("RunMat:SelectedForeignScope")
                        .build(),
                );
                break;
            }
        }
    }
    if let Some(error) = install_error {
        drop(prepared);
        let mut response = recorder.encode_runtime_failure(error, &runtime);
        if let Err(fault) = retire_selected_unit_hosts(
            &unit_hosts,
            &cleanup,
            &driver,
            &recorder,
            &runtime,
            &mut response,
        )
        .await
        {
            return GraphHostExecution::RetirementUnverified {
                message: fault.message,
                owner: Box::new(RetainedGraphHost {
                    _hosts: unit_hosts,
                    _materialized: materialized,
                    _cleanup: cleanup,
                    _session: session,
                    _cause: fault.cause,
                }),
            };
        }
        match session.close_gracefully_retirement().await {
            runmat_core::SessionRetirement::Retained(error) => {
                return GraphHostExecution::RetirementUnverified {
                    message: format!("graph worker session lifecycle could not retire: {error}"),
                    owner: Box::new(RetainedGraphHost {
                        _hosts: unit_hosts,
                        _materialized: materialized,
                        _cleanup: cleanup,
                        _session: session,
                        _cause: Some(error),
                    }),
                };
            }
            runmat_core::SessionRetirement::Retired(Err(error)) => {
                eprintln!("graph worker cleanup failed after execution error: {error}");
            }
            runmat_core::SessionRetirement::Retired(Ok(())) => {}
        }
        drop(cleanup);
        return GraphHostExecution::Completed(response);
    }
    let execution = runmat_vm::execute_prepared_graph_request(prepared);
    let mut response = match driver.participate_foreground(execution).await {
        Ok(response) => response,
        Err(error) => recorder.encode_runtime_failure(error, &runtime),
    };
    // Selected foreign adapters may run authenticated local callbacks during
    // retirement, so close them while the shared local-task host is still live.
    if let Err(fault) = retire_selected_unit_hosts(
        &unit_hosts,
        &cleanup,
        &driver,
        &recorder,
        &runtime,
        &mut response,
    )
    .await
    {
        return GraphHostExecution::RetirementUnverified {
            message: fault.message,
            owner: Box::new(RetainedGraphHost {
                _hosts: unit_hosts,
                _materialized: materialized,
                _cleanup: cleanup,
                _session: session,
                _cause: fault.cause,
            }),
        };
    }
    match session.close_gracefully_retirement().await {
        runmat_core::SessionRetirement::Retained(error) => {
            return GraphHostExecution::RetirementUnverified {
                message: format!("graph worker session lifecycle could not retire: {error}"),
                owner: Box::new(RetainedGraphHost {
                    _hosts: unit_hosts,
                    _materialized: materialized,
                    _cleanup: cleanup,
                    _session: session,
                    _cause: Some(error),
                }),
            };
        }
        runmat_core::SessionRetirement::Retired(Err(error)) => {
            if matches!(&response, ProgramExecutionResponse::GraphSuccess { .. }) {
                response = recorder.encode_runtime_failure(error, &runtime);
            } else {
                eprintln!("graph worker cleanup failed after execution error: {error}");
            }
        }
        runmat_core::SessionRetirement::Retired(Ok(())) => {}
    }
    drop(cleanup);
    GraphHostExecution::Completed(response)
}

async fn retire_selected_unit_hosts(
    hosts: &[Option<runmat_core::ForeignUnitHost>],
    contexts: &runmat_vm::SelectedGraphCleanupContexts,
    driver: &runmat_runtime::context::RuntimeLocalTaskSupervisor,
    recorder: &runmat_vm::GraphFailureSourceRecorder,
    runtime: &runmat_runtime::context::RuntimeContext,
    response: &mut ProgramExecutionResponse,
) -> Result<(), SelectedRetirementFault> {
    for (index, host) in hosts.iter().enumerate() {
        let Some(host) = host else { continue };
        let retirement = contexts
            .with_unit_context(index, |context| host.shutdown_retirement(context, driver))
            .await;
        match retirement {
            Ok(runmat_core::ForeignUnitRetirement::Retired(Ok(()))) => {}
            Ok(runmat_core::ForeignUnitRetirement::Retired(Err(error))) => {
                if matches!(response, ProgramExecutionResponse::GraphSuccess { .. }) {
                    *response = recorder.encode_runtime_failure(error, runtime);
                } else {
                    eprintln!(
                        "selected foreign host cleanup failed after execution error: {error}"
                    );
                }
            }
            Ok(runmat_core::ForeignUnitRetirement::Retained(error)) => {
                return Err(SelectedRetirementFault {
                    message: format!("selected foreign host lifecycle could not retire: {error}"),
                    cause: Some(error),
                });
            }
            Err(message) => {
                return Err(SelectedRetirementFault {
                    message: format!("selected foreign host exact cleanup scope failed: {message}"),
                    cause: None,
                });
            }
        }
    }
    Ok(())
}

fn validate_inprocess_python_unit_namespaces(
    materialized: &[Option<Rc<runmat_core::foreign_artifacts::SelectedForeignMaterialization>>],
) -> Result<(), String> {
    use runmat_core::foreign_artifacts::MaterializedForeignView as _;

    let mut environment = None;
    let mut modules = std::collections::BTreeMap::<String, String>::new();
    for project in materialized {
        let Some(bundle) = project.as_ref().and_then(|project| project.python_bundle()) else {
            continue;
        };
        if bundle.environment.as_ref().is_some_and(|identity| {
            identity.execution_mode == runmat_python::PythonExecutionMode::OutOfProcess
        }) {
            continue;
        }
        if let Some(selected) = &bundle.environment {
            if environment.as_ref().is_some_and(|prior| prior != selected) {
                return Err(
                    "selected in-process Python units require incompatible interpreters; an explicitly admitted isolated mode is required".into(),
                );
            }
            environment = Some(selected.clone());
        }
        for artifact in &bundle.artifacts {
            if let Some(prior) = modules.insert(
                artifact.module.clone(),
                artifact.identity.as_str().to_owned(),
            ) {
                if prior != artifact.identity.as_str() {
                    return Err(
                        "selected in-process Python units contain conflicting module identities; an explicitly admitted isolated mode is required".into(),
                    );
                }
            }
        }
    }
    Ok(())
}

mod program_host;
use program_host::{execute_host_unit_request, HostUnitRequest};

fn execute_test_attempt<'a>(
    request: &'a ProgramExecutionRequest,
    materialized: Option<&'a crate::materialized_project::MaterializedProject>,
    bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
    cancellation: Option<Arc<AtomicBool>>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<WorkerExecution, String>> + 'a>> {
    // Own the test lifecycle independently of the Direct request frame, just
    // as the selected program host owns its interpreter continuation.
    Box::pin(execute_test_attempt_inner(
        request,
        materialized,
        bundle,
        cancellation,
    ))
}

async fn execute_test_attempt_inner(
    request: &ProgramExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
    bundle: Option<Arc<runmat_execution_artifact::ExecutionBundle>>,
    cancellation: Option<Arc<AtomicBool>>,
) -> Result<WorkerExecution, String> {
    let workload = TestAttemptWorkload::from_program_request(request)?;
    let project = materialized.and_then(|materialized| materialized.handoff());
    if project.is_none()
        && (request.recipe.foreign_policy.is_some()
            || request.recipe.interop != runmat_types::InteropManifest::empty())
    {
        return Err("selected test foreign policy requires its exact project bundle".into());
    }
    if project.is_some() && bundle.is_none() {
        return Err("selected test project requires its exact validated bundle".into());
    }
    if let Some(project) = project {
        let revision = project.revision();
        if request.recipe.program_revision.graph_digest().bytes() != revision.graph_digest.bytes()
            || workload.submission.snapshot.base_source_digest
                != revision.source_revision.to_string()
        {
            return Err(
                "test workload base project revision differs from the installed bundle".into(),
            );
        }
    }
    // Validate the selected closure and every rebased private byte before
    // installing any handoff adapter into the session.
    let prepared = match (materialized, project, bundle.as_ref()) {
        (Some(materialized), Some(project), Some(bundle)) => Some(
            runmat_core::foreign_artifacts::PreparedSelectedForeignInput::from_validated_bundle(
                Arc::clone(bundle),
                &request.recipe,
                Arc::new(project.clone()),
                materialized.private_root(),
            )?,
        ),
        _ => None,
    };
    let mut session = runmat_core::RunMatSession::with_frozen_environment(
        true,
        false,
        request.environment.entries().to_vec(),
    )
    .map_err(|error| format!("failed to initialize test execution session: {error}"))?;
    if let Some(bundle) = bundle.as_ref() {
        if let Some(config) = runmat_core::cluster_settings::runtime_config(bundle)? {
            session
                .apply_cluster_runtime_config(&config)
                .map_err(|error| error.to_string())?;
        }
    }
    if let Some(materialized) = materialized {
        session
            .attach_foreign_private_root(materialized.private_root())
            .map_err(|error| format!("failed to bind foreign private bytes: {error}"))?;
    }
    if let Some(project) = project {
        let materialized = materialized.ok_or("selected test project has no materialization")?;
        let original_bundle = bundle
            .as_deref()
            .ok_or("selected test project has no bundle")?;
        let view = crate::materialized_project::SelectedMaterializedView::from_admitted_recipe(
            materialized,
            original_bundle,
            &request.recipe,
        )?;
        let prepared = prepared.ok_or("selected test project has no admitted foreign input")?;
        session
            .install_selected_project_handoff(
                project.clone(),
                prepared,
                &request.recipe.interop,
                &view,
            )
            .await
            .map_err(|error| format!("failed to install exact test project: {error}"))?;
    }
    let execution = session
        .execute_planned_test(
            &workload.submission.snapshot,
            &workload.submission.plan,
            &workload.test_id,
            workload.attempt,
            cancellation
                .map(runmat_execution::CancellationToken::from_shared)
                .unwrap_or_default(),
        )
        .await
        .map_err(|error| error.to_string());
    let shutdown = session
        .shutdown_foreign_runtime()
        .await
        .map_err(|error| format!("failed to shut down test execution session: {error}"));
    match (execution, shutdown) {
        (Ok(execution), Ok(())) => Ok(WorkerExecution {
            result: execution.result,
            events: execution.events,
            coverage: execution.coverage,
        }),
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::process::Command;
    use std::sync::Arc;

    use runmat_execution::value::{InlineValue, PayloadGraph, ValuePayload};
    use runmat_execution::{Digest, ProgramCallable, ProgramFunctionId};
    use runmat_execution_artifact::{
        ExecutableForm, ExecutionBundleBuilder, ForeignArtifactClosure, GraphExecutionRequest,
        GraphInvocationRoots, LogicalObject, ObjectNamespace, ProgramArtifact, ProgramBuildRecipe,
        ProgramExecutionRequest, ProgramExecutionResponse, ProgramTarget,
        GRAPH_EXECUTION_REQUEST_SCHEMA_V11, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    };
    use runmat_test::descriptor::TestSelector;
    use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
    use runmat_test::result::TerminalDisposition;
    use runmat_test_runner::worker::RunSubmission;
    use runmat_test_runner_execution::{decode_execution, TestAttemptWorkload};

    use super::{execute_host_program_request, execute_host_program_request_with_validated_bundle};

    fn original_request(source: &str) -> runmat_execution_artifact::ProgramExecutionRequest {
        let snapshot = FrozenTestRunSnapshot::freeze(
            Digest::sha256(b"graph").to_string(),
            "sha256:base-sources",
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
            Digest::sha256(b"test-config").to_string(),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: "tests/arithmeticTest.m".into(),
                content: source.into(),
            }],
            Vec::new(),
        )
        .unwrap();
        let session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let prepared = session
            .prepare_tests(&snapshot, &TestSelector::default())
            .unwrap();
        let test_id = prepared.plan.tests().next().unwrap().id.clone();
        let workload = TestAttemptWorkload::new(
            RunSubmission::new(prepared.plan, snapshot).unwrap(),
            test_id,
            1,
        )
        .unwrap();
        workload
            .program_request(runmat_types::InteropManifest::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn host_executes_an_exact_test_workload_and_returns_canonical_result() {
        let request = original_request("function tests = arithmeticTest()\n tests = functiontests(localfunctions);\nend\nfunction testAddition(testCase)\n testCase.verifyEqual(1 + 1, 2);\nend\n");
        let response = execute_host_program_request(request).await;
        let ProgramExecutionResponse::Success { outputs } = response else {
            panic!("test-capable host rejected a valid workload: {response:?}");
        };
        let [value] = outputs.as_slice() else {
            panic!("test-capable host returned an invalid output count");
        };
        let execution = decode_execution(value).unwrap();
        assert_eq!(
            execution.result.state.disposition,
            TerminalDisposition::Passed,
            "{execution:#?}"
        );
        assert!(!execution.events.is_empty());
        assert!(!execution.coverage.is_empty());
    }

    #[tokio::test]
    async fn original_test_attempt_uses_request_context_and_rejects_bundle_substitution() {
        let source = "function tests = arithmeticTest()\n tests = functiontests(localfunctions);\nend\nfunction testEnvironment(testCase)\n testCase.verifyEqual(getenv('RUNMAT_TEST_ORIGINAL'), 'captured');\n testCase.verifyEmpty(getenv('HOME'));\n setenv('RUNMAT_TEST_ORIGINAL','child');\nend\n";
        let host = std::env::var_os("RUNMAT_TEST_ORIGINAL");
        let mut request = original_request(source);
        request.environment = runmat_execution::EnvironmentSnapshot::new(vec![(
            "RUNMAT_TEST_ORIGINAL".into(),
            "captured".into(),
        )])
        .unwrap();
        let conflicting = [("RUNMAT_TEST_ORIGINAL".into(), "substituted".into())];
        assert!(matches!(
            super::execute_host_program_request_in_context(
                request.clone(),
                super::HostProgramContext {
                    environment: Some(&conflicting),
                    ..super::HostProgramContext::default()
                },
            )
            .await,
            ProgramExecutionResponse::Failure { .. }
        ));
        let ProgramExecutionResponse::Success { outputs } =
            execute_host_program_request(request).await
        else {
            panic!("original test invocation failed");
        };
        let [value] = outputs.as_slice() else {
            panic!("original test invocation returned an invalid output count");
        };
        assert_eq!(
            decode_execution(value).unwrap().result.state.disposition,
            TerminalDisposition::Passed
        );
        assert_eq!(std::env::var_os("RUNMAT_TEST_ORIGINAL"), host);
    }

    #[tokio::test]
    async fn host_rejects_required_native_interface_without_captured_artifacts_before_execution() {
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::new(
                    "runner-native-interface-test@1",
                    "native_interface.m",
                    "answer = 42;\n",
                ),
                None,
            )
            .await
            .unwrap();
        let interop = runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: runmat_types::ForeignAdapterId::new(
                    runmat_native_ffi::NATIVE_FFI_ADAPTER_ID,
                )
                .unwrap(),
                minimum_version: runmat_native_ffi::NATIVE_FFI_ADAPTER_VERSION,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::NativeCode,
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: vec![runmat_types::ForeignArtifactIdentity::new(
                    "native-ffi:v1:missing-fixture",
                )
                .unwrap()],
            }],
            adapter_contracts: Vec::new(),
        };
        let envelope = unit
            .portable_envelope_for_with_interop(None, interop.clone())
            .unwrap();
        let function = usize::try_from(
            envelope
                .manifest
                .identity
                .entrypoint_function
                .expect("selected native fixture has a callable entrypoint")
                .0,
        )
        .unwrap();
        let entrypoint = envelope.manifest.identity.entrypoint.clone();
        let mut policy_config = runmat_config::runtime::ForeignConfig::default();
        policy_config.native.timeout_ms = Some(std::num::NonZeroU64::new(15_000).unwrap());
        let policy =
            runmat_core::foreign_artifacts::capture_foreign_policy_object(&policy_config).unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: unit.revision().program_revision.clone(),
            entrypoint,
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("runner-native-interface-test"),
            interop,
            foreign_policy: Some(policy.descriptor),
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV4,
            envelope.canonical_bytes().unwrap(),
        )
        .unwrap();
        let response = execute_host_program_request(ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(
                ProgramFunctionId(u32::try_from(function).expect("portable function id")),
                None,
            ),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
        })
        .await;
        let ProgramExecutionResponse::Failure { message } = response else {
            panic!("host executed a program before satisfying its native interface");
        };
        // The captured policy has no immutable native artifacts. Admission must
        // reject that incomplete authority before provider lookup or execution.
        assert_eq!(
            message,
            "selected foreign V3 product has no configured immutable project artifacts"
        );
    }

    #[tokio::test]
    async fn host_executes_a_packaged_mex_module_from_the_exact_bundle() {
        if Command::new(if cfg!(windows) { "gcc" } else { "clang" })
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("remote MEX execution requires a C compiler");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("remote_fixture.c");
        std::fs::write(
            &source,
            r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(73.0);
}
"#,
        )
        .unwrap();
        let output = runmat_mex::MexBuild::new(&source, temp.path())
            .compile()
            .unwrap();
        let manifest_bytes = std::fs::read(&output.manifest).unwrap();
        let module_bytes = std::fs::read(&output.module).unwrap();
        let manifest =
            runmat_mex::MexArtifactManifest::from_canonical_bytes(&manifest_bytes).unwrap();

        let project_root = temp.path().join("project");
        std::fs::create_dir_all(project_root.join("src")).unwrap();
        std::fs::write(
            project_root.join("runmat.toml"),
            "[package]\nname = \"remote-mex-application\"\n[sources]\nroots = [\"src\"]\n",
        )
        .unwrap();
        let project = runmat_package::build_frozen_project(
            &project_root.join("runmat.toml"),
            BTreeSet::new(),
        )
        .unwrap();
        let project_revision = project.revision();
        let revision = runmat_execution::ProgramRevision::new(
            Digest::from_bytes(*project_revision.graph_digest.bytes()),
            Digest::from_bytes(*project_revision.source_revision.bytes()),
            runmat_core::program_environment(runmat_core::CompatMode::RunMat),
        )
        .unwrap();

        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::new(
                    "remote-mex-test@1",
                    "remote_mex.m",
                    "function value = main(); value = remote_fixture(); end\nfunction value = pureHelper(); value = 11; end\n",
                ),
                Some(revision),
            )
            .await
            .unwrap();
        let interop = manifest.interop_manifest();
        let envelope = unit
            .portable_envelope_for_with_interop(Some("main"), interop.clone())
            .unwrap();
        let function = usize::try_from(
            envelope
                .manifest
                .identity
                .entrypoint_function
                .expect("selected MEX fixture has a callable entrypoint")
                .0,
        )
        .unwrap();
        let entrypoint = envelope.manifest.identity.entrypoint.clone();
        let mut policy_config = runmat_config::runtime::ForeignConfig::default();
        // This proves exact bundled native execution, not unsafe preemption of
        // manifested in-process code. Deadline refusal has its own admission test.
        policy_config.mex.timeout_ms = None;
        let policy =
            runmat_core::foreign_artifacts::capture_foreign_policy_object(&policy_config).unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: unit.revision().program_revision.clone(),
            entrypoint,
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("remote-mex-execution"),
            interop,
            foreign_policy: Some(policy.descriptor.clone()),
            accelerators: manifest.accelerator_requirements().unwrap(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let identity = manifest.identity.to_string();
        let root = "mex/fixture";
        let filename = output.module.file_name().unwrap().to_string_lossy();
        let sidecar = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{root}/{filename}.runmat.json"),
            runmat_mex::MEX_ARTIFACT_MANIFEST_MEDIA_TYPE,
            manifest_bytes,
        )
        .unwrap();
        let module = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{root}/{filename}"),
            runmat_mex::MEX_MODULE_MEDIA_TYPE,
            module_bytes,
        )
        .unwrap();
        let closure = ForeignArtifactClosure::new(
            runmat_types::ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID).unwrap(),
            runmat_types::ForeignArtifactIdentity::new(identity).unwrap(),
            vec![sidecar.descriptor.digest, module.descriptor.digest],
        )
        .unwrap();
        let bundle =
            ExecutionBundleBuilder::native(&project, unit.revision().program_revision.clone())
                .unwrap()
                .with_compiled_package_closure()
                .with_foreign_object(sidecar)
                .unwrap()
                .with_foreign_object(module)
                .unwrap()
                .with_foreign_artifact_closure(closure)
                .unwrap()
                .with_foreign_policy_object(policy.clone())
                .unwrap()
                .with_materialized_program(
                    recipe,
                    ExecutableForm::ExecutableUnitV4,
                    envelope.canonical_bytes().unwrap(),
                )
                .build()
                .unwrap();
        std::fs::remove_file(&output.manifest).unwrap();
        std::fs::remove_file(&output.module).unwrap();

        let bundle = Arc::new(bundle);
        let materialized =
            crate::materialized_project::MaterializedProject::from_bundle(&bundle).unwrap();
        let recipe = bundle.manifest.recipes.first().cloned().unwrap();
        let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
        let response = execute_host_program_request_with_validated_bundle(
            ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe,
                artifact,
                callable: ProgramCallable::semantic(
                    ProgramFunctionId(u32::try_from(function).expect("portable function id")),
                    None,
                ),
                context: runmat_execution::ProgramInvocationContext::Direct,
                assignment: None,
                job_id: None,
                arguments: Vec::new(),
                requested_outputs: 1,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console: None,
            },
            Some(&materialized),
            None,
            Some(Arc::clone(&bundle)),
        )
        .await;
        assert_eq!(
            response,
            ProgramExecutionResponse::Success {
                outputs: vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
                    73.0_f64.to_bits(),
                )))],
            }
        );
        let mut archive = Vec::new();
        runmat_execution_artifact::archive::write_bundle(
            &bundle,
            &mut archive,
            runmat_execution_artifact::archive::ArchiveLimits::default(),
        )
        .unwrap();
        let empty_envelope = unit
            .portable_envelope_for_with_interop(
                Some("pureHelper"),
                runmat_types::InteropManifest::empty(),
            )
            .unwrap();
        assert_ne!(
            empty_envelope.manifest.identity.entrypoint_function,
            envelope.manifest.identity.entrypoint_function,
        );
        assert_eq!(
            empty_envelope.manifest.interop,
            runmat_types::InteropManifest::empty(),
        );
        let empty_bytes = empty_envelope.canonical_bytes().unwrap();
        let mut empty_recipe = bundle.manifest.recipes[0].clone();
        empty_recipe.entrypoint = empty_envelope.manifest.identity.entrypoint.clone();
        empty_recipe.interop = runmat_types::InteropManifest::empty();
        empty_recipe.accelerators.clear();
        let empty_bundle =
            ExecutionBundleBuilder::native(&project, unit.revision().program_revision.clone())
                .unwrap()
                .with_compiled_package_closure()
                .with_foreign_policy_object(policy)
                .unwrap()
                .with_materialized_program(
                    empty_recipe,
                    ExecutableForm::ExecutableUnitV4,
                    empty_bytes.clone(),
                )
                .build()
                .unwrap();
        let mut empty_archive = Vec::new();
        runmat_execution_artifact::archive::write_bundle(
            &empty_bundle,
            &mut empty_archive,
            runmat_execution_artifact::archive::ArchiveLimits::default(),
        )
        .unwrap();
        let policy_ref = runmat_execution_artifact::foreign_policy_ref(
            bundle.manifest.recipes[0].foreign_policy.as_ref().unwrap(),
        )
        .unwrap();
        let graph_request = GraphExecutionRequest {
            invocation_controls: None,
            project_runtime_policy: None,
            schema_version: GRAPH_EXECUTION_REQUEST_SCHEMA_V11,
            recipe: bundle.manifest.recipes[0].clone(),
            artifact: bundle.manifest.artifacts[0].clone(),
            retained_units: vec![empty_bytes],
            class_dependencies: Vec::new(),
            class_method_dependencies: Vec::new(),
            static_dependencies: Vec::new(),
            selected_bundle_archives: vec![Some(archive), Some(empty_archive)],
            selected_foreign_policies: vec![Some(policy_ref), Some(policy_ref)],
            graph: PayloadGraph {
                roots: Vec::new(),
                nodes: Vec::new(),
            },
            roots: GraphInvocationRoots::Builtin {
                name: runmat_types::BuiltinId("sum".into()),
                arguments: Vec::new(),
            },
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            requested_outputs: 1,
        };
        let admitted = graph_request.validate_and_admit_bundles().unwrap();
        assert!(admitted[0].is_some());
        assert!(admitted[1].is_some());
        let mut missing = graph_request.clone();
        missing.selected_bundle_archives[1] = None;
        assert!(missing.validate_and_admit_bundles().is_err());
        let mut mismatched = graph_request.clone();
        mismatched.selected_foreign_policies[1] = None;
        assert!(mismatched.validate_and_admit_bundles().is_err());
        let mut tampered = graph_request;
        tampered.selected_bundle_archives[0]
            .as_mut()
            .unwrap()
            .push(0);
        assert!(tampered.validate_and_admit_bundles().is_err());
    }

    #[tokio::test]
    async fn host_executes_a_packaged_java_artifact_from_the_exact_bundle() {
        let Some(java_home) =
            std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(std::path::PathBuf::from)
        else {
            eprintln!("RUNMAT_TEST_JAVA_HOME is unset; remote Java execution was not requested");
            return;
        };
        let javac = java_home
            .join("bin")
            .join(if cfg!(windows) { "javac.exe" } else { "javac" });
        let jar = java_home
            .join("bin")
            .join(if cfg!(windows) { "jar.exe" } else { "jar" });
        if !javac.is_file() || !jar.is_file() {
            eprintln!("remote Java execution requires javac and jar");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let project_root = temp.path().join("project");
        let java_source = temp.path().join("java/fixture/packaged/Value.java");
        let java_classes = temp.path().join("classes");
        std::fs::create_dir_all(java_source.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&java_classes).unwrap();
        std::fs::create_dir_all(project_root.join("src")).unwrap();
        std::fs::create_dir_all(project_root.join("lib")).unwrap();
        std::fs::write(
            &java_source,
            "package fixture.packaged; public final class Value { private Value() {} public static int read() { return 73; } }",
        )
        .unwrap();
        assert!(Command::new(&javac)
            .args(["-d", java_classes.to_str().unwrap()])
            .arg(&java_source)
            .status()
            .unwrap()
            .success());
        let jar_path = project_root.join("lib/fixture.jar");
        assert!(Command::new(&jar)
            .args(["--create", "--file"])
            .arg(&jar_path)
            .args(["-C", java_classes.to_str().unwrap(), "."])
            .status()
            .unwrap()
            .success());
        let source_text = "function value = main(); value = readPackaged() + readPackaged(); end\nfunction value = readPackaged(); value = fixture.packaged.Value.read(); end\n";
        std::fs::write(project_root.join("src/main.m"), source_text).unwrap();
        std::fs::write(
            project_root.join("runmat.toml"),
            "[package]\nname = \"remote-java-application\"\nversion = \"1.0.0\"\n[sources]\nroots = [\"src\"]\n[java-artifacts.fixture]\npath = \"lib/fixture.jar\"\n",
        )
        .unwrap();

        let project = runmat_package::build_frozen_project(
            &project_root.join("runmat.toml"),
            BTreeSet::from([runmat_package::HostCapability::Jvm]),
        )
        .unwrap();
        let project_revision = project.revision();
        let revision = runmat_execution::ProgramRevision::new(
            Digest::from_bytes(*project_revision.graph_digest.bytes()),
            Digest::from_bytes(*project_revision.source_revision.bytes()),
            runmat_core::program_environment(runmat_core::CompatMode::RunMat),
        )
        .unwrap();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        session
            .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
            .await
            .unwrap();
        let primary_path = std::fs::canonicalize(project_root.join("src/main.m"))
            .expect("canonical Java fixture source path");
        let primary = project
            .all_sources()
            .find(|(_, access)| {
                std::fs::canonicalize(access).ok().as_deref() == Some(primary_path.as_path())
            })
            .map(|(descriptor, _)| descriptor.id.clone())
            .expect("Java fixture source belongs to the frozen project");
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::new(
                    primary.package_instance.to_string(),
                    primary.relative_path.to_string(),
                    source_text,
                ),
                Some(revision.clone()),
            )
            .await
            .unwrap();
        let jar_bytes = std::fs::read(&jar_path).unwrap();
        let java_identity = runmat_java::JavaArtifactIdentity::for_bytes(&jar_bytes);
        let interop = runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID).unwrap(),
                minimum_version: runmat_java::JAVA_ADAPTER_VERSION,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: vec![runmat_types::ForeignArtifactIdentity::new(
                    java_identity.to_string(),
                )
                .unwrap()],
            }],
            adapter_contracts: Vec::new(),
        };
        let envelope = unit
            .portable_envelope_for_with_interop(Some("main"), interop.clone())
            .unwrap();
        let function = usize::try_from(
            envelope
                .manifest
                .identity
                .entrypoint_function
                .expect("selected Java fixture has a callable entrypoint")
                .0,
        )
        .unwrap();
        let entrypoint = envelope.manifest.identity.entrypoint.clone();
        let mut policy_config = runmat_config::runtime::ForeignConfig::default();
        policy_config.java.home = Some(java_home);
        let policy =
            runmat_core::foreign_artifacts::capture_foreign_policy_object(&policy_config).unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision.clone(),
            entrypoint,
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("remote-java-execution"),
            interop,
            foreign_policy: Some(policy.descriptor.clone()),
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let java_object = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "java/fixture.jar",
            runmat_java::JAVA_ARCHIVE_MEDIA_TYPE,
            jar_bytes,
        )
        .unwrap();
        let closure = ForeignArtifactClosure::new(
            runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID).unwrap(),
            runmat_types::ForeignArtifactIdentity::new(java_identity.to_string()).unwrap(),
            vec![java_object.descriptor.digest],
        )
        .unwrap();
        let bundle = ExecutionBundleBuilder::native(&project, revision)
            .unwrap()
            .with_compiled_package_closure()
            .with_foreign_object(java_object)
            .unwrap()
            .with_foreign_artifact_closure(closure)
            .unwrap()
            .with_foreign_policy_object(policy)
            .unwrap()
            .with_materialized_program(
                recipe,
                ExecutableForm::ExecutableUnitV4,
                envelope.canonical_bytes().unwrap(),
            )
            .build()
            .unwrap();
        let bundle = Arc::new(bundle);
        let materialized =
            crate::materialized_project::MaterializedProject::from_bundle(&bundle).unwrap();
        let recipe = bundle.manifest.recipes.first().cloned().unwrap();
        let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
        let response = execute_host_program_request_with_validated_bundle(
            ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe,
                artifact,
                callable: ProgramCallable::semantic(
                    ProgramFunctionId(u32::try_from(function).expect("portable function id")),
                    None,
                ),
                context: runmat_execution::ProgramInvocationContext::Direct,
                assignment: None,
                job_id: None,
                arguments: Vec::new(),
                requested_outputs: 1,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console: None,
            },
            Some(&materialized),
            None,
            Some(Arc::clone(&bundle)),
        )
        .await;
        assert_eq!(
            response,
            ProgramExecutionResponse::Success {
                outputs: vec![ValuePayload::Inline(Box::new(InlineValue::I32(146)))],
            }
        );
    }
}
