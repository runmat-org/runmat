use runmat_execution::{
    value::ValuePayload, Digest, OutputContract, ProgramCallable, ProgramEnvironment,
    ProgramFunctionId, ProgramRevision,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_runtime::execution::{
    DeferredCall, DeferredInvocation, DeferredProgram, ExecutionServiceError,
};
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

#[path = "program_execution/callable_admission.rs"]
pub mod callable_admission;

#[path = "program_execution/dynamic_admission.rs"]
pub mod dynamic_admission;

#[path = "program_execution/graph_materialization.rs"]
mod graph_materialization;
#[path = "program_execution/graph_request.rs"]
mod graph_request;
#[path = "program_execution/region_sources.rs"]
mod region_sources;
#[path = "program_execution/request_future.rs"]
mod request_future;
#[path = "program_execution/semantic_identity.rs"]
mod semantic_identity;
#[path = "program_execution/unit_admission.rs"]
mod unit_admission;
#[path = "program_execution/unit_sources.rs"]
mod unit_sources;
pub use crate::bytecode::program::retained_artifacts::{
    prepare_retained_callable_units, PreparedRetainedCallableUnits, RetainedCallableStageError,
    RetainedCallableStageLimits, SelectedFingerprintInvariant,
};
pub use graph_materialization::materialize_deferred_graph_call;
pub use graph_request::{
    compiled_graph_execution_capabilities, execute_graph_request,
    execute_graph_request_with_context, execute_graph_request_with_context_recording,
    execute_graph_request_with_selected_foreign_ports_recording, execute_prepared_graph_request,
    prepare_graph_request_with_selected_foreign_ports_recording, GraphFailureSourceRecorder,
    PreparedSelectedGraph, RestoredSelectedGraph, SelectedGraphCleanupContexts,
    SelectedGraphForeignPorts,
};
pub use region_sources::encode_portable_region_sources_v1;
pub use unit_admission::{
    admit_executable_unit, selected_graph_callable_facts, selected_graph_callable_facts_for_owner,
    AdmittedExecutableUnit, ExecutableUnitAdmissionError, GraphCodecAdmission,
    RetainedExecutableSourceOwner, RuntimeBoundExecutableUnit, SelectedGraphCallableFacts,
    SelectedUnitInteropReceipt,
};

pub(crate) fn capture_region_sources(
    bytecode: &crate::Bytecode,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<Vec<u8>, String> {
    region_sources::capture(bytecode, runtime)
}

pub fn materialize_deferred_call(
    call: &DeferredCall,
    outputs: OutputContract,
    target: runmat_execution_artifact::ProgramTarget,
) -> Result<
    (
        ProgramCallable,
        runmat_execution::ProgramInvocationContext,
        ProgramBuildRecipe,
        ProgramArtifact,
        Vec<ValuePayload>,
    ),
    ExecutionServiceError,
> {
    let (callable, invocation_context, arguments) = match &call.invocation {
        DeferredInvocation::Callable(descriptor) => {
            let runmat_runtime::call::descriptor::CallableTarget::Resolved { identity, .. } =
                &descriptor.target
            else {
                return Err(ExecutionServiceError::Failed(
                    "isolated execution requires a resolved semantic callable".into(),
                ));
            };
            let callable = match identity.as_identity() {
                runmat_hir::CallableIdentity::BoundFunction(function)
                | runmat_hir::CallableIdentity::AnonymousFunction(function)
                | runmat_hir::CallableIdentity::ExternalFunction { function, .. } => {
                    ProgramCallable::semantic(
                        semantic_identity::captured_function(call, *function)
                            .map_err(ExecutionServiceError::Failed)?,
                        descriptor
                            .metadata
                            .display_name
                            .as_ref()
                            .map(ToString::to_string),
                    )
                }
                runmat_hir::CallableIdentity::Builtin(name) => {
                    ProgramCallable::builtin(name.0.clone())
                        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?
                }
                _ => {
                    return Err(ExecutionServiceError::Failed(
                        "isolated execution requires a resolved semantic callable".into(),
                    ))
                }
            };
            (
                callable,
                runmat_execution::ProgramInvocationContext::Direct,
                descriptor.args.as_slice(),
            )
        }
        DeferredInvocation::Program {
            callable,
            context,
            arguments,
            ..
        } => (callable.clone(), context.clone(), arguments.as_slice()),
    };
    let program = match &call.program {
        DeferredProgram::InterpreterV2(program) | DeferredProgram::RegionSourcesV1(program) => {
            program.as_ref()
        }
        DeferredProgram::Local => {
            return Err(ExecutionServiceError::Failed(
                "execution is missing its exact isolated program".into(),
            ))
        }
        DeferredProgram::SelectedGraph(_) => {
            return Err(ExecutionServiceError::Failed(
                "selected graph requires V8 graph materialization".into(),
            ))
        }
    };
    let is_region = matches!(
        &callable,
        ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
    );
    if is_region != matches!(&call.program, DeferredProgram::RegionSourcesV1(_)) {
        return Err(ExecutionServiceError::Failed(
            "isolated callable differs from its typed program form".into(),
        ));
    }
    if is_region && call.program_revision.is_none() {
        return Err(ExecutionServiceError::Failed(
            "isolated region is missing its compiler-bound program revision".into(),
        ));
    }
    let revision = call
        .program_revision
        .clone()
        .unwrap_or_else(|| captured_program_revision(program));
    let accelerators =
        runmat_execution::resource::accelerator_requirements_for_capabilities(&call.capabilities)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: callable.recipe_entrypoint(),
        outputs,
        execution_mode: "interpreter".into(),
        target,
        interop: runmat_types::InteropManifest::empty(),
        accelerators,
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let executable_form = match &call.program {
        DeferredProgram::RegionSourcesV1(_) => ExecutableForm::InterpreterRegionSourcesV1,
        DeferredProgram::InterpreterV2(_) => ExecutableForm::InterpreterBytecodeV2,
        DeferredProgram::Local | DeferredProgram::SelectedGraph(_) => {
            unreachable!("direct materialization selected a validated V2 program above")
        }
    };
    let artifact = ProgramArtifact::materialize(&recipe, executable_form, program.to_vec())
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let arguments = arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::encode_inline_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    Ok((callable, invocation_context, recipe, artifact, arguments))
}

pub async fn execute_deferred_program_in_context(
    call: DeferredCall,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_value::ValueSequence, ExecutionServiceError> {
    let requested_outputs = u16::try_from(call.invocation.requested_outputs())
        .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
    let output_contract = OutputContract {
        requested_outputs,
        owned_work: None,
    };
    let (callable, invocation_context, recipe, artifact, arguments) = materialize_deferred_call(
        &call,
        output_contract,
        runmat_execution_artifact::ProgramTarget::portable("vm-parallel-region-v1"),
    )?;
    let request = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable,
        context: invocation_context,
        assignment: None,
        job_id: None,
        arguments,
        requested_outputs,
        environment: call.environment.clone(),
        console: call.console,
    };
    request
        .validate_for_portable_host()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    match execute_program_request_with_context(request, runtime).await {
        ProgramExecutionResponse::Success { outputs } => {
            let outputs = outputs
                .iter()
                .map(runmat_runtime::execution::value_codec::decode_inline_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
            runmat_value::ValueSequence::comma_separated(outputs)
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
        }
        ProgramExecutionResponse::Failure { message } => {
            Err(ExecutionServiceError::Failed(message))
        }
        ProgramExecutionResponse::RuntimeFailure { failure } => {
            Err(ExecutionServiceError::RuntimeFailure(Box::new(failure)))
        }
        ProgramExecutionResponse::ExternalizedSuccess { .. } => Err(ExecutionServiceError::Failed(
            "parallel region returned an externalized result to an inline executor".into(),
        )),
        ProgramExecutionResponse::SpmdSuccess { .. } => Err(ExecutionServiceError::Failed(
            "SPMD task response requires the typed gang execution path".into(),
        )),
        ProgramExecutionResponse::GraphSuccess { .. } => Err(ExecutionServiceError::Failed(
            "graph result reached the direct V7 execution path".into(),
        )),
    }
}

/// Execute a locally scheduled task against the same selected V3 graph and
/// result receipt used by an isolated worker. The issuing context, not the
/// service's `requires_program_capture` hint, authenticates the product and
/// decodes its result.
pub async fn execute_deferred_graph_in_context(
    call: &DeferredCall,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError> {
    let requested_outputs = u16::try_from(call.invocation.requested_outputs())
        .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
    let outputs = OutputContract {
        requested_outputs,
        owned_work: None,
    };
    let (_, request, receipt) = materialize_deferred_graph_call(
        call,
        outputs,
        runmat_execution_artifact::ProgramTarget::portable("vm-parallel-graph-v1"),
        runmat_execution::value::ValueLimits::default(),
    )?;
    receipt.check_issuing_context(&runtime)?;
    match execute_graph_request_with_context(request, runtime.clone()).await {
        ProgramExecutionResponse::GraphSuccess { graph } => {
            let result = receipt.decode_in_context(&runtime, &graph)?;
            if result.as_slice().len() != usize::from(requested_outputs) {
                return Err(ExecutionServiceError::InvalidOutputContract);
            }
            Ok(result)
        }
        ProgramExecutionResponse::RuntimeFailure { failure } => {
            Err(ExecutionServiceError::LocalizedRuntimeFailure(
                receipt
                    .decode_failure_in_context(&runtime, &failure)?
                    .into(),
            ))
        }
        ProgramExecutionResponse::Failure { message } => {
            Err(ExecutionServiceError::Failed(message))
        }
        _ => Err(ExecutionServiceError::Infrastructure(
            "selected graph returned an incompatible parallel response".into(),
        )),
    }
}

fn captured_program_revision(program: &[u8]) -> ProgramRevision {
    let digest = Digest::sha256(program);
    ProgramRevision::new(
        digest,
        digest,
        ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            runmat_runtime::native::linked_runtime_fingerprint(),
            Digest::sha256(b"runmat-local-captured-catalog-v1"),
            "matlab",
        )
        .expect("captured execution compatibility constants are valid"),
    )
    .expect("captured program revision is valid")
}

pub async fn execute_program_request(request: ProgramExecutionRequest) -> ProgramExecutionResponse {
    // A worker request is an execution host, not a bare RuntimeContext
    // embedding. Keep the workspace and non-cloneable lifecycle owners outside
    // the context and drive the local-task supervisor through completion.
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let workspace_owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let workspace_service = std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
        workspace_owner.clone(),
    ));
    #[cfg(not(target_arch = "wasm32"))]
    let supervisor = std::rc::Rc::new(runmat_runtime::context::RuntimeLocalTaskSupervisor::new());
    #[cfg(target_arch = "wasm32")]
    let supervisor = std::rc::Rc::new(runmat_runtime::context::BrowserLocalTaskSupervisor::new());
    let ports = runtime
        .service_ports()
        .clone()
        .with_workspace(workspace_service.clone())
        .with_base_workspace(workspace_service)
        .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
    let runtime = runtime.with_service_ports(ports);
    let host = match runmat_runtime::context::RuntimeLifecycleHost::new(
        &runtime,
        supervisor.clone(),
    ) {
        Ok(host) => host,
        Err(error) => {
            let failure = runtime_failure_response(error, &runtime);
            if let Err(close_error) = workspace_owner.close() {
                tracing::warn!(%close_error, "workspace close failed after worker host admission error");
            }
            return failure;
        }
    };
    let complete = async {
        let result = execute_program_request_in_context(request, Some(runtime.clone()), None).await;
        match host.retire_workspace_and_drain(&runtime).await {
            Ok(()) => result,
            Err(error)
                if matches!(
                    &result,
                    ProgramExecutionResponse::Failure { .. }
                        | ProgramExecutionResponse::RuntimeFailure { .. }
                ) =>
            {
                tracing::warn!(%error, "cleanup callback failed after worker execution error");
                result
            }
            Err(error) => runtime_failure_response(error, &runtime),
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let result = match supervisor.participate_foreground(complete).await {
        Ok(result) => result,
        Err(error) => runtime_failure_response(error, &runtime),
    };
    #[cfg(target_arch = "wasm32")]
    let result = complete.await;
    host.force_close();
    match workspace_owner.close() {
        Ok(_) => result,
        Err(error)
            if matches!(
                &result,
                ProgramExecutionResponse::Failure { .. }
                    | ProgramExecutionResponse::RuntimeFailure { .. }
            ) =>
        {
            tracing::warn!(%error, "workspace close failed after worker execution error");
            result
        }
        Err(error) => runtime_failure_response(
            runmat_runtime::workspace::base_workspace_error_to_runtime(error),
            &runtime,
        ),
    }
}

pub async fn execute_program_request_with_context(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    execute_program_request_on_owned_local_task(request, runtime, None).await
}

/// Execute a Direct V3 request with the exact authenticated foreign closure
/// selected before adapter effects. The boxed bundle keeps the large archive
/// owner out of the caller's async frame until VM source binding consumes it.
pub async fn execute_program_request_with_context_and_bundle(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    bundle: Box<runmat_execution_artifact::ExecutionBundle>,
) -> ProgramExecutionResponse {
    execute_program_request_on_owned_local_task(request, runtime, Some(bundle)).await
}

/// Keep the deep interpreter poll chain in one host-owned local task instead
/// of nesting it beneath the host's foreground response/cleanup future. The
/// existing canonical supervisor owns cancellation and drops the exact request
/// and bundle before its completion lease settles. A bare runtime without a
/// supervisor retains the direct path.
async fn execute_program_request_on_owned_local_task(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    bundle: Option<Box<runmat_execution_artifact::ExecutionBundle>>,
) -> ProgramExecutionResponse {
    let selected_runtime = runtime.clone();
    let execution = async move {
        Ok::<_, runmat_runtime::RuntimeError>(
            execute_program_request_in_context(request, Some(selected_runtime), bundle).await,
        )
    };
    match runmat_runtime::context::run_owned_request_continuation(&runtime, execution).await {
        Ok(response) => response,
        Err(error) => runtime_failure_response(error, &runtime),
    }
}

async fn execute_program_request_in_context(
    request: ProgramExecutionRequest,
    runtime: Option<runmat_runtime::context::RuntimeContext>,
    bundle: Option<Box<runmat_execution_artifact::ExecutionBundle>>,
) -> ProgramExecutionResponse {
    if request.validate_for_portable_host().is_err() {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a protocol or program identity mismatch".into(),
        };
    }
    let runtime = match runtime
        .map(|runtime| runtime.with_invocation_environment(&request.environment))
        .transpose()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected the invocation environment: {error}"),
            }
        }
    };
    if let Some(controls) = request.console {
        let Some(runtime) = &runtime else {
            return ProgramExecutionResponse::Failure {
                message: "requested console capture requires an owning runtime host".into(),
            };
        };
        if let Err(error) = runtime.validate_invocation_console(Some(controls)) {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected the invocation console: {error}"),
            };
        }
    }
    if let Err(error) = request.recipe.program_revision.validate_current_compiler() {
        return ProgramExecutionResponse::Failure {
            message: format!("worker rejected compiler-incompatible program: {error}"),
        };
    }
    if request.artifact.form == ExecutableForm::TestAttemptV1 {
        return ProgramExecutionResponse::Failure {
            message: "test-attempt programs require a test-capable execution host".into(),
        };
    }
    if request.artifact.form == ExecutableForm::MeshingWorkload {
        return ProgramExecutionResponse::Failure {
            message: "meshing workloads require a meshing-capable execution host".into(),
        };
    }
    if matches!(
        request.artifact.form,
        ExecutableForm::AnalysisWorkload | ExecutableForm::GeometryWorkload
    ) {
        return ProgramExecutionResponse::Failure {
            message: "domain workloads require an analysis- or geometry-capable execution host"
                .into(),
        };
    }
    if request.artifact.form == ExecutableForm::NativeObjectV1 {
        return ProgramExecutionResponse::Failure {
            message: "native object programs require a native AOT execution host".into(),
        };
    }
    let runtime = runtime.unwrap_or_else(|| {
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
    });
    let _assignment = runtime.enter_execution_assignment(request.assignment.clone());
    let _job = runtime.enter_execution_job(request.job_id);
    match request_future::prepare(request, runtime, bundle) {
        Ok(execution) => execution.await,
        Err(error) => error.into_response(),
    }
}

fn recipe_supports_capabilities(
    recipe: &ProgramBuildRecipe,
    capabilities: &runmat_types::CapabilitySet,
) -> bool {
    runmat_execution::resource::accelerator_requirements_for_capabilities(capabilities).is_ok_and(
        |required| {
            runmat_execution::resource::accelerator_requests_satisfy_requirements(
                &recipe.accelerators,
                &required,
            )
        },
    )
}

fn callable_capabilities(
    registry: &crate::FunctionRegistry,
    callable: &ProgramCallable,
    origin_program: Option<Digest>,
) -> Result<runmat_types::CapabilitySet, String> {
    let function = |id: ProgramFunctionId| {
        semantic_identity::resolve(registry, id, origin_program)
            .ok()
            .and_then(|id| registry.get(id))
    };
    match callable {
        ProgramCallable::Semantic { function: id, .. } => function(*id)
            .map(|function| function.capabilities.clone())
            .ok_or_else(|| "worker could not resolve callable requirements".into()),
        ProgramCallable::Builtin { name } if runmat_builtins::builtin_name_is_known(&name.0) => {
            Ok(runmat_builtins::builtin_required_capabilities(&name.0))
        }
        ProgramCallable::Builtin { .. } => {
            Err("worker could not resolve builtin requirements".into())
        }
        ProgramCallable::GraphTarget { .. } => {
            Err("graph scheduler labels are not direct executable callables".into())
        }
        ProgramCallable::ParallelRegion { region } => function(region.0.function)
            .and_then(|function| {
                function
                    .parfor_regions
                    .iter()
                    .find(|candidate| candidate.contract.id == *region)
            })
            .map(|region| region.contract.capabilities.clone())
            .ok_or_else(|| "worker could not resolve parfor region requirements".into()),
        ProgramCallable::SpmdRegion { region } => function(region.0.function)
            .and_then(|function| {
                function
                    .spmd_regions
                    .iter()
                    .find(|candidate| candidate.contract.id == *region)
            })
            .map(|region| region.contract.capabilities.clone())
            .ok_or_else(|| "worker could not resolve SPMD region requirements".into()),
    }
}

async fn execute_spmd_region_request(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    let ProgramCallable::SpmdRegion { region } = &request.callable else {
        return ProgramExecutionResponse::Failure {
            message: "worker received a non-SPMD callable on the SPMD execution path".into(),
        };
    };
    let runmat_execution::ProgramInvocationContext::SpmdTask { task } = &request.context else {
        return ProgramExecutionResponse::Failure {
            message: "SPMD region task is missing its typed execution context".into(),
        };
    };
    let collective = match runtime
        .service_ports()
        .require_collective("isolated SPMD execution")
    {
        Ok(collective) if collective.context() == task => collective.clone(),
        Ok(_) => {
            return ProgramExecutionResponse::Failure {
                message: "worker collective context differs from its SPMD task identity".into(),
            }
        }
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: error.to_string(),
            }
        }
    };
    let (bytecode, runtime) = match region_sources::bind(
        &request.artifact.executable_bytes,
        runtime,
        &request.recipe.program_revision,
    ) {
        Ok(bound) => bound,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected invalid SPMD sources: {error}"),
            }
        }
    };
    let Some(executable) = bytecode
        .spmd_regions
        .iter()
        .find(|candidate| candidate.contract.id == *region)
        .cloned()
    else {
        return ProgramExecutionResponse::Failure {
            message: "worker could not find the requested SPMD region in its exact program".into(),
        };
    };
    if !recipe_supports_capabilities(&request.recipe, &executable.contract.capabilities) {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a recipe that weakens SPMD accelerator requirements".into(),
        };
    }
    if usize::from(request.requested_outputs) != executable.outputs.len() {
        return ProgramExecutionResponse::Failure {
            message: "SPMD task output count differs from its compiler-bound region".into(),
        };
    }
    let mut cooperation = runtime.cpu_cooperation();
    let arguments =
        match runmat_runtime::execution::value_codec::decode_inline_values_cooperatively_in_context(
            &request.arguments,
            runmat_execution::value::ValueLimits::default(),
            None,
            Some(&mut cooperation),
            &runtime,
        )
        .await
        {
            Ok(arguments) => arguments,
            Err(runmat_runtime::execution::value_codec::CooperativeValueCodecError::Execution(
                error,
            )) => {
                return runtime_failure_response(error.into_runtime_error(), &runtime);
            }
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker rejected an invalid SPMD capture: {error}"),
                }
            }
        };
    if arguments.len() != executable.captures.len() {
        return ProgramExecutionResponse::Failure {
            message: format!(
                "SPMD task received {} captures; its executable contract requires {}",
                arguments.len(),
                executable.captures.len()
            ),
        };
    }
    let mut arguments = match arguments.try_into_values() {
        Ok(arguments) => arguments,
        Err(error) => return runtime_failure_response(error, &runtime),
    };
    let mut frame = crate::interpreter::vars::InterpreterVars::from_owned(vec![
        runmat_value::Value::Num(0.0);
        bytecode.var_count
    ]);
    let mut assigned_slots = HashSet::new();
    // Borrow the consuming iterator so its original root lease remains live
    // after the final capture moves into the frame, through VM publication.
    for (capture, value) in executable.captures.iter().zip(&mut arguments) {
        if frame.get(capture.slot).is_none() {
            return ProgramExecutionResponse::Failure {
                message: "SPMD capture lies outside its compiler-bound VM frame".into(),
            };
        };
        let _retired = frame.replace_owned(capture.slot, value);
        assigned_slots.insert(capture.slot);
    }
    let services = runtime.service_ports().clone().with_collective(collective);
    let runtime = runtime.with_service_ports(services);
    let callable_name = request.callable.display_name();
    let result = crate::interpreter::runner::interpret_resume_in_context(
        &bytecode,
        crate::InterpreterResumeState {
            pc: executable.body.pc,
            completion_boundary: Some(
                crate::interpreter::state::InterpreterCompletionBoundary::before(
                    executable.exit.pc,
                ),
            ),
            vars: frame,
            assigned_slots,
            supplied_inputs: 0,
            requested_outputs: 0,
            missing_input_slots: HashSet::new(),
            global_aliases: HashMap::new(),
            persistent_aliases: HashMap::new(),
            side_effect_epoch: 0,
        },
        Some(&callable_name),
        runtime.clone(),
    )
    .await;
    match result {
        Ok(crate::InterpreterOutcome::Completed(completion)) => {
            let mut outputs = Vec::with_capacity(executable.outputs.len());
            let mut failure = None;
            for output in &executable.outputs {
                if !completion.assigned_slots.contains(&output.slot) {
                    outputs.push(None);
                    continue;
                }
                let Some(value) = completion.values.get(output.slot) else {
                    failure =
                        Some("SPMD output lies outside its compiler-bound VM frame".to_string());
                    break;
                };
                match crate::interpreter::dispatch::encode_spmd_interpreter_output(&runtime, value)
                    .await
                {
                    Ok(value) => outputs.push(Some(value)),
                    Err(error) => {
                        failure = Some(error.to_string());
                        break;
                    }
                }
            }
            let outputs = failure.map_or(Ok(outputs), Err);
            match outputs {
                Ok(outputs) => ProgramExecutionResponse::SpmdSuccess { outputs },
                Err(message) => ProgramExecutionResponse::Failure { message },
            }
        }
        Err(error) => runtime_failure_response(error, &runtime),
    }
}

async fn execute_parallel_region_request(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    if !matches!(&request.callable, ProgramCallable::ParallelRegion { .. }) {
        return ProgramExecutionResponse::Failure {
            message: "worker received a non-parallel callable on the parallel execution path"
                .into(),
        };
    }
    if !matches!(
        &request.context,
        runmat_execution::ProgramInvocationContext::ParallelTask { .. }
    ) {
        return ProgramExecutionResponse::Failure {
            message: "parallel region task is missing its typed execution context".into(),
        };
    }
    if request.requested_outputs != 1 {
        return ProgramExecutionResponse::Failure {
            message: "parallel region tasks return one typed result envelope".into(),
        };
    }
    let (bytecode, runtime) = match region_sources::bind(
        &request.artifact.executable_bytes,
        runtime,
        &request.recipe.program_revision,
    ) {
        Ok(bound) => bound,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected invalid parallel sources: {error}"),
            }
        }
    };
    let mut cooperation = runtime.cpu_cooperation();
    let arguments =
        match runmat_runtime::execution::value_codec::decode_inline_values_cooperatively_in_context(
            &request.arguments,
            runmat_execution::value::ValueLimits::default(),
            None,
            Some(&mut cooperation),
            &runtime,
        )
        .await
        {
            Ok(arguments) => arguments,
            Err(runmat_runtime::execution::value_codec::CooperativeValueCodecError::Execution(
                error,
            )) => {
                return runtime_failure_response(error.into_runtime_error(), &runtime);
            }
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker rejected an invalid parallel task input: {error}"),
                }
            }
        };
    match execute_parallel_region_values(
        &request.callable,
        &request.context,
        request.requested_outputs,
        &request.recipe,
        &bytecode,
        &runtime,
        arguments,
    )
    .await
    {
        Ok(value) => match runmat_runtime::execution::value_codec::encode_inline_value_in_context(
            value.value(),
            &runtime,
        ) {
            Ok(value) => ProgramExecutionResponse::Success {
                outputs: vec![value],
            },
            Err(error) => ProgramExecutionResponse::Failure {
                message: format!("worker could not transfer its parallel task result: {error}"),
            },
        },
        Err(ParallelRegionError::Contract(response)) => response,
        Err(ParallelRegionError::Runtime(error)) => runtime_failure_response(error, &runtime),
    }
}

/// One typed region/task engine shared by inline V1 captures and admitted V3
/// graph captures. Both callers must bind their own executable sources before
/// supplying values; this function never interprets transport identities.
enum ParallelRegionError {
    Contract(ProgramExecutionResponse),
    Runtime(runmat_runtime::RuntimeError),
}

async fn execute_parallel_region_values(
    callable: &ProgramCallable,
    context: &runmat_execution::ProgramInvocationContext,
    requested_outputs: u16,
    recipe: &ProgramBuildRecipe,
    bytecode: &crate::Bytecode,
    runtime: &runmat_runtime::context::RuntimeContext,
    arguments: runmat_runtime::execution::RootedValues,
) -> Result<runmat_runtime::execution::RootedValue, ParallelRegionError> {
    let ProgramCallable::ParallelRegion { region } = callable else {
        return Err(ParallelRegionError::Contract(
            ProgramExecutionResponse::Failure {
                message: "worker received a non-parallel callable on the parallel execution path"
                    .into(),
            },
        ));
    };
    let runmat_execution::ProgramInvocationContext::ParallelTask { task } = context else {
        return Err(ParallelRegionError::Contract(
            ProgramExecutionResponse::Failure {
                message: "parallel region task is missing its typed execution context".into(),
            },
        ));
    };
    if requested_outputs != 1 {
        return Err(ParallelRegionError::Contract(
            ProgramExecutionResponse::Failure {
                message: "parallel region tasks return one typed result envelope".into(),
            },
        ));
    }
    let Some(executable) = bytecode
        .parfor_regions
        .iter()
        .find(|candidate| candidate.contract.id == *region)
    else {
        return Err(ParallelRegionError::Contract(
            ProgramExecutionResponse::Failure {
                message: "worker could not find the requested parallel region in its exact program"
                    .into(),
            },
        ));
    };
    if !recipe_supports_capabilities(recipe, &executable.contract.capabilities) {
        return Err(ParallelRegionError::Contract(
            ProgramExecutionResponse::Failure {
                message: "worker rejected a recipe that weakens parfor accelerator requirements"
                    .into(),
            },
        ));
    }
    let expected_arguments = executable.input_variables().count() + 1;
    if arguments.len() != expected_arguments {
        return Err(ParallelRegionError::Contract(ProgramExecutionResponse::Failure {
            message: format!(
                "parallel task received {} arguments; its executable contract requires {expected_arguments}",
                arguments.len()
            ),
        }));
    }
    arguments
        .with_owned_output_storage_async(|storage| async move {
            let mut arguments = storage.try_into_owned().map_err(|_| {
                ParallelRegionError::Runtime(runmat_runtime::runtime_error::semantic_error(
                    "ParallelTaskInput",
                    "parallel arguments require their original unique collection owner",
                ))
            })?;
            let Some(Value::Cell(iterations)) = arguments.try_remove(0) else {
                return Err(ParallelRegionError::Contract(
                    ProgramExecutionResponse::Failure {
                        message: "parallel task iteration input must be a cell vector".into(),
                    },
                ));
            };
            let inputs = arguments.try_into_storage().map_err(|_| {
                ParallelRegionError::Runtime(runmat_runtime::runtime_error::semantic_error(
                    "ParallelTaskInput",
                    "parallel arguments could not republish their original collection owner",
                ))
            })?;
            if iterations.data().len() != task.chunk.len {
                return Err(ParallelRegionError::Contract(
                    ProgramExecutionResponse::Failure {
                        message: "parallel task iteration payload differs from its declared chunk"
                            .into(),
                    },
                ));
            }
            let callable_name = callable.display_name();
            let result = crate::interpreter::runner::interpret_parfor_task_in_context(
                crate::interpreter::runner::ParforTaskExecution {
                    bytecode_owner: None,
                    bytecode,
                    region: executable,
                    inputs,
                    iterations: iterations.into_parts().0,
                    input_roots: None,
                    randomness: task.randomness.clone(),
                    mode: crate::interpreter::runner::ParforTaskMode::IndependentIterations,
                    current_function_name: &callable_name,
                    runtime: runtime.clone(),
                },
            )
            .await
            .and_then(|outputs| {
                outputs.with_owned_output_storage(|outputs| {
                    runmat_value::CellArray::from_shared_storage(
                        outputs,
                        runmat_value::SharedShape::new_unmanaged(vec![
                            1,
                            executable.output_variables().count(),
                        ]),
                    )
                    .map(Value::Cell)
                    .map_err(|error| {
                        runmat_runtime::runtime_error::semantic_error(
                            "ParallelTaskOutput",
                            error.to_string(),
                        )
                    })
                    .and_then(|value| {
                        runmat_runtime::execution::RootedValue::new(value)
                            .map_err(|error| error.into_runtime_error())
                    })
                })
            });
            result.map_err(ParallelRegionError::Runtime)
        })
        .await
}

fn unsupported_executable_revisions(
    revisions: &runmat_execution::ExecutableComponentRevisions,
) -> Option<String> {
    let supported = revisions.mir_schema == runmat_mir::MIR_SCHEMA_VERSION
        && revisions.analysis_schema == runmat_mir::analysis::ANALYSIS_STORE_SCHEMA_VERSION
        && revisions.bytecode_schema == crate::BYTECODE_SCHEMA_VERSION
        && revisions.vm_layout_schema == crate::VM_LAYOUT_SCHEMA_VERSION
        && revisions.function_registry_schema == crate::FUNCTION_REGISTRY_SCHEMA_VERSION
        && revisions.source_map_schema == 2
        && revisions.contract_schema == runmat_types::RUNMAT_TYPES_SCHEMA.major
        && u32::from(revisions.catalog_schema) == runmat_builtins::BUILTIN_CATALOG_SCHEMA;
    (!supported).then(|| {
        format!(
            "worker rejected stale executable component revisions: MIR actual {} expected {}; analysis actual {} expected {}; bytecode actual {} expected {}; VM layout actual {} expected {}; function registry actual {} expected {}; source map actual {} expected 2; contract actual {} expected {}; catalog actual {} expected {}. Rebuild the program with this RunMat version",
            revisions.mir_schema,
            runmat_mir::MIR_SCHEMA_VERSION,
            revisions.analysis_schema,
            runmat_mir::analysis::ANALYSIS_STORE_SCHEMA_VERSION,
            revisions.bytecode_schema,
            crate::BYTECODE_SCHEMA_VERSION,
            revisions.vm_layout_schema,
            crate::VM_LAYOUT_SCHEMA_VERSION,
            revisions.function_registry_schema,
            crate::FUNCTION_REGISTRY_SCHEMA_VERSION,
            revisions.source_map_schema,
            revisions.contract_schema,
            runmat_types::RUNMAT_TYPES_SCHEMA.major,
            revisions.catalog_schema,
            runmat_builtins::BUILTIN_CATALOG_SCHEMA,
        )
    })
}

async fn execute_function_request(
    request: ProgramExecutionRequest,
    registry: crate::FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    selected_function: Option<runmat_hir::FunctionId>,
) -> ProgramExecutionResponse {
    let arguments = match request
        .arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::decode_inline_value)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(arguments) => arguments,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid argument: {error}"),
            }
        }
    };
    let requested_outputs = usize::from(request.requested_outputs);
    let result = match &request.callable {
        runmat_execution::ProgramCallable::Semantic { function, .. } => {
            let local = match selected_function.map(Ok).unwrap_or_else(|| {
                let origin_program = semantic_identity::revision_owner(
                    &request.recipe.program_revision,
                    &request.artifact.executable_bytes,
                )?;
                semantic_identity::resolve(&registry, *function, origin_program)
            }) {
                Ok(local) => local,
                Err(message) => return ProgramExecutionResponse::Failure { message },
            };
            crate::invoke_semantic_function_value_in_context(
                local.0,
                &arguments,
                requested_outputs,
                &registry,
                runtime.clone(),
            )
            .await
        }
        runmat_execution::ProgramCallable::Builtin { name } => {
            let descriptor = runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                runmat_hir::CallableIdentity::Builtin(name.clone()),
                arguments,
                requested_outputs,
                runmat_hir::CallableFallbackPolicy::None,
                runmat_runtime::call::descriptor::CallableCallKind::Direct,
            );
            runtime
                .scope(crate::call::execute_callable_descriptor(
                    &runtime, descriptor,
                ))
                .await
        }
        runmat_execution::ProgramCallable::ParallelRegion { .. }
        | runmat_execution::ProgramCallable::SpmdRegion { .. }
        | runmat_execution::ProgramCallable::GraphTarget { .. } => {
            return ProgramExecutionResponse::Failure {
                message:
                    "structured parallel callable reached the semantic function execution path"
                        .into(),
            }
        }
    };
    match result {
        Ok(sequence) => match sequence
            .iter()
            .map(|value| {
                runmat_runtime::execution::value_codec::encode_inline_value_in_context(
                    value, &runtime,
                )
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(outputs) => ProgramExecutionResponse::Success { outputs },
            Err(error) => ProgramExecutionResponse::Failure {
                message: format!("worker could not transfer its result: {error}"),
            },
        },
        Err(error) => runtime_failure_response(error, &runtime),
    }
}

async fn execute_unit_script(
    bytecode: crate::Bytecode,
    requested_outputs: u16,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    if requested_outputs > 1 {
        return ProgramExecutionResponse::Failure {
            message: "scripts do not define multiple output ordinals".into(),
        };
    }
    let result = crate::interpret_workspace(
        &bytecode,
        crate::interpreter::vars::InterpreterVars::from_owned(Vec::new()),
        Some("<main>"),
        runtime.clone(),
        None,
        None,
    )
    .await
    .outcome;
    match result {
        Ok(crate::InterpreterOutcome::Completed(completion)) => {
            if requested_outputs == 0 {
                return ProgramExecutionResponse::Success {
                    outputs: Vec::new(),
                };
            }
            // Allocated slots are not results: never fabricate an unassigned `ans`.
            let Some(value) = completion
                .ans_slot
                .filter(|slot| completion.assigned_slots.contains(slot))
                .and_then(|slot| completion.values.get(slot).map(|value| value.clone()))
            else {
                return ProgramExecutionResponse::Failure {
                    message: "script did not produce the requested ans value".into(),
                };
            };
            match runmat_runtime::execution::value_codec::encode_inline_value_in_context(
                &value, &runtime,
            ) {
                Ok(value) => ProgramExecutionResponse::Success {
                    outputs: vec![value],
                },
                Err(error) => ProgramExecutionResponse::Failure {
                    message: format!("worker could not transfer its result: {error}"),
                },
            }
        }
        Err(error) => runtime_failure_response(error, &runtime),
    }
}

pub fn runtime_failure_response(
    error: runmat_runtime::RuntimeError,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    match runmat_runtime::execution::encode_runtime_failure_in_context(&error, runtime) {
        Ok(failure) => ProgramExecutionResponse::RuntimeFailure { failure },
        Err(protocol_error) => ProgramExecutionResponse::Failure {
            message: format!(
                "worker could not encode its structured runtime failure: {protocol_error}"
            ),
        },
    }
}

async fn execute_script_request(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    let bytecode = match crate::decode_interpreter_script_v2(&request.artifact.executable_bytes) {
        Ok(bytecode) => bytecode,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid script program: {error}"),
            }
        }
    };
    execute_unit_script(bytecode, request.requested_outputs, runtime).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_execution::{
        Digest, OutputContract, ProgramCallable, ProgramEnvironment, ProgramFunctionId,
        ProgramRevision,
    };
    use runmat_execution_artifact::{
        ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
        ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    };
    use runmat_value::Value;

    use super::{
        admit_executable_unit, execute_program_request, recipe_supports_capabilities,
        ExecutableUnitAdmissionError,
    };
    use crate::{Bytecode, Instr};

    fn script_request(
        environment: ProgramEnvironment,
        form: ExecutableForm,
        executable_bytes: Vec<u8>,
    ) -> ProgramExecutionRequest {
        let revision = ProgramRevision::new(
            Digest::sha256(b"fixture-graph"),
            Digest::sha256(b"fixture-source"),
            environment,
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "script".into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-script-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(&recipe, form, executable_bytes).unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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
        }
    }

    fn executable_unit_request(bytes: &[u8]) -> ProgramExecutionRequest {
        let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(bytes)
            .expect("frozen executable-unit fixture remains canonically decodable");
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: envelope.manifest.identity.program.clone(),
            entrypoint: envelope.manifest.identity.entrypoint.clone(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("frozen-unit-test"),
            interop: envelope.manifest.interop.clone(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact =
            ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, bytes.to_vec())
                .unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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
        }
    }

    fn dependency_free_frozen_envelope(frozen: &[u8]) -> runmat_execution::ExecutableUnitEnvelope {
        // These historical fixtures predate selected dependency products. Lift
        // only that absent transport shape, leaving compiler/component revisions
        // and every opaque component byte unchanged for independent rejection.
        let mut envelope: serde_json::Value =
            serde_json::from_slice(frozen).expect("frozen envelope JSON");
        let manifest = envelope["manifest"]
            .as_object_mut()
            .expect("frozen manifest");
        assert!(!manifest.contains_key("dependencies"));
        manifest.insert("dependencies".into(), serde_json::json!([]));
        let envelope_object = envelope.as_object_mut().expect("frozen envelope");
        assert!(!envelope_object.contains_key("dependency_products"));
        envelope_object.insert("dependency_products".into(), serde_json::json!([]));
        serde_json::from_value(envelope).expect("dependency-free frozen envelope shape")
    }

    fn with_current_envelope_schema(frozen: &[u8]) -> Vec<u8> {
        let mut envelope = dependency_free_frozen_envelope(frozen);
        envelope.schema_version = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION;
        envelope.manifest.schema_version = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION;
        envelope.canonical_bytes().expect("schema-lifted envelope")
    }

    fn with_stale_envelope_schema(frozen: &[u8]) -> Vec<u8> {
        let mut envelope = dependency_free_frozen_envelope(frozen);
        let stale = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION
            .checked_sub(1)
            .expect("current envelope has a predecessor");
        envelope.schema_version = stale;
        envelope.manifest.schema_version = stale;
        // Deliberately invalid input cannot use the validating canonical encoder.
        serde_json::to_vec(&envelope).expect("encode stale-header fixture")
    }

    #[test]
    fn exact_script_program_executes_top_level_bytecode() {
        let mut bytecode = Bytecode::with_instructions(
            vec![Instr::LoadConst(42.0), Instr::StoreVar(0), Instr::Return],
            1,
        );
        bytecode.var_names = HashMap::from([(0, "ans".into())]);
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
            entrypoint: "script".into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-script-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterScriptV2,
            crate::encode_interpreter_script_v2(&bytecode).unwrap(),
        )
        .unwrap();
        let response =
            futures::executor::block_on(execute_program_request(ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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
            }));
        let ProgramExecutionResponse::Success { outputs } = response else {
            panic!("standalone worker must execute and retire its workspace: {response:?}");
        };
        assert_eq!(outputs.len(), 1);
        let output = runmat_runtime::execution::value_codec::decode_inline_value(&outputs[0])
            .expect("script output remains a valid inline value");
        assert_eq!(output, Value::Num(42.0));
    }

    #[test]
    fn worker_rejects_stale_compiler_and_frozen_v1_before_decoding() {
        let bytecode = crate::encode_interpreter_script_v2(&Bytecode::with_instructions(
            vec![Instr::Return],
            0,
        ))
        .unwrap();
        let stale_compiler = ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap();
        let response = futures::executor::block_on(execute_program_request(script_request(
            stale_compiler,
            ExecutableForm::InterpreterScriptV2,
            bytecode,
        )));
        assert!(matches!(
            response,
            ProgramExecutionResponse::Failure { message }
                if message.contains("compiler schema actual 1 expected 2")
        ));

        let current = ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap();
        let response = futures::executor::block_on(execute_program_request(script_request(
            current,
            ExecutableForm::InterpreterScriptV1,
            include_bytes!("../tests/fixtures/interpreter-script-legacy-raw.json").to_vec(),
        )));
        assert!(matches!(
            response,
            ProgramExecutionResponse::Failure { message }
                if message.contains("legacy script bytecode")
        ));
    }

    #[test]
    fn worker_rejects_frozen_unit_components_and_compiler_independently() {
        let stale_components = include_bytes!(
            "../../runmat-execution/tests/fixtures/executable-unit-stale-components.json"
        );
        assert!(matches!(
            admit_executable_unit(&with_stale_envelope_schema(stale_components)),
            Err(ExecutableUnitAdmissionError::InvalidHeader(_))
        ));
        let stale_components = with_current_envelope_schema(stale_components);
        assert!(matches!(
            admit_executable_unit(&stale_components),
            Err(ExecutableUnitAdmissionError::ComponentRevisions(_))
        ));
        let response = futures::executor::block_on(execute_program_request(
            executable_unit_request(&stale_components),
        ));
        assert!(matches!(
            response,
            ProgramExecutionResponse::Failure { message }
                if message.contains(&format!("MIR actual 2 expected {}", runmat_mir::MIR_SCHEMA_VERSION))
                    && message.contains(&format!("analysis actual 2 expected {}", runmat_mir::analysis::ANALYSIS_STORE_SCHEMA_VERSION))
                    && message.contains(&format!(
                        "bytecode actual 6 expected {}",
                        crate::BYTECODE_SCHEMA_VERSION
                    ))
        ));

        let compiler_one =
            include_bytes!("../../runmat-execution/tests/fixtures/executable-unit-compiler-1.json");
        assert!(matches!(
            admit_executable_unit(&with_stale_envelope_schema(compiler_one)),
            Err(ExecutableUnitAdmissionError::InvalidHeader(_))
        ));
        let compiler_one = with_current_envelope_schema(compiler_one);
        assert!(matches!(
            admit_executable_unit(&compiler_one),
            Err(ExecutableUnitAdmissionError::CompilerRevision(_))
        ));
        let response = futures::executor::block_on(execute_program_request(
            executable_unit_request(&compiler_one),
        ));
        assert!(matches!(
            response,
            ProgramExecutionResponse::Failure { message }
                if message.contains("compiler schema actual 1 expected 2")
        ));
    }

    #[test]
    fn generic_vm_rejects_meshing_workload_for_specialized_host() {
        let revision = ProgramRevision::new(
            Digest::sha256(b"mesh-graph"),
            Digest::sha256(b"mesh-source"),
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
            entrypoint: "meshing_workload".into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "meshing".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-meshing-host-v2"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::MeshingWorkload,
            b"inert-host-contract".to_vec(),
        )
        .unwrap();
        let response =
            futures::executor::block_on(execute_program_request(ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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
            }));
        assert_eq!(
            response,
            ProgramExecutionResponse::Failure {
                message: "meshing workloads require a meshing-capable execution host".into(),
            }
        );
    }

    #[test]
    fn worker_requires_the_recipe_to_cover_typed_callable_capabilities() {
        let revision = ProgramRevision::new(
            Digest::sha256(b"capability-graph"),
            Digest::sha256(b"capability-source"),
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
        let mut recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "accelerated".into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("capability-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let capabilities = runmat_types::CapabilitySet(std::collections::BTreeSet::from([
            runmat_types::CapabilityRequirement::Accelerator,
        ]));

        assert!(!recipe_supports_capabilities(&recipe, &capabilities));
        recipe.accelerators =
            vec![runmat_execution::resource::AcceleratorRequest::generic_compute(1).unwrap()];
        assert!(recipe_supports_capabilities(&recipe, &capabilities));
    }

    #[test]
    fn image_reader_builtin_callable_requires_both_source_services_at_host_admission() {
        use std::collections::BTreeSet;

        use runmat_execution::host::{
            ExecutionHostInventory, ExecutionHostRequirement, ExecutionHostTarget,
            EXECUTION_HOST_SCHEMA_VERSION,
        };
        use runmat_execution::security::ExecutionTrustTier;
        use runmat_execution::LanguageCompatibilityMode;
        use runmat_types::{CapabilityRequirement, CapabilitySet, InteropManifest};

        // This test requires the composed typed imread catalog entry. A
        // builtin worker callable has no call-site source literal to narrow.
        let registry = crate::FunctionRegistry::new(HashMap::new());
        let callable = ProgramCallable::builtin("imread").expect("builtin callable");
        let capabilities = super::callable_capabilities(&registry, &callable, None)
            .expect("resolve builtin callable requirements");
        assert!(capabilities.0.contains(&CapabilityRequirement::Filesystem));
        assert!(capabilities.0.contains(&CapabilityRequirement::Network));

        let environment = ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            Digest::sha256(b"image-reader-runtime"),
            Digest::sha256(b"image-reader-catalog"),
            "matlab",
        )
        .unwrap();
        let requirement = ExecutionHostRequirement::portable(
            environment.clone(),
            capabilities,
            InteropManifest::empty(),
            BTreeSet::from([ExecutionTrustTier::CustomerTrusted]),
        )
        .expect("portable builtin requirement");
        let mut host = ExecutionHostInventory {
            schema_version: EXECUTION_HOST_SCHEMA_VERSION,
            semantic_schema: environment.semantic_schema,
            compiler_schema: environment.compiler_schema,
            runtime_fingerprint: environment.runtime_fingerprint,
            catalog_fingerprint: environment.catalog_fingerprint,
            compatibility_modes: BTreeSet::from([LanguageCompatibilityMode::Matlab]),
            target: ExecutionHostTarget::BrowserWasm,
            capabilities: CapabilitySet(BTreeSet::from([CapabilityRequirement::Filesystem])),
            adapters: Vec::new(),
            process_stack_available: false,
            host_bridge_available: false,
            trust_tier: ExecutionTrustTier::CustomerTrusted,
        };
        assert!(requirement.is_satisfied_by(&host).is_err());
        host.capabilities = CapabilitySet(BTreeSet::from([CapabilityRequirement::Network]));
        assert!(requirement.is_satisfied_by(&host).is_err());
        host.capabilities = CapabilitySet(BTreeSet::from([
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::Network,
        ]));
        requirement
            .is_satisfied_by(&host)
            .expect("both source services admit the callable");
    }
}
