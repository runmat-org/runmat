use runmat_execution::{
    value::ValuePayload, Digest, OutputContract, ProgramCallable, ProgramEnvironment,
    ProgramFunctionId, ProgramRevision,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
};
use runmat_runtime::execution::{DeferredCall, DeferredInvocation, ExecutionServiceError};
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

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
            let callable = match &descriptor.target {
                runmat_runtime::call::descriptor::CallableTarget::Resolved {
                    identity:
                        runmat_hir::CallableIdentity::BoundFunction(function)
                        | runmat_hir::CallableIdentity::AnonymousFunction(function)
                        | runmat_hir::CallableIdentity::ExternalFunction { function, .. },
                    ..
                } => ProgramCallable::semantic(
                    ProgramFunctionId(u32::try_from(function.0).map_err(|_| {
                        ExecutionServiceError::Failed(
                            "semantic function identity exceeds its portable representation".into(),
                        )
                    })?),
                    descriptor.metadata.display_name.clone(),
                ),
                runmat_runtime::call::descriptor::CallableTarget::Resolved {
                    identity: runmat_hir::CallableIdentity::Builtin(name),
                    ..
                } => ProgramCallable::builtin(name.0.clone())
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?,
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
    let program = call.program.as_deref().ok_or_else(|| {
        ExecutionServiceError::Failed("execution is missing its exact program".into())
    })?;
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
        expected_artifact_id: None,
    };
    let executable_form = if matches!(
        &callable,
        ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
    ) {
        ExecutableForm::InterpreterScriptV2
    } else {
        ExecutableForm::InterpreterBytecodeV2
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
) -> Result<Value, ExecutionServiceError> {
    let requested_outputs = u16::try_from(call.invocation.requested_outputs())
        .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
    let output_contract = OutputContract { requested_outputs };
    let (callable, invocation_context, recipe, artifact, arguments) = materialize_deferred_call(
        &call,
        output_contract,
        runmat_execution_artifact::ProgramTarget::portable("vm-parallel-region-v1"),
    )?;
    let request = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
        recipe,
        artifact,
        callable,
        context: invocation_context,
        assignment: None,
        job_id: None,
        arguments,
        requested_outputs,
    };
    request
        .validate_for_portable_host()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    match execute_program_request_with_context(request, runtime).await {
        ProgramExecutionResponse::Success { value } => {
            runmat_runtime::execution::value_codec::decode_inline_value(&value)
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
            Digest::sha256(format!(
                "runmat-runtime-abi-v1\0{}",
                env!("CARGO_PKG_VERSION")
            )),
            Digest::sha256(b"runmat-local-captured-catalog-v1"),
            "matlab",
        )
        .expect("captured execution compatibility constants are valid"),
    )
    .expect("captured program revision is valid")
}

pub async fn execute_program_request(request: ProgramExecutionRequest) -> ProgramExecutionResponse {
    execute_program_request_in_context(request, None).await
}

pub async fn execute_program_request_with_context(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    execute_program_request_in_context(request, Some(runtime)).await
}

async fn execute_program_request_in_context(
    request: ProgramExecutionRequest,
    runtime: Option<runmat_runtime::context::RuntimeContext>,
) -> ProgramExecutionResponse {
    if request.validate_for_portable_host().is_err() {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a protocol or program identity mismatch".into(),
        };
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
    if request.artifact.form == ExecutableForm::InterpreterScriptV1 {
        return ProgramExecutionResponse::Failure {
            message: format!(
                "worker rejected legacy script bytecode; rebuild the program for bytecode schema {}",
                crate::BYTECODE_SCHEMA_VERSION
            ),
        };
    }
    if request.artifact.form == ExecutableForm::InterpreterScriptV2
        && !matches!(
            &request.callable,
            ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
        )
    {
        return execute_script_request(request, runtime).await;
    }
    if request.artifact.form == ExecutableForm::ExecutableUnitV3 {
        return execute_unit_request(request, runtime).await;
    }
    if request.artifact.form == ExecutableForm::InterpreterBytecodeV1 {
        return ProgramExecutionResponse::Failure {
            message: format!(
                "worker rejected legacy interpreter bytecode; expected schema {}. Rebuild the program with this RunMat version",
                crate::BYTECODE_SCHEMA_VERSION
            ),
        };
    }
    if request.artifact.form != ExecutableForm::InterpreterBytecodeV2
        && request.artifact.form != ExecutableForm::InterpreterScriptV2
    {
        return ProgramExecutionResponse::Failure {
            message:
                "worker rejected an unsupported interpreter artifact form; rebuild the program"
                    .into(),
        };
    }
    if matches!(request.callable, ProgramCallable::ParallelRegion { .. }) {
        return execute_parallel_region_request(request, runtime).await;
    }
    if matches!(request.callable, ProgramCallable::SpmdRegion { .. }) {
        return execute_spmd_region_request(request, runtime).await;
    }
    let registry: crate::FunctionRegistry =
        match crate::decode_interpreter_program_v2(&request.artifact.executable_bytes) {
            Ok(registry) => registry,
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker rejected an invalid program: {error}"),
                }
            }
        };
    let capabilities = match callable_capabilities(&registry, &request.callable) {
        Ok(capabilities) => capabilities,
        Err(message) => return ProgramExecutionResponse::Failure { message },
    };
    if !recipe_supports_capabilities(&request.recipe, &capabilities) {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a recipe that weakens callable accelerator requirements"
                .into(),
        };
    }
    execute_function_request(&request, &registry, &runtime).await
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
) -> Result<runmat_types::CapabilitySet, String> {
    let function = |id: ProgramFunctionId| {
        usize::try_from(id.0)
            .ok()
            .map(runmat_hir::FunctionId)
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
    let bytecode: crate::Bytecode =
        match crate::decode_interpreter_script_v2(&request.artifact.executable_bytes) {
            Ok(bytecode) => bytecode,
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker rejected invalid SPMD bytecode: {error}"),
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
    let arguments = match request
        .arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::decode_inline_value)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(arguments) => arguments,
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
    let mut frame = vec![None; bytecode.var_count];
    for (capture, value) in executable.captures.iter().zip(arguments) {
        let Some(slot) = frame.get_mut(capture.slot) else {
            return ProgramExecutionResponse::Failure {
                message: "SPMD capture lies outside its compiler-bound VM frame".into(),
            };
        };
        *slot = Some(value);
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
                match crate::interpreter::dispatch::encode_spmd_output(&runtime, value).await {
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
        Err(error) => runtime_failure_response(error),
    }
}

async fn execute_parallel_region_request(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    let ProgramCallable::ParallelRegion { region } = &request.callable else {
        return ProgramExecutionResponse::Failure {
            message: "worker received a non-parallel callable on the parallel execution path"
                .into(),
        };
    };
    let runmat_execution::ProgramInvocationContext::ParallelTask { task } = &request.context else {
        return ProgramExecutionResponse::Failure {
            message: "parallel region task is missing its typed execution context".into(),
        };
    };
    if request.requested_outputs != 1 {
        return ProgramExecutionResponse::Failure {
            message: "parallel region tasks return one typed result envelope".into(),
        };
    }
    let bytecode: crate::Bytecode =
        match crate::decode_interpreter_script_v2(&request.artifact.executable_bytes) {
            Ok(bytecode) => bytecode,
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker rejected invalid parallel bytecode: {error}"),
                }
            }
        };
    let Some(executable) = bytecode
        .parfor_regions
        .iter()
        .find(|candidate| candidate.contract.id == *region)
    else {
        return ProgramExecutionResponse::Failure {
            message: "worker could not find the requested parallel region in its exact program"
                .into(),
        };
    };
    if !recipe_supports_capabilities(&request.recipe, &executable.contract.capabilities) {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a recipe that weakens parfor accelerator requirements".into(),
        };
    }
    let mut arguments = match request
        .arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::decode_inline_value)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(arguments) => arguments,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid parallel task input: {error}"),
            }
        }
    };
    let expected_arguments = executable.input_variables().count() + 1;
    if arguments.len() != expected_arguments {
        return ProgramExecutionResponse::Failure {
            message: format!(
                "parallel task received {} arguments; its executable contract requires {expected_arguments}",
                arguments.len()
            ),
        };
    }
    let Value::Cell(iterations) = arguments.remove(0) else {
        return ProgramExecutionResponse::Failure {
            message: "parallel task iteration input must be a cell vector".into(),
        };
    };
    if iterations.data.len() != task.chunk.len {
        return ProgramExecutionResponse::Failure {
            message: "parallel task iteration payload differs from its declared chunk".into(),
        };
    }
    let callable_name = request.callable.display_name();
    let result = crate::interpreter::runner::interpret_parfor_task_in_context(
        crate::interpreter::runner::ParforTaskExecution {
            bytecode: &bytecode,
            region: executable,
            inputs: arguments,
            iterations: iterations.data,
            randomness: task.randomness.clone(),
            mode: crate::interpreter::runner::ParforTaskMode::IndependentIterations,
            current_function_name: &callable_name,
            runtime,
        },
    )
    .await
    .and_then(|outputs| {
        runmat_value::CellArray::new(outputs, 1, executable.output_variables().count())
            .map(Value::Cell)
            .map_err(|error| {
                runmat_runtime::runtime_error::semantic_error(
                    "ParallelTaskOutput",
                    error.to_string(),
                )
            })
    });
    match result {
        Ok(value) => match runmat_runtime::execution::value_codec::encode_inline_value(&value) {
            Ok(value) => ProgramExecutionResponse::Success { value },
            Err(error) => ProgramExecutionResponse::Failure {
                message: format!("worker could not transfer its parallel task result: {error}"),
            },
        },
        Err(error) => runtime_failure_response(error),
    }
}

async fn execute_unit_request(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    let admission = match runmat_execution::ExecutableUnitEnvelope::admission(
        &request.artifact.executable_bytes,
    ) {
        Ok(admission) => admission,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid executable unit header: {error}"),
            }
        }
    };
    if let Some(message) = unsupported_executable_revisions(&admission.revisions) {
        return ProgramExecutionResponse::Failure { message };
    }
    let envelope = match request.artifact.executable_unit() {
        Ok(Some(envelope)) => envelope,
        Ok(None) => {
            return ProgramExecutionResponse::Failure {
                message: "worker received a non-unit artifact on the unit execution path".into(),
            }
        }
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid executable unit: {error}"),
            }
        }
    };
    let revisions = &envelope.manifest.revisions;
    if let Some(message) = unsupported_executable_revisions(revisions) {
        return ProgramExecutionResponse::Failure { message };
    }
    let Some(bytecode_payload) =
        envelope.component(runmat_execution::ExecutableComponentKind::Bytecode)
    else {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected executable unit without bytecode".into(),
        };
    };
    let mut bytecode: crate::Bytecode = match serde_json::from_slice(&bytecode_payload.bytes) {
        Ok(bytecode) => bytecode,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected invalid executable bytecode: {error}"),
            }
        }
    };

    if !bytecode.bound_functions.is_empty()
        || !bytecode.function_registry.functions.is_empty()
        || bytecode.layout.is_some()
    {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected executable bytecode with duplicate component authorities"
                .into(),
        };
    }
    let Some(registry_payload) =
        envelope.component(runmat_execution::ExecutableComponentKind::FunctionRegistry)
    else {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected executable unit without a function registry".into(),
        };
    };
    let registry: crate::FunctionRegistry = match serde_json::from_slice(&registry_payload.bytes) {
        Ok(registry) => registry,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected invalid executable registry: {error}"),
            }
        }
    };
    let Some(layout_payload) =
        envelope.component(runmat_execution::ExecutableComponentKind::VmLayout)
    else {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected executable unit without a VM layout".into(),
        };
    };
    let layout: crate::VmAssemblyLayout = match serde_json::from_slice(&layout_payload.bytes) {
        Ok(layout) => layout,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected invalid executable VM layout: {error}"),
            }
        }
    };
    bytecode.bound_functions = registry.functions.clone();
    bytecode.function_registry = registry.clone();
    bytecode.layout = Some(layout);

    match envelope.manifest.identity.entrypoint_kind {
        runmat_execution::ExecutableEntrypointKind::Script => {
            execute_unit_script(bytecode, runtime).await
        }
        runmat_execution::ExecutableEntrypointKind::Function => {
            execute_function_request(&request, &registry, &runtime).await
        }
    }
}

fn unsupported_executable_revisions(
    revisions: &runmat_execution::ExecutableComponentRevisions,
) -> Option<String> {
    let supported = revisions.mir_schema == runmat_mir::MIR_SCHEMA_VERSION
        && revisions.analysis_schema == runmat_mir::analysis::ANALYSIS_STORE_SCHEMA_VERSION
        && revisions.bytecode_schema == crate::BYTECODE_SCHEMA_VERSION
        && revisions.vm_layout_schema == crate::VM_LAYOUT_SCHEMA_VERSION
        && revisions.function_registry_schema == crate::FUNCTION_REGISTRY_SCHEMA_VERSION
        && revisions.contract_schema == runmat_types::RUNMAT_TYPES_SCHEMA.major
        && u32::from(revisions.catalog_schema) == runmat_builtins::BUILTIN_CATALOG_SCHEMA;
    (!supported).then(|| {
        format!(
            "worker rejected stale executable component revisions: MIR actual {} expected {}; analysis actual {} expected {}; bytecode actual {} expected {}; VM layout actual {} expected {}; function registry actual {} expected {}; contract actual {} expected {}; catalog actual {} expected {}. Rebuild the program with this RunMat version",
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
            revisions.contract_schema,
            runmat_types::RUNMAT_TYPES_SCHEMA.major,
            revisions.catalog_schema,
            runmat_builtins::BUILTIN_CATALOG_SCHEMA,
        )
    })
}

async fn execute_function_request(
    request: &ProgramExecutionRequest,
    registry: &crate::FunctionRegistry,
    runtime: &runmat_runtime::context::RuntimeContext,
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
            crate::invoke_semantic_function_value_in_context(
                function.0 as usize,
                &arguments,
                requested_outputs,
                registry,
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
                .scope(runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor))
                .await
        }
        runmat_execution::ProgramCallable::ParallelRegion { .. }
        | runmat_execution::ProgramCallable::SpmdRegion { .. } => {
            return ProgramExecutionResponse::Failure {
                message:
                    "structured parallel callable reached the semantic function execution path"
                        .into(),
            }
        }
    };
    match result {
        Ok(value) => match runmat_runtime::execution::value_codec::encode_inline_value(&value) {
            Ok(value) => ProgramExecutionResponse::Success { value },
            Err(error) => ProgramExecutionResponse::Failure {
                message: format!("worker could not transfer its result: {error}"),
            },
        },
        Err(error) => runtime_failure_response(error),
    }
}

async fn execute_unit_script(
    bytecode: crate::Bytecode,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    let result_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "ans").then_some(*slot));
    let mut variables = vec![runmat_value::Value::Num(0.0); bytecode.var_count];
    let result =
        crate::interpret_with_vars_in_context(&bytecode, &mut variables, Some("<main>"), runtime)
            .await
            .map(|outcome| match outcome {
                crate::InterpreterOutcome::Completed(completion) => completion.values,
            });
    match result {
        Ok(values) => {
            let value = result_slot
                .and_then(|slot| values.get(slot).cloned())
                .unwrap_or(runmat_value::Value::Num(0.0));
            match runmat_runtime::execution::value_codec::encode_inline_value(&value) {
                Ok(value) => ProgramExecutionResponse::Success { value },
                Err(error) => ProgramExecutionResponse::Failure {
                    message: format!("worker could not transfer its result: {error}"),
                },
            }
        }
        Err(error) => runtime_failure_response(error),
    }
}

fn runtime_failure_response(error: runmat_runtime::RuntimeError) -> ProgramExecutionResponse {
    match runmat_runtime::execution::encode_runtime_failure(&error) {
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
    let bytecode: crate::Bytecode = match serde_json::from_slice(&request.artifact.executable_bytes)
    {
        Ok(bytecode) => bytecode,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid script program: {error}"),
            }
        }
    };
    execute_unit_script(bytecode, runtime).await
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
        ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
    };

    use super::{execute_program_request, recipe_supports_capabilities};
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
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-script-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(&recipe, form, executable_bytes).unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
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
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("frozen-unit-test"),
            interop: envelope.manifest.interop.clone(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact =
            ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV3, bytes.to_vec())
                .unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
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
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-script-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
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
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
                recipe,
                artifact,
                callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
                context: runmat_execution::ProgramInvocationContext::Direct,
                assignment: None,
                job_id: None,
                arguments: Vec::new(),
                requested_outputs: 1,
            }));
        assert!(matches!(response, ProgramExecutionResponse::Success { .. }));
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
        let response = futures::executor::block_on(execute_program_request(
            executable_unit_request(stale_components),
        ));
        assert!(matches!(
            response,
            ProgramExecutionResponse::Failure { message }
                if message.contains("MIR actual 2 expected 3")
                    && message.contains("analysis actual 2 expected 3")
                    && message.contains("bytecode actual 6 expected 7")
        ));

        let compiler_one =
            include_bytes!("../../runmat-execution/tests/fixtures/executable-unit-compiler-1.json");
        let response = futures::executor::block_on(execute_program_request(
            executable_unit_request(compiler_one),
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
            },
            execution_mode: "meshing".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("portable-meshing-host-v2"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
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
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
                recipe,
                artifact,
                callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
                context: runmat_execution::ProgramInvocationContext::Direct,
                assignment: None,
                job_id: None,
                arguments: Vec::new(),
                requested_outputs: 1,
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
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("capability-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
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
}
