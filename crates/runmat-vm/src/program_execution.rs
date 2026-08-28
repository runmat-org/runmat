use runmat_execution::{
    value::ValuePayload, Digest, OutputContract, ProgramCallable, ProgramEnvironment,
    ProgramFunctionId, ProgramRevision,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V4,
};
use runmat_runtime::execution::{DeferredCall, DeferredInvocation, ExecutionServiceError};
use runmat_value::Value;

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
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: callable.recipe_entrypoint(),
        outputs,
        execution_mode: "interpreter".into(),
        target,
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV1,
        program.to_vec(),
    )
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
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V4,
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
    }
}

fn captured_program_revision(program: &[u8]) -> ProgramRevision {
    let digest = Digest::sha256(program);
    ProgramRevision::new(
        digest,
        digest,
        ProgramEnvironment::new(
            1,
            1,
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
        return execute_script_request(request, runtime).await;
    }
    if request.artifact.form == ExecutableForm::ExecutableUnitV3 {
        return execute_unit_request(request, runtime).await;
    }
    if matches!(request.callable, ProgramCallable::ParallelRegion { .. }) {
        return execute_parallel_region_request(request, runtime).await;
    }
    let registry: crate::FunctionRegistry =
        match serde_json::from_slice(&request.artifact.executable_bytes) {
            Ok(registry) => registry,
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker rejected an invalid program: {error}"),
                }
            }
        };
    execute_function_request(&request, &registry, &runtime).await
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
    let bytecode: crate::Bytecode = match serde_json::from_slice(&request.artifact.executable_bytes)
    {
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
                runmat_hir::CallableIdentity::Builtin(runmat_hir::BuiltinId(name.clone())),
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
                crate::InterpreterOutcome::Completed(values) => values,
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
        ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V4,
    };

    use super::execute_program_request;
    use crate::{Bytecode, Instr};

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
                1,
                1,
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
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterScriptV1,
            serde_json::to_vec(&bytecode).unwrap(),
        )
        .unwrap();
        let response =
            futures::executor::block_on(execute_program_request(ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V4,
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
    fn generic_vm_rejects_meshing_workload_for_specialized_host() {
        let revision = ProgramRevision::new(
            Digest::sha256(b"mesh-graph"),
            Digest::sha256(b"mesh-source"),
            ProgramEnvironment::new(
                1,
                1,
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
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V4,
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
}
