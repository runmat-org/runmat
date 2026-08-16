use runmat_execution::{
    value::ValuePayload, Digest, OutputContract, ProgramEnvironment, ProgramRevision,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse,
};
use runmat_runtime::execution::{DeferredCall, ExecutionServiceError};

pub fn materialize_deferred_call(
    call: &DeferredCall,
    outputs: OutputContract,
    target: runmat_execution_artifact::ProgramTarget,
) -> Result<(ProgramBuildRecipe, ProgramArtifact, Vec<ValuePayload>), ExecutionServiceError> {
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
        entrypoint: call.function.to_string(),
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
    let arguments = call
        .arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::encode_inline_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    Ok((recipe, artifact, arguments))
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
    if request.validate_for_portable_host().is_err() {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a protocol or program identity mismatch".into(),
        };
    }
    if request.artifact.form == ExecutableForm::InterpreterScriptV1 {
        return execute_script_request(request).await;
    }
    if request.artifact.form == ExecutableForm::TestAttemptV1 {
        return ProgramExecutionResponse::Failure {
            message: "test-attempt programs require a test-capable execution host".into(),
        };
    }
    if request.artifact.form == ExecutableForm::MeshingWorkloadV2 {
        return ProgramExecutionResponse::Failure {
            message: "meshing workloads require a meshing-capable execution host".into(),
        };
    }
    if request.artifact.form == ExecutableForm::ExecutableUnitV3 {
        return execute_unit_request(request).await;
    }
    if request.artifact.form == ExecutableForm::NativeObjectV1 {
        return ProgramExecutionResponse::Failure {
            message: "native object programs require a native AOT execution host".into(),
        };
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
    execute_function_request(&request, &registry).await
}

async fn execute_unit_request(request: ProgramExecutionRequest) -> ProgramExecutionResponse {
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
        runmat_execution::ExecutableEntrypointKind::Script => execute_unit_script(bytecode).await,
        runmat_execution::ExecutableEntrypointKind::Function => {
            execute_function_request(&request, &registry).await
        }
    }
}

async fn execute_function_request(
    request: &ProgramExecutionRequest,
    registry: &crate::FunctionRegistry,
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
    match crate::invoke_semantic_function_value(
        request.function,
        &arguments,
        usize::from(request.requested_outputs),
        registry,
    )
    .await
    {
        Ok(value) => match runmat_runtime::execution::value_codec::encode_inline_value(&value) {
            Ok(value) => ProgramExecutionResponse::Success { value },
            Err(error) => ProgramExecutionResponse::Failure {
                message: format!("worker could not transfer its result: {error}"),
            },
        },
        Err(error) => ProgramExecutionResponse::Failure {
            message: error.to_string(),
        },
    }
}

async fn execute_unit_script(bytecode: crate::Bytecode) -> ProgramExecutionResponse {
    let result_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "ans").then_some(*slot));
    match crate::interpret(&bytecode).await {
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
        Err(error) => ProgramExecutionResponse::Failure {
            message: error.to_string(),
        },
    }
}

async fn execute_script_request(request: ProgramExecutionRequest) -> ProgramExecutionResponse {
    let bytecode: crate::Bytecode = match serde_json::from_slice(&request.artifact.executable_bytes)
    {
        Ok(bytecode) => bytecode,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid script program: {error}"),
            }
        }
    };
    execute_unit_script(bytecode).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{
        ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
        ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
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
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                recipe,
                artifact,
                function: 0,
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
            ExecutableForm::MeshingWorkloadV2,
            b"inert-host-contract".to_vec(),
        )
        .unwrap();
        let response =
            futures::executor::block_on(execute_program_request(ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                recipe,
                artifact,
                function: 0,
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
