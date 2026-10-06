use runmat_execution::{value::ValuePayload, OutputContract};
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
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(program)
        .map_err(|error| {
            ExecutionServiceError::Failed(format!("invalid complete deferred program: {error}"))
        })?;
    if call.program_revision.as_ref() != Some(&envelope.manifest.identity.program)
        || usize::try_from(envelope.manifest.identity.entrypoint_function.0).ok()
            != Some(call.function)
        || usize::from(outputs.requested_outputs) != call.requested_outputs
    {
        return Err(ExecutionServiceError::Failed(
            "deferred program revision, function or output contract mismatch".into(),
        ));
    }
    let (recipe, artifact) =
        ProgramArtifact::materialize_executable_unit(&envelope, outputs, target)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let arguments = call
        .arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::encode_inline_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    Ok((recipe, artifact, arguments))
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
    if request.artifact.form == ExecutableForm::MeshingWorkload {
        return ProgramExecutionResponse::Failure {
            message: "meshing workloads require a meshing-capable execution host".into(),
        };
    }
    if request.artifact.form == ExecutableForm::AnalysisWorkload {
        return ProgramExecutionResponse::Failure {
            message: "analysis workloads require an analysis-capable execution host".into(),
        };
    }
    if request.artifact.form == ExecutableForm::ExecutableUnitV4 {
        return ProgramExecutionResponse::Failure {
            message: "executable units require a session-capable execution host".into(),
        };
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
        Ok(value) => match runmat_runtime::execution::value_codec::encode_inline_outputs(
            &value,
            usize::from(request.requested_outputs),
        ) {
            Ok(outputs) => ProgramExecutionResponse::from_success_outputs(outputs, vec![]),
            Err(error) => ProgramExecutionResponse::Failure {
                message: format!("worker could not transfer its result: {error}"),
            },
        },
        Err(error) => ProgramExecutionResponse::Failure {
            message: error.to_string(),
        },
    }
}

async fn execute_script_bytecode(
    bytecode: crate::Bytecode,
    requested_outputs: u16,
) -> ProgramExecutionResponse {
    if requested_outputs > 1 {
        return ProgramExecutionResponse::Failure {
            message: "scripts do not define multiple output ordinals".into(),
        };
    }
    // A fresh script has no preassigned workspace bindings. Use the same VM-owned
    // assignment authority as session execution; allocated slots are not results.
    let names = bytecode
        .var_names
        .iter()
        .map(|(slot, name)| (name.clone(), *slot))
        .collect();
    let _pending_workspace = crate::push_pending_workspace(names, Default::default());
    let result = crate::interpret(&bytecode).await;
    let workspace = crate::take_updated_workspace_state();
    let _ = crate::take_updated_workspace_assigned_report();
    match result {
        Ok(values) => {
            let result_slot = workspace.as_ref().and_then(|workspace| {
                workspace
                    .assigned
                    .contains("ans")
                    .then(|| workspace.names.get("ans").copied())
                    .flatten()
            });
            let value = if requested_outputs == 0 {
                runmat_value::Value::OutputList(vec![])
            } else if let Some(value) = result_slot.and_then(|slot| values.get(slot).cloned()) {
                value
            } else {
                return ProgramExecutionResponse::Failure {
                    message: "script did not produce the requested ans value".into(),
                };
            };
            match runmat_runtime::execution::value_codec::encode_inline_outputs(
                &value,
                usize::from(requested_outputs),
            ) {
                Ok(outputs) => ProgramExecutionResponse::from_success_outputs(outputs, vec![]),
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
    execute_script_bytecode(bytecode, request.requested_outputs).await
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
            ExecutableForm::MeshingWorkload,
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

#[cfg(test)]
#[path = "program_execution/outputs.rs"]
mod output_tests;
