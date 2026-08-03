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
    target_profile: impl Into<String>,
) -> Result<(ProgramBuildRecipe, ProgramArtifact, Vec<ValuePayload>), ExecutionServiceError> {
    let program = call.program.as_deref().ok_or_else(|| {
        ExecutionServiceError::Failed("execution is missing its exact program".into())
    })?;
    let revision = call
        .program_revision
        .clone()
        .unwrap_or_else(|| captured_program_revision(program));
    let recipe = ProgramBuildRecipe {
        schema_version: 1,
        program_revision: revision,
        entrypoint: call.function.to_string(),
        outputs,
        execution_mode: "interpreter".into(),
        target_profile: target_profile.into(),
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
    if request.validate().is_err() {
        return ProgramExecutionResponse::Failure {
            message: "worker rejected a protocol or program identity mismatch".into(),
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
        &registry,
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
