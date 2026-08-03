use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};

use crate::protocol::{WorkerRequest, WorkerResponse, PROTOCOL};
use crate::{NativeExecutionError, NativeExecutionResult};

pub async fn run_worker_stdio() -> NativeExecutionResult<()> {
    let (mut reader, mut writer) = runmat_process_host::ipc::stdio::endpoint();
    let limits = FrameLimits {
        max_message_bytes: 64 * 1024 * 1024,
    };
    let payload = read_payload(&mut reader, limits).await?;
    let request: WorkerRequest = serde_json::from_slice(&payload)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let response = execute(request).await;
    let payload = serde_json::to_vec(&response)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    write_payload(&mut writer, &payload, limits).await?;
    Ok(())
}

async fn execute(request: WorkerRequest) -> WorkerResponse {
    if request.protocol != PROTOCOL || request.artifact.validate_against(&request.recipe).is_err() {
        return WorkerResponse::Failure {
            message: "worker rejected a protocol or program identity mismatch".into(),
        };
    }
    let registry: runmat_vm::FunctionRegistry =
        match serde_json::from_slice(&request.artifact.executable_bytes) {
            Ok(registry) => registry,
            Err(error) => {
                return WorkerResponse::Failure {
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
            return WorkerResponse::Failure {
                message: format!("worker rejected an invalid argument: {error}"),
            }
        }
    };
    match runmat_vm::invoke_semantic_function_value(
        request.function,
        &arguments,
        request.requested_outputs,
        &registry,
    )
    .await
    {
        Ok(value) => match runmat_runtime::execution::value_codec::encode_inline_value(&value) {
            Ok(value) => WorkerResponse::Success { value },
            Err(error) => WorkerResponse::Failure {
                message: format!("worker could not transfer its result: {error}"),
            },
        },
        Err(error) => WorkerResponse::Failure {
            message: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

    use super::*;

    #[tokio::test]
    async fn worker_rejects_a_tampered_materialized_program_before_decoding() {
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
            schema_version: 1,
            program_revision: revision,
            entrypoint: "0".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target_profile: "test-interpreter-bytecode-v1".into(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let mut artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"not reached".to_vec(),
        )
        .unwrap();
        artifact.executable_bytes.push(0);
        let response = execute(WorkerRequest {
            protocol: PROTOCOL.into(),
            recipe,
            artifact,
            function: 0,
            arguments: Vec::new(),
            requested_outputs: 1,
        })
        .await;
        assert!(
            matches!(response, WorkerResponse::Failure { message } if message.contains("identity mismatch"))
        );
    }
}
