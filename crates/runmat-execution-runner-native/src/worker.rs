use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};

use crate::protocol::{WorkerRequest, WorkerResponse};
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

pub(crate) async fn execute(request: WorkerRequest) -> WorkerResponse {
    crate::execute_host_program_request(request).await
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
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "0".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable(
                "test-interpreter-bytecode-v1",
            ),
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
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
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
