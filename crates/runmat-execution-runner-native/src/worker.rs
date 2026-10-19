use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};

use crate::protocol::{WorkerRequest, WorkerResponse};
use crate::{NativeExecutionError, NativeExecutionResult};

mod console;

pub async fn run_worker_stdio() -> NativeExecutionResult<()> {
    if std::env::var_os(crate::NATIVE_ACCELERATOR_REQUIRED_ENV).is_some() {
        let encoded = std::env::var(crate::NATIVE_ACCELERATOR_REQUESTS_ENV).map_err(|_| {
            NativeExecutionError::Configuration(
                "local accelerator request environment is missing".into(),
            )
        })?;
        let requests: Vec<runmat_execution::resource::AcceleratorRequest> =
            serde_json::from_str(&encoded).map_err(|error| {
                NativeExecutionError::Configuration(format!(
                    "local accelerator request environment is invalid: {error}"
                ))
            })?;
        crate::accelerator_placement::validate_local_provider_for(&requests)?;
    }
    let (mut reader, mut writer) = runmat_process_host::ipc::stdio::endpoint()?;
    let limits = FrameLimits {
        max_message_bytes: 64 * 1024 * 1024,
    };
    let payload = read_payload(&mut reader, limits).await?;
    let request: WorkerRequest = serde_json::from_slice(&payload)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    if !reader.buffer().is_empty() {
        return Err(NativeExecutionError::Protocol(
            "unexpected input follows one-request worker invocation".into(),
        ));
    }
    drop(reader);
    let parent = runmat_process_host::ipc::WorkerParentGuard::start(std::io::stdin())?;
    console::execute_streamed(request, &mut writer, limits, &parent).await
}

pub(crate) async fn execute(request: WorkerRequest) -> WorkerResponse {
    if let Err(error) = request.validate() {
        return WorkerResponse::Failure {
            message: error.to_string(),
        };
    }
    let (program, resources) = request.into_parts();
    crate::test_workload::execute_host_program_request_in_context(
        program,
        None,
        None,
        None,
        Some(&resources),
    )
    .await
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
        let response = crate::execute_host_program_request(
            runmat_execution_artifact::ProgramExecutionRequest {
                console: None,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
                recipe,
                artifact,
                function: 0,
                arguments: Vec::new(),
                requested_outputs: 1,
            },
        )
        .await;
        assert!(
            matches!(response, WorkerResponse::Failure { message } if message.contains("identity mismatch"))
        );
    }
}
