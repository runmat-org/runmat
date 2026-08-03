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
    if request.protocol != PROTOCOL
        || runmat_execution::Digest::sha256(&request.program) != request.program_digest
    {
        return WorkerResponse::Failure {
            message: "worker rejected a protocol or program identity mismatch".into(),
        };
    }
    let registry: runmat_vm::FunctionRegistry = match serde_json::from_slice(&request.program) {
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
