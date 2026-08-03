use runmat_test::protocol::{ProtocolLimits, WorkerRequest, WorkerResponse};
use tokio::io::AsyncWrite;

use crate::{NativeRunnerError, NativeRunnerResult};

use super::framing::{encode_request_frame, encode_worker_response_frame, host_limits};

pub async fn write_request(
    writer: &mut (impl AsyncWrite + Unpin),
    request: &WorkerRequest,
    limits: ProtocolLimits,
) -> NativeRunnerResult<()> {
    let frame = encode_request_frame(request, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    runmat_process_host::ipc::write_frame(writer, &frame, host_limits(limits))
        .await
        .map_err(map_host_error)?;
    Ok(())
}

pub async fn write_response(
    writer: &mut (impl AsyncWrite + Unpin),
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> NativeRunnerResult<()> {
    let frame = encode_worker_response_frame(response, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    runmat_process_host::ipc::write_frame(writer, &frame, host_limits(limits))
        .await
        .map_err(map_host_error)?;
    Ok(())
}

fn map_host_error(error: runmat_process_host::ProcessHostError) -> NativeRunnerError {
    match error {
        runmat_process_host::ProcessHostError::Io(error) => NativeRunnerError::Io(error),
        error => NativeRunnerError::Protocol(error.to_string()),
    }
}
