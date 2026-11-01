use runmat_test::protocol::{ProtocolLimits, WorkerRequest, WorkerResponse};
use tokio::io::AsyncRead;

use crate::{ControlPeer, NativeRunnerError, NativeRunnerResult};

use super::framing::{decode_response_frame, decode_worker_request_frame, host_limits};

pub async fn read_response(
    reader: &mut (impl AsyncRead + Unpin),
    limits: ProtocolLimits,
) -> NativeRunnerResult<WorkerResponse> {
    let frame = runmat_process_host::ipc::read_frame(reader, host_limits(limits))
        .await
        .map_err(|error| map_read_error(error, ControlPeer::Worker))?;
    decode_response_frame(&frame, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))
}

pub async fn read_request(
    reader: &mut (impl AsyncRead + Unpin),
    limits: ProtocolLimits,
) -> NativeRunnerResult<WorkerRequest> {
    let frame = runmat_process_host::ipc::read_frame(reader, host_limits(limits))
        .await
        .map_err(|error| map_read_error(error, ControlPeer::Coordinator))?;
    decode_worker_request_frame(&frame, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))
}

fn map_read_error(
    error: runmat_process_host::ProcessHostError,
    peer: ControlPeer,
) -> NativeRunnerError {
    match error {
        runmat_process_host::ProcessHostError::Io(error)
            if error.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            NativeRunnerError::ControlClosed {
                peer,
                source: error,
            }
        }
        runmat_process_host::ProcessHostError::Io(error) => NativeRunnerError::Io(error),
        error => NativeRunnerError::Protocol(error.to_string()),
    }
}
