use runmat_test::protocol::{ProtocolLimits, WorkerRequest, WorkerResponse};

use crate::{NativeRunnerError, NativeRunnerResult};

pub(super) fn frame_length(header: [u8; 4], limits: ProtocolLimits) -> NativeRunnerResult<usize> {
    let length = u32::from_be_bytes(header) as usize;
    if length > limits.max_message_bytes as usize {
        return Err(NativeRunnerError::Protocol(format!(
            "worker frame is {length} bytes; negotiated maximum is {}",
            limits.max_message_bytes
        )));
    }
    Ok(length)
}

pub(super) fn decode_response_frame(
    frame: &[u8],
    limits: ProtocolLimits,
) -> runmat_test_runner::RunnerResult<WorkerResponse> {
    runmat_test_runner::worker::decode_frame(frame, limits)
}

pub(super) fn encode_request_frame(
    request: &WorkerRequest,
    limits: ProtocolLimits,
) -> runmat_test_runner::RunnerResult<Vec<u8>> {
    runmat_test_runner::worker::encode_frame(request, limits)
}

pub(super) fn decode_worker_request_frame(
    frame: &[u8],
    limits: ProtocolLimits,
) -> runmat_test_runner::RunnerResult<WorkerRequest> {
    runmat_test_runner::worker::decode_request_frame(frame, limits)
}

pub(super) fn encode_worker_response_frame(
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> runmat_test_runner::RunnerResult<Vec<u8>> {
    runmat_test_runner::worker::encode_response_frame(response, limits)
}
