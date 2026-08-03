use runmat_test::protocol::{ProtocolLimits, WorkerRequest, WorkerResponse};

pub(super) fn host_limits(limits: ProtocolLimits) -> runmat_process_host::ipc::FrameLimits {
    runmat_process_host::ipc::FrameLimits {
        max_message_bytes: limits.max_message_bytes,
    }
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
