use runmat_test::protocol::{
    decode_request, decode_response, encode_request, encode_response, ProtocolLimits,
    WorkerRequest, WorkerResponse,
};

use crate::{RunnerError, RunnerResult};

pub fn encode_frame(request: &WorkerRequest, limits: ProtocolLimits) -> RunnerResult<Vec<u8>> {
    let payload = encode_request(request, limits)
        .map_err(|error| RunnerError::Protocol(error.to_string()))?;
    let length = u32::try_from(payload.len())
        .map_err(|_| RunnerError::Protocol("worker frame length exceeds u32".into()))?;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

pub fn decode_frame(frame: &[u8], limits: ProtocolLimits) -> RunnerResult<WorkerResponse> {
    if frame.len() < 4 {
        return Err(RunnerError::Protocol(
            "worker frame is missing its length prefix".into(),
        ));
    }
    let length = u32::from_be_bytes(frame[..4].try_into().expect("four length bytes")) as usize;
    if length != frame.len() - 4 {
        return Err(RunnerError::Protocol(
            "worker frame length prefix does not match payload".into(),
        ));
    }
    decode_response(&frame[4..], limits).map_err(|error| RunnerError::Protocol(error.to_string()))
}

pub fn decode_request_frame(frame: &[u8], limits: ProtocolLimits) -> RunnerResult<WorkerRequest> {
    if frame.len() < 4 {
        return Err(RunnerError::Protocol(
            "worker frame is missing its length prefix".into(),
        ));
    }
    let length = u32::from_be_bytes(frame[..4].try_into().expect("four length bytes")) as usize;
    if length != frame.len() - 4 {
        return Err(RunnerError::Protocol(
            "worker frame length prefix does not match payload".into(),
        ));
    }
    decode_request(&frame[4..], limits).map_err(|error| RunnerError::Protocol(error.to_string()))
}

pub fn encode_response_frame(
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> RunnerResult<Vec<u8>> {
    let payload = encode_response(response, limits)
        .map_err(|error| RunnerError::Protocol(error.to_string()))?;
    let length = u32::try_from(payload.len())
        .map_err(|_| RunnerError::Protocol("worker frame length exceeds u32".into()))?;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}
