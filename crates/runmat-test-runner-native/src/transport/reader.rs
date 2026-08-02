use runmat_test::protocol::{ProtocolLimits, WorkerRequest, WorkerResponse};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::{NativeRunnerError, NativeRunnerResult};

use super::framing::{decode_response_frame, decode_worker_request_frame, frame_length};

pub async fn read_response(
    reader: &mut (impl AsyncRead + Unpin),
    limits: ProtocolLimits,
) -> NativeRunnerResult<WorkerResponse> {
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            NativeRunnerError::Protocol("worker closed its response stream".into())
        } else {
            NativeRunnerError::Io(error)
        }
    })?;
    let length = frame_length(header, limits)?;
    let mut frame = Vec::with_capacity(4 + length);
    frame.extend_from_slice(&header);
    frame.resize(4 + length, 0);
    reader.read_exact(&mut frame[4..]).await?;
    decode_response_frame(&frame, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))
}

pub async fn read_request(
    reader: &mut (impl AsyncRead + Unpin),
    limits: ProtocolLimits,
) -> NativeRunnerResult<WorkerRequest> {
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            NativeRunnerError::Protocol("coordinator closed its request stream".into())
        } else {
            NativeRunnerError::Io(error)
        }
    })?;
    let length = frame_length(header, limits)?;
    let mut frame = Vec::with_capacity(4 + length);
    frame.extend_from_slice(&header);
    frame.resize(4 + length, 0);
    reader.read_exact(&mut frame[4..]).await?;
    decode_worker_request_frame(&frame, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))
}
