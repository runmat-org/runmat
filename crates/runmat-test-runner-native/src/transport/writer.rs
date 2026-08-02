use runmat_test::protocol::{ProtocolLimits, WorkerRequest, WorkerResponse};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{NativeRunnerError, NativeRunnerResult};

use super::framing::{encode_request_frame, encode_worker_response_frame};

pub async fn write_request(
    writer: &mut (impl AsyncWrite + Unpin),
    request: &WorkerRequest,
    limits: ProtocolLimits,
) -> NativeRunnerResult<()> {
    let frame = encode_request_frame(request, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn write_response(
    writer: &mut (impl AsyncWrite + Unpin),
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> NativeRunnerResult<()> {
    let frame = encode_worker_response_frame(response, limits)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}
