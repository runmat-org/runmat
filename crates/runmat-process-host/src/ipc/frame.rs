use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameLimits {
    pub max_message_bytes: u32,
}

impl FrameLimits {
    pub fn validate(self) -> ProcessHostResult<Self> {
        if self.max_message_bytes == 0 {
            return Err(ProcessHostError::Configuration(
                "frame limit must be greater than zero".into(),
            ));
        }
        Ok(self)
    }
}

pub async fn read_payload(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
) -> ProcessHostResult<Vec<u8>> {
    limits.validate()?;
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header).await?;
    let length = checked_length(header, limits)?;
    let mut payload = vec![0; length];
    reader.read_exact(&mut payload).await?;
    Ok(payload)
}

pub async fn read_frame(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
) -> ProcessHostResult<Vec<u8>> {
    let payload = read_payload(reader, limits).await?;
    let length = u32::try_from(payload.len())
        .map_err(|_| ProcessHostError::Protocol("frame length exceeds u32".into()))?;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

pub async fn write_payload(
    writer: &mut (impl AsyncWrite + Unpin),
    payload: &[u8],
    limits: FrameLimits,
) -> ProcessHostResult<()> {
    limits.validate()?;
    if payload.len() > limits.max_message_bytes as usize {
        return Err(ProcessHostError::Protocol(format!(
            "frame is {} bytes; negotiated maximum is {}",
            payload.len(),
            limits.max_message_bytes
        )));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| ProcessHostError::Protocol("frame length exceeds u32".into()))?;
    writer.write_all(&length.to_be_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn write_frame(
    writer: &mut (impl AsyncWrite + Unpin),
    frame: &[u8],
    limits: FrameLimits,
) -> ProcessHostResult<()> {
    if frame.len() < 4 {
        return Err(ProcessHostError::Protocol(
            "encoded frame is missing its length header".into(),
        ));
    }
    let header: [u8; 4] = frame[..4]
        .try_into()
        .expect("four-byte slice has a four-byte array representation");
    let length = checked_length(header, limits)?;
    if frame.len() != 4 + length {
        return Err(ProcessHostError::Protocol(
            "encoded frame length header does not match its payload".into(),
        ));
    }
    writer.write_all(frame).await?;
    writer.flush().await?;
    Ok(())
}

fn checked_length(header: [u8; 4], limits: FrameLimits) -> ProcessHostResult<usize> {
    let length = u32::from_be_bytes(header) as usize;
    if length > limits.max_message_bytes as usize {
        return Err(ProcessHostError::Protocol(format!(
            "frame is {length} bytes; negotiated maximum is {}",
            limits.max_message_bytes
        )));
    }
    Ok(length)
}
