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
    let length = checked_payload_length(payload, limits)?;
    writer.write_all(&length.to_be_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Synchronous source callback: bounded framing and backpressure without needing
/// the interpreter to yield to an async writer on the same thread.
pub fn write_payload_sync(
    writer: &mut impl std::io::Write,
    payload: &[u8],
    limits: FrameLimits,
) -> ProcessHostResult<()> {
    let length = checked_payload_length(payload, limits)?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

fn checked_payload_length(payload: &[u8], limits: FrameLimits) -> ProcessHostResult<u32> {
    limits.validate()?;
    if payload.len() > limits.max_message_bytes as usize {
        return Err(ProcessHostError::Protocol(format!(
            "frame is {} bytes; negotiated maximum is {}",
            payload.len(),
            limits.max_message_bytes
        )));
    }
    u32::try_from(payload.len())
        .map_err(|_| ProcessHostError::Protocol("frame length exceeds u32".into()))
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

#[cfg(test)]
mod source_write_tests {
    use super::*;
    #[tokio::test]
    async fn synchronous_source_frames_use_the_same_codec_and_reject_before_writing() {
        let limits = FrameLimits {
            max_message_bytes: 64,
        };
        let mut bytes = Vec::new();
        write_payload_sync(&mut bytes, b"original\0\n", limits).unwrap();
        assert_eq!(
            read_payload(&mut bytes.as_slice(), limits).await.unwrap(),
            b"original\0\n"
        );
        let mut oversized = Vec::new();
        assert!(write_payload_sync(&mut oversized, &[1; 65], limits).is_err());
        assert!(oversized.is_empty());
    }
}
