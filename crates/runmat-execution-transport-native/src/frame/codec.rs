use runmat_execution_artifact::encryption::{
    decode_transfer_wire_frame, encode_transfer_wire_frame, TransferWireFrame,
};

use crate::{TransportError, TransportResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    Control = 0,
    Artifact = 1,
    Cancellation = 2,
    Telemetry = 3,
}

impl TryFrom<u8> for FrameKind {
    type Error = TransportError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Control),
            1 => Ok(Self::Artifact),
            2 => Ok(Self::Cancellation),
            3 => Ok(Self::Telemetry),
            _ => Err(TransportError::MalformedFrame(
                "unknown frame kind".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLimits {
    pub maximum_payload_bytes: usize,
    pub maximum_frame_bytes: usize,
}

impl Default for FrameLimits {
    fn default() -> Self {
        Self {
            maximum_payload_bytes: 8 * 1024 * 1024,
            maximum_frame_bytes: 8 * 1024 * 1024 + 128,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFrame {
    pub session_id: [u8; 16],
    pub sequence: u64,
    pub kind: FrameKind,
    /// Application ciphertext for secure routes; control bootstrap frames may
    /// carry bounded public metadata.
    pub payload: Vec<u8>,
}

impl WireFrame {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn encode(&self, limits: FrameLimits) -> TransportResult<Vec<u8>> {
        if self.payload.len() > limits.maximum_payload_bytes {
            return Err(TransportError::FrameTooLarge);
        }
        encode_transfer_wire_frame(
            &TransferWireFrame {
                session_id: self.session_id,
                sequence: self.sequence,
                frame_kind: self.kind as u8,
                encrypted_payload: self.payload.clone(),
            },
            limits.maximum_frame_bytes,
        )
        .map_err(map_artifact)
    }

    pub fn decode(bytes: &[u8], limits: FrameLimits) -> TransportResult<Self> {
        if bytes.len() > limits.maximum_frame_bytes {
            return Err(TransportError::FrameTooLarge);
        }
        let frame =
            decode_transfer_wire_frame(bytes, limits.maximum_frame_bytes).map_err(map_artifact)?;
        if frame.encrypted_payload.len() > limits.maximum_payload_bytes {
            return Err(TransportError::FrameTooLarge);
        }
        Ok(Self {
            session_id: frame.session_id,
            sequence: frame.sequence,
            kind: FrameKind::try_from(frame.frame_kind)?,
            payload: frame.encrypted_payload,
        })
    }
}

fn map_artifact(error: impl std::fmt::Display) -> TransportError {
    TransportError::MalformedFrame(error.to_string())
}
