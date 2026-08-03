use minicbor::{Decoder, Encoder};

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
        let mut bytes = Vec::with_capacity(self.payload.len() + 64);
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .map(5)
            .and_then(|encoder| encoder.u8(0))
            .and_then(|encoder| encoder.u16(Self::SCHEMA_VERSION))
            .and_then(|encoder| encoder.u8(1))
            .and_then(|encoder| encoder.bytes(&self.session_id))
            .and_then(|encoder| encoder.u8(2))
            .and_then(|encoder| encoder.u64(self.sequence))
            .and_then(|encoder| encoder.u8(3))
            .and_then(|encoder| encoder.u8(self.kind as u8))
            .and_then(|encoder| encoder.u8(4))
            .and_then(|encoder| encoder.bytes(&self.payload))
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        if bytes.len() > limits.maximum_frame_bytes {
            return Err(TransportError::FrameTooLarge);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8], limits: FrameLimits) -> TransportResult<Self> {
        if bytes.len() > limits.maximum_frame_bytes {
            return Err(TransportError::FrameTooLarge);
        }
        let mut decoder = Decoder::new(bytes);
        if decoder
            .map()
            .map_err(malformed)?
            .ok_or_else(|| TransportError::MalformedFrame("indefinite map".to_string()))?
            != 5
        {
            return Err(TransportError::MalformedFrame(
                "frame field count".to_string(),
            ));
        }
        expect_key(&mut decoder, 0)?;
        if decoder.u16().map_err(malformed)? != Self::SCHEMA_VERSION {
            return Err(TransportError::MalformedFrame(
                "unsupported frame schema".to_string(),
            ));
        }
        expect_key(&mut decoder, 1)?;
        let session = decoder.bytes().map_err(malformed)?;
        let session_id: [u8; 16] = session
            .try_into()
            .map_err(|_| TransportError::MalformedFrame("session id length".to_string()))?;
        expect_key(&mut decoder, 2)?;
        let sequence = decoder.u64().map_err(malformed)?;
        expect_key(&mut decoder, 3)?;
        let kind = FrameKind::try_from(decoder.u8().map_err(malformed)?)?;
        expect_key(&mut decoder, 4)?;
        let payload = decoder.bytes().map_err(malformed)?.to_vec();
        if payload.len() > limits.maximum_payload_bytes || decoder.position() != bytes.len() {
            return Err(TransportError::FrameTooLarge);
        }
        Ok(Self {
            session_id,
            sequence,
            kind,
            payload,
        })
    }
}

fn expect_key(decoder: &mut Decoder<'_>, expected: u8) -> TransportResult<()> {
    let actual = decoder.u8().map_err(malformed)?;
    if actual != expected {
        return Err(TransportError::MalformedFrame(
            "non-canonical frame key order".to_string(),
        ));
    }
    Ok(())
}

fn malformed(error: minicbor::decode::Error) -> TransportError {
    TransportError::MalformedFrame(error.to_string())
}
