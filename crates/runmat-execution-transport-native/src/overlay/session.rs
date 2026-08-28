use crate::frame::{FrameKind, WireFrame};
use crate::{TransportError, TransportResult};

#[derive(Debug, Clone)]
pub struct OverlaySession {
    session_id: [u8; 16],
    next_sequence: u64,
}

impl OverlaySession {
    pub fn new(session_id: [u8; 16]) -> Self {
        Self {
            session_id,
            next_sequence: 0,
        }
    }

    pub fn frame(&mut self, kind: FrameKind, payload: Vec<u8>) -> TransportResult<WireFrame> {
        let sequence = self.next_sequence;
        self.next_sequence = sequence.checked_add(1).ok_or(TransportError::Overflow)?;
        Ok(WireFrame {
            session_id: self.session_id,
            sequence,
            kind,
            payload,
        })
    }
}
