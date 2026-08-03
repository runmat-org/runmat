use crate::{TransportError, TransportResult};

#[derive(Debug, Clone)]
pub struct ResumeState {
    total_bytes: u64,
    next_offset: u64,
}

impl ResumeState {
    pub fn new(total_bytes: u64) -> TransportResult<Self> {
        if total_bytes == 0 {
            return Err(TransportError::Integrity);
        }
        Ok(Self {
            total_bytes,
            next_offset: 0,
        })
    }

    pub fn accept(&mut self, offset: u64, byte_len: usize) -> TransportResult<()> {
        if offset != self.next_offset {
            return Err(TransportError::Integrity);
        }
        let end = offset
            .checked_add(u64::try_from(byte_len).map_err(|_| TransportError::Overflow)?)
            .ok_or(TransportError::Overflow)?;
        if end > self.total_bytes {
            return Err(TransportError::Integrity);
        }
        self.next_offset = end;
        Ok(())
    }

    pub fn next_offset(&self) -> u64 {
        self.next_offset
    }

    pub fn is_complete(&self) -> bool {
        self.next_offset == self.total_bytes
    }
}
