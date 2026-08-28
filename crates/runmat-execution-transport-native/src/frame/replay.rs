use crate::{TransportError, TransportResult};

#[derive(Debug, Clone, Default)]
pub struct ReplayWindow {
    highest: Option<u64>,
    seen: u64,
}

impl ReplayWindow {
    pub const WIDTH: u64 = 64;

    pub fn accept(&mut self, sequence: u64) -> TransportResult<()> {
        let Some(highest) = self.highest else {
            self.highest = Some(sequence);
            self.seen = 1;
            return Ok(());
        };
        if sequence > highest {
            let shift = sequence - highest;
            self.seen = if shift >= Self::WIDTH {
                1
            } else {
                (self.seen << shift) | 1
            };
            self.highest = Some(sequence);
            return Ok(());
        }
        let distance = highest - sequence;
        if distance >= Self::WIDTH {
            return Err(TransportError::Replay);
        }
        let bit = 1_u64 << distance;
        if self.seen & bit != 0 {
            return Err(TransportError::Replay);
        }
        self.seen |= bit;
        Ok(())
    }
}
