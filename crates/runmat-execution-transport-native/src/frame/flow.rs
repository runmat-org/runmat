use crate::{TransportError, TransportResult};

#[derive(Debug, Clone)]
pub struct FlowWindow {
    maximum_in_flight_bytes: u64,
    in_flight_bytes: u64,
}

impl FlowWindow {
    pub fn new(maximum_in_flight_bytes: u64) -> TransportResult<Self> {
        if maximum_in_flight_bytes == 0 {
            return Err(TransportError::FlowControl);
        }
        Ok(Self {
            maximum_in_flight_bytes,
            in_flight_bytes: 0,
        })
    }

    pub fn reserve(&mut self, bytes: u64) -> TransportResult<()> {
        let next = self
            .in_flight_bytes
            .checked_add(bytes)
            .ok_or(TransportError::Overflow)?;
        if next > self.maximum_in_flight_bytes {
            return Err(TransportError::FlowControl);
        }
        self.in_flight_bytes = next;
        Ok(())
    }

    pub fn acknowledge(&mut self, bytes: u64) {
        self.in_flight_bytes = self.in_flight_bytes.saturating_sub(bytes);
    }
}
