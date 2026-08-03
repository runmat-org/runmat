use std::time::Duration;

use crate::{TransportError, TransportResult};

#[derive(Debug, Clone)]
pub struct ReconnectBackoff {
    initial: Duration,
    maximum: Duration,
    current: Duration,
}

impl ReconnectBackoff {
    pub fn new(initial: Duration, maximum: Duration) -> TransportResult<Self> {
        if initial.is_zero() || maximum < initial {
            return Err(TransportError::Unavailable(
                "invalid reconnect backoff".to_string(),
            ));
        }
        Ok(Self {
            initial,
            maximum,
            current: initial,
        })
    }

    pub fn next_delay(&mut self) -> Duration {
        let value = self.current;
        self.current = self.current.saturating_mul(2).min(self.maximum);
        value
    }

    pub fn reset(&mut self) {
        self.current = self.initial;
    }
}
