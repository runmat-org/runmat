use std::time::Duration;

use crate::{TransportError, TransportResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeartbeatSchedule {
    pub interval: Duration,
    pub ttl: Duration,
    pub maximum_clock_skew: Duration,
}

impl HeartbeatSchedule {
    pub fn new(
        interval: Duration,
        ttl: Duration,
        maximum_clock_skew: Duration,
    ) -> TransportResult<Self> {
        let margin = interval
            .checked_add(maximum_clock_skew)
            .ok_or(TransportError::Overflow)?;
        if interval.is_zero() || ttl <= margin {
            return Err(TransportError::Unavailable(
                "heartbeat TTL has no reconnect/skew margin".to_string(),
            ));
        }
        Ok(Self {
            interval,
            ttl,
            maximum_clock_skew,
        })
    }
}
