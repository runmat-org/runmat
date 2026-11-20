//! Rate limiting for live progress counts.
//!
//! A host draws a progress bar from `Advanced` events. Each phase may report several counts
//! (faces meshed, points inserted, edges created), so the limiter keeps one stream per phase and
//! `unit` metadata value. A stream passes an event when all of these hold:
//!
//! - at least [`MINIMUM_EVENT_INTERVAL`] has passed since its previous event (about 10 events
//!   per second at most);
//! - its fraction moved by at least [`MINIMUM_FRACTION_STEP`], or its count grew by that share
//!   when there is no total, or [`HEARTBEAT_INTERVAL`] passed, or its count dropped (a new run
//!   of the same loop, such as the next linear solve, restarts the count).
//!
//! The final count of a stream (`current == total`) always passes, and so do `Started`,
//! `Completed`, `Failed` and `Cancelled` events and heartbeats.
use std::collections::BTreeMap;
use std::time::Duration;

use runmat_time::Instant;

use super::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};

/// Shortest spacing between two count events of one stream, in milliseconds.
pub const MINIMUM_EVENT_INTERVAL_MS: u64 = 100;
/// Shortest spacing between two count events of one stream.
pub const MINIMUM_EVENT_INTERVAL: Duration = Duration::from_millis(MINIMUM_EVENT_INTERVAL_MS);
/// Spacing of heartbeats while a phase reports no new count, in milliseconds.
pub const HEARTBEAT_INTERVAL_MS: u64 = 2000;
/// Spacing of heartbeats while a phase reports no new count.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(HEARTBEAT_INTERVAL_MS);
/// How often a host's heartbeat timer calls [`super::ProgressRelay::heartbeat`], in
/// milliseconds. The relay sends a heartbeat only when [`HEARTBEAT_INTERVAL`] has passed, so
/// heartbeats arrive at most this much later than due.
pub const HEARTBEAT_POLL_INTERVAL_MS: u64 = 250;
/// How often a host's heartbeat timer calls [`super::ProgressRelay::heartbeat`].
pub const HEARTBEAT_POLL_INTERVAL: Duration = Duration::from_millis(HEARTBEAT_POLL_INTERVAL_MS);
/// Smallest fraction (or relative count) change that is worth an event.
pub const MINIMUM_FRACTION_STEP: f64 = 0.005;

/// Metadata key naming the counted quantity of an `Advanced` event.
pub const UNIT_METADATA: &str = "unit";
/// Metadata key marking an event that carries no new count.
pub const HEARTBEAT_METADATA: &str = "heartbeat";
/// Metadata key with the milliseconds since the phase started.
pub const ELAPSED_METADATA: &str = "elapsedMs";

#[derive(Clone, Copy)]
struct StreamState {
    emitted_at: Instant,
    current: Option<u64>,
    fraction: Option<f64>,
}

/// Admits or drops progress events by the rules in the module documentation.
#[derive(Default)]
pub struct ProgressRateLimiter {
    streams: BTreeMap<(FeaProgressPhase, String), StreamState>,
}

impl ProgressRateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `event`, observed at `now`, should reach the host.
    pub fn admit(&mut self, event: &FeaProgressEvent, now: Instant) -> bool {
        if event.status != FeaProgressStatus::Advanced {
            // A phase boundary restarts its counts.
            self.streams.retain(|(phase, _), _| *phase != event.phase);
            return true;
        }
        if event.metadata.contains_key(HEARTBEAT_METADATA) {
            return true;
        }
        let key = (
            event.phase,
            event
                .metadata
                .get(UNIT_METADATA)
                .cloned()
                .unwrap_or_default(),
        );
        let next = StreamState {
            emitted_at: now,
            current: event.current,
            fraction: event.fraction,
        };
        let Some(previous) = self.streams.get(&key).copied() else {
            self.streams.insert(key, next);
            return true;
        };
        let is_final = matches!((event.current, event.total), (Some(current), Some(total)) if current == total)
            && previous.current != event.current;
        let elapsed = now.saturating_duration_since(previous.emitted_at);
        let admitted = is_final
            || (elapsed >= MINIMUM_EVENT_INTERVAL
                && (elapsed >= HEARTBEAT_INTERVAL || moved_enough(&previous, event)));
        if admitted {
            self.streams.insert(key, next);
        }
        admitted
    }
}

fn moved_enough(previous: &StreamState, event: &FeaProgressEvent) -> bool {
    if let (Some(before), Some(after)) = (previous.current, event.current) {
        if after < before {
            return true;
        }
    }
    match (previous.fraction, event.fraction) {
        (Some(before), Some(after)) => after - before >= MINIMUM_FRACTION_STEP,
        _ => match (previous.current, event.current) {
            (Some(before), Some(after)) => {
                after > before && (after - before) as f64 >= MINIMUM_FRACTION_STEP * before as f64
            }
            (None, Some(_)) => true,
            _ => false,
        },
    }
}

#[cfg(test)]
#[path = "rate_tests.rs"]
mod tests;
