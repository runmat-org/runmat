//! A host-side progress relay: rate limiting plus heartbeats.
//!
//! The relay wraps the host's handler. Every event passes through [`ProgressRateLimiter`]. While
//! a phase is open (its last event was `Started` or `Advanced`) and no event reached the host for
//! [`HEARTBEAT_INTERVAL`], [`ProgressRelay::heartbeat`] sends an `Advanced` event that repeats
//! the phase's last count and message and adds `heartbeat = "true"` and `elapsedMs` (time since
//! the phase started). A host can then tell a phase that is still working from one that stopped
//! reporting. Native hosts call [`ProgressRelay::spawn_heartbeat`] to send heartbeats from a
//! background thread; a browser host calls `heartbeat` from its own timer.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use runmat_time::Instant;

use super::rate::{
    ProgressRateLimiter, ELAPSED_METADATA, HEARTBEAT_INTERVAL, HEARTBEAT_METADATA,
};
use super::{FeaProgressEvent, FeaProgressHandler, FeaProgressStatus};

struct OpenPhase {
    started_at: Instant,
    last: FeaProgressEvent,
}

struct RelayState {
    limiter: ProgressRateLimiter,
    delivered_at: Option<Instant>,
    open: Option<OpenPhase>,
}

/// Rate-limited, heartbeat-capable forwarding to one host handler.
pub struct ProgressRelay {
    handler: FeaProgressHandler,
    state: Mutex<RelayState>,
}

impl ProgressRelay {
    pub fn new(handler: FeaProgressHandler) -> Arc<Self> {
        Arc::new(Self {
            handler,
            state: Mutex::new(RelayState {
                limiter: ProgressRateLimiter::new(),
                delivered_at: None,
                open: None,
            }),
        })
    }

    /// A handler that sends every event through this relay.
    pub fn handler(self: &Arc<Self>) -> FeaProgressHandler {
        let relay = Arc::clone(self);
        Arc::new(move |event| relay.emit(event))
    }

    pub fn emit(&self, event: FeaProgressEvent) {
        self.emit_at(event, Instant::now());
    }

    pub(crate) fn emit_at(&self, event: FeaProgressEvent, now: Instant) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.limiter.admit(&event, now) {
            return;
        }
        match event.status {
            FeaProgressStatus::Started => {
                state.open = Some(OpenPhase {
                    started_at: now,
                    last: event.clone(),
                });
            }
            FeaProgressStatus::Advanced => match state.open.as_mut() {
                Some(open) if open.last.phase == event.phase => open.last = event.clone(),
                _ => {
                    state.open = Some(OpenPhase {
                        started_at: now,
                        last: event.clone(),
                    })
                }
            },
            FeaProgressStatus::Completed
            | FeaProgressStatus::Failed
            | FeaProgressStatus::Cancelled => {
                if state
                    .open
                    .as_ref()
                    .is_some_and(|open| open.last.phase == event.phase)
                {
                    state.open = None;
                }
            }
        }
        state.delivered_at = Some(now);
        // Delivered under the lock so a heartbeat never overtakes a newer count.
        (self.handler)(event);
    }

    /// Sends a heartbeat when a phase is open and the host heard nothing for
    /// [`HEARTBEAT_INTERVAL`]. Returns whether one was sent.
    pub fn heartbeat(&self) -> bool {
        self.heartbeat_at(Instant::now())
    }

    pub(crate) fn heartbeat_at(&self, now: Instant) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let quiet = state
            .delivered_at
            .is_none_or(|at| now.saturating_duration_since(at) >= HEARTBEAT_INTERVAL);
        let Some(open) = state.open.as_ref().filter(|_| quiet) else {
            return false;
        };
        let elapsed_ms = now.saturating_duration_since(open.started_at).as_millis();
        let mut metadata = open
            .last
            .metadata
            .iter()
            .filter(|(key, _)| key.as_str() != ELAPSED_METADATA)
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        metadata.insert(HEARTBEAT_METADATA.into(), "true".into());
        metadata.insert(ELAPSED_METADATA.into(), elapsed_ms.to_string());
        let event = FeaProgressEvent {
            status: FeaProgressStatus::Advanced,
            metadata,
            ..open.last.clone()
        };
        state.delivered_at = Some(now);
        (self.handler)(event);
        true
    }

    /// Sends heartbeats from a background thread until the guard drops.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn spawn_heartbeat(self: &Arc<Self>) -> HeartbeatGuard {
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        let relay = Arc::clone(self);
        let thread = std::thread::Builder::new()
            .name("runmat-fea-progress-heartbeat".into())
            .spawn(move || {
                while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                    stopped.recv_timeout(HEARTBEAT_POLL_INTERVAL)
                {
                    relay.heartbeat();
                }
            })
            .ok();
        HeartbeatGuard {
            stop: Some(stop),
            thread,
        }
    }
}

/// How often the heartbeat thread checks for a quiet phase.
#[cfg(not(target_arch = "wasm32"))]
const HEARTBEAT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Stops the heartbeat thread when dropped.
#[cfg(not(target_arch = "wasm32"))]
pub struct HeartbeatGuard {
    stop: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
#[path = "relay_tests.rs"]
mod tests;
