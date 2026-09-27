//! Provider-owned evidence for one candidate execution, not a scheduler.
//!
//! Begin before candidate-specific staging. Providers must track every submitted
//! operation (including transfers/adapters), then the caller publishes only after
//! output validation/adoption succeeds. Returning a handle is not completion.
//! Callbacks retain only thread-safe recorders; the session owning the receipt
//! consumes the result on its own thread. No callback invokes runtime services.
//! Nested operations should share the current recorder, not record a second total.
mod activation;
mod state;

pub use activation::{
    current_provider_observation, observe_provider_future, with_provider_observation,
};
pub use state::{ProviderOperationReceipt, ProviderOperationRecorder, ProviderSubmission};

use std::sync::Arc;
use std::time::Duration;

/// Explicit absence of evidence is not synchronous completion.
pub enum ProviderObservationSupport {
    Unsupported,
    Observed(ProviderOperationReceipt),
}

/// Monotonic timestamps from one epoch. None means trustworthy timing is absent.
/// Implementations must be nonblocking and safe on provider callback threads.
pub trait ProviderObservationClock: Send + Sync {
    fn now(&self) -> Option<Duration>;
}

struct Clock(runmat_time::Instant);
impl ProviderObservationClock for Clock {
    fn now(&self) -> Option<Duration> {
        Some(self.0.elapsed())
    }
}

pub(super) fn clock() -> Arc<dyn ProviderObservationClock> {
    Arc::new(Clock(runmat_time::Instant::now()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderObservationFailure {
    Cancelled,
    Abandoned,
    ProviderFailed,
    PublicationFailed,
    InvalidTiming,
    TrackingOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderObservationStatus {
    Pending,
    Completed {
        total_elapsed: Duration,
        submissions: u64,
    },
    Failed(ProviderObservationFailure),
}

/// Stage measurements are separate from the end-to-end total: intervals may
/// overlap. These are measured durations, never enqueue-time execution estimates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderObservationStage {
    Preparation,
    Allocation,
    Upload,
    Queue,
    Execution,
    Synchronization,
    Download,
    Materialization,
}

/// Bounded aggregates per stage; no per-submission event log is retained.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProviderStageMeasurement {
    pub samples: u64,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderOperationObservation {
    pub status: ProviderObservationStatus,
    pub stages: [ProviderStageMeasurement; 8],
    /// Actual selected pipeline events, not global cache-counter differences.
    pub pipelines: ProviderPipelineObservations,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProviderPipelineObservations {
    pub ready: u64,
    pub warm: u64,
    pub cold: u64,
    pub preparing: u64,
}

#[cfg(test)]
mod tests;
