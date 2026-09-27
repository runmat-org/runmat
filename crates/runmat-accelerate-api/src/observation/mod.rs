//! Provider-owned evidence for one candidate execution, not a scheduler.
//!
//! Begin before candidate-specific staging. Providers must track every submitted
//! operation (including transfers/adapters), then the caller publishes only after
//! output validation/adoption succeeds. Returning a handle is not completion.
//! Callbacks retain only thread-safe recorders; the session owning the receipt
//! consumes the result on its own thread. No callback invokes runtime services.
//! Nested operations should share the current recorder, not record a second total.
mod activation;
mod budget;
mod state;
#[cfg(feature = "wgpu")]
mod wgpu_progress;
#[cfg(feature = "wgpu")]
pub use wgpu_progress::{
    WgpuObservationProgress, WgpuObservationProgressError, WgpuObservationProgressReservation,
};

pub use activation::{
    current_provider_observation, observe_provider_future, with_provider_observation,
};
pub use budget::ProviderObservationBudget;
pub use state::{ProviderOperationReceipt, ProviderOperationRecorder, ProviderSubmission};

use std::sync::Arc;
use std::time::Duration;

/// Explicit absence of evidence is not synchronous completion. A provider may
/// return Observed only if it positively supports complete operation tracking,
/// or positively identifies this route as synchronous/empty. An opaque provider
/// must return Unsupported, never an empty receipt inferred from missing events.
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
    ResourceLimit,
}

/// Physical instrumentation demand for the admitted operation. Requested buffer
/// bytes are not driver allocation/VRAM granularity. No timing estimate is implied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderObservationOverhead {
    pub submissions: u32,
    pub requested_peak_buffer_bytes: u64,
    pub clear_commands: u32,
    pub map_callbacks: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderObservationCapability {
    Unsupported,
    Supported(ProviderObservationOverhead),
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

impl ProviderObservationStage {
    pub const ALL: [Self; 8] = [
        Self::Preparation,
        Self::Allocation,
        Self::Upload,
        Self::Queue,
        Self::Execution,
        Self::Synchronization,
        Self::Download,
        Self::Materialization,
    ];
    pub const COUNT: usize = Self::ALL.len();
    pub const fn index(self) -> usize {
        match self {
            Self::Preparation => 0,
            Self::Allocation => 1,
            Self::Upload => 2,
            Self::Queue => 3,
            Self::Execution => 4,
            Self::Synchronization => 5,
            Self::Download => 6,
            Self::Materialization => 7,
        }
    }
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
    pub stages: [ProviderStageMeasurement; ProviderObservationStage::COUNT],
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
