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
mod resource_activation;
mod state;
mod vm_region;
#[cfg(feature = "wgpu")]
pub use vm_region::record_source_region_wgpu_submission;
pub use vm_region::{
    record_source_region_provider_selection, record_source_region_unattributed_device_effect,
    set_source_region_for_current_poll, with_source_region_audit, with_source_region_poll,
    SourceRegionAuditAllocationError, SourceRegionRouteAudit, SourceRegionRouteAuditSlot,
};
#[cfg(feature = "wgpu")]
mod wgpu_progress;
mod wgpu_query;
#[cfg(feature = "wgpu")]
pub use wgpu_progress::{
    WgpuObservationCompletion, WgpuObservationProgress, WgpuObservationProgressError,
    WgpuObservationProgressReservation,
};

pub use activation::{
    current_provider_observation, observe_provider_future, with_provider_observation,
};
pub use budget::ProviderObservationBudget;
#[cfg(feature = "wgpu")]
pub use budget::{WgpuQueryPermit, WgpuTailPermit};
pub use resource_activation::{current_provider_resource_scope, observe_provider_resource_future};
pub use state::{
    stochastic_evolution_source_transfer_retirement_host_upper_bytes, ProviderCompletedSubmission,
    ProviderOperationReceipt, ProviderOperationRecorder, ProviderResourcePhasePlan,
    ProviderResourcePhaseToken, ProviderResourceScope, ProviderResourceSnapshot,
    ProviderResourceTicket, ProviderScopedResourceSnapshot, ProviderSubmission,
    StochasticEvolutionSourceTransferRetirement,
    StochasticEvolutionSourceTransferRetirementNotifier,
};
pub use wgpu_query::{
    WgpuEncoderQueryProspective, WgpuQueryAllocationDisposition, WgpuQueryAllocationObservation,
};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderObservationFailure {
    Cancelled,
    Abandoned,
    ProviderFailed,
    PublicationFailed,
    InvalidTiming,
    TrackingOverflow,
    ResourceLimit,
    ExecutionUnproven,
    SharedRuntimeFallback,
}

impl std::fmt::Display for ProviderObservationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "provider observation failed: {self:?}")
    }
}

impl std::error::Error for ProviderObservationFailure {}

/// Reported by the admitted operation owner, never inferred from output
/// residency or the presence of transfer submissions. Selected execution means
/// the selected provider's actual operation (including an audited identity/empty
/// route) succeeded; it does not claim device completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderExecutionDisposition {
    SelectedProvider,
    SharedRuntimeFallback,
    Unproven,
}

/// Physical instrumentation demand for the admitted operation. Requested buffer
/// bytes are not driver allocation/VRAM granularity. No timing estimate is implied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProviderObservationOverhead {
    pub submissions: u32,
    pub requested_peak_buffer_bytes: u64,
    pub clear_commands: u32,
    pub map_callbacks: u32,
    /// One conditional encoder-query demand (compute passes and, when
    /// supported, an outer whole-encoder interval). It is not a mandatory
    /// fence resource ticket. Absent for unsupported routes/devices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wgpu_compute_query: Option<WgpuEncoderQueryProspective>,
}

impl ProviderObservationOverhead {
    pub fn requested_total_peak_bytes(self) -> Option<u64> {
        self.requested_peak_buffer_bytes.checked_add(
            self.wgpu_compute_query
                .map_or(0, |query| query.requested_peak_buffer_bytes),
        )
    }
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

/// The route actually published by a conditional provider-first invocation.
/// A host fallback is never evidence that the selected provider kernel ran.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderRouteOutcome {
    SelectedProvider,
    /// Provider-owned exact hook ran its typed native readback/host compute/
    /// resident upload route rather than its direct shader.
    ProviderHostMediated,
    ConditionalHostFallback,
    /// The selected hook returned Unsupported; the admitted conditional host
    /// continuation published a validated result on the original provider.
    ConditionalProviderUpload,
    /// H preparation returned Unsupported before the selected hook began.
    PreparationHostFallback,
    PreparationProviderUpload,
    /// A consumed domain probe selected the original host continuation;
    /// this is not an Unsupported hook result or provider-kernel execution.
    DomainHostContinuation,
    DomainProviderUpload,
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
    pub route_outcome: Option<ProviderRouteOutcome>,
    pub stages: [ProviderStageMeasurement; ProviderObservationStage::COUNT],
    /// Aggregate device timestamp intervals for every registered WGPU
    /// submission. None includes unsupported or incomplete readback.
    pub verified_wgpu_compute_elapsed: Option<Duration>,
    /// Actual encoded-and-submitted dispatch calls, never queue submissions.
    /// None means at least one registered submission was not covered.
    pub verified_wgpu_dispatches: Option<u64>,
    /// Actual optional query allocations across completed submissions. None
    /// means not all submitted routes supplied allocation disposition.
    pub wgpu_query_allocations: Option<WgpuQueryAllocationObservation>,
    /// Actual selected pipeline events, not global cache-counter differences.
    pub pipelines: ProviderPipelineObservations,
    /// Explicitly issued resource tickets only; None means no coverage.
    pub resources: Option<ProviderResourceSnapshot>,
}

/// One physical WGPU submission, supplied only by its completion owner after
/// queue submission and safe readback retirement. The two measurements fail
/// independently: an unsupported timestamp query need not erase an exact
/// dispatch count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WgpuSubmissionMeasurement {
    /// Whole submitted encoder interval, excluding telemetry resolve/fence work.
    /// None is not a zero-duration transfer or a compute-only substitute.
    pub device_elapsed: Option<Duration>,
    pub compute_elapsed: Option<Duration>,
    pub dispatches: Option<u64>,
    pub query_allocation: Option<WgpuQueryAllocationDisposition>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProviderPipelineObservations {
    pub ready: u64,
    pub warm: u64,
    pub cold: u64,
    pub preparing: u64,
    pub unknown: u64,
}

#[cfg(test)]
mod tests;
