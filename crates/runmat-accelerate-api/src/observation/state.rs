use super::*;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

#[path = "state/resources.rs"]
mod resources;
#[path = "state/wgpu_measurement.rs"]
mod wgpu_measurement;
use resources::ResourceState;
pub use resources::{
    ProviderResourcePhasePlan, ProviderResourcePhaseToken, ProviderResourceScope,
    ProviderResourceSnapshot, ProviderResourceTicket, ProviderScopedResourceSnapshot,
};
use wgpu_measurement::WgpuMeasurementAggregate;
mod compound;
mod real_unary;
mod stochastic_transfer;
pub use stochastic_transfer::{
    stochastic_evolution_source_transfer_retirement_host_upper_bytes,
    StochasticEvolutionSourceTransferRetirement,
    StochasticEvolutionSourceTransferRetirementNotifier,
};
mod host_metadata;

struct State {
    compound: Option<compound::CompoundObservation>,
    start: Option<Duration>,
    publication: Option<Duration>,
    completion: Option<Duration>,
    registered: u64,
    pending: u64,
    failure: Option<ProviderObservationFailure>,
    taken: bool,
    selected_execution: Option<bool>,
    selected_image_upload_device: Option<u32>,
    conditional_host_fallback: bool,
    conditional_provider_upload: bool,
    conditional_provider_upload_active: bool,
    preparation_fallback: bool,
    provider_host_mediated_allowed: bool,
    route_outcome: Option<ProviderRouteOutcome>,
    expected_execution: Option<(u32, crate::ProviderOperationIdentity)>,
    stages: [ProviderStageMeasurement; ProviderObservationStage::COUNT],
    pipelines: ProviderPipelineObservations,
    wgpu: WgpuMeasurementAggregate,
    resources: ResourceState,
    waiter: Option<Waker>,
    resource_waiter: Option<Waker>,
    operation_host_metadata:
        Option<Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}

impl State {
    fn observation(&self, status: ProviderObservationStatus) -> ProviderOperationObservation {
        let mut observation = self.timing_observation(status);
        observation.resources = self.resources.snapshot();
        observation
    }

    fn timing_observation(
        &self,
        status: ProviderObservationStatus,
    ) -> ProviderOperationObservation {
        let wgpu = self.wgpu.snapshot(self.registered, status);
        ProviderOperationObservation {
            status,
            route_outcome: self.route_outcome,
            stages: self.stages,
            verified_wgpu_compute_elapsed: wgpu.compute_elapsed,
            verified_wgpu_dispatches: wgpu.dispatches,
            wgpu_query_allocations: wgpu.query_allocations,
            pipelines: self.pipelines,
            resources: None,
        }
    }

    fn status(&self) -> ProviderObservationStatus {
        if let Some(failure) = self.failure {
            return ProviderObservationStatus::Failed(failure);
        }
        match (self.start, self.publication) {
            (Some(start), Some(publication))
                if self.pending == 0
                    && self.resources.live_count() == 0
                    && !self.resources.scope_pending() =>
            {
                match publication
                    .max(self.completion.unwrap_or(publication))
                    .checked_sub(start)
                {
                    Some(total_elapsed) => ProviderObservationStatus::Completed {
                        total_elapsed,
                        submissions: self.registered,
                    },
                    None => {
                        ProviderObservationStatus::Failed(ProviderObservationFailure::InvalidTiming)
                    }
                }
            }
            _ => ProviderObservationStatus::Pending,
        }
    }

    fn fail(&mut self, reason: ProviderObservationFailure) {
        if !self.taken && self.failure.is_none() {
            self.failure = Some(reason);
        }
    }

    fn timestamp(&mut self, time: Option<Duration>) -> Option<Duration> {
        let time = time.filter(|time| self.start.is_some_and(|start| *time >= start));
        if time.is_none() {
            self.fail(ProviderObservationFailure::InvalidTiming);
        }
        time
    }
}

struct Shared {
    state: Mutex<State>,
    clock: Arc<dyn ProviderObservationClock>,
    budget: Arc<ProviderObservationBudget>,
    // Only the default clock can be compared to an outer runmat_time::Instant.
    // Synthetic/custom clocks remain valid for provider-local v2 receipts.
    host_start: Option<runmat_time::Instant>,
}

// Kept outside callback-shared state: an outstanding map callback must not hold
// the worker's channel sender alive while that worker waits for channel closure.
#[derive(Clone, Default)]
#[cfg(feature = "wgpu")]
struct ProgressOwner {
    port: Option<Arc<dyn WgpuObservationProgress>>,
}

impl Shared {
    fn with<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = f(&mut state);
        let wake = if state.status() != ProviderObservationStatus::Pending {
            state.waiter.take()
        } else {
            None
        };
        let resource_wake = if state.resources.live_count() == 0 {
            state.resource_waiter.take()
        } else {
            None
        };
        drop(state);
        if let Some(wake) = wake {
            wake.wake();
        }
        if let Some(wake) = resource_wake {
            wake.wake();
        }
        result
    }
}

/// Unique session-side owner. Dropping an unconsumed receipt abandons it;
/// provider callbacks cannot resurrect it or deliver feedback themselves.
pub struct ProviderOperationReceipt(Arc<Shared>, #[cfg(feature = "wgpu")] ProgressOwner);

impl Default for ProviderOperationReceipt {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderOperationReceipt {
    pub fn new() -> Self {
        Self::with_budget(Arc::new(ProviderObservationBudget::new(
            std::num::NonZeroU32::new(64).expect("nonzero local observation limit"),
        )))
    }

    pub fn with_clock(clock: Arc<dyn ProviderObservationClock>) -> Self {
        Self::with_clock_and_budget(
            clock,
            Arc::new(ProviderObservationBudget::new(
                std::num::NonZeroU32::new(64).expect("nonzero local observation limit"),
            )),
        )
    }

    pub fn with_budget(budget: Arc<ProviderObservationBudget>) -> Self {
        let host_start = runmat_time::Instant::now();
        Self::construct(
            Arc::new(Clock(host_start)),
            budget,
            Some(host_start),
            Some(Duration::ZERO),
        )
    }

    #[cfg(feature = "wgpu")]
    pub fn with_wgpu_progress(
        budget: Arc<ProviderObservationBudget>,
        progress: Arc<dyn WgpuObservationProgress>,
    ) -> Self {
        let mut receipt = Self::with_budget(budget);
        receipt.1.port = Some(progress);
        receipt
    }

    pub fn with_clock_and_budget(
        clock: Arc<dyn ProviderObservationClock>,
        budget: Arc<ProviderObservationBudget>,
    ) -> Self {
        Self::construct(clock, budget, None, None)
    }

    fn construct(
        clock: Arc<dyn ProviderObservationClock>,
        budget: Arc<ProviderObservationBudget>,
        host_start: Option<runmat_time::Instant>,
        initial_start: Option<Duration>,
    ) -> Self {
        let start = initial_start.or_else(|| clock.now());
        Self(
            Arc::new(Shared {
                state: Mutex::new(State {
                    start,
                    publication: None,
                    completion: None,
                    registered: 0,
                    pending: 0,
                    failure: start
                        .is_none()
                        .then_some(ProviderObservationFailure::InvalidTiming),
                    taken: false,
                    selected_execution: None,
                    selected_image_upload_device: None,
                    compound: None,
                    conditional_host_fallback: false,
                    conditional_provider_upload: false,
                    conditional_provider_upload_active: false,
                    preparation_fallback: false,
                    provider_host_mediated_allowed: false,
                    route_outcome: None,
                    expected_execution: None,
                    stages: [ProviderStageMeasurement::default(); ProviderObservationStage::COUNT],
                    pipelines: ProviderPipelineObservations::default(),
                    wgpu: WgpuMeasurementAggregate::default(),
                    resources: ResourceState::default(),
                    waiter: None,
                    resource_waiter: None,
                    operation_host_metadata: None,
                }),
                clock,
                budget,
                host_start,
            }),
            #[cfg(feature = "wgpu")]
            ProgressOwner::default(),
        )
    }

    /// The exact default-clock prelude preceding this receipt. A custom clock
    /// has no shared epoch and must not contribute a full-call positive sample.
    pub fn default_clock_prelude_since(
        &self,
        outer_started: runmat_time::Instant,
    ) -> Option<Duration> {
        self.0.host_start?.checked_duration_since(outer_started)
    }

    pub fn recorder(&self) -> ProviderOperationRecorder {
        ProviderOperationRecorder(
            self.0.clone(),
            #[cfg(feature = "wgpu")]
            self.1.clone(),
        )
    }

    /// Direct admitted bodies may fall back internally. Require their explicit
    /// route witness before publication, in addition to provider completion.
    /// Call before staging; a late change cannot retroactively certify work.
    pub fn require_selected_execution(&mut self) {
        self.0.with(|state| {
            if state.registered != 0 || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.selected_execution = Some(false);
            }
        });
    }

    /// Bind one host image upload to this selected receipt before any provider
    /// submission. The upload is not proof that the selected stages executed.
    pub fn require_selected_image_upload(&mut self, device_id: u32) {
        self.0.with(|state| {
            if state.registered != 0
                || state.publication.is_some()
                || state.taken
                || state.selected_execution != Some(false)
                || state.selected_image_upload_device.is_some()
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.selected_image_upload_device = Some(device_id);
            }
        });
    }

    /// Bind a direct body's route witness to the selected device registration
    /// and canonical operation. Device IDs use the provider registry's existing
    /// process-lifetime uniqueness authority; no provider pointer is retained.
    pub fn require_provider_execution(
        &mut self,
        device_id: u32,
        operation: crate::ProviderOperationIdentity,
    ) {
        self.0.with(|state| {
            if state.registered != 0
                || state.publication.is_some()
                || state.taken
                || state.expected_execution.is_some()
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.selected_execution = Some(false);
                state.expected_execution = Some((device_id, operation));
            }
        });
    }

    /// A public form may require a provider attempt while expressly allowing
    /// a host result only after that hook returns Unsupported. Admit that
    /// conditional route before any staging; it is not a second selectable
    /// host candidate or a provider-execution witness.
    pub fn allow_conditional_host_fallback(&mut self) {
        self.0.with(|state| {
            if state.compound.is_some()
                || state.registered != 0
                || state.publication.is_some()
                || state.taken
                || state.conditional_provider_upload
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.conditional_host_fallback = true;
            }
        });
    }

    /// The provider's own exact hook may execute a typed native readback,
    /// host computation and resident upload. Require its distinct completion
    /// witness; neither an upload alone nor a direct-shader witness relabels it.
    pub fn allow_provider_host_mediated(&mut self) {
        self.0.with(|state| {
            if state.compound.is_some()
                || state.registered != 0
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.provider_host_mediated_allowed = true;
            }
        });
    }

    /// Admit a host-compute continuation whose final publication is uploaded
    /// to the same selected provider. It is legal only after an Unsupported
    /// hook result, and its physical stage is budgeted with the route.
    pub fn allow_conditional_provider_upload(&mut self) {
        self.0.with(|state| {
            if state.compound.is_some()
                || state.registered != 0
                || state.publication.is_some()
                || state.taken
                || state.conditional_host_fallback
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.conditional_provider_upload = true;
            }
        });
    }

    /// Activate the already admitted upload stage only after the provider
    /// hook returned Unsupported. Preparation uploads cannot certify or fence
    /// this continuation merely because they share the same outer receipt.
    pub fn begin_conditional_provider_upload(&mut self) {
        self.0.with(|state| {
            if !state.conditional_provider_upload
                || state.expected_execution.is_none()
                || state.selected_execution != Some(false)
                || state.publication.is_some()
                || state.taken
                || state.conditional_provider_upload_active
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.conditional_provider_upload_active = true;
            }
        });
    }

    /// A preparatory H upload returned Unsupported before the selected hook
    /// was invoked. Admit its already-budgeted host-to-provider continuation
    /// without misreporting that outcome as a hook fallback.
    pub fn begin_preparation_provider_upload(&mut self) {
        self.begin_conditional_provider_upload();
        self.0.with(|state| {
            if state.conditional_provider_upload_active && state.failure.is_none() {
                state.preparation_fallback = true;
            }
        });
    }

    /// Called by the owning runtime only after an actual Unsupported result
    /// and successful validation of the fallback host publication. Pending
    /// provider submissions, if any, still gate completion of this receipt.
    pub fn confirm_conditional_host_fallback(&mut self) {
        self.0.with(|state| {
            if !state.conditional_host_fallback
                || state.expected_execution.is_none()
                || state.selected_execution != Some(false)
                || state.publication.is_some()
                || state.taken
                || state.preparation_fallback
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::ConditionalHostFallback);
            }
        });
    }

    pub fn confirm_preparation_host_fallback(&mut self) {
        self.0.with(|state| {
            if !state.conditional_host_fallback
                || state.expected_execution.is_none()
                || state.selected_execution != Some(false)
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::PreparationHostFallback);
            }
        });
    }

    /// Called only after the Unsupported continuation uploaded and validated
    /// every published provider output. Pending submissions still gate the
    /// completed whole-call observation.
    pub fn confirm_conditional_provider_upload(&mut self) {
        self.0.with(|state| {
            if !state.conditional_provider_upload
                || !state.conditional_provider_upload_active
                || state.preparation_fallback
                || state.expected_execution.is_none()
                || state.selected_execution != Some(false)
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::ConditionalProviderUpload);
            }
        });
    }

    pub fn confirm_preparation_provider_upload(&mut self) {
        self.0.with(|state| {
            if !state.conditional_provider_upload
                || !state.conditional_provider_upload_active
                || !state.preparation_fallback
                || state.expected_execution.is_none()
                || state.selected_execution != Some(false)
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::PreparationProviderUpload);
            }
        });
    }

    /// Seal submission registration after all selected-route work and successful
    /// output publication. Positively identified synchronous/empty work with no
    /// submissions completes here. Missing instrumentation is not such evidence;
    /// an opaque provider must use ProviderObservationSupport::Unsupported.
    pub fn publish(&mut self) -> Result<(), ProviderObservationFailure> {
        let now = self.0.clock.now();
        self.0.with(|state| {
            if let Some(reason) = state.failure {
                return Err(reason);
            }
            if state.selected_execution == Some(false) {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
                return Err(ProviderObservationFailure::ExecutionUnproven);
            }
            if state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::PublicationFailed);
                return Err(ProviderObservationFailure::PublicationFailed);
            }
            state.publication = state.timestamp(now);
            state.failure.map_or(Ok(()), Err)
        })
    }

    pub fn fail(&mut self, reason: ProviderObservationFailure) {
        self.0.with(|state| state.fail(reason));
    }

    /// Readiness only: does not consume evidence or call any session service.
    pub fn status(&self) -> ProviderObservationStatus {
        self.0.with(|state| state.status())
    }

    /// Deliver at most once on the session's owning thread. The total was fixed
    /// at publication/completion, never at this potentially delayed drain.
    pub fn take_completed(&mut self) -> Option<ProviderOperationObservation> {
        self.0.with(|state| {
            let status = state.status();
            if state.taken || status == ProviderObservationStatus::Pending {
                return None;
            }
            state.taken = true;
            Some(state.observation(status))
        })
    }

    /// Race-safe single-owner wait. Callback threads only wake this caller;
    /// session feedback is still delivered by the caller on its owning thread.
    /// Ready(None) means evidence was already consumed. No custom Waker method
    /// (including dropping a replaced waker) runs while the state lock is held.
    pub fn poll_completed(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ProviderOperationObservation>> {
        self.poll_completion(cx, true)
    }

    /// Consume timing evidence without allocating a diagnostic resource
    /// snapshot. Shares the ordinary completion authority and taken bit;
    /// callers needing per-scope evidence must use `poll_completed` instead.
    pub fn poll_completed_timing(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ProviderOperationObservation>> {
        self.poll_completion(cx, false)
    }

    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
        include_resources: bool,
    ) -> Poll<Option<ProviderOperationObservation>> {
        let mut next = Some(cx.waker().clone());
        let (result, old) = self.0.with(|state| {
            if state.taken {
                return (Poll::Ready(None), None);
            }
            let status = state.status();
            if status == ProviderObservationStatus::Pending {
                return (
                    Poll::Pending,
                    std::mem::replace(&mut state.waiter, next.take()),
                );
            }
            state.taken = true;
            let observation = if include_resources {
                state.observation(status)
            } else {
                state.timing_observation(status)
            };
            (Poll::Ready(Some(observation)), None)
        });
        drop(old);
        result
    }

    /// Wait only for issued resource-ticket retirement, independently of the
    /// operation's success, failure, publication, or consumed feedback. `None`
    /// means no resource scope was ever issued; an unfinished or cancelled
    /// scope remains unfinished/cancelled in the returned snapshot.
    pub fn poll_resource_retirement(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ProviderResourceSnapshot>> {
        let mut next = Some(cx.waker().clone());
        let (result, old) = self.0.with(|state| {
            if state.resources.live_count() != 0 {
                // A pending poll checks only the aggregate owner count. Clone
                // the potentially many scoped facts once, when ready.
                (
                    Poll::Pending,
                    std::mem::replace(&mut state.resource_waiter, next.take()),
                )
            } else {
                (Poll::Ready(state.resources.snapshot()), None)
            }
        });
        // A replaced Waker may itself run code in Drop. Never drop it locked.
        drop(old);
        result
    }
}

impl Drop for ProviderOperationReceipt {
    fn drop(&mut self) {
        self.0
            .with(|state| state.fail(ProviderObservationFailure::Abandoned));
    }
}

/// Thread-safe callback authority; cannot publish outputs or consume feedback.
#[derive(Clone)]
pub struct ProviderOperationRecorder(Arc<Shared>, #[cfg(feature = "wgpu")] ProgressOwner);

impl ProviderOperationRecorder {
    /// Compare the actual observation owner, never an operation label or a
    /// later poll-local selection.
    pub fn same_observation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn confirm_provider_host_mediated_after(
        &self,
        device_id: u32,
        operation: &crate::ProviderOperationIdentity,
        checkpoint: u64,
    ) {
        self.0.with(|state| {
            if !state.provider_host_mediated_allowed
                || state
                    .expected_execution
                    .as_ref()
                    .is_none_or(|(expected, identity)| {
                        *expected != device_id || identity != operation
                    })
                || state.selected_execution != Some(false)
                || state.registered <= checkpoint
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::ProviderHostMediated);
            }
        });
    }

    /// True only in the Unsupported continuation, after the owning runtime
    /// activated its admitted provider-upload stage. A raw upload can then
    /// attach a completion fence without claiming the original shader ran.
    pub fn observes_conditional_provider_upload(&self, device_id: u32) -> bool {
        self.0.with(|state| {
            state.conditional_provider_upload_active
                && state
                    .expected_execution
                    .as_ref()
                    .is_some_and(|(expected, _)| *expected == device_id)
                && state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
        })
    }

    /// Physical completion of the selected source upload remains distinct
    /// from the later selected-stage execution witness.
    pub fn observes_selected_image_upload(&self, device_id: u32) -> bool {
        self.0.with(|state| {
            state.selected_image_upload_device == Some(device_id)
                && state.selected_execution == Some(false)
                && state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
        })
    }

    /// Domain owners can ignore an unrelated outer operation's receipt when
    /// they execute as a nested helper. A matching operation still validates
    /// its actual provider registration when it confirms execution.
    pub fn expects_operation(&self, operation: &crate::ProviderOperationIdentity) -> bool {
        self.0.with(|state| {
            state
                .expected_execution
                .as_ref()
                .is_some_and(|(_, expected)| expected == operation)
        })
    }

    /// Snapshot submission registration immediately before an operation hook.
    /// Earlier uploads and unrelated preparation cannot satisfy its witness.
    pub fn submission_checkpoint(&self) -> u64 {
        self.0.with(|state| state.registered)
    }

    /// The selected provider hook, not the public binding, certifies that it
    /// registered a new submission while executing this exact operation.
    /// The hook must also know that its successful nonempty path launches the
    /// required kernel; a cast or transfer alone is not that proof.
    pub fn confirm_submitted_provider_execution_after(
        &self,
        device_id: u32,
        operation: &crate::ProviderOperationIdentity,
        checkpoint: u64,
    ) {
        self.0.with(|state| {
            if state.registered <= checkpoint {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            }
        });
        self.confirm_provider_execution(device_id, operation);
    }

    /// The operation owner must mark fallback even if it subsequently restores
    /// a resident result. A later selected-route event cannot undo invalidation.
    pub fn execution_disposition(&self, disposition: ProviderExecutionDisposition) {
        self.0.with(|state| match disposition {
            ProviderExecutionDisposition::SelectedProvider => {
                if state.expected_execution.is_some() {
                    state.fail(ProviderObservationFailure::ExecutionUnproven);
                } else if state.publication.is_none() && state.selected_execution.is_some() {
                    state.selected_execution = Some(true);
                    state.route_outcome = Some(ProviderRouteOutcome::SelectedProvider);
                }
            }
            ProviderExecutionDisposition::SharedRuntimeFallback => {
                state.fail(ProviderObservationFailure::SharedRuntimeFallback);
            }
            ProviderExecutionDisposition::Unproven => {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            }
        });
    }

    /// Called only by the actual operation owner after validated execution.
    /// Same-owner facades share the registry identity, not a vtable address.
    /// A mismatched or missing planned route fails closed and cannot be repaired.
    pub fn confirm_provider_execution(
        &self,
        device_id: u32,
        operation: &crate::ProviderOperationIdentity,
    ) {
        self.0.with(|state| {
            let compound_rejects = state.compound.as_ref().is_some_and(|compound| {
                !compound.allows_provider_execution_confirmation(device_id)
            });
            if compound_rejects
                || state.publication.is_some()
                || state.taken
                || state.provider_host_mediated_allowed
                || state.expected_execution.as_ref().is_none_or(
                    |(expected_device, expected_operation)| {
                        *expected_device != device_id || expected_operation != operation
                    },
                )
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::SelectedProvider);
            }
        });
    }

    pub fn fail(&self, reason: ProviderObservationFailure) {
        self.0.with(|state| state.fail(reason));
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu_progress(&self) -> Option<Arc<dyn WgpuObservationProgress>> {
        self.1.port.clone()
    }
    /// Register before submission. Each token is completed exactly once, or its
    /// drop marks observation failure. Registration after publication is rejected.
    pub fn submission(&self) -> Result<ProviderSubmission, ProviderObservationFailure> {
        let Some(permit) = self.0.budget.acquire() else {
            self.0
                .with(|state| state.fail(ProviderObservationFailure::ResourceLimit));
            return Err(ProviderObservationFailure::ResourceLimit);
        };
        self.0.with(|state| {
            if let Some(reason) = state.failure {
                return Err(reason);
            }
            if state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::PublicationFailed);
                return Err(ProviderObservationFailure::PublicationFailed);
            }
            let Some(registered) = state.registered.checked_add(1) else {
                state.fail(ProviderObservationFailure::TrackingOverflow);
                return Err(ProviderObservationFailure::TrackingOverflow);
            };
            state.registered = registered;
            state.pending += 1; // pending <= registered, whose increment was checked
            Ok(ProviderSubmission {
                shared: self.0.clone(),
                consumed: false,
                _permit: permit,
            })
        })
    }

    /// Explicit provider measurement only. Do not supply queue-submit latency as
    /// execution, or sum these overlapping stage aggregates into total elapsed.
    /// Record a submission's stages before completing its token. Evidence freezes
    /// once publication and all submissions complete (or failure occurs), not at
    /// session delivery; late measurements are ignored.
    pub fn record_stage(&self, stage: ProviderObservationStage, elapsed: Duration) {
        self.0.with(|state| {
            if state.taken || state.status() != ProviderObservationStatus::Pending {
                return;
            }
            let slot = &mut state.stages[stage.index()];
            match (
                slot.samples.checked_add(1),
                slot.elapsed.checked_add(elapsed),
            ) {
                (Some(samples), Some(elapsed)) => {
                    *slot = ProviderStageMeasurement { samples, elapsed }
                }
                _ => state.fail(ProviderObservationFailure::TrackingOverflow),
            }
        });
    }

    /// Record readiness at the actual pipeline owner/cache lookup. This does not
    /// establish pre-execution cost-query readiness or replace exact cache keys.
    /// Record measured preparation duration separately via record_stage. As with
    /// stage measurements, events after terminal completion are ignored.
    pub fn record_pipeline(&self, preparation: runmat_execution::CandidatePreparationState) {
        use runmat_execution::CandidatePreparationState;
        self.0.with(|state| {
            if state.taken || state.status() != ProviderObservationStatus::Pending {
                return;
            }
            let count = match preparation {
                CandidatePreparationState::Ready => &mut state.pipelines.ready,
                CandidatePreparationState::Warm => &mut state.pipelines.warm,
                CandidatePreparationState::Cold => &mut state.pipelines.cold,
                CandidatePreparationState::Preparing => &mut state.pipelines.preparing,
                CandidatePreparationState::Unknown => &mut state.pipelines.unknown,
            };
            match count.checked_add(1) {
                Some(next) => *count = next,
                None => state.fail(ProviderObservationFailure::TrackingOverflow),
            }
        });
    }
}

pub struct ProviderSubmission {
    shared: Arc<Shared>,
    consumed: bool,
    _permit: budget::Permit,
}

impl ProviderSubmission {
    /// Call only at actual completion of this submission, never at enqueue.
    pub fn complete(self) {
        self.capture_completion().complete();
    }

    /// Capture callback delivery time without publishing successful evidence.
    /// Keep the returned token until provider-owned fence cleanup succeeds.
    pub fn capture_completion(self) -> ProviderCompletedSubmission {
        let now = self.shared.clock.now();
        ProviderCompletedSubmission {
            submission: self,
            timestamp: now,
        }
    }

    fn complete_at(
        mut self,
        now: Option<Duration>,
        measurement: Option<WgpuSubmissionMeasurement>,
    ) {
        self.shared.with(|state| {
            if state.failure.is_some() || state.taken {
                return;
            }
            if let Some(now) = state.timestamp(now) {
                state.wgpu.record(measurement);
                state.completion = Some(state.completion.map_or(now, |previous| previous.max(now)));
                state.pending -= 1;
            }
        });
        self.consumed = true;
    }

    pub fn fail(mut self) {
        self.shared
            .with(|state| state.fail(ProviderObservationFailure::ProviderFailed));
        self.consumed = true;
    }
}

/// Actual callback timing, still pending until safe provider cleanup. This does
/// not expose timestamp construction or a second clock. Dropping the token
/// fails observation through the original submission's abandonment contract.
pub struct ProviderCompletedSubmission {
    submission: ProviderSubmission,
    timestamp: Option<Duration>,
}

impl ProviderCompletedSubmission {
    pub fn complete(self) {
        self.submission.complete_at(self.timestamp, None);
    }

    /// Settle actual optional WGPU facts only after fence/readback resources
    /// have retired. Missing fields remain independently unknown.
    pub fn complete_with_wgpu(self, measurement: WgpuSubmissionMeasurement) {
        self.submission
            .complete_at(self.timestamp, Some(measurement));
    }

    pub fn fail(self) {
        self.submission.fail();
    }
}

impl Drop for ProviderSubmission {
    fn drop(&mut self) {
        if !self.consumed {
            self.shared
                .with(|state| state.fail(ProviderObservationFailure::Abandoned));
        }
    }
}
