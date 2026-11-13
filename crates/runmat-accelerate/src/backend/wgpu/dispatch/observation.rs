//! Provider-owned private fences. Map callbacks never perform WGPU emission.
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderObservationBudget, ProviderObservationFailure,
    StochasticEvolutionSourceTransferRetirementNotifier, WgpuSubmissionMeasurement,
};
use runmat_accelerate_api::{WgpuDeviceAccess, WgpuEmission};
use runmat_execution::profiling::current_command_profiling;
use std::sync::Arc;

#[path = "observation/admission.rs"]
mod admission;
use admission::{fail_mismatched_admission, prepare, ProviderAdmission};
#[path = "observation/completion_signal.rs"]
mod completion_signal;
#[path = "observation/compute_pass.rs"]
mod compute_pass;
#[path = "observation/probe_ticket.rs"]
mod probe_ticket;
use compute_pass::{EncodedMeasurement, EncoderObservation};
pub(crate) use compute_pass::{
    ProfiledEncoder, SubmissionEncoder, TimestampCapability, MAX_QUERY_REQUESTED_BYTES,
};
pub(crate) fn prospective_encoder_query(
    max_compute_encoders: u32,
    max_total_encoders: u32,
) -> Option<runmat_accelerate_api::observation::WgpuEncoderQueryProspective> {
    if max_total_encoders == 0 || max_compute_encoders > max_total_encoders {
        return None;
    }
    Some(
        runmat_accelerate_api::observation::WgpuEncoderQueryProspective {
            max_compute_encoders,
            max_total_encoders,
            max_query_sets: max_total_encoders,
            max_timestamp_slots: u64::from(max_total_encoders)
                .checked_mul(u64::from(compute_pass::MAX_TIMESTAMP_SLOTS_PER_ENCODER))?,
            requested_peak_buffer_bytes: u64::from(max_total_encoders)
                .checked_mul(MAX_QUERY_REQUESTED_BYTES)?,
            max_map_callbacks: max_total_encoders,
        },
    )
}

pub(crate) fn gate_prospective_encoder_query(
    capability: runmat_accelerate_api::observation::ProviderObservationCapability,
    timestamp: Option<TimestampCapability>,
    budget: &ProviderObservationBudget,
) -> runmat_accelerate_api::observation::ProviderObservationCapability {
    use runmat_accelerate_api::observation::ProviderObservationCapability;
    match capability {
        ProviderObservationCapability::Supported(mut overhead) => {
            let outer = timestamp.is_some_and(TimestampCapability::inside_encoders)
                && current_command_profiling().is_some();
            let per_encoder = if outer {
                MAX_QUERY_REQUESTED_BYTES
            } else {
                compute_pass::MAX_COMPUTE_QUERY_REQUESTED_BYTES
            };
            let optional_limit = budget.wgpu_query_requested_byte_limit();
            let admitted_peak = (optional_limit / per_encoder) * per_encoder;
            if timestamp.is_none() || admitted_peak == 0 {
                overhead.wgpu_compute_query = None;
            } else if let Some(query) = &mut overhead.wgpu_compute_query {
                let encoders = if outer {
                    query.max_total_encoders
                } else {
                    query.max_compute_encoders
                };
                if encoders == 0 {
                    overhead.wgpu_compute_query = None;
                } else {
                    query.max_query_sets = encoders;
                    query.max_map_callbacks = encoders;
                    query.max_timestamp_slots = u64::from(encoders)
                        * u64::from(if outer {
                            compute_pass::MAX_TIMESTAMP_SLOTS_PER_ENCODER
                        } else {
                            compute_pass::MAX_TIMESTAMP_SLOTS_PER_ENCODER - 2
                        });
                    // Creation checked the larger whole-encoder bound; this
                    // selected per-encoder demand cannot overflow afterward.
                    query.requested_peak_buffer_bytes = u64::from(encoders)
                        .checked_mul(per_encoder)
                        .expect("checked prospective query geometry")
                        .min(admitted_peak);
                }
            }
            ProviderObservationCapability::Supported(overhead)
        }
        unsupported => unsupported,
    }
}
#[cfg(test)]
pub(crate) use probe_ticket::ProbeFenceState;
pub(crate) use probe_ticket::ProbeFenceTicket;

pub(crate) const FENCE_REQUESTED_BYTES: u64 = 8;

/// Physical output rejoins the same accounting/retirement owner regardless of
/// whether emission was ordinary compute or the closed numeric-readback op.
struct SubmittedPhysical {
    submission: wgpu::SubmissionIndex,
    physical: Option<runmat_execution::profiling::CommandPhysicalSubmission>,
    readback: Option<compute_pass::TimestampReadback>,
    fence: Option<compute_pass::ObservationBuffer>,
    #[cfg(target_arch = "wasm32")]
    direct_fence: Option<compute_pass::ObservationBuffer>,
}

struct ProbeSubmitGuard(Option<ProbeFenceTicket>);

impl ProbeSubmitGuard {
    fn fence(&mut self, bytes: u64) -> Option<ProbeFenceTicket> {
        let ticket = self.0.take();
        if let Some(ticket) = &ticket {
            ticket.submitted_with_fence(bytes);
        }
        ticket
    }

    fn absent(&mut self, observation_failed: bool) {
        if let Some(ticket) = self.0.take() {
            ticket.submitted_without_fence(observation_failed);
        }
    }
}

impl Drop for ProbeSubmitGuard {
    fn drop(&mut self) {
        if let Some(ticket) = self.0.take() {
            ticket.fail();
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod browser_completion;
mod completion;
#[path = "observation/host_control.rs"]
mod host_control;
#[path = "observation/inline_slots.rs"]
mod inline_slots;
#[path = "observation/retention_admission.rs"]
mod retention_admission;
#[cfg(not(target_arch = "wasm32"))]
mod scoped_retirement;
mod submitted_retention;
#[cfg(target_arch = "wasm32")]
use completion::map_direct_fence;
use completion::{
    map_timing, DirectCompletion, FenceCompletion, PendingDirectObservation, PendingObservation,
    SourceTransferRetirement,
};
#[cfg(not(target_arch = "wasm32"))]
use host_control::observed_completion_callback_fits_bound;
pub(crate) use host_control::observed_submission_dispatch_host_control_upper_bytes;
pub(crate) use host_control::typed_submission_dispatch_host_control_upper_bytes;
use inline_slots::InlineSlots;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use scoped_retirement::{ScopeConfirmation, ScopedRetirement, ScopedRetirementProof};

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use submitted_retention::retirement::SubmittedAllocationRetirement;
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use submitted_retention::with_retirement_for_test;
pub(crate) use submitted_retention::{
    PreparedSubmittedAllocationRetention, SubmittedAllocationRetention, SubmittedReductionInput,
    SubmittedReductionInputs, SubmittedReductionResources, SubmittedResourceOwners,
};

/// Construct outside the access closure, borrow through nested emitters, then
/// commit immediately after emission returns. Drop on unwind invalidates evidence
/// and still hands cleanup to the existing progress owner. The receipt budget
/// bounds fences; this accumulator is not a new scheduler.
pub(crate) struct PendingSubmissions {
    access: WgpuDeviceAccess,
    timestamp: Option<TimestampCapability>,
    budget: Option<Arc<ProviderObservationBudget>>,
    #[cfg(not(target_arch = "wasm32"))]
    progress_port: Option<Arc<dyn runmat_accelerate_api::observation::WgpuObservationProgress>>,
    observations: InlineSlots<PendingObservation, 1>,
    direct_observations: InlineSlots<PendingDirectObservation, 1>,
    legacy_poll: Option<wgpu::SubmissionIndex>,
    last_fence_requested_bytes: Option<u64>,
    armed_probe: Option<ProbeFenceTicket>,
    armed_resources: InlineSlots<runmat_accelerate_api::observation::ProviderResourceTicket, 3>,
    armed_allocation_retention: Option<SubmittedAllocationRetention>,
    armed_source_transfer_retirement: Option<StochasticEvolutionSourceTransferRetirementNotifier>,
    prepared_allocation_admission: Option<ProviderAdmission>,
    #[cfg(not(target_arch = "wasm32"))]
    scoped_retirement: bool,
    #[cfg(not(target_arch = "wasm32"))]
    prepared_direct_progress:
        Option<Box<dyn runmat_accelerate_api::observation::WgpuObservationProgressReservation>>,
    #[cfg(target_arch = "wasm32")]
    prepared_direct_fence: Option<runmat_accelerate_api::observation::WgpuTailPermit>,
}

impl PendingSubmissions {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn set_progress_port_for_test(
        &mut self,
        progress: Arc<dyn runmat_accelerate_api::observation::WgpuObservationProgress>,
    ) {
        self.progress_port = Some(progress);
    }
    /// Retire only after a non-indexed queue callback or actual device-loss
    /// proof authenticated physical retirement. A rejected or unconfirmed command never earns
    /// positive timing evidence. Existing map/query owners remain here until
    /// that callback, independently of logical future cancellation.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn retire_scoped(mut self, proof: ScopedRetirementProof, confirmed: bool) {
        let confirmed = confirmed && proof.queue_completed();
        self.legacy_poll = None;
        for observation in self.direct_observations.drain() {
            observation.completion.retire_scoped(confirmed);
        }
        for observation in self.observations.drain() {
            observation.completion.retire_scoped(confirmed);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn scoped_retirement_host_upper_bytes() -> Option<u64> {
        // The queued physical completion is inline in the preallocated slot.
        // Only the driver's signal callback capture allocates per request.
        let callback = std::mem::size_of::<ScopedRetirement>()
            .checked_add(std::mem::align_of::<ScopedRetirement>())?;
        ScopedRetirement::host_upper_bytes()?.checked_add(callback.try_into().ok()?)
    }

    pub(crate) fn new(access: WgpuDeviceAccess) -> Self {
        Self {
            access,
            timestamp: None,
            budget: None,
            #[cfg(not(target_arch = "wasm32"))]
            progress_port: None,
            observations: InlineSlots::default(),
            direct_observations: InlineSlots::default(),
            legacy_poll: None,
            last_fence_requested_bytes: None,
            armed_probe: None,
            armed_resources: InlineSlots::default(),
            armed_allocation_retention: None,
            armed_source_transfer_retirement: None,
            prepared_allocation_admission: None,
            #[cfg(not(target_arch = "wasm32"))]
            scoped_retirement: false,
            #[cfg(not(target_arch = "wasm32"))]
            prepared_direct_progress: None,
            #[cfg(target_arch = "wasm32")]
            prepared_direct_fence: None,
        }
    }

    /// Provider construction supplies bounded optional instrumentation ports;
    /// this method never reserves a slot or allocates device resources.
    pub(crate) fn with_profiling(
        mut self,
        timestamp: Option<TimestampCapability>,
        budget: Arc<ProviderObservationBudget>,
        #[cfg(not(target_arch = "wasm32"))] progress_port: Option<
            Arc<dyn runmat_accelerate_api::observation::WgpuObservationProgress>,
        >,
    ) -> Self {
        self.timestamp = timestamp;
        self.budget = Some(budget);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.progress_port = progress_port;
        }
        self
    }

    pub(crate) fn profiled_encoder<'a>(
        &mut self,
        emission: &'a WgpuEmission<'_>,
        descriptor: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> ProfiledEncoder<'a> {
        let selected_provider = current_provider_observation();
        let command = current_command_profiling();
        let provider_active = selected_provider.is_some();
        let command_active = command.is_some();
        let active = provider_active || command_active;
        let admission = self
            .prepared_allocation_admission
            .take()
            .unwrap_or_else(|| selected_provider.map_or(ProviderAdmission::Absent, prepare));
        let provider_prepared = matches!(&admission, ProviderAdmission::Prepared(_));
        #[cfg(not(target_arch = "wasm32"))]
        let completion_prepared = self.scoped_retirement || self.prepared_direct_progress.is_some();
        ProfiledEncoder::new(
            emission,
            descriptor,
            EncoderObservation {
                command,
                admission,
                timestamp: (provider_prepared || !provider_active)
                    .then_some(self.timestamp)
                    .flatten(),
                budget: active.then(|| self.budget.clone()).flatten(),
                #[cfg(not(target_arch = "wasm32"))]
                progress_port: (active
                    && !self.scoped_retirement
                    && self.prepared_direct_progress.is_none())
                .then(|| self.progress_port.clone())
                .flatten(),
                #[cfg(not(target_arch = "wasm32"))]
                completion_prepared,
            },
        )
    }

    /// One-shot passive observation for the next ordinary submission.
    pub(crate) fn arm_probe_completion(&mut self) -> ProbeFenceTicket {
        if let Some(previous) = self.armed_probe.take() {
            previous.fail();
        }
        let ticket = ProbeFenceTicket::armed();
        self.armed_probe = Some(ticket.clone());
        ticket
    }

    /// Keep a transient owner's issued byte ticket until the already-existing
    /// optional completion fence for its final consumer is cleaned up.
    pub(crate) fn retain_probe_resource_until_next_submit(
        &mut self,
        ticket: runmat_accelerate_api::observation::ProviderResourceTicket,
    ) {
        self.retain_provider_resource_until_next_submit(ticket);
    }

    /// Retain one exact transient physical allocation ticket through the next
    /// submitted command's canonical completion owner.
    pub(crate) fn retain_provider_resource_until_next_submit(
        &mut self,
        ticket: runmat_accelerate_api::observation::ProviderResourceTicket,
    ) {
        self.armed_resources.push(ticket);
    }

    /// Transfer one operation-owned allocation bundle into the next exact
    /// submission completion.  This does not create another completion path.
    pub(crate) fn retain_allocations_until_next_submit(
        &mut self,
        prepared: PreparedSubmittedAllocationRetention,
        retention: SubmittedAllocationRetention,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.armed_allocation_retention.is_none(),
            "submission allocation retention is already armed"
        );
        anyhow::ensure!(
            self.prepared_allocation_admission.is_none(),
            "submission completion admission is already armed"
        );
        let PreparedSubmittedAllocationRetention {
            admission,
            retirement,
        } = prepared;
        let retention = retention.with_retirement(retirement)?;
        anyhow::ensure!(
            matches!(admission, ProviderAdmission::Prepared(_)),
            "submission completion admission was not prepared"
        );
        self.prepared_allocation_admission = Some(admission);
        self.armed_allocation_retention = Some(retention);
        Ok(())
    }

    pub(crate) fn retire_source_transfer_with_next_submit(
        &mut self,
        notifier: StochasticEvolutionSourceTransferRetirementNotifier,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.armed_source_transfer_retirement.is_none(),
            "source-transfer retirement is already armed"
        );
        self.armed_source_transfer_retirement = Some(notifier);
        Ok(())
    }

    /// Emission fact only. None is an absent fence, not completed GPU work.
    pub(crate) fn last_fence_requested_bytes(&self) -> Option<u64> {
        self.last_fence_requested_bytes
    }

    pub(crate) fn submit_numeric_readback(
        &mut self,
        source: &runmat_accelerate_api::WgpuNumericReadbackSource,
        stage: &runmat_accelerate_api::WgpuNumericReadbackBuffer,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        source.ensure_device(&self.access)?;
        use runmat_accelerate_api::observation::WgpuQueryAllocationDisposition;
        let selected = current_provider_observation();
        let provider_active = selected.is_some();
        let command = current_command_profiling();
        let active = provider_active || command.is_some();
        let admission = self
            .prepared_allocation_admission
            .take()
            .unwrap_or_else(|| selected.map_or(ProviderAdmission::Absent, prepare));
        let prepared = matches!(&admission, ProviderAdmission::Prepared(_));
        let timestamp = (prepared || !provider_active)
            .then_some(self.timestamp)
            .flatten();
        let outer_requested =
            command.is_some() && timestamp.is_some_and(TimestampCapability::inside_encoders);
        let permit = outer_requested
            .then(|| {
                self.budget
                    .as_ref()
                    .and_then(|budget| budget.try_reserve_wgpu_query(MAX_QUERY_REQUESTED_BYTES))
            })
            .flatten();
        #[cfg(not(target_arch = "wasm32"))]
        let (permit, progress) = if permit.is_some()
            && !prepared
            && !self.scoped_retirement
            && self.prepared_direct_progress.is_none()
        {
            let progress = self
                .progress_port
                .as_ref()
                .and_then(|port| port.reserve().ok());
            (progress.as_ref().and(permit), progress)
        } else {
            (permit, None)
        };
        let timing = permit.is_some();
        let metadata = EncodedMeasurement {
            encoder: (),
            dispatches: active.then_some(0),
            empty_compute_elapsed: active.then_some(std::time::Duration::ZERO),
            empty_device_elapsed: None,
            readback: None,
            query_allocation: active.then_some(if timing {
                WgpuQueryAllocationDisposition::Allocated {
                    timestamp_slots: compute_pass::MAX_TIMESTAMP_SLOTS_PER_ENCODER,
                    resolve_requested_bytes: 16,
                    staging_requested_bytes: 16,
                    map_callbacks: 1,
                }
            } else {
                WgpuQueryAllocationDisposition::NoComputePass
            }),
            admission,
            command,
            #[cfg(not(target_arch = "wasm32"))]
            progress,
        };
        self.submit_prepared(metadata, timing, |(), _, fence, direct_fence, command| {
            let submitted = source.submit(stage, timing, fence, direct_fence, command.as_ref())?;
            let readback = submitted.timestamps.zip(permit).map(|(resources, permit)| {
                compute_pass::TimestampReadback::numeric(
                    resources,
                    permit,
                    timestamp.expect("admitted numeric timestamps have capability"),
                )
            });
            Ok(SubmittedPhysical {
                submission: submitted.submission,
                physical: submitted.physical,
                readback,
                fence: submitted
                    .fence
                    .map(compute_pass::ObservationBuffer::Readback),
                #[cfg(target_arch = "wasm32")]
                direct_fence: submitted
                    .direct_fence
                    .map(compute_pass::ObservationBuffer::Readback),
            })
        })
    }

    pub(crate) fn submit(
        &mut self,
        emission: &WgpuEmission<'_>,
        encoder: impl SubmissionEncoder,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        emission.ensure_device(&self.access)?;
        self.submit_prepared(
            encoder.into_measurement(),
            false,
            |mut encoder, readback, fence, direct_fence, command| {
                let make_fence = |encoder: &mut wgpu::CommandEncoder, label| {
                    let buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                        emission.device().create_buffer(&wgpu::BufferDescriptor {
                            label: Some(label),
                            size: FENCE_REQUESTED_BYTES,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    );
                    encoder.clear_buffer(&buffer, 0, None);
                    compute_pass::ObservationBuffer::Ordinary(buffer)
                };
                let fence =
                    fence.then(|| make_fence(&mut encoder, "runmat-operation-completion-fence"));
                #[cfg(target_arch = "wasm32")]
                let direct_fence = direct_fence
                    .then(|| make_fence(&mut encoder, "runmat-direct-completion-fence"));
                #[cfg(not(target_arch = "wasm32"))]
                let _ = direct_fence;
                let encoded = encoder.finish();
                let physical = command.and_then(|owner| owner.begin_physical_submission());
                let submission = emission.submit_profiled(Some(encoded));
                Ok(SubmittedPhysical {
                    submission,
                    physical,
                    readback,
                    fence,
                    #[cfg(target_arch = "wasm32")]
                    direct_fence,
                })
            },
        )
    }

    // Private stack-only emission seam: callers cannot acquire device/encoder
    // authority here. The numeric caller supplies only the API's fixed copy op.
    fn submit_prepared<E>(
        &mut self,
        encoded: EncodedMeasurement<E>,
        planned_readback: bool,
        emit: impl FnOnce(
            E,
            Option<compute_pass::TimestampReadback>,
            bool,
            bool,
            Option<runmat_execution::profiling::CommandProfilingReceipt>,
        ) -> anyhow::Result<SubmittedPhysical>,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        let EncodedMeasurement {
            encoder,
            dispatches,
            empty_compute_elapsed,
            empty_device_elapsed,
            readback,
            query_allocation,
            admission,
            command: captured_command,
            #[cfg(not(target_arch = "wasm32"))]
                progress: direct_progress,
        } = encoded;
        let has_readback = readback.is_some() || planned_readback;
        let measurement = WgpuSubmissionMeasurement {
            device_elapsed: empty_device_elapsed,
            compute_elapsed: empty_compute_elapsed,
            dispatches,
            query_allocation,
        };
        self.last_fence_requested_bytes = None;
        let mut probe = ProbeSubmitGuard(self.armed_probe.take());
        let submitted_resources = std::mem::take(&mut self.armed_resources);
        #[cfg(not(target_arch = "wasm32"))]
        let submitted_allocation_retention = if self.scoped_retirement {
            // The exact prearmed scoped slot already owns this accumulator;
            // retain the frame there until its non-indexed physical proof.
            None
        } else {
            self.armed_allocation_retention.take()
        };
        #[cfg(target_arch = "wasm32")]
        let submitted_allocation_retention = self.armed_allocation_retention.take();
        let source_transfer_retirement = self
            .armed_source_transfer_retirement
            .take()
            .map(SourceTransferRetirement::new);
        let current_recorder = current_provider_observation();
        let command = current_command_profiling();
        let admission_matched = fail_mismatched_admission(
            &admission,
            current_recorder.as_ref(),
            captured_command.as_ref(),
            command.as_ref(),
        );
        let captured_direct = matches!(&admission, ProviderAdmission::Absent);
        let observation_requested = admission.recorder().is_some() || current_recorder.is_some();
        let observed = match admission {
            ProviderAdmission::Prepared(observed) => Some(observed),
            ProviderAdmission::Absent | ProviderAdmission::Failed(_) => None,
        };
        let retained_allocations_need_direct_completion =
            submitted_allocation_retention.is_some() && observed.is_none();
        let source_transfer_needs_direct_completion =
            source_transfer_retirement.is_some() && observed.is_none();
        // A failed provider recorder cannot silently become a direct route.
        // Its physical contribution remains Missing after the real submit.
        let needs_direct_completion =
            (admission_matched && command.is_some() && !observation_requested)
                || retained_allocations_need_direct_completion;
        let needs_direct_completion =
            needs_direct_completion || source_transfer_needs_direct_completion;
        // Even an unattributable command-only query must retain its readback
        // and permit through submitted-work retirement. Scope mismatch removes
        // positive evidence, not the device resource's lifetime owner.
        let retires_direct_resources = captured_direct && !admission_matched;
        #[cfg(not(target_arch = "wasm32"))]
        let direct_progress = if self.scoped_retirement {
            None
        } else if needs_direct_completion {
            self.prepared_direct_progress
                .take()
                .or(direct_progress)
                .or_else(|| {
                    self.progress_port
                        .as_ref()
                        .and_then(|port| port.reserve().ok())
                })
        } else {
            direct_progress
        };
        #[cfg(target_arch = "wasm32")]
        let direct_tail_permit = self.prepared_direct_fence.take();
        #[cfg(target_arch = "wasm32")]
        let direct_fence_permit = if needs_direct_completion && !has_readback {
            direct_tail_permit
                .is_none()
                .then(|| {
                    self.budget
                        .as_ref()
                        .and_then(|budget| budget.try_reserve_wgpu_query(FENCE_REQUESTED_BYTES))
                })
                .flatten()
        } else {
            None
        };
        #[cfg(not(target_arch = "wasm32"))]
        if (submitted_allocation_retention.is_some() || source_transfer_retirement.is_some())
            && observed.is_none()
            && direct_progress.is_none()
            && !self.scoped_retirement
        {
            return Err(anyhow::anyhow!(
                "exact submission progress is unavailable for retained allocations"
            ));
        }
        #[cfg(target_arch = "wasm32")]
        if (submitted_allocation_retention.is_some() || source_transfer_retirement.is_some())
            && observed.is_none()
            && !has_readback
            && direct_fence_permit.is_none()
            && direct_tail_permit.is_none()
        {
            return Err(anyhow::anyhow!(
                "exact submission fence is unavailable for retained allocations"
            ));
        }
        #[cfg(target_arch = "wasm32")]
        let direct_fence_requested =
            !has_readback && (direct_fence_permit.is_some() || direct_tail_permit.is_some());
        #[cfg(not(target_arch = "wasm32"))]
        let direct_fence_requested = false;
        #[cfg(not(target_arch = "wasm32"))]
        let _ = has_readback;
        // Issue the exact fixed fence ticket before any physical effect.
        let resource_ticket = observed
            .as_ref()
            .and_then(|_| {
                runmat_accelerate_api::observation::current_provider_resource_scope()
                    .map(|scope| scope.issue_resource_ticket(FENCE_REQUESTED_BYTES))
            })
            .transpose()?;
        let SubmittedPhysical {
            submission,
            mut physical,
            readback,
            fence,
            #[cfg(target_arch = "wasm32")]
            direct_fence,
        } = emit(
            encoder,
            readback,
            observed.is_some(),
            direct_fence_requested,
            command,
        )?;
        self.last_fence_requested_bytes = fence.as_ref().map(|buffer| buffer.size());
        if !admission_matched {
            // Preserve the observed owner solely for readback/fence retirement.
            // The selected command has an actual emission but no trustworthy
            // attribution, so its linear token records Missing.
            drop(physical.take());
        }
        #[cfg(not(target_arch = "wasm32"))]
        let has_direct_retirement =
            physical.is_some() || readback.is_some() || direct_progress.is_some();
        #[cfg(target_arch = "wasm32")]
        let has_direct_retirement =
            physical.is_some() || readback.is_some() || direct_fence.is_some();
        let probe_for_fence = if let Some(bytes) = self.last_fence_requested_bytes {
            probe.fence(bytes)
        } else {
            probe.absent(observation_requested);
            None
        };
        let mut submitted_probe = ProbeSubmitGuard(probe_for_fence);
        if let (Some(observed), Some(fence)) = (observed, fence) {
            let readback_result = readback.as_ref().map(map_timing);
            let (send, result) = completion_signal::channel();
            let token = observed.token;
            fence.map_async(move |status| {
                // No gate acquisition, unmap, or runtime callback here.
                let completed = if status.is_ok() {
                    Some(token.capture_completion())
                } else {
                    token.fail();
                    None
                };
                send.send(completed);
            });
            self.observations.push(PendingObservation {
                completion: FenceCompletion {
                    access: self.access.clone(),
                    #[cfg(not(target_arch = "wasm32"))]
                    unretired_submission: Some(submission.clone()),
                    fence: Some(fence),
                    recorder: observed.recorder,
                    result,
                    measurement,
                    readback,
                    readback_result,
                    probe: submitted_probe.0.clone(),
                    resource_ticket,
                    submitted_resources,
                    submitted_allocation_retention,
                    physical: physical.take(),
                    source_transfer_retirement,
                },
                #[cfg(not(target_arch = "wasm32"))]
                submission: submission.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                progress: observed.progress,
            });
            submitted_probe.0.take();
        } else if (needs_direct_completion || retires_direct_resources) && has_direct_retirement {
            #[cfg(not(target_arch = "wasm32"))]
            if direct_progress.is_some() || self.scoped_retirement {
                let result = readback.as_ref().map(map_timing);
                self.direct_observations.push(PendingDirectObservation {
                    completion: DirectCompletion {
                        access: self.access.clone(),
                        unretired_submission: Some(submission.clone()),
                        readback,
                        fence: None,
                        result,
                        physical: physical.take(),
                        measurement,
                        submitted_allocation_retention,
                        source_transfer_retirement,
                    },
                    submission: submission.clone(),
                    progress: direct_progress,
                });
            } else {
                // No exact submission-progress owner: the linear submission
                // drops Missing, while the semantic submission remains valid.
                drop(physical.take());
            }
            #[cfg(target_arch = "wasm32")]
            if readback.is_some() || direct_fence.is_some() {
                let result = readback
                    .as_ref()
                    .map(map_timing)
                    .or_else(|| direct_fence.as_ref().map(map_direct_fence));
                self.direct_observations.push(PendingDirectObservation {
                    completion: DirectCompletion {
                        access: self.access.clone(),
                        readback,
                        fence: direct_fence,
                        _fence_permit: direct_fence_permit,
                        _tail_permit: direct_tail_permit,
                        result,
                        physical: physical.take(),
                        measurement,
                        submitted_allocation_retention,
                        source_transfer_retirement,
                    },
                });
            } else {
                drop(physical.take());
            }
        }
        Ok(submission)
    }

    pub(crate) fn commit(mut self) {
        if let Some(probe) = self.armed_probe.take() {
            probe.fail();
        }
        // An armed owner with no physical submit has no device work to await.
        drop(self.armed_allocation_retention.take());
        drop(self.armed_source_transfer_retirement.take());
        drop(self.prepared_allocation_admission.take());
        self.drain(false);
        if let Some(submission) = self.legacy_poll.take() {
            super::common::poll_submitted(&self.access, submission);
        }
    }

    /// Only combine same-queue chunks with no intervening CPU/readback data
    /// dependency. A host-dependent barrier must end its emission first.
    pub(crate) fn request_legacy_poll(&mut self, submission: wgpu::SubmissionIndex) {
        self.legacy_poll = Some(submission);
    }

    fn drain(&mut self, abandoned: bool) {
        for mut observation in self.direct_observations.drain() {
            let control_claim = observation
                .completion
                .submitted_allocation_retention
                .as_mut()
                .and_then(|retention| retention.take_completion_host_claim());
            if abandoned {
                // The command is abandoned, not the already-submitted device
                // work. Retire its physical evidence as Missing now, but keep
                // query/readback resources with the progress callback.
                drop(observation.completion.physical.take());
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let PendingDirectObservation {
                    completion,
                    submission,
                    progress,
                } = observation;
                let Some(progress) = progress else {
                    // Only native scoped owners omit a reservation. Their
                    // accumulator must be retired by its prearmed proof path.
                    log::error!("scoped physical completion reached ordinary drain");
                    continue;
                };
                let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    progress.submit(
                        submission,
                        runmat_accelerate_api::observation::WgpuObservationCompletion::new(
                            Box::new(move |status| completion.after_poll(status)),
                            [None, control_claim],
                        ),
                    )
                }));
                match &delivered {
                    Ok(Ok(())) => log::debug!("WGPU direct profiling progress callback enqueued"),
                    Ok(Err(reason)) => {
                        log::debug!("WGPU direct profiling progress enqueue failed: {reason:?}")
                    }
                    Err(_) => log::debug!("WGPU direct profiling progress enqueue panicked"),
                }
                if !matches!(delivered, Ok(Ok(()))) {
                    // The callback owns the physical submission; failure to enqueue
                    // drops it as Missing, without replaying the submission.
                    log::warn!("WGPU direct profiling progress unavailable");
                }
            }
            #[cfg(target_arch = "wasm32")]
            browser_completion::direct(observation.completion, [None, control_claim]);
        }
        for mut observation in self.observations.drain() {
            let claims = [
                observation
                    .completion
                    .recorder
                    .completion_host_metadata_claim(),
                observation
                    .completion
                    .submitted_allocation_retention
                    .as_mut()
                    .and_then(|retention| retention.take_completion_host_claim()),
            ];
            if abandoned {
                if let Some(probe) = &observation.completion.probe {
                    probe.fail();
                }
                observation
                    .completion
                    .recorder
                    .fail(ProviderObservationFailure::Abandoned);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let PendingObservation {
                    completion,
                    submission,
                    progress,
                } = observation;
                let Some(progress) = progress else {
                    completion
                        .recorder
                        .fail(ProviderObservationFailure::ProviderFailed);
                    log::error!("scoped observed completion reached ordinary drain");
                    continue;
                };
                let recorder = completion.recorder.clone();
                let callback = move |status| completion.after_poll(status);
                if !observed_completion_callback_fits_bound(&callback) {
                    recorder.fail(ProviderObservationFailure::TrackingOverflow);
                    continue;
                }
                let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    progress.submit(
                        submission,
                        runmat_accelerate_api::observation::WgpuObservationCompletion::new(
                            Box::new(callback),
                            claims,
                        ),
                    )
                }));
                match &delivered {
                    Ok(Ok(())) => {
                        log::debug!("WGPU provider observation progress callback enqueued")
                    }
                    Ok(Err(reason)) => {
                        log::debug!("WGPU provider observation progress enqueue failed: {reason:?}")
                    }
                    Err(_) => {
                        log::debug!("WGPU provider observation progress enqueue panicked")
                    }
                }
                if !matches!(delivered, Ok(Ok(()))) {
                    recorder.fail(ProviderObservationFailure::ProviderFailed);
                }
            }
            #[cfg(target_arch = "wasm32")]
            browser_completion::observed(observation.completion, claims);
        }
    }
}

impl Drop for PendingSubmissions {
    fn drop(&mut self) {
        if let Some(probe) = self.armed_probe.take() {
            probe.fail();
        }
        drop(self.armed_allocation_retention.take());
        drop(self.armed_source_transfer_retirement.take());
        drop(self.prepared_allocation_admission.take());
        self.drain(true);
    }
}

#[cfg(test)]
mod tests;
