//! Provider-owned private fences. Map callbacks never perform WGPU emission.
use futures::channel::oneshot;
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderObservationBudget, ProviderObservationFailure,
    WgpuSubmissionMeasurement,
};
use runmat_accelerate_api::{WgpuDeviceAccess, WgpuEmission};
use runmat_execution::profiling::current_command_profiling;
use std::sync::Arc;

#[path = "observation/admission.rs"]
mod admission;
use admission::{fail_mismatched_admission, prepare, ProviderAdmission};
#[path = "observation/probe_ticket.rs"]
mod probe_ticket;
#[path = "observation/compute_pass.rs"]
mod compute_pass;
use compute_pass::EncodedMeasurement;
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

mod completion;
#[cfg(target_arch = "wasm32")]
use completion::map_direct_fence;
use completion::{
    map_timing, DirectCompletion, FenceCompletion, PendingDirectObservation, PendingObservation,
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
    observations: Vec<PendingObservation>,
    direct_observations: Vec<PendingDirectObservation>,
    legacy_poll: Option<wgpu::SubmissionIndex>,
    last_fence_requested_bytes: Option<u64>,
    armed_probe: Option<ProbeFenceTicket>,
    armed_resources: Vec<runmat_accelerate_api::observation::ProviderResourceTicket>,
}

impl PendingSubmissions {
    pub(crate) fn new(access: WgpuDeviceAccess) -> Self {
        Self {
            access,
            timestamp: None,
            budget: None,
            #[cfg(not(target_arch = "wasm32"))]
            progress_port: None,
            observations: Vec::new(),
            direct_observations: Vec::new(),
            legacy_poll: None,
            last_fence_requested_bytes: None,
            armed_probe: None,
            armed_resources: Vec::new(),
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
        &self,
        emission: &'a WgpuEmission<'_>,
        descriptor: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> ProfiledEncoder<'a> {
        let selected_provider = current_provider_observation();
        let command = current_command_profiling();
        let provider_active = selected_provider.is_some();
        let command_active = command.is_some();
        let active = provider_active || command_active;
        let admission = selected_provider.map_or(ProviderAdmission::Absent, prepare);
        let provider_prepared = matches!(&admission, ProviderAdmission::Prepared(_));
        ProfiledEncoder::new(
            emission,
            descriptor,
            command,
            admission,
            (provider_prepared || !provider_active)
                .then_some(self.timestamp)
                .flatten(),
            active.then(|| self.budget.clone()).flatten(),
            #[cfg(not(target_arch = "wasm32"))]
            active.then(|| self.progress_port.clone()).flatten(),
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
        self.armed_resources.push(ticket);
    }

    /// Emission fact only. None is an absent fence, not completed GPU work.
    pub(crate) fn last_fence_requested_bytes(&self) -> Option<u64> {
        self.last_fence_requested_bytes
    }

    pub(crate) fn submit(
        &mut self,
        emission: &WgpuEmission<'_>,
        encoder: impl SubmissionEncoder,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        emission.ensure_device(&self.access)?;
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
        } = encoder.into_measurement();
        let mut encoder = encoder;
        let measurement = WgpuSubmissionMeasurement {
            device_elapsed: empty_device_elapsed,
            compute_elapsed: empty_compute_elapsed,
            dispatches,
            query_allocation,
        };
        self.last_fence_requested_bytes = None;
        let mut probe = ProbeSubmitGuard(self.armed_probe.take());
        let submitted_resources = std::mem::take(&mut self.armed_resources);
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
        // A failed provider recorder cannot silently become a direct route.
        // Its physical contribution remains Missing after the real submit.
        let needs_direct_completion =
            admission_matched && command.is_some() && !observation_requested;
        // Even an unattributable command-only query must retain its readback
        // and permit through submitted-work retirement. Scope mismatch removes
        // positive evidence, not the device resource's lifetime owner.
        let retires_direct_resources = captured_direct && !admission_matched;
        #[cfg(not(target_arch = "wasm32"))]
        let direct_progress = if needs_direct_completion {
            direct_progress.or_else(|| {
                self.progress_port
                    .as_ref()
                    .and_then(|port| port.reserve().ok())
            })
        } else {
            direct_progress
        };
        #[cfg(target_arch = "wasm32")]
        let direct_fence_permit = if needs_direct_completion && readback.is_none() {
            self.budget
                .as_ref()
                .and_then(|budget| budget.try_reserve_wgpu_query(FENCE_REQUESTED_BYTES))
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let direct_fence = direct_fence_permit.as_ref().map(|_| {
            let buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-direct-completion-fence"),
                    size: FENCE_REQUESTED_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            encoder.clear_buffer(&buffer, 0, None);
            buffer
        });
        let fence = observed.as_ref().map(|_| {
            let buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-operation-completion-fence"),
                    size: FENCE_REQUESTED_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            encoder.clear_buffer(&buffer, 0, None);
            buffer
        });
        let resource_ticket = fence
            .as_ref()
            .and_then(|buffer| {
                runmat_accelerate_api::observation::current_provider_resource_scope()
                    .map(|scope| scope.issue_resource_ticket(buffer.size()))
            })
            .transpose()?;
        self.last_fence_requested_bytes = fence.as_ref().map(|buffer| buffer.size());
        let encoded = encoder.finish();
        let mut physical = command.and_then(|owner| owner.begin_physical_submission());
        let submission = emission.submit_profiled(Some(encoded));
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
            let (send, result) = oneshot::channel();
            let token = observed.token;
            fence
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |status| {
                    // No gate acquisition, unmap, or runtime callback here.
                    let completed = if status.is_ok() {
                        Some(token.capture_completion())
                    } else {
                        token.fail();
                        None
                    };
                    let _ = send.send(completed);
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
                    physical: physical.take(),
                },
                #[cfg(not(target_arch = "wasm32"))]
                submission: submission.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                progress: observed.progress,
            });
            submitted_probe.0.take();
        } else if (needs_direct_completion || retires_direct_resources) && has_direct_retirement {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(progress) = direct_progress {
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
                    },
                    submission: submission.clone(),
                    progress,
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
                        result,
                        physical: physical.take(),
                        measurement,
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
        for mut observation in self.direct_observations.drain(..) {
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
                let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    progress.submit(
                        submission,
                        Box::new(move |status| completion.after_poll(status)),
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
            wasm_bindgen_futures::spawn_local(observation.completion.after_callback());
        }
        for observation in self.observations.drain(..) {
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
                let recorder = completion.recorder.clone();
                let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    progress.submit(
                        submission,
                        Box::new(move |status| completion.after_poll(status)),
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
            wasm_bindgen_futures::spawn_local(observation.completion.after_callback());
        }
    }
}

impl Drop for PendingSubmissions {
    fn drop(&mut self) {
        if let Some(probe) = self.armed_probe.take() {
            probe.fail();
        }
        self.drain(true);
    }
}

#[cfg(test)]
mod tests;
