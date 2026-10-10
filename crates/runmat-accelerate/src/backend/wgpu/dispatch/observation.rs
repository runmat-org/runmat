//! Provider-owned private fences. Map callbacks never perform WGPU emission.
use futures::channel::oneshot;
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderCompletedSubmission, ProviderObservationFailure,
    ProviderOperationRecorder, ProviderSubmission,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::{
    WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_accelerate_api::{WgpuDeviceAccess, WgpuEmission};

#[path = "observation/probe_ticket.rs"]
mod probe_ticket;
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

struct PreparedObservation {
    token: ProviderSubmission,
    recorder: ProviderOperationRecorder,
    #[cfg(not(target_arch = "wasm32"))]
    progress: Box<dyn WgpuObservationProgressReservation>,
}

fn prepare() -> Option<PreparedObservation> {
    let recorder = current_provider_observation()?;
    #[cfg(not(target_arch = "wasm32"))]
    let progress = match recorder
        .wgpu_progress()
        .ok_or(WgpuObservationProgressError::Stopped)
        .and_then(|driver| driver.reserve())
    {
        Ok(reservation) => reservation,
        Err(reason) => {
            recorder.fail(match reason {
                WgpuObservationProgressError::Capacity => ProviderObservationFailure::ResourceLimit,
                WgpuObservationProgressError::Stopped => ProviderObservationFailure::ProviderFailed,
            });
            return None;
        }
    };
    let token = recorder.submission().ok()?;
    Some(PreparedObservation {
        token,
        recorder,
        #[cfg(not(target_arch = "wasm32"))]
        progress,
    })
}

struct FenceCompletion {
    access: WgpuDeviceAccess,
    fence: Option<runmat_accelerate_api::WgpuBufferOwner>,
    recorder: ProviderOperationRecorder,
    result: oneshot::Receiver<Option<ProviderCompletedSubmission>>,
    probe: Option<ProbeFenceTicket>,
    resource_ticket: Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
    submitted_resources: Vec<runmat_accelerate_api::observation::ProviderResourceTicket>,
}

impl FenceCompletion {
    fn release_fence(&mut self) -> bool {
        let access = self.access.clone();
        let released = matches!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                access.with_cleanup(|cleanup| {
                    if let Some(fence) = self.fence.take() {
                        cleanup.unmap_buffer(&fence);
                        drop(fence);
                    }
                })
            })),
            Ok(Ok(()))
        );
        if !released {
            if let Some(ticket) = &self.resource_ticket {
                ticket.fail_owner();
            }
            for ticket in &self.submitted_resources {
                ticket.fail_owner();
            }
        }
        released
    }

    fn cleanup(mut self, completed: Option<ProviderCompletedSubmission>, progress_ok: bool) {
        if !progress_ok || completed.is_none() {
            self.recorder
                .fail(ProviderObservationFailure::ProviderFailed);
        }
        // Outside the original emission. unmap can deliver a cancelled map
        // callback synchronously; that callback only sends its result.
        let cleaned = self.release_fence();
        if progress_ok && cleaned {
            if let Some(completed) = completed {
                completed.complete();
                if let Some(probe) = &self.probe {
                    probe.complete();
                }
            } else if let Some(probe) = &self.probe {
                probe.fail();
            }
        } else {
            self.recorder
                .fail(ProviderObservationFailure::ProviderFailed);
            if let Some(completed) = completed {
                completed.fail();
            }
            if let Some(probe) = &self.probe {
                probe.fail();
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn after_poll(mut self, status: Result<(), WgpuObservationProgressError>) {
        let completed = self.result.try_recv().ok().flatten().flatten();
        self.cleanup(completed, status.is_ok());
    }

    #[cfg(target_arch = "wasm32")]
    async fn after_callback(mut self) {
        let completed = (&mut self.result).await.ok().flatten();
        self.cleanup(completed, true);
    }
}

impl Drop for FenceCompletion {
    fn drop(&mut self) {
        if self.fence.is_some() {
            if let Some(probe) = &self.probe {
                probe.fail();
            }
            self.recorder.fail(ProviderObservationFailure::Abandoned);
            if !self.release_fence() {
                log::error!("WGPU observation fence teardown failed; no retry attempted");
            }
        }
    }
}

struct PendingObservation {
    completion: FenceCompletion,
    #[cfg(not(target_arch = "wasm32"))]
    submission: wgpu::SubmissionIndex,
    #[cfg(not(target_arch = "wasm32"))]
    progress: Box<dyn WgpuObservationProgressReservation>,
}

/// Construct outside the access closure, borrow through nested emitters, then
/// commit immediately after emission returns. Drop on unwind invalidates evidence
/// and still hands cleanup to the existing progress owner. The receipt budget
/// bounds fences; this accumulator is not a new scheduler.
pub(crate) struct PendingSubmissions {
    access: WgpuDeviceAccess,
    observations: Vec<PendingObservation>,
    legacy_poll: Option<wgpu::SubmissionIndex>,
    last_fence_requested_bytes: Option<u64>,
    armed_probe: Option<ProbeFenceTicket>,
    armed_resources: Vec<runmat_accelerate_api::observation::ProviderResourceTicket>,
}

impl PendingSubmissions {
    pub(crate) fn new(access: WgpuDeviceAccess) -> Self {
        Self {
            access,
            observations: Vec::new(),
            legacy_poll: None,
            last_fence_requested_bytes: None,
            armed_probe: None,
            armed_resources: Vec::new(),
        }
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
        mut encoder: wgpu::CommandEncoder,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        emission.ensure_device(&self.access)?;
        self.last_fence_requested_bytes = None;
        let mut probe = ProbeSubmitGuard(self.armed_probe.take());
        let submitted_resources = std::mem::take(&mut self.armed_resources);
        let observation_requested = current_provider_observation().is_some();
        let observed = prepare();
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
        let submission = emission.queue().submit(Some(encoder.finish()));
        let probe_for_fence = if let Some(bytes) = self.last_fence_requested_bytes {
            probe.fence(bytes)
        } else {
            probe.absent(observation_requested);
            None
        };
        let mut submitted_probe = ProbeSubmitGuard(probe_for_fence);
        if let (Some(observed), Some(fence)) = (observed, fence) {
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
                    fence: Some(fence),
                    recorder: observed.recorder,
                    result,
                    probe: submitted_probe.0.clone(),
                    resource_ticket,
                    submitted_resources,
                },
                #[cfg(not(target_arch = "wasm32"))]
                submission: submission.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                progress: observed.progress,
            });
            submitted_probe.0.take();
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
