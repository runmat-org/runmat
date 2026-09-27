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
use std::sync::Arc;

pub(crate) const FENCE_REQUESTED_BYTES: u64 = 8;

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
    fence: Option<Arc<wgpu::Buffer>>,
    recorder: ProviderOperationRecorder,
    result: oneshot::Receiver<Option<ProviderCompletedSubmission>>,
}

impl FenceCompletion {
    fn release_fence(&mut self) -> bool {
        let access = self.access.clone();
        matches!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                access.with_cleanup(|cleanup| {
                    if let Some(fence) = self.fence.take() {
                        cleanup.unmap_buffer(&fence);
                        cleanup.release_shared_buffer(fence);
                    }
                })
            })),
            Ok(Ok(()))
        )
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
            }
        } else {
            self.recorder
                .fail(ProviderObservationFailure::ProviderFailed);
            if let Some(completed) = completed {
                completed.fail();
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
}

impl PendingSubmissions {
    pub(crate) fn new(access: WgpuDeviceAccess) -> Self {
        Self {
            access,
            observations: Vec::new(),
        }
    }

    pub(crate) fn submit(
        &mut self,
        emission: &WgpuEmission<'_>,
        mut encoder: wgpu::CommandEncoder,
    ) -> anyhow::Result<wgpu::SubmissionIndex> {
        emission.ensure_device(&self.access)?;
        let observed = prepare();
        let fence = observed.as_ref().map(|_| {
            let buffer = Arc::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-operation-completion-fence"),
                size: FENCE_REQUESTED_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            encoder.clear_buffer(&buffer, 0, None);
            buffer
        });
        let submission = emission.queue().submit(Some(encoder.finish()));
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
                },
                #[cfg(not(target_arch = "wasm32"))]
                submission: submission.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                progress: observed.progress,
            });
        }
        Ok(submission)
    }

    pub(crate) fn commit(mut self) {
        self.drain(false);
    }

    fn drain(&mut self, abandoned: bool) {
        for observation in self.observations.drain(..) {
            if abandoned {
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
        self.drain(true);
    }
}

#[cfg(test)]
mod tests;
