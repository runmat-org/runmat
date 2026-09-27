//! Private-resource completion fences. Queue-prefix callbacks are insufficient:
//! exported queues can submit unrelated work between submission and registration.
use runmat_accelerate_api::observation::{current_provider_observation, ProviderSubmission};
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderOperationRecorder, WgpuObservationProgressError,
    WgpuObservationProgressReservation,
};
use std::sync::Arc;

/// Requested MAP_READ buffer size/alignment, not actual driver VRAM granularity.
pub(crate) const FENCE_REQUESTED_BYTES: u64 = 8;

struct PreparedObservation {
    token: ProviderSubmission,
    #[cfg(not(target_arch = "wasm32"))]
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
        #[cfg(not(target_arch = "wasm32"))]
        recorder,
        #[cfg(not(target_arch = "wasm32"))]
        progress,
    })
}

/// Preserve the caller's polling policy. Observation never adds a blocking wait
/// or an additional queue submission. Each observed submission ends with a clear
/// of its private fence; mapping waits for that resource, not later queue users.
/// Completion timestamp means observed map-completion delivery, including native
/// progress-worker/browser event-loop service latency, not a device GPU timestamp.
pub(crate) fn submit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
) -> wgpu::SubmissionIndex {
    let observed = prepare();
    let fence = observed.as_ref().map(|_| {
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-operation-completion-fence"),
            size: FENCE_REQUESTED_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        encoder.clear_buffer(buffer.as_ref(), 0, None);
        buffer
    });
    let submission = queue.submit(Some(encoder.finish()));
    if let (Some(observed), Some(fence)) = (observed, fence) {
        let token = observed.token;
        let retained = fence.clone();
        fence
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // Cancellation of the receipt does not release this buffer/permit
                // prematurely. No mapping payload or runtime/session is captured.
                if result.is_ok() {
                    retained.unmap();
                }
                drop(retained);
                if result.is_ok() {
                    token.complete();
                } else {
                    token.fail();
                }
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let recorder = observed.recorder;
            let failed_recorder = recorder.clone();
            // The map callback retains the fence/token until safe teardown.
            // Failure cancellation never recycles its permit independently.
            let cleanup = Box::new(move || {
                failed_recorder.fail(ProviderObservationFailure::ProviderFailed);
                fence.unmap();
            });
            if observed
                .progress
                .submit(submission.clone(), cleanup)
                .is_err()
            {
                recorder.fail(ProviderObservationFailure::ProviderFailed);
            }
        }
    }
    submission
}

#[cfg(test)]
mod tests;
