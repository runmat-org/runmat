//! Private-resource completion fences. Queue-prefix callbacks are insufficient:
//! exported queues can submit unrelated work between submission and registration.
use runmat_accelerate_api::observation::current_provider_observation;
use std::sync::Arc;

/// Requested MAP_READ buffer size/alignment, not actual driver VRAM granularity.
pub(crate) const FENCE_REQUESTED_BYTES: u64 = 8;

/// Preserve the caller's polling policy. Observation never adds a blocking wait
/// or an additional queue submission. Each observed submission ends with a clear
/// of its private fence; mapping waits for that resource, not later queue users.
pub(crate) fn submit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
) -> wgpu::SubmissionIndex {
    let token = current_provider_observation().and_then(|recorder| recorder.submission().ok());
    let fence = token.as_ref().map(|_| {
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
    if let (Some(token), Some(fence)) = (token, fence) {
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
    }
    submission
}

#[cfg(test)]
mod tests;
