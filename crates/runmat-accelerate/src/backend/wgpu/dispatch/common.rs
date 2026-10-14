use super::observation::PendingSubmissions;
use runmat_accelerate_api::{WgpuDeviceAccess, WgpuEmission};

pub fn noop_then_poll(emission: &WgpuEmission<'_>, pipeline: &wgpu::ComputePipeline) {
    let mut enc = emission
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("runmat-noop"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    // submit
    // Note: queue submission is performed by caller to preserve provider semantics
}

/// Emission only; the enclosing owner commits pending observation and invokes
/// `poll_submitted` after its access closure has returned.
pub(crate) fn submit(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    enc: impl super::observation::SubmissionEncoder,
) -> anyhow::Result<wgpu::SubmissionIndex> {
    let submission = pending.submit(emission, enc)?;
    pending.request_legacy_poll(submission.clone());
    Ok(submission)
}

pub(crate) fn poll_submitted(access: &WgpuDeviceAccess, submission: wgpu::SubmissionIndex) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = submission;
        access.poll(wgpu::Maintain::Poll);
    }
    #[cfg(not(target_arch = "wasm32"))]
    access.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
}

pub fn dispatch_groups(groups: u32) -> u32 {
    if groups == 0 {
        0
    } else {
        groups.max(1)
    }
}

pub fn dispatch_size(elements: u32, workgroup: u32) -> u32 {
    if elements == 0 || workgroup == 0 {
        0
    } else {
        elements.div_ceil(workgroup).max(1)
    }
}

pub fn dispatch_size_reduce(elements: u32, workgroup: u32) -> u32 {
    let denom = workgroup.saturating_mul(2);
    if elements == 0 || denom == 0 {
        0
    } else {
        elements.div_ceil(denom).max(1)
    }
}

pub fn dispatch_size_dim(elements: u32, tile: u32) -> u32 {
    if elements == 0 || tile == 0 {
        0
    } else {
        elements.div_ceil(tile).max(1)
    }
}
