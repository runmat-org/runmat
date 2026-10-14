use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;
pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
) -> anyhow::Result<()> {
    let mut encoder = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-find-encoder"),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-find-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    super::common::submit(emission, pending, encoder)?;
    Ok(())
}
