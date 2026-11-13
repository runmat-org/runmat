use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;
pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) -> anyhow::Result<()> {
    let mut encoder = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-sub2ind-encoder"),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-sub2ind-pass"),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if workgroups > 0 {
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
    pending.submit(emission, encoder)?;
    Ok(())
}
