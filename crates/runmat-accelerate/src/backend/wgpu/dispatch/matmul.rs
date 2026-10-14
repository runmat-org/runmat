use super::common::submit;
use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;

pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups_x: u32,
    groups_y: u32,
) -> anyhow::Result<()> {
    let mut enc = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-matmul-encoder"),
        },
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-matmul-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    submit(emission, pending, enc)?;
    Ok(())
}
