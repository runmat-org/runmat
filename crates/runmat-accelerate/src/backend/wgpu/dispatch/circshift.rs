use super::common::submit;
use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;

pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
) -> anyhow::Result<()> {
    let mut encoder = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-circshift-encoder"),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-circshift-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if workgroups_x > 0 {
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
    }
    submit(emission, pending, encoder)?;
    Ok(())
}
