use super::common::submit;
use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;

pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups_x: u32,
    encoder_label: &str,
    pass_label: &str,
) -> anyhow::Result<()> {
    if groups_x == 0 {
        return Ok(());
    }
    let mut encoder = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some(encoder_label),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    submit(emission, pending, encoder)?;
    Ok(())
}
