use super::common::submit;

pub(crate) fn run(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    pending: &mut super::observation::PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
    label: &str,
) -> anyhow::Result<()> {
    let mut encoder = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor { label: Some(label) },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-fft-pass"),
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
