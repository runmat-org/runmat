use super::common::submit;
use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;

pub(crate) fn warmup_noop(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
) -> anyhow::Result<()> {
    let mut enc = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-noop-elementwise"),
        },
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass-elementwise"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    submit(emission, pending, enc)?;
    Ok(())
}

pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
) -> anyhow::Result<()> {
    let mut enc = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-elementwise-encoder"),
        },
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-elementwise-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if workgroups_x > 0 {
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
    }
    submit(emission, pending, enc)?;
    Ok(())
}
