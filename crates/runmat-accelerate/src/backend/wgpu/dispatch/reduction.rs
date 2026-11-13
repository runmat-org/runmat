use super::common::submit;
use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;

pub(crate) fn warmup_noop_single(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
) -> anyhow::Result<()> {
    let mut enc = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-noop-single-pass"),
        },
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass-single"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    submit(emission, pending, enc)?;
    Ok(())
}

pub(crate) fn warmup_noop_after_pass2(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
) -> anyhow::Result<()> {
    let mut enc = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-noop-after-pass2"),
        },
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    submit(emission, pending, enc)?;
    Ok(())
}

pub(crate) fn run_single_pass(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups_x: u32,
) -> anyhow::Result<()> {
    let mut enc = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-reduction-encoder"),
        },
    );
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-reduction-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    submit(emission, pending, enc)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_two_pass(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline_p1: &wgpu::ComputePipeline,
    pipeline_p2: &wgpu::ComputePipeline,
    bg1: &wgpu::BindGroup,
    bg2: &wgpu::BindGroup,
    g0: u32,
    g1: u32,
    g2: u32,
) -> anyhow::Result<()> {
    // Pass 1 in its own encoder
    {
        let mut enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-reduction-pass1-enc"),
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-reduction-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline_p1);
            pass.set_bind_group(0, bg1, &[]);
            pass.dispatch_workgroups(g0, g1, g2);
        }
        submit(emission, pending, enc)?;
    }
    // Pass 2 in a fresh encoder to avoid any usage scope overlap on Metal
    {
        let mut enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-reduction-pass2-enc"),
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-reduction-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline_p2);
            pass.set_bind_group(0, bg2, &[]);
            pass.dispatch_workgroups(g0, 1, 1);
        }
        submit(emission, pending, enc)?;
    }
    Ok(())
}
