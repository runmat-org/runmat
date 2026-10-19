use super::common;
use super::observation::PendingSubmissions;
use runmat_accelerate_api::WgpuEmission;

pub(crate) fn run(
    emission: &WgpuEmission<'_>,
    pending: &mut PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    batches: u32,
    batch_tile: u32,
) -> anyhow::Result<()> {
    if batches == 0 {
        return Ok(());
    }

    let mut encoder = pending.profiled_encoder(
        emission,
        &wgpu::CommandEncoderDescriptor {
            label: Some("runmat-image-normalize-encoder"),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-image-normalize-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let groups = common::dispatch_size(batches, batch_tile.max(1));
        pass.dispatch_workgroups(groups, 1, 1);
    }

    super::common::submit(emission, pending, encoder)?;
    Ok(())
}
