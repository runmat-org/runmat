use super::common;

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    batches: u32,
    batch_tile: u32,
) {
    if batches == 0 {
        return;
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-image-normalize-encoder"),
    });
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

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
}
