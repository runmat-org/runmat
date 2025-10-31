pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-ind2sub-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-ind2sub-pass"),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if workgroups > 0 {
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
    queue.submit(Some(encoder.finish()));
}
