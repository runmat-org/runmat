use super::common::submit;

pub fn warmup_noop(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-solve-noop"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-solve-noop-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    submit(device, queue, encoder);
}

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-solve-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-solve-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if workgroups_x > 0 {
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
    }
    submit(device, queue, encoder);
}
