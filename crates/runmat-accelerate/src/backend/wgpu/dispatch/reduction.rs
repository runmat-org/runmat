use super::common::submit;

pub fn warmup_noop_single(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-noop-single-pass"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass-single"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    submit(device, queue, enc);
}

pub fn warmup_noop_after_pass2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-noop-after-pass2"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    submit(device, queue, enc);
}

pub fn run_single_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups_x: u32,
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-reduction-encoder"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-reduction-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    submit(device, queue, enc);
}

pub fn run_two_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_p1: &wgpu::ComputePipeline,
    pipeline_p2: &wgpu::ComputePipeline,
    bg1: &wgpu::BindGroup,
    bg2: &wgpu::BindGroup,
    g0: u32,
    g1: u32,
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-reduction-2pass-encoder"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-reduction-pass1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_p1);
        pass.set_bind_group(0, bg1, &[]);
        pass.dispatch_workgroups(g0, g1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-reduction-pass2"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_p2);
        pass.set_bind_group(0, bg2, &[]);
        pass.dispatch_workgroups(g0, 1, 1);
    }
    submit(device, queue, enc);
}
