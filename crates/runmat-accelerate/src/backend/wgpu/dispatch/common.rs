pub fn noop_then_poll(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("runmat-noop"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runmat-noop-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    // submit
    // Note: queue submission is performed by caller to preserve provider semantics
}

pub fn submit(device: &wgpu::Device, queue: &wgpu::Queue, enc: wgpu::CommandEncoder) {
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
}

pub fn dispatch_groups(groups: u32) -> u32 {
    if groups == 0 {
        0
    } else {
        groups.max(1)
    }
}

pub fn dispatch_size(elements: u32, workgroup: u32) -> u32 {
    if elements == 0 {
        0
    } else {
        ((elements + workgroup - 1) / workgroup).max(1)
    }
}

pub fn dispatch_size_reduce(elements: u32, workgroup: u32) -> u32 {
    if elements == 0 {
        0
    } else {
        ((elements + workgroup * 2 - 1) / (workgroup * 2)).max(1)
    }
}

pub fn dispatch_size_dim(elements: u32, tile: u32) -> u32 {
    if elements == 0 {
        0
    } else {
        ((elements + tile - 1) / tile).max(1)
    }
}
