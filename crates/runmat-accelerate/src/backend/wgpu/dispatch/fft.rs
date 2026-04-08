use super::common::submit;

#[cfg(not(target_arch = "wasm32"))]
fn pop_validation_scope(device: &wgpu::Device, label: &str) {
    if std::env::var_os("RUNMAT_FFT_DEBUG").is_none() {
        return;
    }
    if let Some(err) = futures::executor::block_on(device.pop_error_scope()) {
        eprintln!("[fft-debug] wgpu validation in {label}: {err}");
    }
}

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
    label: &str,
) {
    if std::env::var_os("RUNMAT_FFT_DEBUG").is_some() {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
    }
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
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
    submit(device, queue, encoder);
    #[cfg(not(target_arch = "wasm32"))]
    pop_validation_scope(device, label);
}
