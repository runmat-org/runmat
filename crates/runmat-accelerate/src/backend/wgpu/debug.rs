use runmat_time::Instant;
use wgpu::util::DeviceExt;

pub fn try_compile_kernel(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
    wgsl_src: &str,
) {
    let device = emission.device();
    let t0 = Instant::now();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{}-module", label)),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_src)),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{}-pl", label)),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let _pipeline = device.create_compute_pipeline(
        &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
            label: Some(&format!("{}-pipeline", label)),
            layout: Some(&pl),
            module: &module,
            entry_point: "main",
        },
    );
    log::info!(
        "try_compile_kernel: '{}' compiled in {:.3} ms",
        label,
        t0.elapsed().as_secs_f64() * 1000.0
    );
}

pub(crate) fn probe_kernel_with_buffers(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    pending: &mut super::dispatch::observation::PendingSubmissions,
    label: &str,
    wgsl_src: &str,
    wg: u32,
) -> anyhow::Result<()> {
    let device = emission.device();
    let t0 = Instant::now();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{}-module", label)),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_src)),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{}-bgl", label)),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{}-pl", label)),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(
        &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
            label: Some(&format!("{}-pipeline", label)),
            layout: Some(&pl),
            module: &module,
            entry_point: "main",
        },
    );

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{}-in", label)),
        contents: bytemuck::cast_slice(&[0.0f32; 4]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{}-out", label)),
        contents: bytemuck::cast_slice(&[0.0f32; 4]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{}-bg", label)),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{}-enc", label)),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{}-pass", label)),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let groups_x = wg.max(1);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    super::dispatch::common::submit(emission, pending, enc)?;
    log::info!(
        "probe_kernel_with_buffers: '{}' compiled+submitted in {:.3} ms",
        label,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
