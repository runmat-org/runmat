use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};

/// Inputs required to compute histogram counts directly on the GPU.
pub struct HistogramGpuInputs {
    pub samples: Arc<wgpu::Buffer>,
    pub sample_count: u32,
    pub scalar: ScalarType,
}

/// Parameters describing how the histogram should be computed.
pub struct HistogramGpuParams {
    pub min_value: f32,
    pub inv_bin_width: f32,
    pub bin_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramUniforms {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvertUniforms {
    bin_count: u32,
    _pad: [u32; 3],
    scale: f32,
    _pad2: [u32; 3],
}

/// Computes histogram bin counts for the given samples and converts them into an `f32` buffer that
/// downstream code (e.g., the bar packer) can consume directly.
pub fn histogram_values_buffer(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &HistogramGpuInputs,
    params: &HistogramGpuParams,
    normalization_scale: f32,
) -> Result<Arc<wgpu::Buffer>, String> {
    if params.bin_count == 0 {
        return Err("hist: bin count must be positive".to_string());
    }

    let bin_count = params.bin_count as usize;
    let zero_counts = vec![0u32; bin_count];
    let counts_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("histogram-counts"),
            contents: bytemuck::cast_slice(&zero_counts),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        }),
    );

    if inputs.sample_count > 0 {
        run_histogram_pass(device, queue, inputs, params, &counts_buffer)?;
    }

    let values_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram-counts-f32"),
        size: (bin_count * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));

    run_convert_pass(
        device,
        queue,
        params.bin_count,
        normalization_scale,
        &counts_buffer,
        &values_buffer,
    )?;

    Ok(values_buffer)
}

fn run_histogram_pass(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &HistogramGpuInputs,
    params: &HistogramGpuParams,
    counts_buffer: &Arc<wgpu::Buffer>,
) -> Result<(), String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_counts_shader(device, workgroup_size, inputs.scalar);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("histogram-counts-bind-layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("histogram-counts-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("histogram-counts-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let uniforms = HistogramUniforms {
        min_value: params.min_value,
        inv_bin_width: params.inv_bin_width,
        sample_count: inputs.sample_count,
        bin_count: params.bin_count,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("histogram-counts-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram-counts-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.samples.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("histogram-counts-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram-counts-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (inputs.sample_count + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(())
}

fn run_convert_pass(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    bin_count: u32,
    normalization_scale: f32,
    counts_buffer: &Arc<wgpu::Buffer>,
    values_buffer: &Arc<wgpu::Buffer>,
) -> Result<(), String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_convert_shader(device, workgroup_size);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("histogram-convert-bind-layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("histogram-convert-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("histogram-convert-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let uniforms = ConvertUniforms {
        bin_count,
        _pad: [0; 3],
        scale: normalization_scale,
        _pad2: [0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("histogram-convert-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram-convert-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("histogram-convert-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram-convert-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (bin_count + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(())
}

fn compile_counts_shader(
    device: &Arc<wgpu::Device>,
    workgroup_size: u32,
    scalar: ScalarType,
) -> wgpu::ShaderModule {
    let template = match scalar {
        ScalarType::F32 => shaders::histogram_counts::F32,
        ScalarType::F64 => shaders::histogram_counts::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("histogram-counts-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn compile_convert_shader(device: &Arc<wgpu::Device>, workgroup_size: u32) -> wgpu::ShaderModule {
    let source = shaders::histogram_convert::TEMPLATE
        .replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("histogram-convert-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
