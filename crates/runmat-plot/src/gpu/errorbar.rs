use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use crate::plots::line::LineStyle;
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct ErrorBarGpuInputs {
    pub x_buffer: Arc<wgpu::Buffer>,
    pub y_buffer: Arc<wgpu::Buffer>,
    pub x_neg_buffer: Option<Arc<wgpu::Buffer>>,
    pub x_pos_buffer: Option<Arc<wgpu::Buffer>>,
    pub y_neg_buffer: Arc<wgpu::Buffer>,
    pub y_pos_buffer: Arc<wgpu::Buffer>,
    pub len: u32,
    pub scalar: ScalarType,
}

pub struct ErrorBarGpuParams {
    pub color: Vec4,
    pub cap_size_data: f32,
    pub line_style: LineStyle,
    pub orientation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ErrorBarUniforms {
    color: [f32; 4],
    count: u32,
    line_style: u32,
    cap_half_width: f32,
    orientation: u32,
}

pub fn pack_vertical_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &ErrorBarGpuInputs,
    params: &ErrorBarGpuParams,
) -> Result<GpuVertexBuffer, String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("errorbar-pack-bind-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, false),
            uniform_entry(7),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("errorbar-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("errorbar-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let segments_per_point = match params.orientation {
        0 => 6u64,
        1 => 6u64,
        _ => 12u64,
    };
    let max_vertices = inputs.len as u64 * segments_per_point;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("errorbar-gpu-vertices"),
        size: max_vertices * std::mem::size_of::<Vertex>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let uniforms = ErrorBarUniforms {
        color: params.color.to_array(),
        count: inputs.len,
        line_style: line_style_code(params.line_style),
        cap_half_width: params.cap_size_data.max(0.0) * 0.5,
        orientation: params.orientation,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("errorbar-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("errorbar-pack-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inputs.y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs
                    .x_neg_buffer
                    .as_ref()
                    .map(|b| b.as_entire_binding())
                    .unwrap_or(inputs.x_buffer.as_entire_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs
                    .x_pos_buffer
                    .as_ref()
                    .map(|b| b.as_entire_binding())
                    .unwrap_or(inputs.x_buffer.as_entire_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: inputs.y_neg_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: inputs.y_pos_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("errorbar-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("errorbar-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(inputs.len.div_ceil(workgroup_size), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    Ok(GpuVertexBuffer::new(output_buffer, max_vertices as usize))
}

fn compile_shader(
    device: &Arc<wgpu::Device>,
    workgroup_size: u32,
    scalar: ScalarType,
) -> wgpu::ShaderModule {
    let template = match scalar {
        ScalarType::F32 => shaders::errorbar::F32,
        ScalarType::F64 => shaders::errorbar::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("errorbar-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn line_style_code(style: LineStyle) -> u32 {
    match style {
        LineStyle::Solid => 0,
        LineStyle::Dashed => 1,
        LineStyle::Dotted => 2,
        LineStyle::DashDot => 3,
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
