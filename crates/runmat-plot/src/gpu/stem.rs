use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use crate::plots::line::LineStyle;
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct StemGpuInputs {
    pub x_buffer: Arc<wgpu::Buffer>,
    pub y_buffer: Arc<wgpu::Buffer>,
    pub len: u32,
    pub scalar: ScalarType,
}

pub struct StemGpuParams {
    pub color: Vec4,
    pub baseline_color: Vec4,
    pub baseline: f32,
    pub baseline_visible: bool,
    pub min_x: f32,
    pub max_x: f32,
    pub line_style: LineStyle,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StemUniforms {
    color: [f32; 4],
    baseline_color: [f32; 4],
    baseline: f32,
    min_x: f32,
    max_x: f32,
    point_count: u32,
    line_style: u32,
    baseline_visible: u32,
}

pub fn pack_vertices_from_xy(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &StemGpuInputs,
    params: &StemGpuParams,
) -> Result<GpuVertexBuffer, String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stem-pack-bind-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stem-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stem-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let baseline_count = if params.baseline_visible { 2 } else { 0 };
    let vertex_count = baseline_count as u64 + inputs.len as u64 * 2;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stem-gpu-vertices"),
        size: vertex_count * std::mem::size_of::<Vertex>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let uniforms = StemUniforms {
        color: params.color.to_array(),
        baseline_color: params.baseline_color.to_array(),
        baseline: params.baseline,
        min_x: params.min_x,
        max_x: params.max_x,
        point_count: inputs.len,
        line_style: line_style_code(params.line_style),
        baseline_visible: if params.baseline_visible { 1 } else { 0 },
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stem-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stem-pack-bind-group"),
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
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("stem-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stem-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(inputs.len.div_ceil(workgroup_size), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    Ok(GpuVertexBuffer::new(output_buffer, vertex_count as usize))
}

fn compile_shader(
    device: &Arc<wgpu::Device>,
    workgroup_size: u32,
    scalar: ScalarType,
) -> wgpu::ShaderModule {
    let template = match scalar {
        ScalarType::F32 => shaders::stem::F32,
        ScalarType::F64 => shaders::stem::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stem-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
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
fn line_style_code(style: LineStyle) -> u32 {
    match style {
        LineStyle::Solid => 0,
        LineStyle::Dashed => 1,
        LineStyle::Dotted => 2,
        LineStyle::DashDot => 3,
    }
}
