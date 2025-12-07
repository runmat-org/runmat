use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;
const VERTICES_PER_BAR: u32 = 6;

/// Inputs required to pack bar vertices directly on the GPU.
pub struct BarGpuInputs {
    pub values_buffer: Arc<wgpu::Buffer>,
    pub len: u32,
    pub scalar: ScalarType,
}

/// Parameters describing how the GPU vertices should be generated.
pub struct BarGpuParams {
    pub color: Vec4,
    pub bar_width: f32,
    pub group_index: u32,
    pub group_count: u32,
    pub orientation: BarOrientation,
}

#[derive(Clone, Copy, Debug)]
pub enum BarOrientation {
    Vertical,
    Horizontal,
}

impl BarOrientation {
    fn as_u32(self) -> u32 {
        match self {
            BarOrientation::Vertical => 0,
            BarOrientation::Horizontal => 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BarUniforms {
    color: [f32; 4],
    bar_width: f32,
    count: u32,
    group_index: u32,
    group_count: u32,
    orientation: u32,
    _pad: u32,
}

/// Builds a GPU-resident vertex buffer for bar charts directly from provider-owned value arrays.
pub fn pack_vertices_from_values(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &BarGpuInputs,
    params: &BarGpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.len == 0 {
        return Err("bar: input cannot be empty".to_string());
    }

    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bar-pack-bind-layout"),
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
        label: Some("bar-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bar-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let vertex_count = inputs.len as u64 * VERTICES_PER_BAR as u64;
    let output_size = vertex_count * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bar-gpu-vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let uniforms = BarUniforms {
        color: params.color.to_array(),
        bar_width: params.bar_width,
        count: inputs.len,
        group_index: params.group_index,
        group_count: params.group_count.max(1),
        orientation: params.orientation.as_u32(),
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bar-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bar-pack-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bar-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bar-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (inputs.len + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(workgroups, 1, 1);
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
        ScalarType::F32 => shaders::bar::F32,
        ScalarType::F64 => shaders::bar::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bar-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
