use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Inputs required to pack surface vertices directly on the GPU.
pub struct SurfaceGpuInputs<'a> {
    pub x_axis: &'a [f32],
    pub y_axis: &'a [f32],
    pub z_buffer: Arc<wgpu::Buffer>,
    pub color_table: &'a [[f32; 4]],
    pub x_len: u32,
    pub y_len: u32,
    pub scalar: ScalarType,
}

/// Parameters describing how the GPU vertices should be generated.
pub struct SurfaceGpuParams {
    pub min_z: f32,
    pub max_z: f32,
    pub alpha: f32,
    pub flatten_z: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceUniforms {
    min_z: f32,
    max_z: f32,
    alpha: f32,
    flatten: u32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    _pad: u32,
}

/// Builds a GPU-resident vertex buffer for surface plots directly from provider-owned Z data.
pub fn pack_surface_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &SurfaceGpuInputs<'_>,
    params: &SurfaceGpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.x_len < 2 || inputs.y_len < 2 {
        return Err("surf: axis vectors must contain at least two elements".to_string());
    }

    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);

    let x_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface-x-axis"),
            contents: bytemuck::cast_slice(inputs.x_axis),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let y_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface-y-axis"),
            contents: bytemuck::cast_slice(inputs.y_axis),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let color_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface-color-table"),
            contents: bytemuck::cast_slice(inputs.color_table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let vertex_count = inputs
        .x_len
        .checked_mul(inputs.y_len)
        .ok_or_else(|| "surf: grid dimensions overflowed vertex count".to_string())?;
    let output_size = vertex_count as u64 * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface-gpu-vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let uniforms = SurfaceUniforms {
        min_z: params.min_z,
        max_z: params.max_z.max(params.min_z + 1e-6),
        alpha: params.alpha,
        flatten: if params.flatten_z { 1 } else { 0 },
        x_len: inputs.x_len,
        y_len: inputs.y_len,
        color_table_len: inputs.color_table.len() as u32,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("surface-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface-pack-bind-layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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
        label: Some("surface-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("surface-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("surface-pack-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs.z_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: color_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("surface-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("surface-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (vertex_count + workgroup_size - 1) / workgroup_size;
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
        ScalarType::F32 => shaders::surface::F32,
        ScalarType::F64 => shaders::surface::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("surface-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
