use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Inputs required to pack contour vertices directly on the GPU.
pub struct ContourGpuInputs<'a> {
    pub x_axis: &'a [f32],
    pub y_axis: &'a [f32],
    pub z_buffer: Arc<wgpu::Buffer>,
    pub color_table: &'a [[f32; 4]],
    pub level_values: &'a [f32],
    pub x_len: u32,
    pub y_len: u32,
    pub scalar: ScalarType,
}

/// Parameters describing the contours to generate.
pub struct ContourGpuParams {
    pub min_z: f32,
    pub max_z: f32,
    pub base_z: f32,
    pub level_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContourUniforms {
    min_z: f32,
    max_z: f32,
    base_z: f32,
    level_count: u32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    cell_count: u32,
}

const VERTICES_PER_INVOCATION: u64 = 4;

/// Builds a GPU-resident vertex buffer for contour plots directly from provider-owned Z data.
pub fn pack_contour_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &ContourGpuInputs<'_>,
    params: &ContourGpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.x_len < 2 || inputs.y_len < 2 {
        return Err("contour: axis vectors must contain at least two elements".to_string());
    }
    if inputs.level_values.is_empty() {
        return Err("contour: level buffer must not be empty".to_string());
    }
    if params.level_count == 0 {
        return Err("contour: level count must be positive".to_string());
    }
    if inputs.level_values.len() != params.level_count as usize {
        return Err("contour: level buffer length mismatch".to_string());
    }
    let cells_x = inputs.x_len - 1;
    let cells_y = inputs.y_len - 1;
    let cell_count = cells_x
        .checked_mul(cells_y)
        .ok_or_else(|| "contour: grid dimensions overflowed cell count".to_string())?;
    if cell_count == 0 {
        return Err("contour: no cells available for contour generation".to_string());
    }
    let total_invocations = cell_count
        .checked_mul(params.level_count)
        .ok_or_else(|| "contour: invocation count overflowed".to_string())?;
    let vertex_count = total_invocations
        .checked_mul(VERTICES_PER_INVOCATION as u32)
        .ok_or_else(|| "contour: vertex count overflowed".to_string())?;

    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);

    let x_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contour-x-axis"),
            contents: bytemuck::cast_slice(inputs.x_axis),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let y_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contour-y-axis"),
            contents: bytemuck::cast_slice(inputs.y_axis),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let color_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contour-color-table"),
            contents: bytemuck::cast_slice(inputs.color_table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let level_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contour-level-values"),
            contents: bytemuck::cast_slice(inputs.level_values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let output_size = (vertex_count as u64) * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contour-gpu-vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let uniforms = ContourUniforms {
        min_z: params.min_z,
        max_z: if params.max_z <= params.min_z {
            params.min_z + 1e-6
        } else {
            params.max_z
        },
        base_z: params.base_z,
        level_count: params.level_count,
        x_len: inputs.x_len,
        y_len: inputs.y_len,
        color_table_len: inputs.color_table.len() as u32,
        cell_count,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contour-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("contour-pack-bind-layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("contour-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("contour-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("contour-pack-bind-group"),
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
            wgpu::BindGroupEntry {
                binding: 6,
                resource: level_buffer.as_ref().as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("contour-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("contour-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (total_invocations + workgroup_size - 1) / workgroup_size;
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
        ScalarType::F32 => shaders::contour::F32,
        ScalarType::F64 => shaders::contour::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("contour-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
