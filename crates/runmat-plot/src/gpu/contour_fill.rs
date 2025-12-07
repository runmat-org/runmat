use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct ContourFillGpuInputs<'a> {
    pub x_axis: &'a [f32],
    pub y_axis: &'a [f32],
    pub z_buffer: Arc<wgpu::Buffer>,
    pub color_table: &'a [[f32; 4]],
    pub level_values: &'a [f32],
    pub x_len: u32,
    pub y_len: u32,
    pub scalar: ScalarType,
}

pub struct ContourFillGpuParams {
    pub base_z: f32,
    pub alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContourFillUniforms {
    base_z: f32,
    alpha: f32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    level_count: u32,
    cell_count: u32,
    _pad: u32,
}

const TRI_VERTICES: usize = 6;

pub fn pack_contour_fill_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &ContourFillGpuInputs<'_>,
    params: &ContourFillGpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.x_len < 2 || inputs.y_len < 2 {
        return Err("contourf: axis vectors must contain at least two elements".to_string());
    }
    if inputs.level_values.len() < 2 {
        return Err("contourf: level vector must contain at least two entries".to_string());
    }
    if inputs.color_table.is_empty() {
        return Err("contourf: color table must contain at least one entry".to_string());
    }

    let cells_x = inputs.x_len - 1;
    let cells_y = inputs.y_len - 1;
    let cell_count = cells_x
        .checked_mul(cells_y)
        .ok_or_else(|| "contourf: grid dimensions overflowed cell count".to_string())?;
    if cell_count == 0 {
        return Err("contourf: no cells available for fill generation".to_string());
    }
    let vertex_count = cell_count
        .checked_mul(TRI_VERTICES as u32)
        .ok_or_else(|| "contourf: vertex count overflowed".to_string())?;

    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);

    let x_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contourf-x-axis"),
            contents: bytemuck::cast_slice(inputs.x_axis),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let y_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contourf-y-axis"),
            contents: bytemuck::cast_slice(inputs.y_axis),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let color_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contourf-color-table"),
            contents: bytemuck::cast_slice(inputs.color_table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let level_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contourf-level-values"),
            contents: bytemuck::cast_slice(inputs.level_values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let output_size = vertex_count as u64 * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contourf-gpu-vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let uniforms = ContourFillUniforms {
        base_z: params.base_z,
        alpha: params.alpha,
        x_len: inputs.x_len,
        y_len: inputs.y_len,
        color_table_len: inputs.color_table.len() as u32,
        level_count: inputs.level_values.len() as u32,
        cell_count,
        _pad: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contourf-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("contourf-bind-layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
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
        label: Some("contourf-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("contourf-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("contourf-bind-group"),
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
                resource: level_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("contourf-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("contourf-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((cell_count + workgroup_size - 1) / workgroup_size).max(1);
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
        ScalarType::F32 => shaders::contour_fill::F32,
        ScalarType::F64 => shaders::contour_fill::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("contourf-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
