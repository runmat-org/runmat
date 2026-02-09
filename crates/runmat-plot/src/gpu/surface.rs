use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::axis::{axis_storage_buffer, AxisData};
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Axis data source used by the GPU surface vertex packer.
pub type SurfaceAxis<'a> = AxisData<'a>;

/// Inputs required to pack surface vertices directly on the GPU.
pub struct SurfaceGpuInputs<'a> {
    pub x_axis: SurfaceAxis<'a>,
    pub y_axis: SurfaceAxis<'a>,
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
    pub x_stride: u32,
    pub y_stride: u32,
    pub lod_x_len: u32,
    pub lod_y_len: u32,
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
    lod_x_len: u32,
    lod_y_len: u32,
    x_stride: u32,
    y_stride: u32,
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

    let x_buffer = axis_storage_buffer(device, "surface-x-axis", &inputs.x_axis, inputs.scalar)?;
    let y_buffer = axis_storage_buffer(device, "surface-y-axis", &inputs.y_axis, inputs.scalar)?;

    let color_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface-color-table"),
            contents: bytemuck::cast_slice(inputs.color_table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let lod_x_len = params.lod_x_len.max(1);
    let lod_y_len = params.lod_y_len.max(1);
    let vertex_count = lod_x_len
        .checked_mul(lod_y_len)
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
        lod_x_len,
        lod_y_len,
        x_stride: params.x_stride.max(1),
        y_stride: params.y_stride.max(1),
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
        let workgroups = vertex_count.div_ceil(workgroup_size);
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

#[cfg(test)]
mod stress_tests {
    use super::*;
    use pollster::FutureExt;

    fn maybe_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        if std::env::var("RUNMAT_PLOT_SKIP_GPU_TESTS").is_ok()
            || std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").is_err()
        {
            return None;
        }
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("runmat-plot-surface-test-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .block_on()
            .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    #[test]
    fn gpu_packer_handles_large_surface() {
        let Some((device, queue)) = maybe_device() else {
            return;
        };
        let x_len = 2048u32;
        let y_len = 2048u32;
        let total = (x_len * y_len) as usize;
        let x_axis: Vec<f32> = (0..x_len).map(|i| i as f32 * 0.1).collect();
        let y_axis: Vec<f32> = (0..y_len).map(|i| i as f32 * 0.1).collect();
        let mut z_data = vec![0.0f32; total];
        for (idx, value) in z_data.iter_mut().enumerate() {
            let x = (idx % x_len as usize) as f32 * 0.01;
            let y = (idx / x_len as usize) as f32 * 0.01;
            *value = (x.sin() + y.cos()) * 0.5;
        }
        let z_buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface-test-z"),
                contents: bytemuck::cast_slice(&z_data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );

        let color_table: Vec<[f32; 4]> = (0..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                [t, 1.0 - t, 0.5, 1.0]
            })
            .collect();

        let inputs = SurfaceGpuInputs {
            x_axis: SurfaceAxis::F32(&x_axis),
            y_axis: SurfaceAxis::F32(&y_axis),
            z_buffer,
            color_table: &color_table,
            x_len,
            y_len,
            scalar: ScalarType::F32,
        };
        let stride = 8;
        let lod_x_len = x_len.div_ceil(stride);
        let lod_y_len = y_len.div_ceil(stride);
        let params = SurfaceGpuParams {
            min_z: -1.0,
            max_z: 1.0,
            alpha: 1.0,
            flatten_z: false,
            x_stride: stride,
            y_stride: stride,
            lod_x_len,
            lod_y_len,
        };

        let gpu_vertices =
            pack_surface_vertices(&device, &queue, &inputs, &params).expect("surface pack failed");
        assert!(gpu_vertices.vertex_count > 0);
        assert_eq!(gpu_vertices.vertex_count, (lod_x_len * lod_y_len) as usize);
    }
}
