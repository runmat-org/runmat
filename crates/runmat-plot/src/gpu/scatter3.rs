use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::scatter2::{ScatterAttributeBuffer, ScatterColorBuffer};
use crate::gpu::shaders;
use crate::gpu::{tuning, util, ScalarType};
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Inputs required to pack scatter3 vertices directly on the GPU.
pub struct Scatter3GpuInputs {
    pub x_buffer: Arc<wgpu::Buffer>,
    pub y_buffer: Arc<wgpu::Buffer>,
    pub z_buffer: Arc<wgpu::Buffer>,
    pub len: u32,
    pub scalar: ScalarType,
}

/// Parameters describing how the GPU vertices should be generated.
pub struct Scatter3GpuParams {
    pub color: Vec4,
    pub point_size: f32,
    pub sizes: ScatterAttributeBuffer,
    pub colors: ScatterColorBuffer,
    pub lod_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Scatter3Uniforms {
    color: [f32; 4],
    point_size: f32,
    count: u32,
    lod_stride: u32,
    has_sizes: u32,
    has_colors: u32,
    color_stride: u32,
    _pad: u32,
}

/// Builds a GPU-resident vertex buffer for scatter3 plots directly from
/// provider-owned XYZ arrays with either single- or double-precision inputs.
pub fn pack_vertices_from_xyz(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &Scatter3GpuInputs,
    params: &Scatter3GpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.len == 0 {
        return Err("scatter3: empty input tensors".to_string());
    }

    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scatter3-pack-bind-layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            wgpu::BindGroupLayoutEntry {
                binding: 7,
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scatter3-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scatter3-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let lod_stride = params.lod_stride.max(1);
    let max_points = (inputs.len + lod_stride - 1) / lod_stride;
    let output_size = max_points as u64 * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter3-gpu-vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter3-gpu-counter"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&counter_buffer, 0, bytemuck::bytes_of(&0u32));

    let (size_buffer, has_sizes) = prepare_size_buffer(device, params);
    let (color_buffer, has_colors, color_stride) = prepare_color_buffer(device, params);

    let uniforms = Scatter3Uniforms {
        color: params.color.to_array(),
        point_size: params.point_size,
        count: inputs.len,
        lod_stride,
        has_sizes: if has_sizes { 1 } else { 0 },
        has_colors: if has_colors { 1 } else { 0 },
        color_stride,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scatter3-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter3-pack-bind-group"),
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
                resource: inputs.z_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: color_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: counter_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scatter3-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter3-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (inputs.len + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter3-pack-counter-readback"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &counter_buffer,
        0,
        &readback_buffer,
        0,
        std::mem::size_of::<u32>() as u64,
    );
    queue.submit(Some(encoder.finish()));

    let drawn_vertices = util::readback_u32(device, &readback_buffer)
        .map_err(|e| format!("scatter3: failed to read GPU vertex count: {e}"))?;

    Ok(GpuVertexBuffer::new(output_buffer, drawn_vertices as usize))
}

fn prepare_size_buffer(
    device: &Arc<wgpu::Device>,
    params: &Scatter3GpuParams,
) -> (Arc<wgpu::Buffer>, bool) {
    match &params.sizes {
        ScatterAttributeBuffer::None => (
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scatter3-size-fallback"),
                    contents: bytemuck::cast_slice(&[0.0f32]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            ),
            false,
        ),
        ScatterAttributeBuffer::Host(data) => {
            if data.is_empty() {
                (
                    Arc::new(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-size-fallback"),
                            contents: bytemuck::cast_slice(&[0.0f32]),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        }),
                    ),
                    false,
                )
            } else {
                (
                    Arc::new(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-size-host"),
                            contents: bytemuck::cast_slice(data.as_slice()),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        }),
                    ),
                    true,
                )
            }
        }
        ScatterAttributeBuffer::Gpu(buffer) => (buffer.clone(), true),
    }
}

fn prepare_color_buffer(
    device: &Arc<wgpu::Device>,
    params: &Scatter3GpuParams,
) -> (Arc<wgpu::Buffer>, bool, u32) {
    match &params.colors {
        ScatterColorBuffer::None => (
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scatter3-color-fallback"),
                    contents: bytemuck::cast_slice(&[
                        params.color.x,
                        params.color.y,
                        params.color.z,
                        params.color.w,
                    ]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            ),
            false,
            4,
        ),
        ScatterColorBuffer::Host(colors) => {
            if colors.is_empty() {
                (
                    Arc::new(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-color-fallback"),
                            contents: bytemuck::cast_slice(&[
                                params.color.x,
                                params.color.y,
                                params.color.z,
                                params.color.w,
                            ]),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        }),
                    ),
                    false,
                    4,
                )
            } else {
                (
                    Arc::new(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-color-host"),
                            contents: bytemuck::cast_slice(colors.as_slice()),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        }),
                    ),
                    true,
                    4,
                )
            }
        }
        ScatterColorBuffer::Gpu { buffer, components } => (buffer.clone(), true, *components),
    }
}

fn compile_shader(
    device: &Arc<wgpu::Device>,
    workgroup_size: u32,
    scalar: ScalarType,
) -> wgpu::ShaderModule {
    let template = match scalar {
        ScalarType::F32 => shaders::scatter3::F32,
        ScalarType::F64 => shaders::scatter3::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scatter3-pack-shader"),
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
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("scatter3-test-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                },
                None,
            )
            .block_on()
            .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    #[test]
    fn lod_stride_limits_vertex_count() {
        let Some((device, queue)) = maybe_device() else {
            return;
        };
        let point_count = 1_200_000u32;
        let stride = 4u32;
        let max_points = (point_count + stride - 1) / stride;

        let x: Vec<f32> = (0..point_count).map(|i| i as f32 * 0.001).collect();
        let y: Vec<f32> = x.iter().map(|v| v.cos()).collect();
        let z: Vec<f32> = x.iter().map(|v| v.sin()).collect();

        let make_buffer = |label: &str, data: &[f32]| {
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            )
        };

        let inputs = Scatter3GpuInputs {
            x_buffer: make_buffer("scatter3-test-x", &x),
            y_buffer: make_buffer("scatter3-test-y", &y),
            z_buffer: make_buffer("scatter3-test-z", &z),
            len: point_count,
            scalar: ScalarType::F32,
        };
        let params = Scatter3GpuParams {
            color: Vec4::new(0.2, 0.6, 0.9, 1.0),
            point_size: 6.0,
            sizes: ScatterAttributeBuffer::None,
            colors: ScatterColorBuffer::None,
            lod_stride: stride,
        };

        let gpu_vertices =
            pack_vertices_from_xyz(&device, &queue, &inputs, &params).expect("gpu scatter3 pack");
        assert_eq!(gpu_vertices.vertex_count, max_points as usize);
    }
}
