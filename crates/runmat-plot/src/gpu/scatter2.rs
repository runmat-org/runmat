use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::shaders;
use crate::gpu::{tuning, util, ScalarType};
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Optional per-point scalar attribute buffer (marker sizes).
pub enum ScatterAttributeBuffer {
    None,
    Host(Vec<f32>),
    Gpu(Arc<wgpu::Buffer>),
}

impl ScatterAttributeBuffer {
    pub fn has_data(&self) -> bool {
        !matches!(self, ScatterAttributeBuffer::None)
    }
}

/// Optional per-point color buffer. Host data is supplied as RGBA tuples; GPU buffers
/// may contain RGB (3 floats) or RGBA (4 floats) sequences.
pub enum ScatterColorBuffer {
    None,
    Host(Vec<[f32; 4]>),
    Gpu {
        buffer: Arc<wgpu::Buffer>,
        components: u32,
    },
}

impl ScatterColorBuffer {
    pub fn has_data(&self) -> bool {
        !matches!(self, ScatterColorBuffer::None)
    }

    pub fn stride(&self) -> u32 {
        match self {
            ScatterColorBuffer::None => 4,
            ScatterColorBuffer::Host(_) => 4,
            ScatterColorBuffer::Gpu { components, .. } => *components,
        }
    }

    pub fn buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        match self {
            ScatterColorBuffer::Gpu { buffer, .. } => Some(buffer.clone()),
            _ => None,
        }
    }
}

/// Inputs required to pack scatter vertices directly on the GPU.
pub struct Scatter2GpuInputs {
    pub x_buffer: Arc<wgpu::Buffer>,
    pub y_buffer: Arc<wgpu::Buffer>,
    pub len: u32,
    pub scalar: ScalarType,
}

/// Parameters describing how the GPU vertices should be generated.
pub struct Scatter2GpuParams {
    pub color: Vec4,
    pub point_size: f32,
    pub sizes: ScatterAttributeBuffer,
    pub colors: ScatterColorBuffer,
    pub lod_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Scatter2Uniforms {
    color: [f32; 4],
    point_size: f32,
    count: u32,
    lod_stride: u32,
    has_sizes: u32,
    has_colors: u32,
    color_stride: u32,
}

/// Builds a GPU-resident vertex buffer for scatter plots directly from
/// provider-owned XY arrays. Only single-precision tensors are supported
/// today; callers should fall back to host packing for other types.
pub fn pack_vertices_from_xy(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &Scatter2GpuInputs,
    params: &Scatter2GpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.len == 0 {
        return Err("scatter: empty input tensors".to_string());
    }

    let lod_stride = params.lod_stride.max(1);
    let max_points = inputs.len.div_ceil(lod_stride);

    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scatter2-pack-bind-layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scatter2-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scatter2-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let output_size = max_points as u64 * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter2-gpu-vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter2-gpu-counter"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&counter_buffer, 0, bytemuck::bytes_of(&0u32));

    let (size_buffer, has_sizes) = prepare_size_buffer(device, params);
    let (color_buffer, has_colors, color_stride) = prepare_color_buffer(device, params);

    let uniforms = Scatter2Uniforms {
        color: params.color.to_array(),
        point_size: params.point_size,
        count: inputs.len,
        lod_stride,
        has_sizes: if has_sizes { 1 } else { 0 },
        has_colors: if has_colors { 1 } else { 0 },
        color_stride,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scatter2-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter2-pack-bind-group"),
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: color_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: counter_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scatter2-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter2-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = inputs.len.div_ceil(workgroup_size);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter2-pack-counter-readback"),
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
        .map_err(|e| format!("scatter: failed to read GPU vertex count: {e}"))?;
    Ok(GpuVertexBuffer::new(output_buffer, drawn_vertices as usize))
}

fn prepare_size_buffer(
    device: &Arc<wgpu::Device>,
    params: &Scatter2GpuParams,
) -> (Arc<wgpu::Buffer>, bool) {
    match &params.sizes {
        ScatterAttributeBuffer::None => (
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scatter2-size-fallback"),
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
                            label: Some("scatter2-size-fallback"),
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
                            label: Some("scatter2-size-host"),
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
    params: &Scatter2GpuParams,
) -> (Arc<wgpu::Buffer>, bool, u32) {
    match &params.colors {
        ScatterColorBuffer::None => (
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scatter2-color-fallback"),
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
                            label: Some("scatter2-color-fallback"),
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
                            label: Some("scatter2-color-host"),
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
        ScalarType::F32 => shaders::scatter2::F32,
        ScalarType::F64 => shaders::scatter2::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scatter2-pack-shader"),
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
                    label: Some("runmat-plot-scatter-test-device"),
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
    fn gpu_packer_handles_large_point_cloud() {
        let Some((device, queue)) = maybe_device() else {
            return;
        };
        let point_count = 1_200_000u32;
        let x_data: Vec<f32> = (0..point_count).map(|i| i as f32 * 0.001).collect();
        let y_data: Vec<f32> = x_data.iter().map(|v| v.sin()).collect();

        let x_buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scatter2-test-x"),
                contents: bytemuck::cast_slice(&x_data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        let y_buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scatter2-test-y"),
                contents: bytemuck::cast_slice(&y_data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );

        let target = 250_000u32;
        let stride = if point_count <= target {
            1
        } else {
            point_count.div_ceil(target)
        };
        let expected_vertices = point_count.div_ceil(stride) as usize;

        let inputs = Scatter2GpuInputs {
            x_buffer,
            y_buffer,
            len: point_count,
            scalar: ScalarType::F32,
        };
        let params = Scatter2GpuParams {
            color: Vec4::new(0.8, 0.1, 0.3, 1.0),
            point_size: 8.0,
            sizes: ScatterAttributeBuffer::None,
            colors: ScatterColorBuffer::None,
            lod_stride: stride,
        };

        let gpu_vertices =
            pack_vertices_from_xy(&device, &queue, &inputs, &params).expect("gpu packing failed");
        assert!(gpu_vertices.vertex_count > 0);
        assert_eq!(gpu_vertices.vertex_count, expected_vertices);
    }
}
