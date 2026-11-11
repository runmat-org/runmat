use crate::core::renderer::Vertex;
use crate::core::scene::{DrawIndirectArgsRaw, GpuVertexBuffer};
use crate::gpu::scatter2::{
    ScatterAttributeBuffer, ScatterColorBuffer, ScatterColorLayout, ScatterSizeUnit,
};
use crate::gpu::shaders;
use crate::gpu::{
    scatter_error::preflight_native_f64_features, tuning, ScalarType, ScatterPackError,
};
use glam::Vec4;
use wgpu::util::DeviceExt;

/// Inputs required to pack scatter3 vertices directly on the GPU.
#[derive(Clone, Debug)]
pub struct Scatter3GpuInputs {
    pub x_buffer: crate::gpu::GpuInputBuffer,
    pub y_buffer: crate::gpu::GpuInputBuffer,
    pub z_buffer: crate::gpu::GpuInputBuffer,
    pub len: u32,
    pub scalar: ScalarType,
    /// Retained source size authority, including a borrowed GPU export.
    pub sizes: ScatterAttributeBuffer,
    pub size_unit: ScatterSizeUnit,
    pub colors: ScatterColorBuffer,
    pub color_layout: ScatterColorLayout,
}

/// Parameters describing how the GPU vertices should be generated.
pub struct Scatter3GpuParams {
    pub color: Vec4,
    pub point_size: f32,
    pub sizes: ScatterAttributeBuffer,
    pub size_unit: ScatterSizeUnit,
    pub colors: ScatterColorBuffer,
    pub color_layout: ScatterColorLayout,
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
    attribute_layout: u32,
    _pad: u32,
}

/// Builds a GPU-resident vertex buffer for scatter3 plots directly from
/// provider-owned XYZ arrays. Binary64 inputs are loaded and narrowed to F32
/// plot vertices. External raw-device callers must validate the native F64
/// adapter before constructing the device; this function checks enabled
/// features before creating a shader or output buffer.
pub fn pack_vertices_from_xyz(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    inputs: &Scatter3GpuInputs,
    params: &Scatter3GpuParams,
) -> Result<GpuVertexBuffer, ScatterPackError> {
    let device = emission.device();
    let queue = emission.queue();
    if inputs.len == 0 {
        return Err(ScatterPackError::EmptyInputs);
    }
    if inputs.size_unit != params.size_unit || inputs.color_layout != params.color_layout {
        return Err(ScatterPackError::RetainedAttributeMismatch);
    }
    if params.size_unit != ScatterSizeUnit::from_source(&params.sizes)
        || params.color_layout != ScatterColorLayout::from_source(&params.colors)
    {
        return Err(ScatterPackError::SourceAttributeMismatch);
    }
    preflight_native_f64_features(inputs.scalar, device.features())?;

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

    let pipeline =
        device.create_compute_pipeline(&crate::wgpu_compat::wgpu_compute_pipeline_descriptor! {
            label: Some("scatter3-pack-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

    let lod_stride = params.lod_stride.max(1);
    let max_points = inputs.len.div_ceil(lod_stride);
    let output_size = max_points as u64 * 6 * std::mem::size_of::<Vertex>() as u64;
    let output_buffer = runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("scatter3-gpu-vertices"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    ));

    let indirect_args = runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("scatter3-gpu-indirect-args"),
            size: std::mem::size_of::<DrawIndirectArgsRaw>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    ));
    let init = DrawIndirectArgsRaw {
        vertex_count: 0,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&indirect_args, 0, bytemuck::bytes_of(&init));

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
        attribute_layout: params.size_unit.shader_flag() | (params.color_layout.shader_flag() << 1),
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
                resource: inputs
                    .x_buffer
                    .buffer(emission)
                    .map_err(ScatterPackError::BufferAccess)?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inputs
                    .y_buffer
                    .buffer(emission)
                    .map_err(ScatterPackError::BufferAccess)?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs
                    .z_buffer
                    .buffer(emission)
                    .map_err(ScatterPackError::BufferAccess)?
                    .as_entire_binding(),
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
                resource: size_buffer
                    .buffer(emission)
                    .map_err(ScatterPackError::BufferAccess)?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: color_buffer
                    .buffer(emission)
                    .map_err(ScatterPackError::BufferAccess)?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: indirect_args.as_entire_binding(),
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
        let workgroups = inputs.len.div_ceil(workgroup_size);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(GpuVertexBuffer::with_indirect(
        output_buffer,
        (max_points as usize) * 6,
        indirect_args,
    ))
}

fn prepare_size_buffer(
    device: &wgpu::Device,
    params: &Scatter3GpuParams,
) -> (crate::gpu::GpuInputBuffer, bool) {
    match &params.sizes {
        ScatterAttributeBuffer::None => (
            runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("scatter3-size-fallback"),
                    contents: bytemuck::cast_slice(&[0.0f32]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                },
            ))
            .into(),
            false,
        ),
        ScatterAttributeBuffer::Host(data) => {
            if data.is_empty() {
                (
                    runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-size-fallback"),
                            contents: bytemuck::cast_slice(&[0.0f32]),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ))
                    .into(),
                    false,
                )
            } else {
                (
                    runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-size-host"),
                            contents: bytemuck::cast_slice(data.as_slice()),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ))
                    .into(),
                    true,
                )
            }
        }
        ScatterAttributeBuffer::Gpu(buffer) => (buffer.clone(), true),
    }
}

fn prepare_color_buffer(
    device: &wgpu::Device,
    params: &Scatter3GpuParams,
) -> (crate::gpu::GpuInputBuffer, bool, u32) {
    match &params.colors {
        ScatterColorBuffer::None => (
            runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("scatter3-color-fallback"),
                    contents: bytemuck::cast_slice(&[
                        params.color.x,
                        params.color.y,
                        params.color.z,
                        params.color.w,
                    ]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                },
            ))
            .into(),
            false,
            4,
        ),
        ScatterColorBuffer::Host(colors) => {
            if colors.is_empty() {
                (
                    runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-color-fallback"),
                            contents: bytemuck::cast_slice(&[
                                params.color.x,
                                params.color.y,
                                params.color.z,
                                params.color.w,
                            ]),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ))
                    .into(),
                    false,
                    4,
                )
            } else {
                (
                    runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("scatter3-color-host"),
                            contents: bytemuck::cast_slice(colors.as_slice()),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ))
                    .into(),
                    true,
                    4,
                )
            }
        }
        ScatterColorBuffer::Gpu { buffer, components } => (buffer.clone(), true, *components),
    }
}

fn compile_shader(
    device: &wgpu::Device,
    workgroup_size: u32,
    scalar: ScalarType,
) -> wgpu::ShaderModule {
    let template = match scalar {
        ScalarType::F32 => shaders::scatter3::F32,
        ScalarType::F64 => shaders::scatter3::F64,
    };
    let source = shaders::compute_source(template, workgroup_size);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scatter3-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

#[cfg(test)]
mod stress_tests {
    use super::*;
    use pollster::FutureExt;

    #[test]
    fn attribute_uniform_layout_matches_both_scatter3_shaders() {
        assert_eq!(std::mem::size_of::<Scatter3Uniforms>(), 48);
        assert_eq!(std::mem::offset_of!(Scatter3Uniforms, attribute_layout), 40);
    }

    fn maybe_device() -> Option<runmat_accelerate_api::WgpuDeviceAccess> {
        crate::gpu::test_device::maybe_scatter_device("scatter3-test-device")
    }

    #[test]
    fn lod_stride_limits_vertex_count() {
        let Some(access) = maybe_device() else {
            return;
        };
        let point_count = 1_200_000u32;
        let stride = 4u32;
        let max_points = point_count.div_ceil(stride);

        let x: Vec<f32> = (0..point_count).map(|i| i as f32 * 0.001).collect();
        let y: Vec<f32> = x.iter().map(|v| v.cos()).collect();
        let z: Vec<f32> = x.iter().map(|v| v.sin()).collect();

        let make_buffer = |label: &str, data: &[f32]| {
            runmat_accelerate_api::WgpuBufferOwner::new(
                access
                    .with_shared(|emission| {
                        emission
                            .device()
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some(label),
                                contents: bytemuck::cast_slice(data),
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            })
                    })
                    .expect("test device access"),
            )
        };

        let inputs = Scatter3GpuInputs {
            x_buffer: make_buffer("scatter3-test-x", &x).into(),
            y_buffer: make_buffer("scatter3-test-y", &y).into(),
            z_buffer: make_buffer("scatter3-test-z", &z).into(),
            len: point_count,
            scalar: ScalarType::F32,
            sizes: ScatterAttributeBuffer::None,
            size_unit: ScatterSizeUnit::DiameterPixels,
            colors: ScatterColorBuffer::None,
            color_layout: ScatterColorLayout::Interleaved,
        };
        let params = Scatter3GpuParams {
            color: Vec4::new(0.2, 0.6, 0.9, 1.0),
            point_size: 6.0,
            sizes: ScatterAttributeBuffer::None,
            size_unit: ScatterSizeUnit::DiameterPixels,
            colors: ScatterColorBuffer::None,
            color_layout: ScatterColorLayout::Interleaved,
            lod_stride: stride,
        };

        let gpu_vertices = access
            .with_shared(|emission| pack_vertices_from_xyz(emission, &inputs, &params))
            .expect("test device access")
            .expect("gpu scatter3 pack");
        assert_eq!(gpu_vertices.vertex_count, max_points as usize * 6);
    }

    #[test]
    fn gpu_area_sizes_and_column_major_rgb_pack_distinct_points() {
        let Some(access) = maybe_device() else {
            assert!(
                std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").is_err(),
                "forced GPU test requires an adapter"
            );
            return;
        };
        let supports_f64 = access
            .with_shared(|emission| {
                emission
                    .device()
                    .features()
                    .contains(runmat_accelerate_api::native_wgpu_f64::REQUIRED_FEATURES)
            })
            .expect("device feature query");
        if std::env::var("RUNMAT_PLOT_REQUIRE_F64").is_ok() {
            assert!(
                supports_f64,
                "qualification requires a real F64 GPU shader case"
            );
        }
        let make_buffer = |label: &str, data: &[u8]| -> crate::gpu::GpuInputBuffer {
            runmat_accelerate_api::WgpuBufferOwner::new(
                access
                    .with_shared(|emission| {
                        emission
                            .device()
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some(label),
                                contents: data,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            })
                    })
                    .expect("test device access"),
            )
            .into()
        };
        for scalar in [ScalarType::F32, ScalarType::F64] {
            if scalar == ScalarType::F64 && !supports_f64 {
                eprintln!("scatter3 F64 GPU shader unsupported by this adapter");
                continue;
            }
            let sizes = make_buffer(
                "scatter3-area-points2",
                bytemuck::cast_slice(&[36.0f32, 49.0]),
            );
            // A 2-by-3 MATLAB color matrix in physical column-major order:
            // point zero is red, point one is green.
            let colors = make_buffer(
                "scatter3-column-major-rgb",
                bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]),
            );
            let coordinate = |label_f32: &str,
                              label_f64: &str,
                              values_f32: [f32; 2],
                              values_f64: [f64; 2]| match scalar {
                ScalarType::F32 => make_buffer(label_f32, bytemuck::cast_slice(&values_f32)),
                ScalarType::F64 => make_buffer(label_f64, bytemuck::cast_slice(&values_f64)),
            };
            let inputs = Scatter3GpuInputs {
                x_buffer: coordinate(
                    "scatter3-unit-x-f32",
                    "scatter3-unit-x-f64",
                    [0.0, 1.0],
                    [0.0, 1.0],
                ),
                y_buffer: coordinate(
                    "scatter3-unit-y-f32",
                    "scatter3-unit-y-f64",
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
                z_buffer: coordinate(
                    "scatter3-unit-z-f32",
                    "scatter3-unit-z-f64",
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
                len: 2,
                scalar,
                sizes: ScatterAttributeBuffer::Gpu(sizes.clone()),
                size_unit: ScatterSizeUnit::AreaPointsSquared,
                colors: ScatterColorBuffer::Gpu {
                    buffer: colors.clone(),
                    components: 3,
                },
                color_layout: ScatterColorLayout::ColumnMajor,
            };
            let params = Scatter3GpuParams {
                color: Vec4::ONE,
                point_size: 6.0,
                sizes: ScatterAttributeBuffer::Gpu(sizes),
                size_unit: ScatterSizeUnit::AreaPointsSquared,
                colors: ScatterColorBuffer::Gpu {
                    buffer: colors,
                    components: 3,
                },
                color_layout: ScatterColorLayout::ColumnMajor,
                lod_stride: 1,
            };
            let vertices = access
                .with_shared(|emission| pack_vertices_from_xyz(emission, &inputs, &params))
                .expect("test device access")
                .expect("gpu scatter3 pack");
            let bytes = crate::gpu::util::copy_readback_bytes_typed(
                &access,
                &vertices.buffer.clone().into(),
                12 * 48,
            )
            .block_on()
            .expect("vertex readback");
            let mut seen = [0usize; 2];
            for vertex in bytes.chunks_exact(48).take(12) {
                let field = |offset: usize| {
                    f32::from_le_bytes(vertex[offset..offset + 4].try_into().unwrap())
                };
                let index = if field(0) < 0.5 { 0 } else { 1 };
                seen[index] += 1;
                let expected = if index == 0 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                for (channel, expected) in expected.into_iter().enumerate() {
                    assert!((field(12 + 4 * channel) - expected).abs() < 1e-6);
                }
                let area = if index == 0 { 36.0 } else { 49.0 };
                let expected_size = crate::plots::scatter::marker_area_points2_to_diameter_px(area);
                assert!((field(36) - expected_size).abs() < 1e-4);
            }
            assert_eq!(seen, [6, 6]);
            eprintln!("scatter3 {scalar:?} GPU attribute case executed");
        }
    }
}
