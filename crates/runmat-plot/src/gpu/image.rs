use crate::core::scene::GpuVertexBuffer;
use crate::gpu::axis::{axis_storage_buffer, AxisData};
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::core::renderer::Vertex;

pub struct TrueColorImageGpuInputs<'a> {
    pub x_axis: AxisData<'a>,
    pub y_axis: AxisData<'a>,
    pub image_buffer: Arc<wgpu::Buffer>,
    pub rows: u32,
    pub cols: u32,
    pub channels: u32,
    pub scalar: ScalarType,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageUniforms {
    rows: u32,
    cols: u32,
    channels: u32,
    _pad: u32,
}

pub fn pack_truecolor_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &TrueColorImageGpuInputs<'_>,
) -> Result<GpuVertexBuffer, String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);
    let x_buffer = axis_storage_buffer(device, "image-x", &inputs.x_axis, inputs.scalar)?;
    let y_buffer = axis_storage_buffer(device, "image-y", &inputs.y_axis, inputs.scalar)?;
    let vertex_count = inputs.rows as u64 * inputs.cols as u64;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("image-gpu-vertices"),
        size: vertex_count * std::mem::size_of::<Vertex>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let uniforms = ImageUniforms {
        rows: inputs.rows,
        cols: inputs.cols,
        channels: inputs.channels,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("image-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("image-pack-bind-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("image-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("image-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("image-pack-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs.image_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("image-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("image-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.rows * inputs.cols).div_ceil(workgroup_size), 1, 1);
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
        ScalarType::F32 => shaders::image::F32,
        ScalarType::F64 => shaders::image::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("image-pack-shader"),
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

#[cfg(test)]
mod tests {
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
                    label: Some("runmat-plot-image-test-device"),
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
    fn gpu_packer_generates_truecolor_image_vertices() {
        let Some((device, queue)) = maybe_device() else {
            return;
        };
        let x = [1.0f32, 2.0f32];
        let y = [1.0f32, 2.0f32];
        let image = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image-test-truecolor"),
                contents: bytemuck::cast_slice(&[
                    1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                ]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        let packed = pack_truecolor_vertices(
            &device,
            &queue,
            &TrueColorImageGpuInputs {
                x_axis: AxisData::F32(&x),
                y_axis: AxisData::F32(&y),
                image_buffer: image,
                rows: 2,
                cols: 2,
                channels: 3,
                scalar: ScalarType::F32,
            },
        )
        .expect("image pack should succeed");
        assert_eq!(packed.vertex_count, 4);
    }
}
