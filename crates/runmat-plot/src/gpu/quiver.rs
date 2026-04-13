use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::axis::{axis_storage_buffer, AxisData};
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct QuiverGpuInputs<'a> {
    pub x_data: AxisData<'a>,
    pub y_data: AxisData<'a>,
    pub u_buffer: Arc<wgpu::Buffer>,
    pub v_buffer: Arc<wgpu::Buffer>,
    pub count: u32,
    pub rows: u32,
    pub cols: u32,
    pub xy_mode: u32,
    pub scalar: ScalarType,
}

pub struct QuiverGpuParams {
    pub color: Vec4,
    pub scale: f32,
    pub head_size: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuiverUniforms {
    color: [f32; 4],
    count: u32,
    rows: u32,
    cols: u32,
    xy_mode: u32,
    scale: f32,
    head_size: f32,
    _pad: f32,
}

pub fn pack_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &QuiverGpuInputs<'_>,
    params: &QuiverGpuParams,
) -> Result<GpuVertexBuffer, String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);
    let x_buffer = axis_storage_buffer(device, "quiver-x", &inputs.x_data, inputs.scalar)?;
    let y_buffer = axis_storage_buffer(device, "quiver-y", &inputs.y_data, inputs.scalar)?;
    let vertex_count = inputs.count as u64 * 6;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quiver-gpu-vertices"),
        size: vertex_count * std::mem::size_of::<Vertex>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let uniforms = QuiverUniforms {
        color: params.color.to_array(),
        count: inputs.count,
        rows: inputs.rows,
        cols: inputs.cols,
        xy_mode: inputs.xy_mode,
        scale: params.scale,
        head_size: params.head_size,
        _pad: 0.0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quiver-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quiver-pack-bind-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            uniform_entry(5),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quiver-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("quiver-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("quiver-pack-bind-group"),
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
                resource: inputs.u_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs.v_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("quiver-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quiver-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(inputs.count.div_ceil(workgroup_size), 1, 1);
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
        ScalarType::F32 => shaders::quiver::F32,
        ScalarType::F64 => shaders::quiver::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quiver-pack-shader"),
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
                    label: Some("runmat-plot-quiver-test-device"),
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
    fn gpu_packer_generates_quiver_vertices() {
        let Some((device, queue)) = maybe_device() else {
            return;
        };
        let x = [1.0f32, 2.0f32];
        let y = [1.0f32, 2.0f32];
        let u = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quiver-test-u"),
                contents: bytemuck::cast_slice(&[0.5f32, -0.25f32]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        let v = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quiver-test-v"),
                contents: bytemuck::cast_slice(&[1.0f32, 0.75f32]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        let packed = pack_vertices(
            &device,
            &queue,
            &QuiverGpuInputs {
                x_data: AxisData::F32(&x),
                y_data: AxisData::F32(&y),
                u_buffer: u,
                v_buffer: v,
                count: 2,
                rows: 2,
                cols: 1,
                xy_mode: 0,
                scalar: ScalarType::F32,
            },
            &QuiverGpuParams {
                color: Vec4::ONE,
                scale: 1.0,
                head_size: 0.2,
            },
        )
        .expect("quiver pack should succeed");
        assert_eq!(packed.vertex_count, 12);
    }
}
