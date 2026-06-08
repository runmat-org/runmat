use crate::core::renderer::Vertex;
use crate::core::scene::GpuVertexBuffer;
use crate::gpu::axis::{axis_storage_buffer, AxisData};
use crate::gpu::shaders;
use crate::gpu::{tuning, ScalarType};
use glam::Vec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct AreaGpuInputs<'a> {
    pub x_axis: AxisData<'a>,
    pub y_buffer: Arc<wgpu::Buffer>,
    pub rows: u32,
    pub cols: u32,
    pub target_col: u32,
    pub scalar: ScalarType,
}

pub struct AreaGpuParams {
    pub color: Vec4,
    pub baseline: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AreaUniforms {
    color: [f32; 4],
    rows: u32,
    cols: u32,
    target_col: u32,
    baseline: f32,
    _pad: [f32; 3],
}

pub fn pack_vertices(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: &AreaGpuInputs<'_>,
    params: &AreaGpuParams,
) -> Result<GpuVertexBuffer, String> {
    if inputs.rows < 2 {
        return Err("area: GPU path requires at least two points".into());
    }
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_shader(device, workgroup_size, inputs.scalar);
    let x_buffer = axis_storage_buffer(device, "area-x", &inputs.x_axis, inputs.scalar)?;
    let segment_count = (inputs.rows - 1) as u64;
    let vertex_count = segment_count * 6;
    let output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("area-gpu-vertices"),
        size: vertex_count * std::mem::size_of::<Vertex>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let uniforms = AreaUniforms {
        color: params.color.to_array(),
        rows: inputs.rows,
        cols: inputs.cols,
        target_col: inputs.target_col,
        baseline: params.baseline,
        _pad: [0.0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("area-pack-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("area-pack-bind-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("area-pack-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("area-pack-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("area-pack-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buffer.as_entire_binding(),
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
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("area-pack-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("area-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.rows - 1).div_ceil(workgroup_size), 1, 1);
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
        ScalarType::F32 => shaders::area::F32,
        ScalarType::F64 => shaders::area::F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("area-pack-shader"),
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
                    label: Some("runmat-plot-area-test-device"),
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
    fn gpu_packer_generates_area_vertices() {
        let Some((device, queue)) = maybe_device() else {
            return;
        };
        let x = [1.0f32, 2.0f32, 3.0f32];
        let y = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("area-test-y"),
                contents: bytemuck::cast_slice(&[1.0f32, 2.0f32, 3.0f32, 0.5f32, 0.5f32, 0.5f32]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        let packed = pack_vertices(
            &device,
            &queue,
            &AreaGpuInputs {
                x_axis: AxisData::F32(&x),
                y_buffer: y,
                rows: 3,
                cols: 2,
                target_col: 1,
                scalar: ScalarType::F32,
            },
            &AreaGpuParams {
                color: Vec4::ONE,
                baseline: 0.0,
            },
        )
        .expect("area pack should succeed");
        assert_eq!(packed.vertex_count, 12);
    }
}
