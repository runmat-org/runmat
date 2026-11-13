//! Exact numerical histogram counting, separate from f32 vertex publication.
use crate::gpu::{shaders, tuning, GpuInputBuffer, ScalarType};
use runmat_accelerate_api::{
    ProviderExecutionError, ProviderExecutionResult, WgpuBufferOwner, WgpuBufferRef,
    WgpuDeviceAccess,
};
use wgpu::util::DeviceExt;

mod evidence;
use evidence::HistogramError;
pub use evidence::{ExactHistogramCounts, ExactHistogramEdges};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramUniforms {
    sample_count: u32,
    bin_count: u32,
    stride: u32,
    _pad: u32,
}

/// Count every sample once using the CPU owner's strict-upper-edge and clipped
/// outlier/NaN-last policy. No weighted arithmetic or native F64 is claimed.
pub async fn histogram_counts(
    access: &WgpuDeviceAccess,
    samples: WgpuBufferRef,
    edges: &ExactHistogramEdges,
) -> ProviderExecutionResult<ExactHistogramCounts> {
    let handle = samples.owning_handle();
    handle
        .allocation_lease()
        .validate_live()
        .map_err(ProviderExecutionError::from_source)?;
    let precision = runmat_accelerate_api::handle_precision(handle)
        .ok_or_else(|| ProviderExecutionError::from_source(HistogramError::SourceExtent))?;
    let elements = if handle.shape().contains(&0) {
        Some(0)
    } else {
        handle
            .shape()
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
    };
    let element_size = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => 4,
        runmat_accelerate_api::ProviderPrecision::F64 => 8,
    };
    if elements != Some(samples.len)
        || samples.shape != handle.shape()
        || samples.precision != precision
        || samples.element_size != element_size
    {
        return Err(ProviderExecutionError::from_source(
            HistogramError::SourceExtent,
        ));
    }
    let count = u32::try_from(samples.len)
        .map_err(|_| ProviderExecutionError::from_source(HistogramError::SampleRange))?;
    let count_bytes = edges
        .bin_count()
        .checked_mul(4)
        .ok_or_else(|| ProviderExecutionError::from_source(HistogramError::CountByteOverflow))?;
    let edge_bytes = edges.storage_bytes();
    let scalar =
        ScalarType::from_is_f64(samples.precision == runmat_accelerate_api::ProviderPrecision::F64);
    let input = GpuInputBuffer::from(samples);
    if count == 0 {
        // Authenticate the exact source allocation/device even though there is
        // no numerical work. A zero-length storage binding or dispatch is not
        // needed to establish the numerical evidence.
        access
            .with_shared(|emission| input.buffer(emission).map(|_| ()))
            .map_err(ProviderExecutionError::from_source)?
            .map_err(ProviderExecutionError::from_source)?;
        return ExactHistogramCounts::admit(vec![0; edges.bin_count()], 0);
    }
    let allocated = std::cell::Cell::new(false);
    let result = async {
        let counts = access
            .with_shared(|emission| -> ProviderExecutionResult<WgpuBufferOwner> {
                let device = emission.device();
                let source = input
                    .buffer(emission)
                    .map_err(ProviderExecutionError::from_source)?;
                let limits = device.limits();
                for bytes in [count_bytes, edge_bytes.len()] {
                    if bytes as u64 > limits.max_buffer_size
                        || bytes as u64 > u64::from(limits.max_storage_buffer_binding_size)
                    {
                        return Err(ProviderExecutionError::from_source(
                            HistogramError::DeviceBufferLimit,
                        ));
                    }
                }
                let element_size = match scalar {
                    ScalarType::F32 => 4,
                    ScalarType::F64 => 8,
                };
                let required = u64::from(count).checked_mul(element_size).ok_or_else(|| {
                    ProviderExecutionError::from_source(HistogramError::SourceByteOverflow)
                })?;
                if required > source.size()
                    || required > u64::from(limits.max_storage_buffer_binding_size)
                {
                    return Err(ProviderExecutionError::from_source(
                        HistogramError::SourceExtent,
                    ));
                }
                if scalar == ScalarType::F64 && count > u32::MAX / 2 {
                    return Err(ProviderExecutionError::from_source(
                        HistogramError::SourceWordRange,
                    ));
                }
                let workgroup = tuning::effective_workgroup_size()
                    .min(limits.max_compute_workgroup_size_x)
                    .min(limits.max_compute_invocations_per_workgroup)
                    .max(1);
                let groups = count
                    .div_ceil(workgroup)
                    .max(1)
                    .min(limits.max_compute_workgroups_per_dimension);
                let stride = groups.checked_mul(workgroup).ok_or_else(|| {
                    ProviderExecutionError::from_source(HistogramError::DispatchOverflow)
                })?;
                if stride == 0 {
                    return Err(ProviderExecutionError::from_source(
                        HistogramError::NoWorkgroups,
                    ));
                }
                // Allocation begins only after live source/geometry/limit admission.
                allocated.set(true);
                // WebGPU storage buffers are zero-initialized at creation. Do
                // not make a second host zero allocation proportional to bins.
                let counts = WgpuBufferOwner::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("histogram-exact-u32-counts"),
                    size: count_bytes as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
                let edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("histogram-binary64-edges"),
                    contents: &edge_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let params = HistogramUniforms {
                    sample_count: count,
                    bin_count: edges.bin_count_u32(),
                    stride,
                    _pad: 0,
                };
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("histogram-exact-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("histogram-exact-counts-shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        shaders::histogram::source(scalar, workgroup).into(),
                    ),
                });
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("histogram-exact-counts-pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let layout = pipeline.get_bind_group_layout(0);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("histogram-exact-counts-bindings"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: source,
                                offset: 0,
                                size: std::num::NonZeroU64::new(required),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: edge_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: counts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
                let commands = emission.encode("histogram-exact-counts-encoder", |encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("histogram-exact-counts-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                });
                emission.queue().submit(Some(commands));
                Ok(counts)
            })
            .map_err(ProviderExecutionError::from_source)??;
        // Keep original export and numerical buffer alive across real progress,
        // mapping, cancellation and cleanup. No permit survives the await.
        let values = crate::gpu::util::copy_readback_u32_buffer(
            access,
            &GpuInputBuffer::from(counts),
            edges.bin_count(),
        )
        .await
        .map_err(ProviderExecutionError::from_source)?;
        access
            .with_shared(|emission| input.buffer(emission).map(|_| ()))
            .map_err(ProviderExecutionError::from_source)?
            .map_err(ProviderExecutionError::from_source)?;
        ExactHistogramCounts::admit(values, count)
    }
    .await;
    result.map_err(|error: ProviderExecutionError| {
        if allocated.get() {
            error.after_effect()
        } else {
            error
        }
    })
}

/// Deliberate rendering-only projection of already published f64 numerical
/// values. This buffer is never reused as authoritative count evidence.
pub fn histogram_render_values(
    access: &WgpuDeviceAccess,
    values: &[f64],
) -> ProviderExecutionResult<WgpuBufferOwner> {
    let rendering = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    access
        .with_shared(|emission| {
            WgpuBufferOwner::new(emission.device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("histogram-render-values-f32"),
                    contents: bytemuck::cast_slice(&rendering),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ))
        })
        .map_err(ProviderExecutionError::from_source)
}
