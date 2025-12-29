use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu::shaders;
use crate::gpu::util::readback_f32;
use crate::gpu::{tuning, ScalarType};

#[derive(Clone)]
pub struct HistogramGpuInputs {
    pub samples: Arc<wgpu::Buffer>,
    pub sample_count: u32,
    pub scalar: ScalarType,
    pub weights: HistogramGpuWeights,
}

#[derive(Clone)]
pub enum HistogramGpuWeights {
    Uniform { total_weight: f32 },
    HostF32 { data: Vec<f32>, total_weight: f32 },
    HostF64 { data: Vec<f64>, total_weight: f32 },
    GpuF32 { buffer: Arc<wgpu::Buffer> },
    GpuF64 { buffer: Arc<wgpu::Buffer> },
}

pub enum HistogramNormalizationMode {
    Count,
    Probability,
    Pdf { bin_width: f32 },
}

pub struct HistogramGpuParams {
    pub min_value: f32,
    pub inv_bin_width: f32,
    pub bin_count: u32,
}

pub struct HistogramGpuOutput {
    pub values_buffer: Arc<wgpu::Buffer>,
    pub total_weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramUniforms {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvertUniforms {
    bin_count: u32,
    _pad: [u32; 3],
    scale: f32,
    _pad2: [u32; 3],
}

struct MaterializedWeights {
    mode: WeightMode,
    buffer: Option<Arc<wgpu::Buffer>>,
    total_hint: Option<f32>,
    accumulate_total: bool,
}

struct HistogramPassInputs<'a> {
    device: &'a Arc<wgpu::Device>,
    queue: &'a Arc<wgpu::Queue>,
    samples: &'a Arc<wgpu::Buffer>,
    sample_count: u32,
    sample_scalar: ScalarType,
    params: &'a HistogramGpuParams,
    counts_buffer: &'a Arc<wgpu::Buffer>,
    total_weight_buffer: &'a Arc<wgpu::Buffer>,
}

struct HistogramBindGroupInputs<'a> {
    device: &'a Arc<wgpu::Device>,
    samples: &'a Arc<wgpu::Buffer>,
    counts_buffer: &'a Arc<wgpu::Buffer>,
    total_weight_buffer: &'a Arc<wgpu::Buffer>,
    params: &'a HistogramGpuParams,
    sample_count: u32,
    accumulate_total: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WeightMode {
    Uniform,
    F32,
    F64,
}

pub fn histogram_values_buffer(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    inputs: HistogramGpuInputs,
    params: &HistogramGpuParams,
    normalization: HistogramNormalizationMode,
) -> Result<HistogramGpuOutput, String> {
    if params.bin_count == 0 {
        return Err("hist: bin count must be positive".to_string());
    }

    let bin_count_usize = params.bin_count as usize;
    let zero_counts = vec![0u32; bin_count_usize];
    let counts_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("histogram-counts"),
            contents: bytemuck::cast_slice(&zero_counts),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        }),
    );

    let total_weight_buffer = Arc::new(device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("histogram-total-weight"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
        },
    ));

    let HistogramGpuInputs {
        samples,
        sample_count,
        scalar,
        weights,
    } = inputs;
    let materialized = materialize_weights(device, weights)?;

    if sample_count > 0 {
        let pass_inputs = HistogramPassInputs {
            device,
            queue,
            samples: &samples,
            sample_count,
            sample_scalar: scalar,
            params,
            counts_buffer: &counts_buffer,
            total_weight_buffer: &total_weight_buffer,
        };
        run_histogram_pass(&pass_inputs, &materialized)?;
    }

    let total_weight = if let Some(hint) = materialized.total_hint {
        hint
    } else {
        readback_f32(device, &total_weight_buffer)
            .map_err(|e| format!("hist: failed to read GPU weights total: {e}"))?
    };

    let normalization_scale = match normalization {
        HistogramNormalizationMode::Count => 1.0,
        HistogramNormalizationMode::Probability => {
            if total_weight <= f32::EPSILON {
                0.0
            } else {
                1.0 / total_weight
            }
        }
        HistogramNormalizationMode::Pdf { bin_width } => {
            if total_weight <= f32::EPSILON || bin_width <= f32::EPSILON {
                0.0
            } else {
                1.0 / (total_weight * bin_width)
            }
        }
    };

    let values_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram-counts-f32"),
        size: (bin_count_usize * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));

    run_convert_pass(
        device,
        queue,
        params.bin_count,
        normalization_scale,
        &counts_buffer,
        &values_buffer,
    )?;

    Ok(HistogramGpuOutput {
        values_buffer,
        total_weight,
    })
}

fn materialize_weights(
    device: &Arc<wgpu::Device>,
    weights: HistogramGpuWeights,
) -> Result<MaterializedWeights, String> {
    match weights {
        HistogramGpuWeights::Uniform { total_weight } => Ok(MaterializedWeights {
            mode: WeightMode::Uniform,
            buffer: None,
            total_hint: Some(total_weight),
            accumulate_total: false,
        }),
        HistogramGpuWeights::HostF32 { data, total_weight } => {
            let buffer = Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("histogram-weights-host-f32"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
            Ok(MaterializedWeights {
                mode: WeightMode::F32,
                buffer: Some(buffer),
                total_hint: Some(total_weight),
                accumulate_total: false,
            })
        }
        HistogramGpuWeights::HostF64 { data, total_weight } => {
            let buffer = Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("histogram-weights-host-f64"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
            Ok(MaterializedWeights {
                mode: WeightMode::F64,
                buffer: Some(buffer),
                total_hint: Some(total_weight),
                accumulate_total: false,
            })
        }
        HistogramGpuWeights::GpuF32 { buffer } => Ok(MaterializedWeights {
            mode: WeightMode::F32,
            buffer: Some(buffer),
            total_hint: None,
            accumulate_total: true,
        }),
        HistogramGpuWeights::GpuF64 { buffer } => Ok(MaterializedWeights {
            mode: WeightMode::F64,
            buffer: Some(buffer),
            total_hint: None,
            accumulate_total: true,
        }),
    }
}

fn run_histogram_pass(
    inputs: &HistogramPassInputs<'_>,
    weights: &MaterializedWeights,
) -> Result<(), String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_counts_shader(
        inputs.device,
        workgroup_size,
        inputs.sample_scalar,
        weights.mode,
    );

    let bind_inputs = HistogramBindGroupInputs {
        device: inputs.device,
        samples: inputs.samples,
        counts_buffer: inputs.counts_buffer,
        total_weight_buffer: inputs.total_weight_buffer,
        params: inputs.params,
        sample_count: inputs.sample_count,
        accumulate_total: weights.accumulate_total,
    };

    let (bind_group_layout, bind_group) = match weights.mode {
        WeightMode::Uniform => build_uniform_bind_group(
            &bind_inputs,
        ),
        WeightMode::F32 | WeightMode::F64 => build_weighted_bind_group(
            &bind_inputs,
            weights.buffer.as_ref().expect("weights buffer missing"),
        ),
    }?;

    let pipeline_layout = inputs
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("histogram-counts-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = inputs
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("histogram-counts-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut encoder = inputs
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("histogram-counts-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram-counts-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = inputs.sample_count.div_ceil(workgroup_size);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    inputs.queue.submit(Some(encoder.finish()));

    Ok(())
}

fn build_uniform_bind_group(
    inputs: &HistogramBindGroupInputs<'_>,
) -> Result<(Arc<wgpu::BindGroupLayout>, Arc<wgpu::BindGroup>), String> {
    let layout = Arc::new(
        inputs
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hist-counts-layout-uniform"),
            entries: &[
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
        }),
    );

    let uniforms = HistogramUniforms {
        min_value: inputs.params.min_value,
        inv_bin_width: inputs.params.inv_bin_width,
        sample_count: inputs.sample_count,
        bin_count: inputs.params.bin_count,
        accumulate_total: if inputs.accumulate_total { 1 } else { 0 },
        _pad: [0; 3],
    };
    let uniform_buffer = inputs
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hist-counts-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = Arc::new(inputs.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hist-counts-bind-group-uniform"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.samples.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inputs.counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs.total_weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    }));

    Ok((layout, bind_group))
}

fn build_weighted_bind_group(
    inputs: &HistogramBindGroupInputs<'_>,
    weights_buffer: &Arc<wgpu::Buffer>,
) -> Result<(Arc<wgpu::BindGroupLayout>, Arc<wgpu::BindGroup>), String> {
    let layout = Arc::new(
        inputs
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hist-counts-layout-weighted"),
            entries: &[
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                storage_read_write_entry(3),
                uniform_entry(4),
            ],
        }),
    );

    let uniforms = HistogramUniforms {
        min_value: inputs.params.min_value,
        inv_bin_width: inputs.params.inv_bin_width,
        sample_count: inputs.sample_count,
        bin_count: inputs.params.bin_count,
        accumulate_total: if inputs.accumulate_total { 1 } else { 0 },
        _pad: [0; 3],
    };
    let uniform_buffer = inputs
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hist-counts-weighted-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = Arc::new(inputs.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hist-counts-bind-group-weighted"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.samples.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inputs.counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs.total_weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    }));

    Ok((layout, bind_group))
}

fn storage_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_read_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
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

fn run_convert_pass(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    bin_count: u32,
    normalization_scale: f32,
    counts_buffer: &Arc<wgpu::Buffer>,
    values_buffer: &Arc<wgpu::Buffer>,
) -> Result<(), String> {
    let workgroup_size = tuning::effective_workgroup_size();
    let shader = compile_convert_shader(device, workgroup_size);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("histogram-convert-bind-layout"),
        entries: &[
            storage_read_entry(0),
            storage_read_write_entry(1),
            uniform_entry(2),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("histogram-convert-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("histogram-convert-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let uniforms = ConvertUniforms {
        bin_count,
        _pad: [0; 3],
        scale: normalization_scale,
        _pad2: [0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("histogram-convert-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram-convert-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("histogram-convert-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram-convert-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = bin_count.div_ceil(workgroup_size);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(())
}

fn compile_counts_shader(
    device: &Arc<wgpu::Device>,
    workgroup_size: u32,
    scalar: ScalarType,
    weight_mode: WeightMode,
) -> wgpu::ShaderModule {
    let template = match (scalar, weight_mode) {
        (ScalarType::F32, WeightMode::Uniform) => shaders::histogram::counts::F32_UNIFORM,
        (ScalarType::F32, WeightMode::F32) => shaders::histogram::counts::F32_WEIGHTS_F32,
        (ScalarType::F32, WeightMode::F64) => shaders::histogram::counts::F32_WEIGHTS_F64,
        (ScalarType::F64, WeightMode::Uniform) => shaders::histogram::counts::F64_UNIFORM,
        (ScalarType::F64, WeightMode::F32) => shaders::histogram::counts::F64_WEIGHTS_F32,
        (ScalarType::F64, WeightMode::F64) => shaders::histogram::counts::F64_WEIGHTS_F64,
    };
    let source = template.replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("histogram-counts-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn compile_convert_shader(device: &Arc<wgpu::Device>, workgroup_size: u32) -> wgpu::ShaderModule {
    let source = shaders::histogram::convert::TEMPLATE
        .replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string());
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("histogram-convert-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
