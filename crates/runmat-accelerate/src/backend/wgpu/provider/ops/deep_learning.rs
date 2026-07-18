use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamUpdateParamsF64 {
    total: u32,
    offset: u32,
    chunk: u32,
    _pad0: u32,
    iteration: f64,
    learn_rate: f64,
    gradient_decay_factor: f64,
    squared_gradient_decay_factor: f64,
    epsilon: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamUpdateParamsF32 {
    total: u32,
    offset: u32,
    chunk: u32,
    _pad0: u32,
    iteration: f32,
    learn_rate: f32,
    gradient_decay_factor: f32,
    squared_gradient_decay_factor: f32,
    epsilon: f32,
    _pad1: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CrossentropyParams {
    total: u32,
    offset: u32,
    chunk: u32,
    mode: u32,
}

impl WgpuProvider {
    pub(crate) fn crossentropy_terms_exec(
        &self,
        request: &ProviderCrossentropyRequest<'_>,
    ) -> Result<ProviderCrossentropyResult> {
        let predictions_entry = self.get_entry(request.predictions)?;
        let targets_entry = self.get_entry(request.targets)?;
        ensure!(
            predictions_entry.len > 0,
            "crossentropy_terms: predictions must not be empty"
        );
        ensure!(
            targets_entry.shape == predictions_entry.shape
                && targets_entry.len == predictions_entry.len,
            "crossentropy_terms: targets must match prediction shape"
        );
        ensure!(
            predictions_entry.storage != GpuTensorStorage::ComplexInterleaved
                && targets_entry.storage != GpuTensorStorage::ComplexInterleaved,
            "crossentropy_terms: complex inputs are not supported"
        );
        ensure!(
            predictions_entry.len <= u32::MAX as usize,
            "crossentropy_terms: tensor length exceeds GPU dispatch limits"
        );

        let out_losses =
            self.create_storage_buffer_checked(predictions_entry.len, "runmat-crossentropy-out")?;
        let error_buffer =
            self.device_ref()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-crossentropy-error"),
                    contents: bytes_of(&0u32),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
        let shader = crossentropy_terms_shader(self.precision);
        let shader_module = self
            .device_ref()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-crossentropy-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
            });
        let bind_layout =
            self.device_ref()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runmat-crossentropy-layout"),
                    entries: &crossentropy_terms_bind_layout_entries(),
                });
        let pipeline_layout =
            self.device_ref()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("runmat-crossentropy-pipeline-layout"),
                    bind_group_layouts: &[&bind_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = self.device_ref().create_compute_pipeline(
            &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                label: Some("runmat-crossentropy-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            },
        );

        let mode = match request.mode {
            ProviderCrossentropyMode::SingleLabel => 0,
            ProviderCrossentropyMode::MultiLabel => 1,
        };
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < predictions_entry.len {
            let chunk_len = (predictions_entry.len - offset).min(chunk_capacity);
            let params_buffer = self.uniform_buffer(
                &CrossentropyParams {
                    total: predictions_entry.len as u32,
                    offset: offset as u32,
                    chunk: chunk_len as u32,
                    mode,
                },
                "runmat-crossentropy-params",
            );
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-crossentropy-bind"),
                    layout: &bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: predictions_entry.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: targets_entry.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_losses.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: error_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &bind_group,
                workgroups,
                "runmat-crossentropy-encoder",
                "runmat-crossentropy-pass",
            );
            offset += chunk_len;
        }

        let error_size = std::mem::size_of::<u32>() as u64;
        let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-crossentropy-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-crossentropy-error-copy"),
                });
        encoder.copy_buffer_to_buffer(&error_buffer, 0, &staging, 0, error_size);
        self.submit(encoder);
        let code = self
            .map_readback_bytes_sync(staging, error_size, "crossentropy")
            .and_then(|bytes| {
                Ok(u32::from_le_bytes(
                    bytes
                        .get(..4)
                        .ok_or_else(|| anyhow!("crossentropy_terms: short error readback"))?
                        .try_into()
                        .map_err(|_| anyhow!("crossentropy_terms: invalid error readback"))?,
                ))
            })?;
        match code {
            0 => {}
            1 => {
                return Err(anyhow!(
                    "crossentropy_terms: inputs must contain finite values"
                ))
            }
            2 => {
                return Err(anyhow!(
                    "crossentropy_terms: targets must be probabilities in the range [0, 1]"
                ))
            }
            3 => {
                return Err(anyhow!(
                    "crossentropy_terms: loss produced a non-finite value"
                ))
            }
            other => {
                return Err(anyhow!(
                    "crossentropy_terms: validation failed with code {other}"
                ))
            }
        }

        Ok(ProviderCrossentropyResult {
            losses: self.register_existing_buffer(
                out_losses,
                predictions_entry.shape.clone(),
                predictions_entry.len,
            ),
        })
    }

    pub(crate) fn adam_update_exec(
        &self,
        request: &ProviderAdamUpdateRequest<'_>,
    ) -> Result<ProviderAdamUpdateResult> {
        ensure!(
            request.iteration > 0,
            "adam_update: iteration must be positive"
        );
        ensure!(
            request.learn_rate > 0.0 && request.learn_rate.is_finite(),
            "adam_update: learnRate must be positive and finite"
        );
        ensure!(
            (0.0..1.0).contains(&request.gradient_decay_factor)
                && request.gradient_decay_factor.is_finite(),
            "adam_update: gradient decay factor must be in [0, 1)"
        );
        ensure!(
            (0.0..1.0).contains(&request.squared_gradient_decay_factor)
                && request.squared_gradient_decay_factor.is_finite(),
            "adam_update: squared gradient decay factor must be in [0, 1)"
        );
        ensure!(
            request.epsilon > 0.0 && request.epsilon.is_finite(),
            "adam_update: epsilon must be positive and finite"
        );

        let parameters_entry = self.get_entry(request.parameters)?;
        let gradient_entry = self.get_entry(request.gradient)?;
        let average_grad_entry = request
            .average_grad
            .map(|handle| self.get_entry(handle))
            .transpose()?;
        let average_sq_grad_entry = request
            .average_sq_grad
            .map(|handle| self.get_entry(handle))
            .transpose()?;

        ensure!(
            parameters_entry.len > 0,
            "adam_update: parameters must not be empty"
        );
        ensure!(
            parameters_entry.storage != GpuTensorStorage::ComplexInterleaved
                && gradient_entry.storage != GpuTensorStorage::ComplexInterleaved
                && average_grad_entry
                    .as_ref()
                    .is_none_or(|entry| entry.storage != GpuTensorStorage::ComplexInterleaved)
                && average_sq_grad_entry
                    .as_ref()
                    .is_none_or(|entry| entry.storage != GpuTensorStorage::ComplexInterleaved),
            "adam_update: complex optimizer tensors are not supported"
        );
        ensure!(
            gradient_entry.shape == parameters_entry.shape
                && average_grad_entry
                    .as_ref()
                    .is_none_or(|entry| entry.shape == parameters_entry.shape)
                && average_sq_grad_entry
                    .as_ref()
                    .is_none_or(|entry| entry.shape == parameters_entry.shape),
            "adam_update: optimizer tensors must match parameter shape"
        );
        ensure!(
            parameters_entry.len <= u32::MAX as usize,
            "adam_update: tensor length exceeds GPU dispatch limits"
        );

        let mut temporary_inputs = Vec::new();
        let average_grad = match request.average_grad {
            Some(handle) => handle.clone(),
            None => {
                let zero = self.fill_exec(&parameters_entry.shape, 0.0)?;
                temporary_inputs.push(zero.clone());
                zero
            }
        };
        let average_sq_grad = match request.average_sq_grad {
            Some(handle) => handle.clone(),
            None => {
                let zero = self.fill_exec(&parameters_entry.shape, 0.0)?;
                temporary_inputs.push(zero.clone());
                zero
            }
        };
        let average_grad_entry = self.get_entry(&average_grad)?;
        let average_sq_grad_entry = self.get_entry(&average_sq_grad)?;

        let out_parameters = self
            .create_storage_buffer_checked(parameters_entry.len, "runmat-adamupdate-params-out")?;
        let out_average_grad =
            self.create_storage_buffer_checked(parameters_entry.len, "runmat-adamupdate-avg-out")?;
        let out_average_sq_grad = self
            .create_storage_buffer_checked(parameters_entry.len, "runmat-adamupdate-avg-sq-out")?;

        let mut validation_result = Ok(());
        if parameters_entry.len > 0 {
            let error_buffer =
                self.device_ref()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-adamupdate-error"),
                        contents: bytes_of(&0u32),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });
            let shader = adam_update_shader(self.precision);
            let shader_module =
                self.device_ref()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("runmat-adamupdate-shader"),
                        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
                    });
            let bind_layout =
                self.device_ref()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-adamupdate-layout"),
                        entries: &adam_update_bind_layout_entries(),
                    });
            let pipeline_layout =
                self.device_ref()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("runmat-adamupdate-pipeline-layout"),
                        bind_group_layouts: &[&bind_layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = self.device_ref().create_compute_pipeline(
                &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                    label: Some("runmat-adamupdate-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "main",
                },
            );

            let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
                * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
            let mut offset = 0usize;
            while offset < parameters_entry.len {
                let chunk_len = (parameters_entry.len - offset).min(chunk_capacity);
                let params_buffer = match self.precision {
                    NumericPrecision::F64 => self.uniform_buffer(
                        &AdamUpdateParamsF64 {
                            total: parameters_entry.len as u32,
                            offset: offset as u32,
                            chunk: chunk_len as u32,
                            _pad0: 0,
                            iteration: request.iteration as f64,
                            learn_rate: request.learn_rate,
                            gradient_decay_factor: request.gradient_decay_factor,
                            squared_gradient_decay_factor: request.squared_gradient_decay_factor,
                            epsilon: request.epsilon,
                        },
                        "runmat-adamupdate-params",
                    ),
                    NumericPrecision::F32 => self.uniform_buffer(
                        &AdamUpdateParamsF32 {
                            total: parameters_entry.len as u32,
                            offset: offset as u32,
                            chunk: chunk_len as u32,
                            _pad0: 0,
                            iteration: request.iteration as f32,
                            learn_rate: request.learn_rate as f32,
                            gradient_decay_factor: request.gradient_decay_factor as f32,
                            squared_gradient_decay_factor: request.squared_gradient_decay_factor
                                as f32,
                            epsilon: request.epsilon as f32,
                            _pad1: [0.0; 3],
                        },
                        "runmat-adamupdate-params",
                    ),
                };
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-adamupdate-bind"),
                        layout: &bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: parameters_entry.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: gradient_entry.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: average_grad_entry.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: average_sq_grad_entry.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: out_parameters.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: out_average_grad.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: out_average_sq_grad.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: error_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                crate::backend::wgpu::dispatch::creation::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-adamupdate-encoder",
                    "runmat-adamupdate-pass",
                );
                offset += chunk_len;
            }

            let error_size = std::mem::size_of::<u32>() as u64;
            let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-adamupdate-error-staging"),
                size: error_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-adamupdate-error-copy"),
                    });
            encoder.copy_buffer_to_buffer(&error_buffer, 0, &staging, 0, error_size);
            self.submit(encoder);
            validation_result = self
                .map_readback_bytes_sync(staging, error_size, "adamupdate")
                .and_then(|bytes| {
                    let code = u32::from_le_bytes(
                        bytes
                            .get(..4)
                            .ok_or_else(|| anyhow!("adam_update: short error readback"))?
                            .try_into()
                            .map_err(|_| anyhow!("adam_update: invalid error readback"))?,
                    );
                    Ok(code)
                })
                .and_then(|code| match code {
                    0 => Ok(()),
                    1 => Err(anyhow!("adam_update: inputs must contain finite values")),
                    2 => Err(anyhow!("adam_update: update produced a non-finite value")),
                    other => Err(anyhow!("adam_update: validation failed with code {other}")),
                });
        }

        for handle in temporary_inputs {
            let _ = self.free_exec(&handle);
        }
        validation_result?;

        Ok(ProviderAdamUpdateResult {
            parameters: self.register_existing_buffer(
                out_parameters,
                parameters_entry.shape.clone(),
                parameters_entry.len,
            ),
            average_grad: self.register_existing_buffer(
                out_average_grad,
                parameters_entry.shape.clone(),
                parameters_entry.len,
            ),
            average_sq_grad: self.register_existing_buffer(
                out_average_sq_grad,
                parameters_entry.shape.clone(),
                parameters_entry.len,
            ),
        })
    }
}

fn crossentropy_terms_bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    std::array::from_fn(|binding| {
        let read_only = binding <= 1;
        wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 3 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        }
    })
}

fn crossentropy_terms_shader(precision: NumericPrecision) -> String {
    let ty = precision.as_str();
    let (max_finite, eps, one_minus_eps) = match precision {
        NumericPrecision::F64 => ("1.7976931348623157e308", "1.0e-12", "0.999999999999"),
        NumericPrecision::F32 => ("3.4028234663852886e38", "1.0e-7", "0.9999999"),
    };
    let workgroup = crate::backend::wgpu::config::WORKGROUP_SIZE;
    format!(
        r#"
const MAX_FINITE_CROSSENTROPY: {ty} = {ty}({max_finite});
const EPS_CROSSENTROPY: {ty} = {ty}({eps});
const ONE_MINUS_EPS_CROSSENTROPY: {ty} = {ty}({one_minus_eps});

struct Tensor {{
  data: array<{ty}>,
}};

struct Params {{
  total: u32,
  offset: u32,
  chunk: u32,
  mode: u32,
}};

struct ErrorState {{
  code: atomic<u32>,
}};

@group(0) @binding(0) var<storage, read> predictions_in: Tensor;
@group(0) @binding(1) var<storage, read> targets_in: Tensor;
@group(0) @binding(2) var<storage, read_write> losses_out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> errors: ErrorState;

fn is_finite_crossentropy(x: {ty}) -> bool {{
  return (x == x) && (abs(x) < MAX_FINITE_CROSSENTROPY);
}}

fn flag_error(code: u32) {{
  atomicMax(&errors.code, code);
}}

@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= params.chunk) {{
    return;
  }}
  let idx = gid.x + params.offset;
  if (idx >= params.total) {{
    return;
  }}

  let prediction = predictions_in.data[idx];
  let target = targets_in.data[idx];
  if (!(is_finite_crossentropy(prediction) && is_finite_crossentropy(target))) {{
    flag_error(1u);
    return;
  }}
  if (target < {ty}(0.0) || target > {ty}(1.0)) {{
    flag_error(2u);
    return;
  }}

  let clipped = clamp(prediction, EPS_CROSSENTROPY, ONE_MINUS_EPS_CROSSENTROPY);
  var loss = -target * log(clipped);
  if (params.mode == 1u) {{
    loss = loss - ({ty}(1.0) - target) * log({ty}(1.0) - clipped);
  }}
  if (!is_finite_crossentropy(loss)) {{
    flag_error(3u);
    return;
  }}
  losses_out.data[idx] = loss;
}}
"#,
        ty = ty,
        max_finite = max_finite,
        eps = eps,
        one_minus_eps = one_minus_eps,
        workgroup = workgroup,
    )
}

fn adam_update_bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 9] {
    std::array::from_fn(|binding| {
        let read_only = binding <= 3;
        wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 7 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        }
    })
}

fn adam_update_shader(precision: NumericPrecision) -> String {
    let ty = precision.as_str();
    let max_finite = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let workgroup = crate::backend::wgpu::config::WORKGROUP_SIZE;
    format!(
        r#"
const MAX_FINITE_ADAM: {ty} = {ty}({max_finite});

struct Tensor {{
  data: array<{ty}>,
}};

struct Params {{
  total: u32,
  offset: u32,
  chunk: u32,
  pad0: u32,
  iteration: {ty},
  learn_rate: {ty},
  gradient_decay_factor: {ty},
  squared_gradient_decay_factor: {ty},
  epsilon: {ty},
}};

struct ErrorState {{
  code: atomic<u32>,
}};

@group(0) @binding(0) var<storage, read> parameters_in: Tensor;
@group(0) @binding(1) var<storage, read> gradient_in: Tensor;
@group(0) @binding(2) var<storage, read> average_grad_in: Tensor;
@group(0) @binding(3) var<storage, read> average_sq_grad_in: Tensor;
@group(0) @binding(4) var<storage, read_write> parameters_out: Tensor;
@group(0) @binding(5) var<storage, read_write> average_grad_out: Tensor;
@group(0) @binding(6) var<storage, read_write> average_sq_grad_out: Tensor;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read_write> errors: ErrorState;

fn is_finite_adam(x: {ty}) -> bool {{
  return (x == x) && (abs(x) < MAX_FINITE_ADAM);
}}

fn flag_error(code: u32) {{
  atomicMax(&errors.code, code);
}}

@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= params.chunk) {{
    return;
  }}
  let idx = gid.x + params.offset;
  if (idx >= params.total) {{
    return;
  }}

  let parameter = parameters_in.data[idx];
  let grad = gradient_in.data[idx];
  let previous_avg_grad = average_grad_in.data[idx];
  let previous_avg_sq_grad = average_sq_grad_in.data[idx];
  if (!(is_finite_adam(parameter) && is_finite_adam(grad) && is_finite_adam(previous_avg_grad) && is_finite_adam(previous_avg_sq_grad))) {{
    flag_error(1u);
    return;
  }}

  let avg_grad = params.gradient_decay_factor * previous_avg_grad + ({ty}(1.0) - params.gradient_decay_factor) * grad;
  let avg_sq_grad = params.squared_gradient_decay_factor * previous_avg_sq_grad + ({ty}(1.0) - params.squared_gradient_decay_factor) * grad * grad;
  let grad_correction = {ty}(1.0) - pow(params.gradient_decay_factor, params.iteration);
  let sq_grad_correction = {ty}(1.0) - pow(params.squared_gradient_decay_factor, params.iteration);
  let corrected_grad = avg_grad / grad_correction;
  let corrected_sq_grad = avg_sq_grad / sq_grad_correction;
  let step = params.learn_rate * corrected_grad / (sqrt(corrected_sq_grad) + params.epsilon);
  let updated = parameter - step;

  if (!(is_finite_adam(updated) && is_finite_adam(avg_grad) && is_finite_adam(avg_sq_grad))) {{
    flag_error(2u);
    return;
  }}

  parameters_out.data[idx] = updated;
  average_grad_out.data[idx] = avg_grad;
  average_sq_grad_out.data[idx] = avg_sq_grad;
}}
"#,
        ty = ty,
        max_finite = max_finite,
        workgroup = workgroup,
    )
}

#[cfg(test)]
mod tests {
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{
        AccelProvider, HostTensorView, ProviderAdamUpdateRequest, ProviderCrossentropyMode,
        ProviderCrossentropyRequest,
    };

    #[test]
    fn adam_update_wgpu_returns_resident_outputs() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };

        let shape = [1usize, 3usize];
        let parameters = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &shape,
            })
            .expect("upload parameters");
        let gradient = provider
            .upload(&HostTensorView {
                data: &[0.1, -0.2, 0.3],
                shape: &shape,
            })
            .expect("upload gradient");

        let result = provider
            .adam_update(&ProviderAdamUpdateRequest {
                parameters: &parameters,
                gradient: &gradient,
                average_grad: None,
                average_sq_grad: None,
                iteration: 1,
                learn_rate: 0.01,
                gradient_decay_factor: 0.9,
                squared_gradient_decay_factor: 0.999,
                epsilon: 1.0e-8,
            })
            .expect("adam update");

        let updated =
            pollster::block_on(provider.download(&result.parameters)).expect("download params");
        let avg =
            pollster::block_on(provider.download(&result.average_grad)).expect("download avg");
        let avg_sq = pollster::block_on(provider.download(&result.average_sq_grad))
            .expect("download avg sq");

        assert_eq!(updated.shape, shape);
        assert_eq!(avg.shape, shape);
        assert_eq!(avg_sq.shape, shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        let expected_updated = [0.990000001, 2.0099999995, 2.9900000003333334];
        let expected_avg = [0.01, -0.02, 0.03];
        let expected_avg_sq = [0.00001, 0.00004, 0.00009];
        for idx in 0..shape[1] {
            assert!(
                (updated.data[idx] - expected_updated[idx]).abs() < tol,
                "updated lane {idx}: got {}, expected {}",
                updated.data[idx],
                expected_updated[idx]
            );
            assert!(
                (avg.data[idx] - expected_avg[idx]).abs() < tol,
                "avg lane {idx}: got {}, expected {}",
                avg.data[idx],
                expected_avg[idx]
            );
            assert!(
                (avg_sq.data[idx] - expected_avg_sq[idx]).abs() < tol,
                "avg sq lane {idx}: got {}, expected {}",
                avg_sq.data[idx],
                expected_avg_sq[idx]
            );
        }

        for handle in [
            &parameters,
            &gradient,
            &result.parameters,
            &result.average_grad,
            &result.average_sq_grad,
        ] {
            provider.free(handle).ok();
        }
    }

    #[test]
    fn crossentropy_wgpu_returns_resident_loss_terms() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };

        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        let result = provider
            .crossentropy_terms(&ProviderCrossentropyRequest {
                predictions: &predictions,
                targets: &targets,
                mode: ProviderCrossentropyMode::MultiLabel,
            })
            .expect("crossentropy terms");
        let losses = pollster::block_on(provider.download(&result.losses)).expect("download");
        assert_eq!(losses.shape, shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        for (idx, actual) in losses.data.iter().enumerate() {
            let expected = -0.8_f64.ln();
            assert!(
                (*actual - expected).abs() < tol,
                "loss lane {idx}: got {actual}, expected {expected}"
            );
        }

        for handle in [&predictions, &targets, &result.losses] {
            provider.free(handle).ok();
        }
    }
}
