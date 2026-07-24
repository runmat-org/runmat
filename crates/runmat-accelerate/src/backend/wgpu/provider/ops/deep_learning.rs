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
    weights_enabled: u32,
    mask_enabled: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftmaxRowsParams {
    rows: u32,
    cols: u32,
    row_offset: u32,
    _pad0: u32,
}

impl WgpuProvider {
    pub(crate) fn elu_exec(&self, input: &GpuTensorHandle, alpha: f64) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(input)?;
        ensure!(
            entry.storage != GpuTensorStorage::ComplexInterleaved
                && runmat_accelerate_api::handle_storage(input)
                    != GpuTensorStorage::ComplexInterleaved,
            "activation_elu: complex inputs are not supported"
        );
        let alpha_tensor = self.fill_exec(&entry.shape, alpha)?;
        let shader = match self.precision {
            NumericPrecision::F64 => ELU_SHADER_F64,
            NumericPrecision::F32 => ELU_SHADER_F32,
        };
        let result = self.fused_elementwise_with_telemetry_exec(
            shader,
            &[input.clone(), alpha_tensor.clone()],
            &entry.shape,
            entry.len,
        );
        let _ = self.free_exec(&alpha_tensor);
        result
    }

    /// Execute one stable softmax reduction per column-major matrix row. The
    /// output remains device-resident; only the error flag is read back so an
    /// invalid normalization can retain CPU-compatible error semantics.
    pub(crate) fn softmax_rows_exec(&self, input: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::Real,
            "activation_softmax_rows: complex inputs are not supported"
        );
        ensure!(
            entry.shape.len() == 2,
            "activation_softmax_rows: input must be a 2-D tensor"
        );
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        ensure!(
            rows <= u32::MAX as usize && cols <= u32::MAX as usize,
            "activation_softmax_rows: tensor shape exceeds GPU dispatch limits"
        );
        let output = self.create_storage_buffer_checked(entry.len, "runmat-softmax-rows-out")?;
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(output, entry.shape, 0));
        }
        let error_buffer =
            self.device_ref()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-softmax-rows-error"),
                    contents: bytes_of(&0u32),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
        let shader = softmax_rows_shader(self.precision);
        let layout_tag = "runmat-softmax-rows-layout";
        let bind_layout = self.cached_bind_group_layout(layout_tag, |device| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(layout_tag),
                entries: &softmax_rows_bind_layout_entries(),
            })
        });
        let pipeline_layout =
            self.device_ref()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("runmat-softmax-rows-pipeline-layout"),
                    bind_group_layouts: &[bind_layout.as_ref()],
                    push_constant_ranges: &[],
                });
        let shader_module = self
            .device_ref()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-softmax-rows-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader.clone())),
            });
        let workgroup_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let pipeline_key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            pipeline_key,
            &pipeline_layout,
            &shader_module,
            "runmat-softmax-rows-pipeline",
            Some(shader.as_bytes()),
            Some(layout_tag),
            Some(workgroup_size),
        );
        let chunk_rows = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
        for row_offset in (0..rows).step_by(chunk_rows) {
            let params = self.uniform_buffer(
                &SoftmaxRowsParams {
                    rows: rows as u32,
                    cols: cols as u32,
                    row_offset: row_offset as u32,
                    _pad0: 0,
                },
                "runmat-softmax-rows-params",
            );
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-softmax-rows-bind"),
                    layout: bind_layout.as_ref(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: error_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = (rows - row_offset).min(chunk_rows) as u32;
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                pipeline.as_ref(),
                &bind_group,
                workgroups,
                "runmat-softmax-rows-encoder",
                "runmat-softmax-rows-pass",
            );
        }
        let error_size = std::mem::size_of::<u32>() as u64;
        let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-softmax-rows-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-softmax-rows-error-copy"),
                });
        encoder.copy_buffer_to_buffer(&error_buffer, 0, &staging, 0, error_size);
        self.submit(encoder);
        let code = self
            .map_readback_bytes_sync(staging, error_size, "softmax_rows")
            .and_then(|bytes| {
                Ok(u32::from_le_bytes(
                    bytes
                        .get(..4)
                        .ok_or_else(|| anyhow!("activation_softmax_rows: short error readback"))?
                        .try_into()
                        .map_err(|_| anyhow!("activation_softmax_rows: invalid error readback"))?,
                ))
            })?;
        ensure!(
            code == 0,
            "activation_softmax_rows: softmax produced invalid normalization"
        );
        Ok(self.register_existing_buffer(output, entry.shape, entry.len))
    }

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
        let weights_entry = request
            .weights
            .map(|handle| self.get_entry(handle))
            .transpose()?;
        let mask_entry = request
            .mask
            .map(|handle| self.get_entry(handle))
            .transpose()?;
        ensure!(
            weights_entry.as_ref().is_none_or(|entry| {
                entry.shape == predictions_entry.shape && entry.len == predictions_entry.len
            }) && mask_entry.as_ref().is_none_or(|entry| {
                entry.shape == predictions_entry.shape && entry.len == predictions_entry.len
            }),
            "crossentropy_terms: weights and mask must match prediction shape"
        );
        ensure!(
            predictions_entry.storage != GpuTensorStorage::ComplexInterleaved
                && targets_entry.storage != GpuTensorStorage::ComplexInterleaved
                && weights_entry
                    .as_ref()
                    .is_none_or(|entry| entry.storage != GpuTensorStorage::ComplexInterleaved)
                && mask_entry
                    .as_ref()
                    .is_none_or(|entry| entry.storage != GpuTensorStorage::ComplexInterleaved),
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
        let weights_buffer = weights_entry
            .as_ref()
            .map(|entry| entry.buffer.as_ref())
            .unwrap_or_else(|| predictions_entry.buffer.as_ref());
        let mask_buffer = mask_entry
            .as_ref()
            .map(|entry| entry.buffer.as_ref())
            .unwrap_or_else(|| predictions_entry.buffer.as_ref());
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
                    weights_enabled: u32::from(weights_entry.is_some()),
                    mask_enabled: u32::from(mask_entry.is_some()),
                    _pad0: 0,
                    _pad1: 0,
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
                            resource: weights_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: mask_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_losses.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
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
            4 => {
                return Err(anyhow!(
                    "crossentropy_terms: weights must contain finite nonnegative values"
                ))
            }
            5 => {
                return Err(anyhow!(
                    "crossentropy_terms: mask must contain binary 0 or 1 values"
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

fn crossentropy_terms_bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 7] {
    std::array::from_fn(|binding| {
        let read_only = binding <= 3;
        wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 5 {
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

fn softmax_rows_bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
    std::array::from_fn(|binding| wgpu::BindGroupLayoutEntry {
        binding: binding as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: if binding == 2 {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        } else {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding == 0,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        },
        count: None,
    })
}

fn softmax_rows_shader(precision: NumericPrecision) -> String {
    let ty = precision.as_str();
    let max_finite = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let workgroup = crate::backend::wgpu::config::WORKGROUP_SIZE;
    format!(
        r#"
const MAX_FINITE_SOFTMAX: {ty} = {ty}({max_finite});

struct Tensor {{ data: array<{ty}> }};
struct Params {{ rows: u32, cols: u32, row_offset: u32, pad0: u32 }};
struct ErrorState {{ code: atomic<u32> }};

@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> errors: ErrorState;

var<workgroup> scratch: array<{ty}, {workgroup}>;
var<workgroup> denominator: {ty};

fn is_finite_softmax(value: {ty}) -> bool {{
  return (value == value) && (abs(value) < MAX_FINITE_SOFTMAX);
}}

@compute @workgroup_size({workgroup})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {{
  let row = workgroup_id.x + params.row_offset;
  let lane = local_id.x;
  var local_max = -MAX_FINITE_SOFTMAX;
  var column = lane;
  loop {{
    if (column >= params.cols) {{ break; }}
    local_max = max(local_max, input.data[row + column * params.rows]);
    column = column + {workgroup}u;
  }}
  scratch[lane] = local_max;
  workgroupBarrier();

  var stride = {workgroup}u / 2u;
  loop {{
    if (lane < stride) {{ scratch[lane] = max(scratch[lane], scratch[lane + stride]); }}
    workgroupBarrier();
    if (stride == 1u) {{ break; }}
    stride = stride / 2u;
  }}
  let row_max = scratch[0u];

  var local_sum = {ty}(0.0);
  column = lane;
  loop {{
    if (column >= params.cols) {{ break; }}
    let value = exp(input.data[row + column * params.rows] - row_max);
    output.data[row + column * params.rows] = value;
    local_sum = local_sum + value;
    column = column + {workgroup}u;
  }}
  scratch[lane] = local_sum;
  workgroupBarrier();
  stride = {workgroup}u / 2u;
  loop {{
    if (lane < stride) {{ scratch[lane] = scratch[lane] + scratch[lane + stride]; }}
    workgroupBarrier();
    if (stride == 1u) {{ break; }}
    stride = stride / 2u;
  }}
  if (lane == 0u) {{
    denominator = scratch[0u];
    if (!(is_finite_softmax(denominator) && denominator > {ty}(0.0))) {{
      atomicMax(&errors.code, 1u);
    }}
  }}
  workgroupBarrier();

  column = lane;
  loop {{
    if (column >= params.cols) {{ break; }}
    output.data[row + column * params.rows] = output.data[row + column * params.rows] / denominator;
    column = column + {workgroup}u;
  }}
}}
"#,
        ty = ty,
        max_finite = max_finite,
        workgroup = workgroup,
    )
}

const ELU_SHADER_F64: &str = r#"
struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> alpha: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let value = input0.data[idx];
  output.data[idx] = select(alpha.data[idx] * (exp(value) - f64(1.0)), value, value > f64(0.0));
}
"#;

const ELU_SHADER_F32: &str = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> alpha: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let value = input0.data[idx];
  output.data[idx] = select(alpha.data[idx] * (exp(value) - 1.0), value, value > 0.0);
}
"#;

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
  weights_enabled: u32,
  mask_enabled: u32,
  pad0: u32,
  pad1: u32,
}};

struct ErrorState {{
  code: atomic<u32>,
}};

@group(0) @binding(0) var<storage, read> predictions_in: Tensor;
@group(0) @binding(1) var<storage, read> targets_in: Tensor;
@group(0) @binding(2) var<storage, read> weights_in: Tensor;
@group(0) @binding(3) var<storage, read> mask_in: Tensor;
@group(0) @binding(4) var<storage, read_write> losses_out: Tensor;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> errors: ErrorState;

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
  if (params.weights_enabled != 0u) {{
    let weight = weights_in.data[idx];
    if (!(is_finite_crossentropy(weight) && weight >= {ty}(0.0))) {{
      flag_error(4u);
      return;
    }}
    loss = loss * weight;
  }}
  if (params.mask_enabled != 0u) {{
    let mask = mask_in.data[idx];
    if (!(is_finite_crossentropy(mask) && (mask == {ty}(0.0) || mask == {ty}(1.0)))) {{
      flag_error(5u);
      return;
    }}
    loss = loss * mask;
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
    fn elu_wgpu_matches_cpu_branch_semantics() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let shape = [2usize, 2usize];
        let input = provider
            .upload(&HostTensorView {
                data: &[-2.0, 0.0, 3.0, f64::NAN],
                shape: &shape,
            })
            .expect("upload input");
        let output =
            pollster::block_on(provider.activation_elu(&input, 1.25)).expect("resident ELU");
        let actual = pollster::block_on(provider.download(&output)).expect("download output");
        let expected = [1.25 * ((-2.0_f64).exp() - 1.0), 0.0, 3.0, f64::NAN];
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        assert_eq!(actual.shape, shape);
        for (actual, expected) in actual.data.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < tolerance || (actual.is_nan() && expected.is_nan()),
                "got {actual}, expected {expected}"
            );
        }
        provider.free(&input).ok();
        provider.free(&output).ok();
    }

    #[test]
    fn softmax_rows_wgpu_is_stable_and_resident() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let shape = [2usize, 3usize];
        let input = provider
            .upload(&HostTensorView {
                // Column-major rows: [1000, 0] and [1001, -1] and [999, 1].
                data: &[1000.0, 0.0, 1001.0, -1.0, 999.0, 1.0],
                shape: &shape,
            })
            .expect("upload input");
        let output =
            pollster::block_on(provider.activation_softmax_rows(&input)).expect("resident softmax");
        let actual = pollster::block_on(provider.download(&output)).expect("download output");
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        let e = std::f64::consts::E;
        let expected = [
            1.0 / (1.0 + e + 1.0 / e),
            1.0 / (1.0 + 1.0 / e + e),
            e / (1.0 + e + 1.0 / e),
            (1.0 / e) / (1.0 + 1.0 / e + e),
            (1.0 / e) / (1.0 + e + 1.0 / e),
            e / (1.0 + 1.0 / e + e),
        ];
        assert_eq!(actual.shape, shape);
        for (actual, expected) in actual.data.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < tolerance,
                "got {actual}, expected {expected}"
            );
        }
        provider.free(&input).ok();
        provider.free(&output).ok();
    }

    #[test]
    fn softmax_rows_wgpu_reports_invalid_normalization() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let shape = [1usize, 2usize];
        let input = provider
            .upload(&HostTensorView {
                data: &[f64::INFINITY, 0.0],
                shape: &shape,
            })
            .expect("upload input");
        let error = pollster::block_on(provider.activation_softmax_rows(&input))
            .expect_err("infinite input must not produce a normalization");
        assert!(error
            .to_string()
            .contains("softmax produced invalid normalization"));
        provider.free(&input).ok();
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
        let weights = provider
            .upload(&HostTensorView {
                data: &[2.0, 3.0],
                shape: &shape,
            })
            .expect("upload weights");
        let mask = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload mask");

        let result = provider
            .crossentropy_terms(&ProviderCrossentropyRequest {
                predictions: &predictions,
                targets: &targets,
                weights: Some(&weights),
                mask: Some(&mask),
                mode: ProviderCrossentropyMode::MultiLabel,
            })
            .expect("crossentropy terms");
        let losses = pollster::block_on(provider.download(&result.losses)).expect("download");
        assert_eq!(losses.shape, shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        let expected = [-2.0 * 0.8_f64.ln(), 0.0];
        for (idx, actual) in losses.data.iter().enumerate() {
            assert!(
                (*actual - expected[idx]).abs() < tol,
                "loss lane {idx}: got {actual}, expected {}",
                expected[idx]
            );
        }

        for handle in [&predictions, &targets, &weights, &mask, &result.losses] {
            provider.free(handle).ok();
        }
    }
}
