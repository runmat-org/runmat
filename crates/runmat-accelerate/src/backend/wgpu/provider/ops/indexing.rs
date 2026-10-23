use super::*;

#[path = "indexing/functional_scatter.rs"]
mod functional_scatter;
#[path = "indexing/validated_plan.rs"]
mod validated_plan;

fn indexing_unproven(
    error: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> runmat_accelerate_api::ProviderExecutionError {
    error.into().with_unproven_effects()
}

fn decode_indexing_error_words(bytes: &[u8]) -> Result<[u32; 4]> {
    ensure!(
        bytes.len() == 16,
        "indexing: mapped error payload size mismatch"
    );
    Ok([
        u32::from_ne_bytes(bytes[0..4].try_into()?),
        u32::from_ne_bytes(bytes[4..8].try_into()?),
        u32::from_ne_bytes(bytes[8..12].try_into()?),
        u32::from_ne_bytes(bytes[12..16].try_into()?),
    ])
}

#[cfg(test)]
#[path = "indexing/layout_tests.rs"]
mod layout_tests;

#[cfg(test)]
#[path = "indexing/validated_actual_tests.rs"]
mod validated_actual_tests;

impl WgpuProvider {
    pub(crate) async fn scatter_column_exec(
        &self,
        matrix: &GpuTensorHandle,
        col_index: usize,
        values: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let planned = validated_plan::ScatterPlan::new(
            self,
            matrix,
            values,
            col_index,
            validated_plan::ScatterAxis::Column,
        )?;
        let validated = if let Some(frame) = &planned {
            let validated = self
                .prepare_validated_indexing_pipeline(&frame.recipe, frame.kind)
                .await?;
            validated
                .require_indexing(&frame.recipe, frame.kind)
                .map_err(indexing_unproven)?;
            frame
                .require_live_inputs(self, matrix, values)
                .map_err(indexing_unproven)?;
            Some(validated)
        } else {
            None
        };
        let matrix_contiguous = self
            .reshape_exec(matrix, matrix.shape())
            .map_err(indexing_unproven)?;
        let matrix = &matrix_contiguous;
        let values_contiguous = self
            .reshape_exec(values, values.shape())
            .map_err(indexing_unproven)?;
        let values = &values_contiguous;

        let raw_matrix = self.get_entry_raw(matrix).map_err(indexing_unproven)?;
        self.get_entry_raw(values).map_err(indexing_unproven)?;
        if matrix.view().is_logical()
            && !self
                .allocations
                .requires_logical_encoding(values)
                .map_err(indexing_unproven)?
        {
            return Err(indexing_unproven(
                runmat_accelerate_api::unsupported_provider_operation(
                    "logical scatter row/column requires canonical logical values",
                ),
            ));
        }
        if raw_matrix.integer_type().is_some() {
            return (|| -> Result<_> {
                ensure!(
                    raw_matrix.shape.len() == 2,
                    "scatter_column: only 2D tensors supported"
                );
                let rows = raw_matrix.shape[0];
                let cols = raw_matrix.shape[1];
                ensure!(
                    col_index < cols,
                    "scatter_column: column index out of bounds"
                );
                let raw_values = self.get_entry_raw(values)?;
                ensure!(
                    raw_values.integer_type() == raw_matrix.integer_type(),
                    "scatter_column: integer storage mismatch"
                );
                ensure!(
                    raw_values.shape.len() <= 2 && raw_values.shape[0] == rows,
                    "scatter_column: length mismatch"
                );
                let indices: Vec<u32> = (0..rows)
                    .map(|row| {
                        u32::try_from(col_index * rows + row)
                            .map_err(|_| anyhow!("scatter_column: index exceeds u32"))
                    })
                    .collect::<Result<_>>()?;
                let all_indices: Vec<u32> = (0..rows * cols)
                    .map(|index| {
                        u32::try_from(index)
                            .map_err(|_| anyhow!("scatter_column: index exceeds u32"))
                    })
                    .collect::<Result<_>>()?;
                let out = self.gather_linear_exec(matrix, &all_indices, &raw_matrix.shape)?;
                self.scatter_linear_exec(&out, &indices, values)?;
                Ok(out)
            })()
            .map_err(indexing_unproven);
        }
        let frame = planned.expect("noninteger scatter has a frozen frame");
        let validated = validated.expect("noninteger scatter has a validated pipeline");
        let rows = frame.rows;
        let cols = frame.cols;
        let m_entry = self.get_entry(matrix).map_err(indexing_unproven)?;
        let v_entry = self.get_entry(values).map_err(indexing_unproven)?;
        let out_buffer = self
            .with_emission(|emission, pending| -> Result<_> {
                let out_buffer =
                    self.create_storage_buffer(emission, rows * cols, "runmat-scatter-col-out")?;
                {
                    let mut enc = pending.profiled_encoder(
                        emission,
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-scatter-col-copy"),
                        },
                    );
                    enc.copy_buffer_to_buffer(
                        m_entry.buffer.as_ref(),
                        0,
                        out_buffer.as_ref(),
                        0,
                        (rows * cols * self.element_size) as u64,
                    );
                    self.submit(emission, pending, enc)?;
                }
                let bgl = validated.layout.as_ref();
                let pipeline = validated.pipeline.as_ref();
                #[repr(C)]
                #[derive(Clone, Copy, Pod, Zeroable)]
                struct Pm {
                    rows: u32,
                    cols: u32,
                    j: u32,
                }
                let params = Pm {
                    rows: rows as u32,
                    cols: cols as u32,
                    j: col_index as u32,
                };
                let pbuf = self.uniform_buffer(emission, &params, "runmat-scatter-col-params")?;
                let bg = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-scatter-col-bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: v_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: m_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: pbuf.as_entire_binding(),
                            },
                        ],
                    });
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    rows as u32,
                    frame.recipe.workgroup,
                );
                crate::backend::wgpu::dispatch::scatter::run(
                    emission, pending, pipeline, &bg, groups,
                )?;
                Ok(out_buffer)
            })
            .map_err(indexing_unproven)?
            .map_err(indexing_unproven)?;
        if matrix.view().is_logical() {
            self.register_logical_buffer(
                out_buffer,
                vec![rows, cols],
                rows * cols,
                BufferUsageClass::Generic,
            )
            .map_err(indexing_unproven)
        } else {
            self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols)
                .map_err(indexing_unproven)
        }
    }
    pub(crate) async fn sub2ind_exec(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.precision != NumericPrecision::F64 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "sub2ind: Double output requires a binary64 provider",
            )
            .into());
        }
        if dims.is_empty()
            || inputs.len() != dims.len()
            || inputs.len() != strides.len()
            || inputs.len() != scalar_mask.len()
        {
            return Err(anyhow!(
                "sub2ind: expected {} subscripts for {} dimensions",
                dims.len(),
                dims.len()
            )
            .into());
        }
        let expected_len = product_checked(output_shape).ok_or_else(|| {
            runmat_accelerate_api::unsupported_provider_operation(
                "sub2ind: output shape exceeds GPU limits",
            )
        })?;
        if expected_len != len {
            return Err(anyhow!("sub2ind: output shape does not match subscript sizes").into());
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || len > u32::MAX as usize
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "sub2ind: dimensions exceed GPU kernel limits",
            )
            .into());
        }
        let largest_index = dims
            .iter()
            .zip(strides)
            .try_fold(1u64, |sum, (&dim, &stride)| {
                u64::try_from(dim.saturating_sub(1))
                    .ok()
                    .and_then(|base| base.checked_mul(stride as u64))
                    .and_then(|term| sum.checked_add(term))
            });
        if largest_index.is_none_or(|index| index > u64::from(u32::MAX)) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "sub2ind: linear index exceeds GPU kernel limits",
            )
            .into());
        }

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&s| s as u32).collect();
        let mask_u32: Vec<u32> = scalar_mask
            .iter()
            .map(|&m| if m { 1u32 } else { 0u32 })
            .collect();

        let mut input_buffers = Vec::with_capacity(inputs.len());
        for (handle, &scalar) in inputs.iter().zip(scalar_mask) {
            let entry = self.get_entry(handle)?;
            if entry.storage != GpuTensorStorage::Real
                || !self.supports_native_real_element_type(entry.element_type)
                || self.allocations.requires_logical_encoding(handle)?
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "sub2ind: inputs require native real numeric storage",
                )
                .into());
            }
            if entry.len != if scalar { 1 } else { len } {
                return Err(anyhow!("sub2ind: subscript cardinality does not match output").into());
            }
            input_buffers.push(entry.buffer.clone());
        }

        if len == 0 {
            let buffer = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-sub2ind-empty")
                })
                .map_err(indexing_unproven)?
                .map_err(indexing_unproven)?;
            return self
                .register_existing_buffer(buffer, output_shape.to_vec(), 0)
                .map_err(indexing_unproven);
        }
        self.require_ieee_words(self.precision)?;

        let workgroup_size = self.workgroup_config.scalar;
        let limits = &self.adapter_limits;
        let storage_bindings =
            super::backend_shared::checked_binding_count("sub2ind", inputs.len(), 2).map_err(
                |error| runmat_accelerate_api::unsupported_provider_operation(error.to_string()),
            )?;
        let total_bindings =
            super::backend_shared::checked_binding_count("sub2ind", inputs.len(), 3).map_err(
                |error| runmat_accelerate_api::unsupported_provider_operation(error.to_string()),
            )?;
        super::backend_shared::validate_compute_binding_counts(
            "sub2ind",
            storage_bindings,
            total_bindings,
            limits,
        )
        .map_err(|error| {
            runmat_accelerate_api::unsupported_provider_operation(error.to_string())
        })?;
        let max_binding =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        let output_bytes = (len as u64)
            .checked_mul(self.element_size as u64)
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "sub2ind: output byte extent exceeds GPU limits",
                )
            })?;
        if workgroup_size == 0
            || workgroup_size > limits.max_compute_workgroup_size_x
            || workgroup_size > limits.max_compute_invocations_per_workgroup
            || limits.max_uniform_buffers_per_shader_stage == 0
            || limits.max_uniform_buffer_binding_size < 16
            || output_bytes > max_binding
            || input_buffers
                .iter()
                .any(|buffer| buffer.as_ref().size() > max_binding)
            || crate::backend::wgpu::dispatch::common::dispatch_size(len as u32, workgroup_size)
                > limits.max_compute_workgroups_per_dimension
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "sub2ind: enabled device limits do not admit the indexing recipe",
            )
            .into());
        }
        let recipe = super::pipeline_recipe::PipelineRecipe {
            shader: crate::backend::wgpu::shaders::sub2ind::build_sub2ind_shader(
                self.precision,
                &dims_u32,
                &strides_u32,
                &mask_u32,
                workgroup_size,
            ),
            layout: format!("runmat-sub2ind-layout-{}", inputs.len()),
            workgroup: workgroup_size,
        };
        let indexing = super::validated_fusion::IndexingPipelineKind::Sub2Ind {
            inputs: inputs.len(),
        };
        let validated = self
            .prepare_validated_indexing_pipeline(&recipe, indexing)
            .await?;
        validated
            .require_indexing(&recipe, indexing)
            .map_err(indexing_unproven)?;
        for (handle, buffer) in inputs.iter().zip(&input_buffers) {
            let live = self.get_entry(handle).map_err(indexing_unproven)?;
            if !runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, buffer)
                || self
                    .allocations
                    .requires_logical_encoding(handle)
                    .map_err(indexing_unproven)?
            {
                return Err(indexing_unproven(anyhow!(
                    "sub2ind: borrowed subscript changed during preparation"
                )));
            }
        }

        let (output_buffer, staging) = self
            .with_emission(|emission, pending| -> Result<_> {
                let output_buffer =
                    self.create_storage_buffer_checked(emission, len, "runmat-sub2ind-out")?;
                let error_bytes = vec![0u8; std::mem::size_of::<u32>() * 4];
                let error_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-sub2ind-error"),
                        contents: &error_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
                );

                let bgl = validated.layout.as_ref();
                let pipeline = validated.pipeline.as_ref();

                #[repr(C)]
                #[derive(Clone, Copy, Pod, Zeroable)]
                struct Params {
                    len: u32,
                    _pad0: u32,
                    _pad1: u32,
                    _pad2: u32,
                }
                let params = Params {
                    len: len as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-sub2ind-params")?;

                let mut bind_entries = Vec::with_capacity(inputs.len() + 3);
                for (idx, buffer) in input_buffers.iter().enumerate() {
                    bind_entries.push(wgpu::BindGroupEntry {
                        binding: idx as u32,
                        resource: buffer.as_ref().as_entire_binding(),
                    });
                }
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: inputs.len() as u32,
                    resource: output_buffer.as_ref().as_entire_binding(),
                });
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: (inputs.len() + 1) as u32,
                    resource: error_buffer.as_ref().as_entire_binding(),
                });
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: (inputs.len() + 2) as u32,
                    resource: params_buffer.as_entire_binding(),
                });
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-sub2ind-bg"),
                        layout: bgl,
                        entries: &bind_entries,
                    });

                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    len as u32,
                    workgroup_size,
                );
                crate::backend::wgpu::dispatch::sub2ind::run(
                    emission,
                    pending,
                    pipeline,
                    &bind_group,
                    groups,
                )?;

                let error_size = (std::mem::size_of::<u32>() * 4) as u64;
                let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-sub2ind-error-staging"),
                        size: error_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-sub2ind-error-copy"),
                    },
                );
                encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
                self.submit(emission, pending, encoder)?;
                Ok((output_buffer, staging))
            })
            .map_err(indexing_unproven)?
            .map_err(indexing_unproven)?;
        let words = self
            .map_readback_with(staging, "sub2ind", decode_indexing_error_words)
            .await
            .map_err(indexing_unproven)?;
        let code = words.first().copied().unwrap_or(0);
        let dim_word = words.get(1).copied().unwrap_or(0);
        let extra = words.get(2).copied().unwrap_or(0);

        if code != 0 {
            let dim_index = dim_word.max(1) as usize;
            let dim_size = dims.get(dim_index.saturating_sub(1)).copied().unwrap_or(0);
            let err = match code {
                1 => anyhow!(
                    "sub2ind: subscript in dimension {} must be finite",
                    dim_index
                ),
                2 => anyhow!(
                    "sub2ind: subscript in dimension {} must be an integer",
                    dim_index
                ),
                3 => anyhow!(
                    "sub2ind: subscript {} exceeds dimension {} (size {})",
                    extra as isize,
                    dim_index,
                    dim_size
                ),
                _ => anyhow!("sub2ind: kernel reported error code {}", code),
            };
            return Err(indexing_unproven(err));
        }

        self.register_existing_buffer_with_usage(
            output_buffer,
            output_shape.to_vec(),
            len,
            BufferUsageClass::FusionOut,
        )
        .map_err(indexing_unproven)
    }

    pub(crate) async fn ind2sub_exec(
        &self,
        dims: &[usize],
        strides: &[usize],
        indices: &GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Vec<GpuTensorHandle>> {
        if self.precision != NumericPrecision::F64 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ind2sub: Double output requires a binary64 provider",
            )
            .into());
        }
        if dims.len() != strides.len() {
            return Err(anyhow!("ind2sub: size vector mismatch").into());
        }
        let expected_len = product_checked(output_shape).ok_or_else(|| {
            runmat_accelerate_api::unsupported_provider_operation(
                "ind2sub: output shape exceeds GPU limits",
            )
        })?;
        if expected_len != len {
            return Err(anyhow!("ind2sub: output shape does not match index tensor").into());
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || total > u32::MAX as usize
            || len > u32::MAX as usize
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ind2sub: dimensions exceed GPU kernel limits",
            )
            .into());
        }
        let storage_bindings =
            super::backend_shared::checked_binding_count("ind2sub", dims.len(), 2).map_err(
                |error| runmat_accelerate_api::unsupported_provider_operation(error.to_string()),
            )?;
        let total_bindings = super::backend_shared::checked_binding_count("ind2sub", dims.len(), 3)
            .map_err(|error| {
                runmat_accelerate_api::unsupported_provider_operation(error.to_string())
            })?;
        super::backend_shared::validate_compute_binding_counts(
            "ind2sub",
            storage_bindings,
            total_bindings,
            &self.adapter_limits,
        )
        .map_err(|error| {
            runmat_accelerate_api::unsupported_provider_operation(error.to_string())
        })?;

        let entry = self.get_entry(indices)?;
        if entry.storage != GpuTensorStorage::Real
            || !self.supports_native_real_element_type(entry.element_type)
            || self.allocations.requires_logical_encoding(indices)?
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ind2sub: indices require native real numeric storage",
            )
            .into());
        }
        if entry.len != len {
            return Err(
                anyhow!("ind2sub: index tensor length does not match provided shape").into(),
            );
        }
        if len == 0 {
            let mut handles = Vec::with_capacity(dims.len());
            for _ in 0..dims.len() {
                let buffer = self
                    .with_emission(|emission, _| {
                        self.create_storage_buffer(emission, 0, "runmat-ind2sub-empty")
                    })
                    .map_err(indexing_unproven)?
                    .map_err(indexing_unproven)?;
                handles.push(
                    self.register_existing_buffer(buffer, output_shape.to_vec(), 0)
                        .map_err(indexing_unproven)?,
                );
            }
            return Ok(handles);
        }
        self.require_ieee_words(self.precision)?;

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&s| s as u32).collect();

        let workgroup_size = self.workgroup_config.scalar;
        let limits = &self.adapter_limits;
        let max_binding =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        let output_bytes = (len as u64)
            .checked_mul(self.element_size as u64)
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "ind2sub: output byte extent exceeds GPU limits",
                )
            })?;
        if workgroup_size == 0
            || workgroup_size > limits.max_compute_workgroup_size_x
            || workgroup_size > limits.max_compute_invocations_per_workgroup
            || limits.max_uniform_buffers_per_shader_stage == 0
            || limits.max_uniform_buffer_binding_size < 16
            || entry.allocated_bytes > max_binding
            || output_bytes > max_binding
            || crate::backend::wgpu::dispatch::common::dispatch_size(len as u32, workgroup_size)
                > limits.max_compute_workgroups_per_dimension
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ind2sub: enabled device limits do not admit the indexing recipe",
            )
            .into());
        }
        let recipe = super::pipeline_recipe::PipelineRecipe {
            shader: crate::backend::wgpu::shaders::ind2sub::build_ind2sub_shader(
                self.precision,
                &dims_u32,
                &strides_u32,
                total as u32,
                workgroup_size,
            ),
            layout: format!("runmat-ind2sub-layout-{}", dims.len()),
            workgroup: workgroup_size,
        };
        let indexing = super::validated_fusion::IndexingPipelineKind::Ind2Sub {
            outputs: dims.len(),
        };
        let validated = self
            .prepare_validated_indexing_pipeline(&recipe, indexing)
            .await?;
        validated
            .require_indexing(&recipe, indexing)
            .map_err(indexing_unproven)?;
        let live = self.get_entry(indices).map_err(indexing_unproven)?;
        if !runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &entry.buffer)
            || self
                .allocations
                .requires_logical_encoding(indices)
                .map_err(indexing_unproven)?
        {
            return Err(indexing_unproven(anyhow!(
                "ind2sub: borrowed indices changed during preparation"
            )));
        }

        let (output_buffers, staging) = self
            .with_emission(|emission, pending| -> Result<_> {
                let mut output_buffers = Vec::with_capacity(dims.len());
                for _ in 0..dims.len() {
                    output_buffers.push(self.create_storage_buffer_checked(
                        emission,
                        len,
                        "runmat-ind2sub-out",
                    )?);
                }

                let error_bytes = vec![0u8; std::mem::size_of::<u32>() * 4];
                let error_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-ind2sub-error"),
                        contents: &error_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
                );

                let bgl = validated.layout.as_ref();
                let pipeline = validated.pipeline.as_ref();

                #[repr(C)]
                #[derive(Clone, Copy, Pod, Zeroable)]
                struct Params {
                    len: u32,
                    _pad0: u32,
                    _pad1: u32,
                    _pad2: u32,
                }
                let params = Params {
                    len: len as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-ind2sub-params")?;

                let mut bind_entries = Vec::with_capacity(dims.len() + 3);
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry.buffer.as_ref().as_entire_binding(),
                });
                for (idx, buffer) in output_buffers.iter().enumerate() {
                    bind_entries.push(wgpu::BindGroupEntry {
                        binding: (idx + 1) as u32,
                        resource: buffer.as_ref().as_entire_binding(),
                    });
                }
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: (dims.len() + 1) as u32,
                    resource: error_buffer.as_ref().as_entire_binding(),
                });
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: (dims.len() + 2) as u32,
                    resource: params_buffer.as_entire_binding(),
                });
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-ind2sub-bg"),
                        layout: bgl,
                        entries: &bind_entries,
                    });

                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    len as u32,
                    workgroup_size,
                );
                crate::backend::wgpu::dispatch::ind2sub::run(
                    emission,
                    pending,
                    pipeline,
                    &bind_group,
                    groups,
                )?;

                let error_size = (std::mem::size_of::<u32>() * 4) as u64;
                let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-ind2sub-error-staging"),
                        size: error_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-ind2sub-error-copy"),
                    },
                );
                encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
                self.submit(emission, pending, encoder)?;
                Ok((output_buffers, staging))
            })
            .map_err(indexing_unproven)?
            .map_err(indexing_unproven)?;
        let words = self
            .map_readback_with(staging, "ind2sub", decode_indexing_error_words)
            .await
            .map_err(indexing_unproven)?;
        let code = words.first().copied().unwrap_or(0);

        if code != 0 {
            let err = match code {
                1..=3 => anyhow!("Linear indices must be positive integers."),
                4 => anyhow!("Linear index exceeds GPU kernel limits."),
                _ => anyhow!("ind2sub: kernel reported error code {}", code),
            };
            return Err(indexing_unproven(err));
        }

        let mut handles = Vec::with_capacity(output_buffers.len());
        for buffer in output_buffers {
            handles.push(
                self.register_existing_buffer(buffer, output_shape.to_vec(), len)
                    .map_err(indexing_unproven)?,
            );
        }
        Ok(handles)
    }

    pub(crate) async fn scatter_row_exec(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let planned = validated_plan::ScatterPlan::new(
            self,
            matrix,
            values,
            row_index,
            validated_plan::ScatterAxis::Row,
        )?;
        let validated = if let Some(frame) = &planned {
            let validated = self
                .prepare_validated_indexing_pipeline(&frame.recipe, frame.kind)
                .await?;
            validated
                .require_indexing(&frame.recipe, frame.kind)
                .map_err(indexing_unproven)?;
            frame
                .require_live_inputs(self, matrix, values)
                .map_err(indexing_unproven)?;
            Some(validated)
        } else {
            None
        };
        let matrix_contiguous = self
            .reshape_exec(matrix, matrix.shape())
            .map_err(indexing_unproven)?;
        let matrix = &matrix_contiguous;
        let values_contiguous = self
            .reshape_exec(values, values.shape())
            .map_err(indexing_unproven)?;
        let values = &values_contiguous;

        let raw_matrix = self.get_entry_raw(matrix).map_err(indexing_unproven)?;
        self.get_entry_raw(values).map_err(indexing_unproven)?;
        if matrix.view().is_logical()
            && !self
                .allocations
                .requires_logical_encoding(values)
                .map_err(indexing_unproven)?
        {
            return Err(indexing_unproven(
                runmat_accelerate_api::unsupported_provider_operation(
                    "logical scatter row/column requires canonical logical values",
                ),
            ));
        }
        if raw_matrix.integer_type().is_some() {
            return (|| -> Result<_> {
                ensure!(
                    raw_matrix.shape.len() == 2,
                    "scatter_row: only 2D tensors supported"
                );
                let rows = raw_matrix.shape[0];
                let cols = raw_matrix.shape[1];
                ensure!(row_index < rows, "scatter_row: row index out of bounds");
                let raw_values = self.get_entry_raw(values)?;
                ensure!(
                    raw_values.integer_type() == raw_matrix.integer_type(),
                    "scatter_row: integer storage mismatch"
                );
                ensure!(
                    raw_values.shape.len() <= 2
                        && (raw_values.shape.len() == 1 || raw_values.shape[1] == cols),
                    "scatter_row: length mismatch"
                );
                let indices: Vec<u32> = (0..cols)
                    .map(|col| {
                        u32::try_from(col * rows + row_index)
                            .map_err(|_| anyhow!("scatter_row: index exceeds u32"))
                    })
                    .collect::<Result<_>>()?;
                let all_indices: Vec<u32> = (0..rows * cols)
                    .map(|index| {
                        u32::try_from(index).map_err(|_| anyhow!("scatter_row: index exceeds u32"))
                    })
                    .collect::<Result<_>>()?;
                let out = self.gather_linear_exec(matrix, &all_indices, &raw_matrix.shape)?;
                self.scatter_linear_exec(&out, &indices, values)?;
                Ok(out)
            })()
            .map_err(indexing_unproven);
        }
        let frame = planned.expect("noninteger scatter has a frozen frame");
        let validated = validated.expect("noninteger scatter has a validated pipeline");
        let rows = frame.rows;
        let cols = frame.cols;
        let m_entry = self.get_entry(matrix).map_err(indexing_unproven)?;
        let v_entry = self.get_entry(values).map_err(indexing_unproven)?;
        let out_buffer = self
            .with_emission(|emission, pending| -> Result<_> {
                let out_buffer =
                    self.create_storage_buffer(emission, rows * cols, "runmat-scatter-row-out")?;
                {
                    let mut enc = pending.profiled_encoder(
                        emission,
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-scatter-row-copy"),
                        },
                    );
                    enc.copy_buffer_to_buffer(
                        m_entry.buffer.as_ref(),
                        0,
                        out_buffer.as_ref(),
                        0,
                        (rows * cols * self.element_size) as u64,
                    );
                    self.submit(emission, pending, enc)?;
                }
                let bgl = validated.layout.as_ref();
                let pipeline = validated.pipeline.as_ref();
                #[repr(C)]
                #[derive(Clone, Copy, Pod, Zeroable)]
                struct Pm {
                    rows: u32,
                    cols: u32,
                    i: u32,
                }
                let params = Pm {
                    rows: rows as u32,
                    cols: cols as u32,
                    i: row_index as u32,
                };
                let pbuf = self.uniform_buffer(emission, &params, "runmat-scatter-row-params")?;
                let bg = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-scatter-row-bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: v_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: m_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: pbuf.as_entire_binding(),
                            },
                        ],
                    });
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    cols as u32,
                    frame.recipe.workgroup,
                );
                crate::backend::wgpu::dispatch::scatter::run(
                    emission, pending, pipeline, &bg, groups,
                )?;
                Ok(out_buffer)
            })
            .map_err(indexing_unproven)?
            .map_err(indexing_unproven)?;
        if matrix.view().is_logical() {
            self.register_logical_buffer(
                out_buffer,
                vec![rows, cols],
                rows * cols,
                BufferUsageClass::Generic,
            )
            .map_err(indexing_unproven)
        } else {
            self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols)
                .map_err(indexing_unproven)
        }
    }

    pub(crate) fn gather_linear_exec(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.gather_linear_exec_admitted(source, indices, output_shape, None)
    }

    pub(crate) fn gather_linear_exec_admitted(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_for_bitwise_move(source, "gather_linear")?;
        let expected = product_checked(output_shape)
            .ok_or_else(|| anyhow!("gather_linear: output shape product overflow"))?;
        let source_logical_len = product_checked(source.shape())
            .ok_or_else(|| anyhow!("gather_linear: source shape product overflow"))?;
        let transpose = runmat_accelerate_api::handle_transpose_info(source);
        let (source_rows, source_columns) = if let Some(info) = transpose {
            ensure!(
                source.shape() == [info.base_cols, info.base_rows]
                    && info.base_rows.checked_mul(info.base_cols) == Some(source_logical_len),
                "gather_linear: transpose geometry mismatch"
            );
            (
                u32::try_from(info.base_rows)?,
                u32::try_from(info.base_cols)?,
            )
        } else {
            (0, 0)
        };
        let lane_factor = if source_logical_len == 0 {
            1
        } else {
            functional_scatter::word_geometry(source)?.1
        };
        let source_words = source_logical_len
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("gather_linear: source word extent overflow"))?;
        let output_words = expected
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("gather_linear: output word extent overflow"))?;
        ensure!(
            source_words.saturating_sub(1) <= u32::MAX as usize
                && output_words.saturating_sub(1) <= u32::MAX as usize,
            "gather_linear: word address exceeds GPU indexing limits"
        );
        let _span = info_span!(
            "gpu.gather_linear",
            source_len = entry.len,
            source_logical_len,
            index_count = indices.len(),
            output_size = expected
        )
        .entered();
        ensure!(
            expected == indices.len(),
            "gather_linear: index count {} does not match output size {}",
            indices.len(),
            expected
        );
        if expected == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-gather-linear-empty")
            })??;
            let handle = self.register_storage_move_buffer_admitted(
                out.clone(),
                &entry,
                output_shape.to_vec(),
                0,
                out.size(),
                admission,
            )?;
            return Ok(handle);
        }
        ensure!(
            indices.len() <= u32::MAX as usize,
            "gather_linear: index count exceeds GPU limits"
        );
        ensure!(
            expected.checked_mul(lane_factor).is_some(),
            "gather_linear: output raw length overflow"
        );
        ensure!(
            indices
                .iter()
                .all(|&idx| (idx as usize) < source_logical_len),
            "gather_linear: index out of bounds for source tensor"
        );
        let indices_len_bytes = std::mem::size_of_val(indices) as u64;
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let indices_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-gather-linear-indices"),
                    size: indices_len_bytes.max(4),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            if !indices.is_empty() {
                emission.write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
            }
            log::trace!(
                "gather_linear begin source_buffer={} ptr=0x{:x} out_shape={:?} count={}",
                source.buffer_id(),
                entry.buffer.as_ref() as *const wgpu::Buffer as usize,
                output_shape,
                indices.len()
            );

            let raw_output_len = expected * lane_factor;
            let out_buffer = self.create_storage_buffer_checked_with_element_size(
                emission,
                raw_output_len,
                "runmat-gather-linear-out",
                BufferUsageClass::Generic,
                4,
            )?;
            let params = LinearGatherParams {
                count: indices.len() as u32,
                lane_factor: lane_factor as u32,
                source_rows,
                source_columns,
            };
            let params_buffer =
                self.uniform_buffer(emission, &params, "runmat-gather-linear-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-gather-linear-bind"),
                    layout: &self.pipelines.gather_linear.get(emission)?.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: indices_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                indices.len() as u32,
                self.workgroup_config.scalar,
            );
            crate::backend::wgpu::dispatch::creation::run(
                emission,
                pending,
                &self.pipelines.gather_linear.get(emission)?.pipeline,
                &bind_group,
                workgroups,
                "runmat-gather-linear-encoder",
                "runmat-gather-linear-pass",
            )?;
            log::trace!(
                "gather_linear complete source_buffer={} out_ptr=0x{:x} count={}",
                source.buffer_id(),
                out_buffer.as_ref() as *const wgpu::Buffer as usize,
                indices.len()
            );

            Ok(out_buffer)
        })??;
        let registration_len = if entry.integer_type().is_some() {
            expected
        } else {
            expected
                .checked_mul(match entry.storage {
                    GpuTensorStorage::Real => 1,
                    GpuTensorStorage::ComplexInterleaved => 2,
                })
                .ok_or_else(|| anyhow!("gather_linear: registration length overflow"))?
        };
        let handle = self.register_storage_move_buffer_admitted(
            out_buffer.clone(),
            &entry,
            output_shape.to_vec(),
            registration_len,
            out_buffer.size(),
            admission,
        )?;
        Ok(handle)
    }

    pub(crate) fn scatter_linear_exec(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        ensure!(
            indices.len() <= u32::MAX as usize,
            "scatter_linear: index count exceeds GPU limits"
        );
        let target_entry = self.get_entry_for_bitwise_move(target, "scatter_linear")?;
        let values_contiguous = if values.shares_allocation_with(target) {
            // A read/write binding to the same physical buffer is not a snapshot.
            // Preserve assignment values before mutating any aliased target view.
            let value_indices = (0..indices.len())
                .map(|index| u32::try_from(index).map_err(Into::into))
                .collect::<Result<Vec<u32>>>()?;
            self.gather_linear_exec(values, &value_indices, values.shape())?
        } else {
            self.reshape_exec(values, values.shape())?
        };
        let values = &values_contiguous;
        let values_entry = self.get_entry_for_bitwise_move(values, "scatter_linear values")?;
        let physical_indices;
        let indices = if runmat_accelerate_api::handle_is_transposed(target) {
            let view = super::backend_shared::build_matrix_operand_view(target, &target_entry)?;
            let rows = target.shape()[0];
            let len = rows
                .checked_mul(target.shape()[1])
                .ok_or_else(|| anyhow!("scatter_linear: target shape overflow"))?;
            physical_indices = indices
                .iter()
                .map(|&index| {
                    let index = index as usize;
                    ensure!(index < len, "scatter_linear: logical index out of bounds");
                    let physical = (index % rows)
                        .checked_mul(view.lda as usize)
                        .and_then(|offset| offset.checked_add(index / rows))
                        .ok_or_else(|| anyhow!("scatter_linear: physical index overflow"))?;
                    u32::try_from(physical).map_err(Into::into)
                })
                .collect::<Result<Vec<u32>>>()?;
            physical_indices.as_slice()
        } else {
            indices
        };
        if self.allocations.requires_logical_encoding(target)?
            && !self.allocations.requires_logical_encoding(values)?
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "scatter_linear: logical allocation requires canonical logical values",
            ));
        }
        ensure!(
            target_entry.element_type == values_entry.element_type
                && target_entry.logical == values_entry.logical,
            "scatter_linear: physical element or logical storage mismatch"
        );
        ensure!(
            target_entry.storage == values_entry.storage,
            "scatter_linear: storage mismatch target={:?} values={:?}",
            target_entry.storage,
            values_entry.storage
        );
        let (target_logical_len, lane_factor) = functional_scatter::word_geometry(target)?;
        let (values_logical_len, values_lane_factor) = functional_scatter::word_geometry(values)?;
        let _span = info_span!(
            "gpu.scatter_linear",
            target_len = target_entry.len,
            target_logical_len,
            index_count = indices.len(),
            values_len = values_entry.len
        )
        .entered();
        ensure!(
            values_logical_len == indices.len() && values_lane_factor == lane_factor,
            "scatter_linear: values length {} does not match index count {} for word factor {}",
            values_logical_len,
            indices.len(),
            lane_factor
        );
        ensure!(
            indices
                .iter()
                .all(|&idx| (idx as usize) < target_logical_len),
            "scatter_linear: index out of bounds for target tensor"
        );
        let indices_len_bytes = std::mem::size_of_val(indices) as u64;
        self.with_emission(|emission, pending| -> Result<()> {
        let indices_buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-scatter-linear-indices"),
            size: indices_len_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        emission.write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
        log::trace!(
            "scatter_linear begin target_buffer={} target_ptr=0x{:x} values_buffer={} values_ptr=0x{:x} count={}",
            target.buffer_id(),
            target_entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            values.buffer_id(),
            values_entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            indices.len()
        );
        let params = LinearScatterParams {
            count: indices.len() as u32,
            lane_factor: lane_factor as u32,
            rhs_scalar: 0,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(emission, &params, "runmat-scatter-linear-params")?;
        let bind_group = emission.device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-linear-bind"),
                layout: &self.pipelines.scatter_linear.get(emission)?.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: target_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            indices.len() as u32,
            self.workgroup_config.scalar,
        );
        crate::backend::wgpu::dispatch::creation::run(
            emission,
            pending,
            &self.pipelines.scatter_linear.get(emission)?.pipeline,
            &bind_group,
            workgroups,
            "runmat-scatter-linear-encoder",
            "runmat-scatter-linear-pass",
        )?;
        log::trace!(
            "scatter_linear complete target_buffer={} values_buffer={} count={}",
            target.buffer_id(),
            values.buffer_id(),
            indices.len()
        );
        Ok(())
        })??;
        Ok(())
    }

    pub(crate) fn find_exec(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        let entry = self.get_entry(a)?;
        ensure!(
            entry.precision == NumericPrecision::F64,
            "find: exact double index outputs require f64 provider precision"
        );
        let total = entry.len;
        if total == 0 {
            let shape = vec![0, 1];
            let (indices, rows, cols, values) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, 0, "runmat-find-empty-indices")?,
                        self.create_storage_buffer(emission, 0, "runmat-find-empty-rows")?,
                        self.create_storage_buffer(emission, 0, "runmat-find-empty-cols")?,
                        self.create_storage_buffer(emission, 0, "runmat-find-empty-values")?,
                    ))
                })??;
            let linear = self.register_existing_buffer(indices, shape.clone(), 0)?;
            let rows_handle = self.register_existing_buffer(rows, shape.clone(), 0)?;
            let cols_handle = self.register_existing_buffer(cols, shape.clone(), 0)?;
            let values_handle = self.register_existing_buffer(values, shape, 0)?;
            return Ok(ProviderFindResult {
                linear,
                rows: rows_handle,
                cols: cols_handle,
                values: Some(values_handle),
            });
        }

        ensure!(
            total <= u32::MAX as usize,
            "find: tensor length exceeds GPU support"
        );

        let rows_extent = entry.shape.first().copied().unwrap_or(1).max(1);
        ensure!(
            rows_extent <= u32::MAX as usize,
            "find: row extent exceeds GPU support"
        );

        let raw_cap = match direction {
            FindDirection::First => limit.unwrap_or(total),
            FindDirection::Last => limit.unwrap_or(1),
        };
        let cap = raw_cap.min(total);

        if cap == 0 {
            let shape = vec![0, 1];
            let (indices, rows, cols, values) =
                self.with_emission(|emission, _| -> Result<_> {
                    Ok((
                        self.create_storage_buffer(emission, 0, "runmat-find-zero-limit-indices")?,
                        self.create_storage_buffer(emission, 0, "runmat-find-zero-limit-rows")?,
                        self.create_storage_buffer(emission, 0, "runmat-find-zero-limit-cols")?,
                        self.create_storage_buffer(emission, 0, "runmat-find-zero-limit-values")?,
                    ))
                })??;
            let linear = self.register_existing_buffer(indices, shape.clone(), 0)?;
            let rows_handle = self.register_existing_buffer(rows, shape.clone(), 0)?;
            let cols_handle = self.register_existing_buffer(cols, shape.clone(), 0)?;
            let values_handle = self.register_existing_buffer(values, shape, 0)?;
            return Ok(ProviderFindResult {
                linear,
                rows: rows_handle,
                cols: cols_handle,
                values: Some(values_handle),
            });
        }

        ensure!(cap <= u32::MAX as usize, "find: limit exceeds GPU support");

        let (indices_buffer, rows_buffer, cols_buffer, values_buffer, count_staging) = self
            .with_emission(|emission, pending| -> Result<_> {
                let indices_buffer =
                    self.create_storage_buffer(emission, cap, "runmat-find-indices")?;
                let rows_buffer = self.create_storage_buffer(emission, cap, "runmat-find-rows")?;
                let cols_buffer = self.create_storage_buffer(emission, cap, "runmat-find-cols")?;
                let values_buffer =
                    self.create_storage_buffer(emission, cap, "runmat-find-values")?;

                let count_storage = emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-find-count-storage"),
                    size: 8,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                emission.write_buffer(&count_storage, 0, bytemuck::cast_slice(&[0u32, 0u32]));
                let count_staging = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-find-count-staging"),
                        size: 8,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );

                let params = crate::backend::wgpu::params::FindParams {
                    len: total as u32,
                    limit: cap as u32,
                    rows: rows_extent as u32,
                    direction: match direction {
                        FindDirection::First => 0,
                        FindDirection::Last => 1,
                    },
                    include_values: 1,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let params_buffer = self.uniform_buffer(emission, &params, "runmat-find-params")?;

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-find-bind"),
                        layout: &self.pipelines.find.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: indices_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: rows_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: cols_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: values_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: count_storage.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                crate::backend::wgpu::dispatch::find::run(
                    emission,
                    pending,
                    &self.pipelines.find.get(emission)?.pipeline,
                    &bind_group,
                )?;

                let mut copy_encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-find-copy-count"),
                    },
                );
                copy_encoder.copy_buffer_to_buffer(&count_storage, 0, &count_staging, 0, 8);
                self.submit(emission, pending, copy_encoder)?;
                Ok((
                    indices_buffer,
                    rows_buffer,
                    cols_buffer,
                    values_buffer,
                    count_staging,
                ))
            })??;
        let bytes = self.map_readback_bytes_sync(count_staging, 8, "find")?;
        let counts: &[u32] = cast_slice(&bytes);
        let count = counts.first().copied().unwrap_or(0) as usize;

        let shape = vec![count, 1];
        let linear = self.register_existing_buffer(indices_buffer, shape.clone(), count)?;
        let rows_handle = self.register_existing_buffer(rows_buffer, shape.clone(), count)?;
        let cols_handle = self.register_existing_buffer(cols_buffer, shape.clone(), count)?;
        let values_handle = self.register_existing_buffer(values_buffer, shape, count)?;

        Ok(ProviderFindResult {
            linear,
            rows: rows_handle,
            cols: cols_handle,
            values: Some(values_handle),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, HostNumericDataView,
        HostNumericTensorView,
    };

    #[test]
    #[ignore = "requires an actual WGPU device"]
    fn functional_scatter_preserves_logical_output_and_borrowed_source() {
        use runmat_accelerate_api::{AccelProvider, HostLogicalTensorView};
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU qualification requires a device");
        let source = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[0, 1, 1, 0],
                shape: &[2, 2],
            })
            .unwrap();
        let values = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[1, 1],
                shape: &[2, 1],
            })
            .unwrap();
        let result = futures::executor::block_on(provider.scatter_column(&source, 0, &values));
        if provider.precision() == runmat_accelerate_api::ProviderPrecision::F64 {
            assert!(
                result.unwrap_err().is_unsupported(),
                "the current floating scatter shader is F32-only"
            );
        } else {
            let output = result.unwrap();
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&source));
            assert_eq!(
                futures::executor::block_on(provider.download(&output))
                    .unwrap()
                    .data,
                vec![1.0, 1.0, 1.0, 0.0]
            );
        }
        assert_eq!(
            futures::executor::block_on(provider.download(&source))
                .unwrap()
                .data,
            vec![0.0, 1.0, 1.0, 0.0]
        );
        let numeric = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[2.0, 3.0],
                shape: &[2, 1],
            })
            .unwrap();
        assert!(
            futures::executor::block_on(provider.scatter_column(&source, 0, &numeric))
                .unwrap_err()
                .is_unsupported()
        );
    }

    #[test]
    fn wgpu_linear_gather_and_scatter_copy_complex_logical_elements() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let source = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &[1, 4],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("upload");

        let gathered = provider
            .gather_linear_exec(&source, &[1, 3], &[1, 2])
            .expect("gather");
        assert_eq!(
            runmat_accelerate_api::handle_storage(&gathered),
            GpuTensorStorage::ComplexInterleaved
        );
        let host = block_on(provider.download_exec(&gathered)).expect("download gathered");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 2]);
        assert_eq!(host.data, vec![2.0, 20.0, 4.0, 40.0]);

        let target = provider
            .zeros_with_storage_exec(&[1, 4], GpuTensorStorage::ComplexInterleaved)
            .expect("zeros");
        provider
            .scatter_linear_exec(&target, &[0, 2], &gathered)
            .expect("scatter");
        let host = block_on(provider.download_exec(&target)).expect("download target");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 4]);
        assert_eq!(host.data, vec![2.0, 20.0, 0.0, 0.0, 4.0, 40.0, 0.0, 0.0]);
    }

    #[test]
    fn wgpu_linear_gather_and_scatter_preserve_u64_words() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [0_u64, 1_u64 << 63, u64::MAX];
        let source = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[1, 3],
            })
            .expect("upload exact u64");
        let gathered = provider
            .gather_linear_exec(&source, &[2, 1], &[1, 2])
            .expect("gather exact u64");
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&gathered),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let host = block_on(provider.download_integer_exec(&gathered)).expect("download gathered");
        assert_eq!(
            host.data,
            HostIntegerDataOwned::U64(vec![u64::MAX, 1_u64 << 63])
        );

        let target = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0, 0, 0]),
                shape: &[1, 3],
            })
            .expect("upload target");
        provider
            .scatter_linear_exec(&target, &[0, 2], &gathered)
            .expect("scatter exact u64");
        let host = block_on(provider.download_integer_exec(&target)).expect("download target");
        assert_eq!(
            host.data,
            HostIntegerDataOwned::U64(vec![u64::MAX, 0, 1_u64 << 63])
        );
    }

    #[test]
    fn wgpu_linear_gather_and_scatter_preserve_all_native_integer_classes() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $values:expr, $expected:expr) => {{
                let source = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&$values),
                        shape: &[1, 3],
                    })
                    .expect("upload integer source");
                let gathered = provider
                    .gather_linear_exec(&source, &[2, 0], &[1, 2])
                    .expect("gather integer values");
                assert_eq!(
                    block_on(provider.download_integer_exec(&gathered))
                        .expect("download gathered")
                        .data,
                    HostIntegerDataOwned::$owned(vec![($expected)[0], ($expected)[1]])
                );
                let target = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&[0 as _, 0 as _, 0 as _]),
                        shape: &[1, 3],
                    })
                    .expect("upload integer target");
                provider
                    .scatter_linear_exec(&target, &[0, 2], &gathered)
                    .expect("scatter integer values");
                assert_eq!(
                    block_on(provider.download_integer_exec(&target))
                        .expect("download target")
                        .data,
                    HostIntegerDataOwned::$owned(vec![($expected)[0], 0, ($expected)[1]])
                );
                let preserved = target.clone();
                let functional = provider
                    .scatter_linear_functional_exec(&target, &[1, 2], &gathered)
                    .expect("functional integer scatter");
                assert!(!functional.shares_allocation_with(&preserved));
                assert_eq!(
                    block_on(provider.download_integer_exec(&functional))
                        .expect("download functional result")
                        .data,
                    HostIntegerDataOwned::$owned(vec![
                        ($expected)[0],
                        ($expected)[0],
                        ($expected)[1]
                    ])
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&preserved))
                        .expect("download retained source")
                        .data,
                    HostIntegerDataOwned::$owned(vec![($expected)[0], 0, ($expected)[1]])
                );
            }};
        }
        check!(I8, I8, [i8::MIN, -1, i8::MAX], [i8::MAX, i8::MIN]);
        check!(I16, I16, [i16::MIN, -1, i16::MAX], [i16::MAX, i16::MIN]);
        check!(I32, I32, [i32::MIN, -1, i32::MAX], [i32::MAX, i32::MIN]);
        check!(I64, I64, [i64::MIN, -1, i64::MAX], [i64::MAX, i64::MIN]);
        check!(U8, U8, [0, 1, u8::MAX], [u8::MAX, 0]);
        check!(U16, U16, [0, 1, u16::MAX], [u16::MAX, 0]);
        check!(U32, U32, [0, 1, u32::MAX], [u32::MAX, 0]);
        check!(U64, U64, [0, 1, u64::MAX], [u64::MAX, 0]);
    }

    #[test]
    fn wgpu_row_and_column_scatter_preserve_native_u64_words() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let matrix = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0, 0, 0, 0, 0, 0]),
                shape: &[2, 3],
            })
            .expect("upload matrix");
        let column = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[u64::MAX, 1 << 63]),
                shape: &[2, 1],
            })
            .expect("upload column");
        let after_column =
            block_on(provider.scatter_column_exec(&matrix, 1, &column)).expect("scatter column");
        let row = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[7, 8, 9]),
                shape: &[1, 3],
            })
            .expect("upload row");
        let after_row =
            block_on(provider.scatter_row_exec(&after_column, 0, &row)).expect("scatter row");
        assert_eq!(
            block_on(provider.download_integer_exec(&after_row))
                .expect("download")
                .data,
            HostIntegerDataOwned::U64(vec![7, 0, 8, 1 << 63, 9, 0])
        );
        for handle in [matrix, column, after_column, row, after_row] {
            handle.release().unwrap();
        }
    }
}
