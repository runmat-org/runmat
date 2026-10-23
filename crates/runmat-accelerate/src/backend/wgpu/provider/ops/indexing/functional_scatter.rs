//! Functional linear mutation: copy physical storage on-device, then scatter
//! exact u32 words only into the new provider-owned allocation.
use super::*;
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};

pub(super) fn word_geometry(handle: &GpuTensorHandle) -> Result<(usize, usize)> {
    let logical = handle
        .shape()
        .iter()
        .try_fold(1usize, |n, &extent| n.checked_mul(extent))
        .ok_or_else(|| anyhow!("functional scatter logical extent overflow"))?;
    let bytes = handle.physical().payload_bytes();
    ensure!(
        logical > 0 && bytes > 0 && bytes.is_multiple_of(4),
        "functional scatter requires a nonempty word-aligned payload"
    );
    let words = usize::try_from(bytes / 4)?;
    ensure!(
        words % logical == 0,
        "functional scatter payload geometry mismatch"
    );
    Ok((logical, words / logical))
}

impl WgpuProvider {
    pub(crate) fn scatter_linear_functional_exec(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.scatter_linear_functional_exec_admitted(
            target,
            indices,
            values,
            runmat_accelerate_api::ProviderScatterValueMapping::Linear,
            None,
        )
    }

    pub(crate) fn scatter_linear_functional_exec_admitted(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
        mapping: runmat_accelerate_api::ProviderScatterValueMapping,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if admission.is_some()
            && (runmat_accelerate_api::handle_is_transposed(target)
                || runmat_accelerate_api::handle_is_transposed(values))
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "admitted functional scatter requires contiguous operands",
            )
            .into());
        }
        let target_entry = self.get_entry_for_bitwise_move(target, "functional scatter")?;
        let values_entry = self.get_entry_for_bitwise_move(values, "functional scatter values")?;
        if !(target_entry.element_type == values_entry.element_type
            && target_entry.storage == values_entry.storage
            && target_entry.logical == values_entry.logical)
        {
            return Err(
                anyhow!("functional scatter requires matching physical class and storage").into(),
            );
        }
        let (target_len, words_per_element) = word_geometry(target)?;
        let (values_len, values_words) = word_geometry(values)?;
        let required_values = match mapping {
            runmat_accelerate_api::ProviderScatterValueMapping::Linear => indices.len(),
            runmat_accelerate_api::ProviderScatterValueMapping::Scalar => 1,
        };
        if required_values != values_len || words_per_element != values_words {
            return Err(anyhow!("functional scatter RHS word geometry mismatch").into());
        }
        if !indices.iter().all(|&index| (index as usize) < target_len) {
            return Err(anyhow!("functional scatter index out of bounds").into());
        }
        let binding_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        let index_bytes = u64::try_from(indices.len())
            .ok()
            .and_then(|len| len.checked_mul(4))
            .ok_or_else(|| anyhow!("functional scatter index byte overflow"))?;
        if indices.len() > u32::MAX as usize
            || words_per_element > u32::MAX as usize
            || target_entry.buffer.size() > binding_limit
            || values_entry.buffer.size() > binding_limit
            || index_bytes.max(4) > binding_limit
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "functional scatter exceeds WGPU storage binding limits",
            )
            .into());
        }
        let limits = &self.adapter_limits;
        let workgroup = self.workgroup_config.scalar;
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            indices.len() as u32,
            workgroup.max(1),
        );
        if super::super::backend_shared::validate_compute_binding_counts(
            "functional scatter",
            3,
            4,
            limits,
        )
        .is_err()
            || workgroup == 0
            || workgroup > limits.max_compute_workgroup_size_x
            || workgroup > limits.max_compute_invocations_per_workgroup
            || workgroups > limits.max_compute_workgroups_per_dimension
            || limits.max_uniform_buffers_per_shader_stage == 0
            || limits.max_uniform_buffer_binding_size < 16
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "functional scatter exceeds WGPU compute limits",
            )
            .into());
        }
        self.pipelines.scatter_linear.preflight()?;

        // Transposed views must be materialized in logical order before a raw
        // bitwise copy; reshape_exec performs the existing device-side move.
        let contiguous = if runmat_accelerate_api::handle_is_transposed(target) {
            self.reshape_exec(target, target.shape())
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
        } else {
            target.clone()
        };
        let values_contiguous = if runmat_accelerate_api::handle_is_transposed(values) {
            self.reshape_exec(values, values.shape())
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
        } else {
            values.clone()
        };
        let entry = if runmat_accelerate_api::handle_is_transposed(target) {
            drop(target_entry);
            self.get_entry_for_bitwise_move(&contiguous, "functional scatter copy")
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
        } else {
            target_entry
        };
        let values_entry = if runmat_accelerate_api::handle_is_transposed(values) {
            drop(values_entry);
            self.get_entry_for_bitwise_move(&values_contiguous, "functional scatter RHS")
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
        } else {
            values_entry
        };
        let copy_bytes = contiguous.physical().payload_bytes();
        if copy_bytes > entry.buffer.size()
            || copy_bytes % 4 != 0
            || entry.buffer.size() > binding_limit
            || values_entry.buffer.size() > binding_limit
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "functional scatter materialized payload exceeds WGPU binding limits"
            ))
            .with_unproven_effects());
        }
        let copy_words = usize::try_from(copy_bytes / 4).map_err(|error| {
            ProviderExecutionError::from(anyhow!(error)).with_unproven_effects()
        })?;

        let output_buffer = self
            .with_emission(|emission, pending| -> Result<_> {
                let pipeline = self.pipelines.scatter_linear.get(emission)?;
                let output = self.create_storage_buffer_checked_with_element_size(
                    emission,
                    copy_words,
                    "runmat-functional-scatter-copy",
                    BufferUsageClass::Generic,
                    4,
                )?;
                ensure!(
                    output.as_ref().size() <= binding_limit,
                    "functional scatter copied output exceeds WGPU binding limit"
                );
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-functional-scatter-copy"),
                    },
                );
                encoder.copy_buffer_to_buffer(
                    entry.buffer.as_ref(),
                    0,
                    output.as_ref(),
                    0,
                    copy_bytes,
                );
                self.submit(emission, pending, encoder)?;

                let indices_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-functional-scatter-indices"),
                        size: index_bytes.max(4),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                emission.write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
                let params = LinearScatterParams {
                    count: indices.len() as u32,
                    lane_factor: words_per_element as u32,
                    rhs_scalar: u32::from(
                        mapping == runmat_accelerate_api::ProviderScatterValueMapping::Scalar,
                    ),
                    _pad: 0,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-functional-scatter-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-functional-scatter-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: output.as_ref().as_entire_binding(),
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
                    &pipeline.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-functional-scatter-encoder",
                    "runmat-functional-scatter-pass",
                )?;
                Ok(output)
            })
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        self.register_storage_move_buffer_admitted(
            output_buffer.clone(),
            &entry,
            target.shape().to_vec(),
            entry.len,
            output_buffer.size(),
            admission,
        )
        .map_err(|error| ProviderExecutionError::from(error).after_effect())
    }
}
