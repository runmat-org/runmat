//! Dimensional F64 reduction for rank greater than two. The input has already
//! crossed any integer-to-floating boundary and is contiguous column-major.
use super::*;
use runmat_accelerate_api::NumericElementType;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    len: u32,
    offset: u32,
    stride_before: u32,
    reduce_len: u32,
}

struct NdSumGeometry {
    before: u32,
    reduce_len: u32,
    out_len: usize,
    out_shape: Vec<usize>,
}

impl WgpuProvider {
    pub(crate) fn nd_sum_pipeline_key(&self) -> u64 {
        super::super::pipeline_recipe::PipelineRecipe::dimension_n_sum_f64(
            self.workgroup_config.scalar,
        )
        .key()
    }

    fn nd_sum_geometry(
        &self,
        shape: &[usize],
        len: usize,
        dim: usize,
        workgroup: u32,
    ) -> Result<NdSumGeometry> {
        ensure!(
            dim < shape.len(),
            "N-D floating sum dimension out of bounds"
        );
        ensure!(
            product_checked(shape) == Some(len),
            "N-D floating sum allocation shape mismatch"
        );
        let before = shape[..dim]
            .iter()
            .try_fold(1usize, |acc, &size| acc.checked_mul(size))
            .ok_or_else(|| anyhow!("N-D floating sum stride overflow"))?;
        let reduce_len = shape[dim];
        let after = shape[dim + 1..]
            .iter()
            .try_fold(1usize, |acc, &size| acc.checked_mul(size))
            .ok_or_else(|| anyhow!("N-D floating sum shape overflow"))?;
        let out_len = before
            .checked_mul(after)
            .ok_or_else(|| anyhow!("N-D floating sum output overflow"))?;
        u32::try_from(len)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum input", len))?;
        let input_bytes = u64::try_from(len)?
            .checked_mul(8)
            .ok_or_else(|| anyhow!("N-D floating sum input byte extent overflow"))?;
        let output_bytes = u64::try_from(out_len)?
            .checked_mul(8)
            .ok_or_else(|| anyhow!("N-D floating sum output byte extent overflow"))?;
        let binding_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        if input_bytes > binding_limit || output_bytes > binding_limit {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "N-D floating sum exceeds enabled storage limits",
            ));
        }
        if out_len != 0 {
            let limits = &self.adapter_limits;
            super::super::backend_shared::validate_compute_binding_counts(
                "N-D floating sum",
                2,
                3,
                limits,
            )
            .map_err(|error| {
                runmat_accelerate_api::unsupported_provider_operation(error.to_string())
            })?;
            if !(workgroup != 0
                && workgroup <= limits.max_compute_workgroup_size_x
                && workgroup <= limits.max_compute_invocations_per_workgroup
                && limits.max_uniform_buffers_per_shader_stage != 0
                && limits.max_uniform_buffer_binding_size >= std::mem::size_of::<Params>() as u32)
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "N-D floating sum exceeds enabled workgroup or uniform limits",
                ));
            }
            let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup)
                .ok_or_else(|| anyhow!("N-D floating sum has invalid dispatch geometry"))?;
            let first_chunk = out_len.min(capacity as usize) as u32;
            if crate::backend::wgpu::dispatch::common::dispatch_size(first_chunk, workgroup)
                > limits.max_compute_workgroups_per_dimension
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "N-D floating sum exceeds enabled dispatch limits",
                ));
            }
        }
        let before = u32::try_from(before)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum", before))?;
        let reduce_len = u32::try_from(reduce_len)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum", reduce_len))?;
        u32::try_from(out_len)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum", out_len))?;
        let mut out_shape = shape.to_vec();
        out_shape[dim] = 1;
        Ok(NdSumGeometry {
            before,
            reduce_len,
            out_len,
            out_shape,
        })
    }

    pub(crate) fn admit_nd_sum_f64_geometry(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        workgroup: u32,
    ) -> Result<bool> {
        let entry = self.get_entry_raw(input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "N-D floating sum requires real storage"
        );
        ensure!(
            input.shape() == entry.shape,
            "N-D floating sum view shape mismatch"
        );
        if entry.logical {
            ensure!(
                self.allocations.requires_logical_encoding(input)?,
                "N-D floating sum logical view lacks its producer"
            );
        }
        let max_binding = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        if entry.allocated_bytes > max_binding {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "N-D floating sum source exceeds enabled storage limit",
            ));
        }
        self.nd_sum_geometry(input.shape(), entry.len, dim, workgroup)
            .map(|geometry| geometry.out_len != 0)
    }

    pub(crate) async fn reduce_nd_sum_f64_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let workgroup = self.workgroup_config.scalar;
        let entry = self
            .get_entry(input)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        if entry.element_type != NumericElementType::F64
            || entry.precision != NumericPrecision::F64
            || entry.integer_type().is_some()
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "N-D floating sum requires physical F64 storage",
            )
            .into());
        }
        let emits = self
            .admit_nd_sum_f64_geometry(input, dim, workgroup)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        let recipe = super::super::pipeline_recipe::PipelineRecipe::dimension_n_sum_f64(workgroup);
        let validated = if emits {
            Some(
                self.prepare_validated_dimension_n_sum_pipeline(&recipe)
                    .await?,
            )
        } else {
            None
        };
        self.reduce_nd_sum_f64_prepared_exec(input, dim, workgroup, &recipe, validated.as_ref())
            .await
    }

    pub(in super::super) async fn reduce_nd_sum_f64_prepared_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        workgroup: u32,
        recipe: &super::super::pipeline_recipe::PipelineRecipe,
        validated: Option<&Arc<super::super::validated_fusion::ValidatedFusionPipeline>>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (entry, geometry) = (|| -> Result<_> {
            let entry = self.get_entry(input)?;
            ensure!(
                self.precision == NumericPrecision::F64 && entry.storage == GpuTensorStorage::Real,
                "N-D floating sum requires real F64 storage"
            );
            ensure!(
                input.shape() == entry.shape
                    && runmat_accelerate_api::handle_transpose_info(input).is_none(),
                "N-D floating sum requires contiguous logical storage"
            );
            ensure!(
                entry.element_type == NumericElementType::F64
                    && entry.precision == NumericPrecision::F64
                    && entry.integer_type().is_none(),
                "N-D floating sum requires physical double elements"
            );
            if entry.logical {
                ensure!(
                    self.allocations.requires_logical_encoding(input)?,
                    "N-D floating sum logical view lacks its producer"
                );
            }
            let geometry = self.nd_sum_geometry(&entry.shape, entry.len, dim, workgroup)?;
            let max_binding = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
                .min(self.adapter_limits.max_buffer_size);
            if entry.allocated_bytes > max_binding {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "N-D floating sum source exceeds enabled storage limit",
                ));
            }
            Ok((entry, geometry))
        })()
        .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        if recipe.workgroup != workgroup {
            return Err(anyhow!("N-D floating sum selected workgroup changed").into());
        }
        if geometry.out_len != 0 {
            validated
                .ok_or_else(|| {
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow!(
                        "N-D floating sum lacks prepared pipeline"
                    ))
                })?
                .require_dimension_n_sum(recipe)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        }
        let output = self
            .with_emission(
                |emission, pending| -> Result<runmat_accelerate_api::WgpuBufferOwner> {
                    let output = self.create_storage_buffer_checked(
                        emission,
                        geometry.out_len,
                        "runmat-nd-sum-f64",
                    )?;
                    ensure!(
                        !runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&output, &entry.buffer),
                        "N-D floating sum output aliases its input"
                    );
                    if geometry.out_len == 0 {
                        return Ok(output);
                    }
                    let validated = validated.expect("nonempty N-D sum was prepared");
                    let capacity =
                        crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup)
                            .ok_or_else(|| {
                                anyhow!("N-D floating sum has invalid workgroup geometry")
                            })? as usize;
                    let mut offset = 0;
                    while offset < geometry.out_len {
                        let count = (geometry.out_len - offset).min(capacity);
                        let params = self.uniform_buffer(
                            emission,
                            &Params {
                                len: count as u32,
                                offset: offset as u32,
                                stride_before: geometry.before,
                                reduce_len: geometry.reduce_len,
                            },
                            "runmat-nd-sum-f64-params",
                        )?;
                        let group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-nd-sum-f64-bg"),
                                    layout: validated.layout.as_ref(),
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: entry.buffer.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: output.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: params.as_entire_binding(),
                                        },
                                    ],
                                });
                        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                            count as u32,
                            workgroup,
                        );
                        crate::backend::wgpu::dispatch::elementwise::run(
                            emission,
                            pending,
                            validated.pipeline.as_ref(),
                            &group,
                            groups,
                        )?;
                        offset += count;
                    }
                    Ok(output)
                },
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?;
        self.register_existing_buffer(output, geometry.out_shape, geometry.out_len)
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
            })
    }
}
