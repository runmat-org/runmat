//! Prepare integer and single inputs for the ordinary sum's F64 accumulator.
//! Native integer reductions remain in `integer.rs` and keep saturation.
use super::*;
use runmat_accelerate_api::{
    GpuTensorStorage, NumericElementType, ProviderExecutionCompletionError, ProviderExecutionError,
    ProviderExecutionResult,
};

#[cfg(test)]
#[path = "integer_float_sum/tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CastParams {
    len: u32,
    offset: u32,
    source_type: u32,
    padding: u32,
}

#[derive(Clone, Copy)]
enum SumSelection {
    Global,
    Dimension(usize),
}

fn finish_temporary(
    result: ProviderExecutionResult<GpuTensorHandle>,
    temporary: GpuTensorHandle,
) -> ProviderExecutionResult<GpuTensorHandle> {
    let cleanup = temporary.release();
    match (result, cleanup) {
        (Ok(output), Ok(_)) => Ok(output),
        (Err(error), Ok(_)) => Err(error.after_effect()),
        (Err(primary), Err(temporary_cleanup)) => Err(ProviderExecutionError::from_source(
            ProviderExecutionCompletionError::new(Some(primary), vec![temporary_cleanup]),
        )
        .after_effect()),
        (Ok(output), Err(temporary_cleanup)) => {
            let mut cleanup = vec![temporary_cleanup];
            cleanup.extend(output.release().err());
            Err(
                ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                    None, cleanup,
                ))
                .after_effect(),
            )
        }
    }
}

impl WgpuProvider {
    fn admit_sum_conversion_extent(&self, input: &GpuTensorHandle, workgroup: u32) -> Result<()> {
        let entry = self.get_entry_raw(input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && (entry.integer_type().is_some()
                    || (entry.element_type == NumericElementType::F32
                        && entry.precision == NumericPrecision::F32)),
            "double sum conversion requires real integer or physical F32 storage"
        );
        ensure!(
            product_checked(&entry.shape) == Some(entry.len),
            "double sum allocation shape mismatch"
        );
        ensure!(
            input.shape() == entry.shape,
            "double sum view shape mismatch"
        );
        if entry.logical {
            ensure!(
                self.allocations.requires_logical_encoding(input)?,
                "double sum logical view lacks its physical producer"
            );
        }
        u32::try_from(entry.len)
            .map_err(|_| gpu_dispatch_length_limit_error("double sum conversion", entry.len))?;
        let bytes = u64::try_from(entry.len)?
            .checked_mul(8)
            .ok_or_else(|| anyhow!("double sum output byte extent overflow"))?;
        let limits = &self.adapter_limits;
        let binding_limit =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        if bytes > binding_limit || entry.allocated_bytes > binding_limit {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "double sum conversion exceeds enabled storage limits",
            ));
        }
        if entry.len == 0 {
            return Ok(());
        }
        super::backend_shared::validate_compute_binding_counts(
            "double sum conversion",
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
            && limits.max_uniform_buffer_binding_size >= std::mem::size_of::<CastParams>() as u32)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "double sum conversion exceeds enabled workgroup or uniform limits",
            ));
        }
        let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup)
            .ok_or_else(|| anyhow!("double sum conversion has invalid dispatch geometry"))?;
        let first_chunk = entry.len.min(capacity as usize) as u32;
        if crate::backend::wgpu::dispatch::common::dispatch_size(first_chunk, workgroup)
            > limits.max_compute_workgroups_per_dimension
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "double sum conversion exceeds enabled dispatch limits",
            ));
        }
        Ok(())
    }

    async fn input_to_double_for_sum(
        &self,
        input: &GpuTensorHandle,
        cast_recipe: Option<&super::pipeline_recipe::PipelineRecipe>,
        validated_cast: Option<&Arc<super::validated_fusion::ValidatedFusionPipeline>>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let entry = self
            .get_entry_raw(input)
            .map_err(ProviderExecutionError::from)?;
        if entry.integer_type().is_some() {
            self.integer_to_double_for_sum(input, cast_recipe, validated_cast)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())
        } else if entry.storage == GpuTensorStorage::Real
            && entry.element_type == NumericElementType::F32
            && entry.precision == NumericPrecision::F32
        {
            self.floating_cast_exec(input, NumericPrecision::F64).await
        } else {
            Err(ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation(
                    "double sum conversion requires real integer or single storage",
                ),
            ))
        }
    }

    fn integer_to_double_for_sum(
        &self,
        input: &GpuTensorHandle,
        cast_recipe: Option<&super::pipeline_recipe::PipelineRecipe>,
        validated_cast: Option<&Arc<super::validated_fusion::ValidatedFusionPipeline>>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(input)?;
        let integer_type = entry
            .integer_type()
            .ok_or_else(|| anyhow!("integer sum conversion requires a resident integer input"))?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "integer sum requires real storage"
        );
        if self.precision != NumericPrecision::F64 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ordinary integer sum requires shader-f64 support",
            ));
        }
        ensure!(
            input.shape() == entry.shape,
            "integer sum conversion requires a contiguous logical view"
        );
        ensure!(
            runmat_accelerate_api::handle_transpose_info(input).is_none(),
            "integer sum conversion requires a materialized transpose"
        );
        ensure!(
            product_checked(&entry.shape) == Some(entry.len),
            "integer sum allocation shape mismatch"
        );
        if entry.len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(
                "integer sum conversion",
                entry.len,
            ));
        }
        let words = entry
            .len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("integer sum allocation overflow"))?;
        let allocated_bytes = (words as u64)
            .checked_mul(4)
            .ok_or_else(|| anyhow!("integer sum byte count overflow"))?;
        let output = self.with_emission(
            |emission, pending| -> Result<runmat_accelerate_api::WgpuBufferOwner> {
                let output = self.create_integer_word_buffer(
                    emission,
                    words,
                    "runmat-integer-sum-floating-cast",
                )?;
                if entry.len == 0 {
                    return Ok(output);
                }
                let recipe = cast_recipe
                    .ok_or_else(|| anyhow!("nonempty integer sum cast lacks selected recipe"))?;
                let workgroup_size = recipe.workgroup;
                let validated = validated_cast
                    .ok_or_else(|| anyhow!("nonempty integer sum cast lacks validated pipeline"))?;
                validated.require_integer_sum_cast(recipe)?;
                let capacity =
                    crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup_size)
                        .ok_or_else(|| {
                            anyhow!("integer sum conversion has invalid workgroup geometry")
                        })? as usize;
                let mut offset = 0;
                while offset < entry.len {
                    let count = (entry.len - offset).min(capacity);
                    let params = self.uniform_buffer(
                        emission,
                        &CastParams {
                            len: count as u32,
                            offset: offset as u32,
                            source_type: super::integer::integer_type_code(integer_type),
                            padding: 0,
                        },
                        "runmat-integer-sum-floating-cast-params",
                    )?;
                    let group = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-integer-sum-floating-cast-bg"),
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
                        workgroup_size,
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
        )??;
        self.register_numeric_buffer(
            output,
            NumericBufferRegistration {
                shape: entry.shape,
                len: entry.len,
                physical_element_type: NumericElementType::F64,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        )
    }

    pub(crate) async fn integer_floating_sum_global_exec(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Global).await
    }

    pub(crate) async fn integer_floating_sum_dim_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Dimension(dim))
            .await
    }

    pub(crate) async fn single_double_sum_global_exec(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Global).await
    }

    pub(crate) async fn single_double_sum_dim_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Dimension(dim))
            .await
    }

    async fn double_sum_exec(
        &self,
        input: &GpuTensorHandle,
        selection: SumSelection,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let entry = self
            .get_entry_raw(input)
            .map_err(ProviderExecutionError::from)?;
        if self.precision != NumericPrecision::F64 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "double sum requires shader-f64 support",
            )
            .into());
        }
        if entry.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "double sum conversion requires real storage",
            )
            .into());
        }
        if let SumSelection::Dimension(dim) = selection {
            if dim >= input.shape().len() {
                return Err(ProviderExecutionError::from(anyhow!(
                    "double sum dimension out of bounds"
                )));
            }
        }
        let workgroup = self.workgroup_config.scalar;
        // Stage all selected physical facts before either pipeline preparation.
        self.admit_sum_conversion_extent(input, workgroup)
            .map_err(ProviderExecutionError::from)?;
        let (nd_recipe, nd_emits) = if let SumSelection::Dimension(dim) = selection {
            if input.shape().len() > 2 {
                let emits = self
                    .admit_nd_sum_f64_geometry(input, dim, workgroup)
                    .map_err(ProviderExecutionError::from)?;
                (
                    Some(super::pipeline_recipe::PipelineRecipe::dimension_n_sum_f64(
                        workgroup,
                    )),
                    emits,
                )
            } else {
                (None, false)
            }
        } else {
            (None, false)
        };
        let cast_recipe = (entry.integer_type().is_some() && entry.len != 0)
            .then(|| super::pipeline_recipe::PipelineRecipe::integer_sum_cast(workgroup));
        // A scoped driver error remains pre-effect because neither recipe has
        // yet caused a view materialization, cast, or reduction submission.
        let validated_nd = if nd_emits {
            Some(
                self.prepare_validated_dimension_n_sum_pipeline(
                    nd_recipe.as_ref().expect("rank-N sum selected its recipe"),
                )
                .await?,
            )
        } else {
            None
        };
        let validated_cast = if let Some(recipe) = &cast_recipe {
            Some(
                self.prepare_validated_integer_sum_cast_pipeline(recipe)
                    .await?,
            )
        } else {
            None
        };
        // Floating reductions read physical column-major order. A lazy
        // transpose must be materialized even for the global route so the
        // cast publishes truthful contiguous shape/layout metadata.
        let contiguous = self
            .reshape_exec(input, input.shape())
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let materialized = !contiguous.shares_allocation_with(input);
        let converted = self
            .input_to_double_for_sum(&contiguous, cast_recipe.as_ref(), validated_cast.as_ref())
            .await;
        let converted = if materialized {
            finish_temporary(converted, contiguous)?
        } else {
            converted?
        };
        let result = match selection {
            SumSelection::Global => self
                .reduce_global_exec(&converted, crate::backend::wgpu::types::GlobalReduceOp::Sum)
                .map_err(ProviderExecutionError::from),
            SumSelection::Dimension(dim) if converted.shape().len() == 2 => self
                .reduce_dim_sum_mean_exec(
                    &converted,
                    dim,
                    crate::backend::wgpu::types::DimReduceOp::Sum,
                )
                .map_err(ProviderExecutionError::from),
            SumSelection::Dimension(dim) => {
                self.reduce_nd_sum_f64_prepared_exec(
                    &converted,
                    dim,
                    workgroup,
                    nd_recipe.as_ref().expect("rank-N sum selected its recipe"),
                    validated_nd.as_ref(),
                )
                .await
            }
        };
        finish_temporary(result, converted)
    }
}
