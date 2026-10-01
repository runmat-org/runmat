//! Prepare integer and single inputs for the ordinary sum's F64 accumulator.
//! Native integer reductions remain in `integer.rs` and keep saturation.
use super::*;
use runmat_accelerate_api::allocation::GpuHandleReleaseError;
use runmat_accelerate_api::{
    GpuTensorStorage, NumericElementType, ProviderExecutionError, ProviderExecutionResult,
};
use std::{error::Error, fmt};

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

/// Preserve the numerical/provider cause and both independent retirement
/// failures without converting an owner-retaining release error into text.
#[derive(Debug)]
struct TemporaryCompletionError {
    stage: &'static str,
    primary: Option<anyhow::Error>,
    temporary_cleanup: GpuHandleReleaseError,
    output_cleanup: Option<GpuHandleReleaseError>,
}

impl fmt::Display for TemporaryCompletionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(primary) = &self.primary {
            write!(formatter, "{} failed: {primary}; ", self.stage)?;
        } else {
            write!(formatter, "{}: ", self.stage)?;
        }
        write!(
            formatter,
            "temporary cleanup failed: {}",
            self.temporary_cleanup
        )?;
        if let Some(output_cleanup) = &self.output_cleanup {
            write!(formatter, "; output cleanup also failed: {output_cleanup}")?;
        }
        Ok(())
    }
}

impl Error for TemporaryCompletionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // A failed retirement is operational even when the numerical hook
        // returned Unsupported. Keep the primary owned in this carrier, but
        // do not advertise its fallback marker through the source chain.
        Some(&self.temporary_cleanup)
    }
}

fn finish_temporary(
    result: Result<GpuTensorHandle>,
    temporary: GpuTensorHandle,
    stage: &'static str,
) -> ProviderExecutionResult<GpuTensorHandle> {
    let cleanup = temporary.release();
    match (result, cleanup) {
        (Ok(output), Ok(_)) => Ok(output),
        (Err(error), Ok(_)) => Err(error.into()),
        (Err(primary), Err(temporary_cleanup)) => Err(ProviderExecutionError::from_source(
            TemporaryCompletionError {
                stage,
                primary: Some(primary),
                temporary_cleanup,
                output_cleanup: None,
            },
        )),
        (Ok(output), Err(cleanup)) => Err(ProviderExecutionError::from_source(
            TemporaryCompletionError {
                stage,
                primary: None,
                temporary_cleanup: cleanup,
                output_cleanup: output.release().err(),
            },
        )),
    }
}

impl WgpuProvider {
    fn input_to_double_for_sum(&self, input: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(input)?;
        if entry.integer_type().is_some() {
            self.integer_to_double_for_sum(input)
        } else if entry.storage == GpuTensorStorage::Real
            && entry.precision == NumericPrecision::F32
        {
            self.floating_cast_exec(input, NumericPrecision::F64)
        } else {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "double sum conversion requires real integer or single storage",
            ))
        }
    }

    fn integer_to_double_for_sum(&self, input: &GpuTensorHandle) -> Result<GpuTensorHandle> {
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
        let output = self.with_emission(|emission, pending| -> Result<Arc<wgpu::Buffer>> {
            let output = self.create_integer_word_buffer(
                emission,
                words,
                "runmat-integer-sum-floating-cast",
            )?;
            if entry.len == 0 {
                return Ok(output);
            }
            let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
            let recipe = super::pipeline_recipe::PipelineRecipe::integer_sum_cast(workgroup_size);
            let shader = &recipe.shader;
            let layout =
                emission
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-integer-sum-floating-cast-bgl"),
                        entries: &[
                            crate::backend::wgpu::bindings::storage_read_entry(0),
                            crate::backend::wgpu::bindings::storage_read_write_entry(1),
                            crate::backend::wgpu::bindings::uniform_entry(2),
                        ],
                    });
            let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
                emission,
                "runmat-integer-sum-floating-cast-pl",
                &layout,
            );
            let module = crate::backend::wgpu::pipelines::create_shader_module(
                emission,
                "runmat-integer-sum-floating-cast-module",
                &shader,
            );
            let key = recipe.key();
            let pipeline = self.get_or_create_pipeline(
                emission,
                key,
                &pipeline_layout,
                &module,
                "runmat-integer-sum-floating-cast",
                Some(shader.as_bytes()),
                Some("runmat-integer-sum-floating-cast-bgl"),
                Some(workgroup_size),
            )?;
            let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup_size)
                .ok_or_else(|| anyhow!("integer sum conversion has invalid workgroup geometry"))?
                as usize;
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
                        layout: &layout,
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
                    emission, pending, &pipeline, &group, groups,
                )?;
                offset += count;
            }
            Ok(output)
        })??;
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

    pub(crate) fn integer_floating_sum_global_exec(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Global)
    }

    pub(crate) fn integer_floating_sum_dim_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Dimension(dim))
    }

    pub(crate) fn single_double_sum_global_exec(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Global)
    }

    pub(crate) fn single_double_sum_dim_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.double_sum_exec(input, SumSelection::Dimension(dim))
    }

    fn double_sum_exec(
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
        // Floating reductions read physical column-major order. A lazy
        // transpose must be materialized even for the global route so the
        // cast publishes truthful contiguous shape/layout metadata.
        let contiguous = self
            .reshape_exec(input, input.shape())
            .map_err(ProviderExecutionError::from)?;
        let materialized = !contiguous.shares_allocation_with(input);
        let converted = self.input_to_double_for_sum(&contiguous);
        let converted = if materialized {
            finish_temporary(converted, contiguous, "integer sum view materialization")?
        } else {
            converted.map_err(ProviderExecutionError::from)?
        };
        let result = match selection {
            SumSelection::Global => self
                .reduce_global_exec(&converted, crate::backend::wgpu::types::GlobalReduceOp::Sum),
            SumSelection::Dimension(dim) if converted.shape().len() == 2 => self
                .reduce_dim_sum_mean_exec(
                    &converted,
                    dim,
                    crate::backend::wgpu::types::DimReduceOp::Sum,
                ),
            SumSelection::Dimension(dim) => self.reduce_nd_sum_f64_exec(&converted, dim),
        };
        finish_temporary(result, converted, "integer floating sum reduction")
    }
}
