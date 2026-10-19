//! Physical admission and scoped execution for public extrema reductions.
use super::super::backend_types::BufferEntry;
use super::super::validated_fusion::ReductionExtremaPipelineKind;
use super::*;
use runmat_accelerate_api::{
    unsupported_provider_operation, ProviderExecutionError, ProviderExecutionResult,
};
use std::cell::Cell;
mod recipe;

struct ExtremaInputFrame {
    entry: BufferEntry,
    kind: ReductionExtremaPipelineKind,
    recipe: Option<super::super::pipeline_recipe::PipelineRecipe>,
}

#[derive(Clone, Copy)]
enum ExtremaRoute {
    Global,
    Dimensional,
}

fn admit_class(
    storage: GpuTensorStorage,
    logical: bool,
    integer: bool,
    route: ExtremaRoute,
) -> Result<()> {
    if storage != GpuTensorStorage::Real || logical {
        return Err(unsupported_provider_operation(
            "extrema requires a real numeric input",
        ));
    }
    if integer && matches!(route, ExtremaRoute::Global) {
        return Err(unsupported_provider_operation(
            "global extrema requires a floating input",
        ));
    }
    Ok(())
}

impl ExtremaInputFrame {
    fn global(provider: &WgpuProvider, input: &GpuTensorHandle) -> Result<Self> {
        let entry = provider.get_entry_raw(input)?;
        admit_class(
            entry.storage,
            input.view().is_logical(),
            entry.integer_type().is_some(),
            ExtremaRoute::Global,
        )?;
        provider.get_entry(input)?;
        if entry.len > 0 {
            provider.require_ieee_words(entry.precision)?;
            provider.pipelines.reduce_global.preflight()?;
        }
        super::global_resources::admit(
            entry.len,
            entry.buffer.size(),
            provider.element_size,
            provider.global_reduce_launch.elements_per_group(),
            provider.global_reduce_launch.max_input_per_pass(),
            u64::from(provider.adapter_limits.max_storage_buffer_binding_size),
            provider.adapter_limits.max_buffer_size,
        )?;
        let kind = ReductionExtremaPipelineKind::Global;
        Ok(Self {
            recipe: (entry.len > 0).then(|| kind.recipe(provider)),
            entry,
            kind,
        })
    }

    fn dimensional(provider: &WgpuProvider, input: &GpuTensorHandle, dim: usize) -> Result<Self> {
        let entry = provider.get_entry_raw(input)?;
        admit_class(
            entry.storage,
            input.view().is_logical(),
            entry.integer_type().is_some(),
            ExtremaRoute::Dimensional,
        )?;
        let geometry = super::extrema_geometry::geometry(
            &entry.shape,
            dim,
            provider.precision,
            provider.workgroup_config.scalar,
            provider.adapter_limits.max_compute_workgroups_per_dimension,
        )?;
        let kind = if entry.integer_type().is_some() {
            ReductionExtremaPipelineKind::IntegerDimensional
        } else {
            provider.get_entry(input)?;
            provider.require_ieee_words(entry.precision)?;
            provider.pipelines.reduce_dim_minmax.preflight()?;
            ReductionExtremaPipelineKind::Dimensional
        };
        if geometry.length > 0 {
            let limit = u64::from(provider.adapter_limits.max_storage_buffer_binding_size)
                .min(provider.adapter_limits.max_buffer_size);
            let output_width = if let Some(integer) = entry.integer_type() {
                crate::backend::wgpu_storage::integer_words_per_element(integer) as u64 * 4
            } else {
                provider.element_size as u64
            }
            .max(provider.element_size as u64); // Values and floating indices share the limit.
            let output_bytes = (geometry.length as u64)
                .checked_mul(output_width)
                .ok_or_else(|| {
                    unsupported_provider_operation("dimensional extrema output extent overflow")
                })?;
            if entry.buffer.size() > limit || output_bytes > limit {
                return Err(unsupported_provider_operation(
                    "dimensional extrema storage binding exceeds adapter limits",
                ));
            }
        }
        Ok(Self {
            recipe: (geometry.length > 0).then(|| kind.recipe(provider)),
            entry,
            kind,
        })
    }

    fn revalidate(&self, provider: &WgpuProvider, input: &GpuTensorHandle) -> Result<()> {
        let current = provider.get_entry_raw(input)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&current.buffer, &self.entry.buffer)
                && current.shape == self.entry.shape
                && current.len == self.entry.len
                && current.element_type == self.entry.element_type
                && current.storage == self.entry.storage
                && current.allocated_bytes == self.entry.allocated_bytes,
            "extrema input changed during scoped pipeline preparation"
        );
        input.allocation_lease().validate_live()?;
        Ok(())
    }
}

fn terminal_error(error: anyhow::Error, allocated: bool) -> ProviderExecutionError {
    let typed = ProviderExecutionError::from(error);
    if allocated {
        typed.after_effect()
    } else {
        typed.with_unproven_effects()
    }
}

impl WgpuProvider {
    pub(crate) async fn typed_global_extrema_exec(
        &self,
        input: &GpuTensorHandle,
        operation: crate::backend::wgpu::types::GlobalReduceOp,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.typed_global_extrema_with_fault(input, operation, None)
            .await
    }

    async fn typed_global_extrema_with_fault(
        &self,
        input: &GpuTensorHandle,
        operation: crate::backend::wgpu::types::GlobalReduceOp,
        after_allocation: Option<&super::phase_context::AllocationFault<'_>>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let frame = ExtremaInputFrame::global(self, input)?;
        let validated = if let Some(recipe) = frame.recipe.as_ref() {
            Some(
                self.prepare_validated_reduction_extrema_pipeline(recipe, frame.kind)
                    .await?,
            )
        } else {
            None
        };
        frame
            .revalidate(self, input)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if let (Some(recipe), Some(validated)) = (frame.recipe.as_ref(), validated.as_ref()) {
            validated
                .require_reduction_extrema(recipe, frame.kind)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        }
        let allocated = Cell::new(false);
        self.reduce_global_with_selected_pipeline_exec(
            input,
            operation,
            super::phase_context::SelectedEmission {
                selected: validated.as_deref(),
                effect: Some(&allocated),
                after_allocation,
            },
        )
        .map_err(|error| terminal_error(error, allocated.get()))
    }

    pub(crate) async fn typed_dimensional_extrema_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        operation: crate::backend::wgpu::types::DimReduceExtrema,
    ) -> ProviderExecutionResult<ReduceDimResult> {
        self.typed_dimensional_extrema_with_fault(input, dim, operation, None)
            .await
    }

    async fn typed_dimensional_extrema_with_fault(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        operation: crate::backend::wgpu::types::DimReduceExtrema,
        after_allocation: Option<&super::phase_context::AllocationFault<'_>>,
    ) -> ProviderExecutionResult<ReduceDimResult> {
        let frame = ExtremaInputFrame::dimensional(self, input, dim)?;
        let validated = if let Some(recipe) = frame.recipe.as_ref() {
            Some(
                self.prepare_validated_reduction_extrema_pipeline(recipe, frame.kind)
                    .await?,
            )
        } else {
            None
        };
        frame
            .revalidate(self, input)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if let (Some(recipe), Some(validated)) = (frame.recipe.as_ref(), validated.as_ref()) {
            validated
                .require_reduction_extrema(recipe, frame.kind)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        }
        let allocated = Cell::new(false);
        self.reduce_dim_minmax_with_selected_pipeline_exec(
            input,
            dim,
            operation,
            super::phase_context::SelectedEmission {
                selected: validated.as_deref(),
                effect: Some(&allocated),
                after_allocation,
            },
        )
        .map_err(|error| terminal_error(error, allocated.get()))
    }
}

#[cfg(test)]
mod tests;
