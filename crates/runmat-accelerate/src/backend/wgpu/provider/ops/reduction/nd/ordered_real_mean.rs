//! Shared exact-class, source-axis-order real mean proposal and consumption.
//! The existing N-D moments cache is a different arithmetic route.

use super::{geometry::NdReductionGeometry, GpuTensorHandle, WgpuProvider};
use crate::backend::wgpu::{
    params::ReduceNdParams, shaders::reduction::ordered_mean::LANES, types::NumericPrecision,
};
use anyhow::Result;
use runmat_accelerate_api::{
    GpuTensorStorage, ProviderElementType, ProviderExecutedOrderedRealMean, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderOrderedRealMean,
    ProviderOrderedRealMeanRecipe, ProviderPrecision, ProviderRealMeanNaNMode,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};

#[path = "ordered_real_mean/execute.rs"]
mod execute;
#[path = "ordered_real_mean/plan.rs"]
mod plan;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "ordered_real_mean/tests.rs"]
mod tests;

type OrderedMeanPlan = ProviderOrderedRealMeanRecipe;

fn provider_precision(precision: NumericPrecision) -> ProviderPrecision {
    match precision {
        NumericPrecision::F32 => ProviderPrecision::F32,
        NumericPrecision::F64 => ProviderPrecision::F64,
    }
}

fn physical_precision(precision: ProviderPrecision) -> NumericPrecision {
    match precision {
        ProviderPrecision::F32 => NumericPrecision::F32,
        ProviderPrecision::F64 => NumericPrecision::F64,
    }
}

impl WgpuProvider {
    pub(in crate::backend::wgpu::provider::backend) fn plan_ordered_real_mean_exec(
        &self,
        source: &ProviderRepresentation,
        source_buffer_bytes: u64,
        axes: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> Result<Option<ProviderOrderedRealMeanRecipe>> {
        self.derive_ordered_mean_recipe(
            source,
            source_buffer_bytes,
            axes,
            output_precision,
            nan_mode,
        )
    }

    /// The live-source projection retains the actual provider allocation's
    /// buffer extent. A future-source proposal cannot manufacture this proof.
    fn ordered_mean_live_source(
        &self,
        source: &GpuTensorHandle,
    ) -> Result<(ProviderRepresentation, u64)> {
        let entry = self.real_floating_entry(source)?;
        if entry.storage != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(source)
            || runmat_accelerate_api::handle_is_transposed(source)
        {
            anyhow::bail!("ordered mean requires an exact contiguous real floating view");
        }
        let shape = entry
            .shape
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let element_type = match entry.precision {
            NumericPrecision::F32 => ProviderElementType::F32,
            NumericPrecision::F64 => ProviderElementType::F64,
        };
        Ok((
            ProviderRepresentation {
                element_type,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape,
                residency: ProviderResidency::Device,
            },
            entry.buffer.size(),
        ))
    }

    fn ordered_mean_plan(
        &self,
        source: &GpuTensorHandle,
        axes: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> Result<Option<OrderedMeanPlan>> {
        let (representation, source_buffer_bytes) = self.ordered_mean_live_source(source)?;
        self.derive_ordered_mean_recipe(
            &representation,
            source_buffer_bytes,
            axes,
            output_precision,
            nan_mode,
        )
    }

    pub(in crate::backend::wgpu::provider::backend) fn prepare_ordered_real_mean_exec(
        &self,
        source: &GpuTensorHandle,
        axes: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> Result<Option<ProviderOrderedRealMean>> {
        Ok(self
            .ordered_mean_plan(source, axes, output_precision, nan_mode)?
            .map(|recipe| ProviderOrderedRealMean::new(source.clone(), recipe)))
    }

    pub(in crate::backend::wgpu::provider::backend) fn prepare_ordered_real_mean_against_recipe_exec(
        &self,
        source: &GpuTensorHandle,
        selected: &ProviderOrderedRealMeanRecipe,
    ) -> Result<Option<ProviderOrderedRealMean>> {
        let current = self.ordered_mean_plan(
            source,
            selected.axes(),
            selected.output_precision(),
            selected.nan_mode(),
        )?;
        match current {
            Some(recipe) if &recipe == selected => {
                Ok(Some(ProviderOrderedRealMean::new(source.clone(), recipe)))
            }
            _ => Ok(None),
        }
    }

    pub(in crate::backend::wgpu::provider::backend) fn execute_ordered_real_mean_exec(
        &self,
        selected: ProviderOrderedRealMean,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedOrderedRealMean> {
        let source = selected.source();
        let recipe = selected.recipe();
        if recipe.steps().is_empty() {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "ordered mean has no stages"
            )));
        }
        let plan = self
            .ordered_mean_plan(
                source,
                recipe.axes(),
                recipe.output_precision(),
                recipe.nan_mode(),
            )
            .map_err(ProviderExecutionError::from)?
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow::anyhow!(
                    "ordered mean source or device route changed"
                ))
            })?;
        if &plan != recipe {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "ordered mean selected physical route changed"
            )));
        }
        let (result, observed_incremental_peak_bytes) =
            self.execute_ordered_mean_plan(source, &plan, recipe.nan_mode(), admission)?;
        let observed_output_bytes = self
            .real_floating_entry(&result)
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?
            .buffer
            .size();
        let selected_submissions = plan.queue_submissions();
        Ok(ProviderExecutedOrderedRealMean {
            output: result,
            requested_incremental_peak_bytes: plan.requested_incremental_peak_bytes(),
            observed_incremental_peak_bytes,
            observed_output_bytes,
            selected_submissions,
        })
    }
}
