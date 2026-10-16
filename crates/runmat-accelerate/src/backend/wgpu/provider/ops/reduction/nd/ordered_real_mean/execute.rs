use super::{
    physical_precision, NdReductionGeometry, OrderedMeanPlan, ProviderExecutionError,
    ProviderExecutionResult, ProviderRealMeanNaNMode, ReduceNdParams, WgpuProvider,
};
use crate::backend::wgpu::{params::AlignedU32, residency::BufferUsageClass};
use anyhow::{ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, WgpuBufferOwner};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OrderedMeanCheckpoint {
    OutputAllocated,
    StageSubmitted,
}

fn params_for_axis(
    geometry: &NdReductionGeometry,
    nan_mode: ProviderRealMeanNaNMode,
) -> ReduceNdParams {
    let mut params = ReduceNdParams {
        rank: geometry.rank as u32,
        kept_count: geometry.kept.len() as u32,
        reduce_count: geometry.reduce.len() as u32,
        _pad: u32::from(nan_mode == ProviderRealMeanNaNMode::Omit),
        rows: geometry.rows as u32,
        cols: geometry.cols as u32,
        _pad2: [0; 2],
        kept_sizes: [AlignedU32::default(); crate::backend::wgpu::params::BCAST_MAX_RANK],
        reduce_sizes: [AlignedU32::default(); crate::backend::wgpu::params::BCAST_MAX_RANK],
        kept_strides: [AlignedU32::default(); crate::backend::wgpu::params::BCAST_MAX_RANK],
        reduce_strides: [AlignedU32::default(); crate::backend::wgpu::params::BCAST_MAX_RANK],
    };
    for (dst, &value) in params.kept_sizes.iter_mut().zip(&geometry.kept_sizes) {
        *dst = AlignedU32::new(value);
    }
    for (dst, &value) in params.reduce_sizes.iter_mut().zip(&geometry.reduce_sizes) {
        *dst = AlignedU32::new(value);
    }
    for (dst, &value) in params.kept_strides.iter_mut().zip(&geometry.kept_strides) {
        *dst = AlignedU32::new(value);
    }
    for (dst, &value) in params
        .reduce_strides
        .iter_mut()
        .zip(&geometry.reduce_strides)
    {
        *dst = AlignedU32::new(value);
    }
    params
}

impl WgpuProvider {
    pub(super) fn execute_ordered_mean_plan(
        &self,
        source: &GpuTensorHandle,
        plan: &OrderedMeanPlan,
        nan_mode: ProviderRealMeanNaNMode,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<(GpuTensorHandle, u64)> {
        self.execute_ordered_mean_plan_with_admission(source, plan, nan_mode, admission, |_| Ok(()))
    }

    /// The request-local checkpoint is a no-op in production. Tests inject a
    /// failure at the real allocation/submission boundary without a global
    /// provider fault switch or a second execution algorithm.
    #[cfg(test)]
    pub(super) fn execute_ordered_mean_plan_with(
        &self,
        source: &GpuTensorHandle,
        plan: &OrderedMeanPlan,
        nan_mode: ProviderRealMeanNaNMode,
        checkpoint: impl FnMut(OrderedMeanCheckpoint) -> Result<()>,
    ) -> ProviderExecutionResult<(GpuTensorHandle, u64)> {
        self.execute_ordered_mean_plan_with_admission(source, plan, nan_mode, None, checkpoint)
    }

    fn execute_ordered_mean_plan_with_admission(
        &self,
        source: &GpuTensorHandle,
        plan: &OrderedMeanPlan,
        nan_mode: ProviderRealMeanNaNMode,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        mut checkpoint: impl FnMut(OrderedMeanCheckpoint) -> Result<()>,
    ) -> ProviderExecutionResult<(GpuTensorHandle, u64)> {
        self.require_ieee_words(crate::backend::wgpu::types::NumericPrecision::F64)
            .map_err(ProviderExecutionError::from)?;
        let entry = self
            .real_floating_entry(source)
            .map_err(ProviderExecutionError::from)?;
        let pipelines = self.ordered_real_mean.as_ref().ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!("ordered real mean pipeline unavailable"))
        })?;
        let mut completed_dispatches = 0usize;
        let emitted = self.with_emission(|emission, pending| -> Result<_> {
            let mut outputs: Vec<WgpuBufferOwner> = Vec::with_capacity(plan.steps().len());
            let mut uniforms: Vec<wgpu::Buffer> = Vec::with_capacity(plan.steps().len());
            for step in plan.steps() {
                let geometry = NdReductionGeometry::new(
                    step.input_shape(),
                    &[step.axis()],
                    "ordered real mean",
                )?;
                let input_precision = physical_precision(step.input_precision());
                let output_precision = physical_precision(step.output_precision());
                let pipeline = pipelines
                    .select(input_precision, output_precision)
                    .get(emission)?;
                let input_buffer = outputs.last().unwrap_or(&entry.buffer);
                ensure!(
                    input_buffer.size() >= step.input_bytes()
                        && geometry.out_shape.as_slice() == step.output_shape(),
                    "ordered mean input or output geometry changed"
                );
                let output = self.create_storage_buffer_checked_with_element_size(
                    emission,
                    geometry.cols,
                    "runmat-ordered-real-mean-output",
                    BufferUsageClass::Generic,
                    match output_precision {
                        crate::backend::wgpu::types::NumericPrecision::F32 => 4,
                        crate::backend::wgpu::types::NumericPrecision::F64 => 8,
                    },
                )?;
                checkpoint(OrderedMeanCheckpoint::OutputAllocated)?;
                let storage_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size);
                ensure!(
                    output.size() >= step.output_bytes()
                        && output.size() <= storage_limit
                        && input_buffer.size() <= storage_limit
                        && !std::ptr::eq(output.as_ref(), input_buffer.as_ref()),
                    "ordered mean output allocation is aliased or too small"
                );
                let uniform = self.uniform_buffer(
                    emission,
                    &params_for_axis(&geometry, nan_mode),
                    "runmat-ordered-real-mean-params",
                )?;
                ensure!(
                    uniform.size()
                        <= u64::from(self.adapter_limits.max_uniform_buffer_binding_size),
                    "ordered mean uniform exceeds the selected device limit"
                );
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-ordered-real-mean-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: output.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: uniform.as_entire_binding(),
                            },
                        ],
                    });
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &pipeline.pipeline,
                    &bind_group,
                    step.workgroups(),
                )?;
                completed_dispatches += 1;
                checkpoint(OrderedMeanCheckpoint::StageSubmitted)?;
                outputs.push(output);
                uniforms.push(uniform);
            }
            Ok((outputs, uniforms))
        });
        let (mut outputs, uniforms) = emitted
            .map_err(|error| {
                let error = ProviderExecutionError::from(error);
                if completed_dispatches > 0 {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                }
            })?
            .map_err(|error| {
                let error = ProviderExecutionError::from(error);
                if completed_dispatches > 0 {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                }
            })?;
        if usize::try_from(plan.queue_submissions()).ok() != Some(completed_dispatches) {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "ordered mean submitted work differs from selected physical route"
            ))
            .after_effect());
        }
        let observed_incremental_peak_bytes = outputs
            .iter()
            .map(|buffer| buffer.size())
            .chain(uniforms.iter().map(wgpu::Buffer::size))
            .try_fold(0u64, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow::anyhow!(
                    "ordered mean observed allocation byte overflow"
                ))
                .after_effect()
            })?;
        if observed_incremental_peak_bytes != plan.requested_incremental_peak_bytes() {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "ordered mean allocations differ from selected physical extents"
            ))
            .after_effect());
        }
        let final_step = plan.steps().last().ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!("ordered mean lost final stage"))
                .after_effect()
        })?;
        let output = outputs.pop().ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!("ordered mean produced no buffer"))
                .after_effect()
        })?;
        let len = final_step
            .output_shape()
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow::anyhow!("ordered mean output overflow"))
                    .after_effect()
            })?;
        let result = self
            .register_real_floating_output_admitted(
                output,
                final_step.output_shape().to_vec(),
                len,
                physical_precision(final_step.output_precision()),
                admission,
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        Ok((result, observed_incremental_peak_bytes))
    }
}
