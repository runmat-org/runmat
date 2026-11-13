//! Frozen physical requests for the two scoped elementwise generic kernels.
use super::super::backend_types::BufferEntry;
use super::super::pipeline_recipe::PipelineRecipe;
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use runmat_accelerate_api::{unsupported_provider_operation, NumericElementType};

pub(super) struct IntegerComparisonFrame {
    pub(super) lhs: BufferEntry,
    pub(super) rhs: BufferEntry,
    pub(super) broadcast: super::super::integer::IntegerBroadcastPlan,
    pub(super) integer_type: runmat_accelerate_api::IntegerElementType,
    pub(super) launch: FrozenLinearLaunch,
    pub(super) chunk_capacity: usize,
    pub(super) recipe: PipelineRecipe,
}

pub(super) struct RoundDigitsFrame {
    pub(super) input: BufferEntry,
    pub(super) storage: GpuTensorStorage,
    pub(super) scalar_len: usize,
    pub(super) allocated_bytes: u64,
    pub(super) launch: FrozenLinearLaunch,
    pub(super) chunk_capacity: usize,
    pub(super) recipe: PipelineRecipe,
    pub(super) significant: bool,
}

pub(crate) fn check_kernel_limits(
    operation: &str,
    limits: &wgpu::Limits,
    storage_bindings: usize,
    total_bindings: usize,
    uniform_bytes: u64,
    buffers: &[u64],
    launch: FrozenLinearLaunch,
) -> Result<usize> {
    validate_compute_binding_counts(operation, storage_bindings, total_bindings, limits)?;
    ensure!(
        limits.max_uniform_buffers_per_shader_stage >= 1
            && uniform_bytes <= u64::from(limits.max_uniform_buffer_binding_size),
        "{operation}: uniform binding exceeds enabled limits"
    );
    ensure!(
        launch.workgroup() <= limits.max_compute_workgroup_size_x
            && launch.workgroup() <= limits.max_compute_invocations_per_workgroup,
        "{operation}: selected workgroup exceeds enabled limits"
    );
    let max_bytes = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    ensure!(
        buffers.iter().all(|&bytes| bytes <= max_bytes),
        "{operation}: storage binding exceeds enabled limits"
    );
    launch
        .chunk_capacity()
        .ok_or_else(|| anyhow!("{operation}: selected dispatch capacity overflows"))
}

impl IntegerComparisonFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        operation_name: &str,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> Result<Self> {
        let lhs_entry = provider.get_entry_raw(lhs)?;
        let rhs_entry = provider.get_entry_raw(rhs)?;
        let integer_type = lhs_entry
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            rhs_entry.integer_type() == Some(integer_type),
            "{operation_name}: integer operands must have the same class"
        );
        ensure!(
            lhs_entry.storage == GpuTensorStorage::Real
                && rhs_entry.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray comparison is not supported"
        );
        let broadcast = super::super::integer::integer_broadcast_plan(
            operation_name,
            &lhs_entry.shape,
            lhs_entry.len,
            &rhs_entry.shape,
            rhs_entry.len,
        )?;
        let lhs_bytes = (lhs_entry.len as u64)
            .checked_mul(integer_type.element_size() as u64)
            .ok_or_else(|| anyhow!("{operation_name}: lhs byte extent overflows"))?;
        let rhs_bytes = (rhs_entry.len as u64)
            .checked_mul(integer_type.element_size() as u64)
            .ok_or_else(|| anyhow!("{operation_name}: rhs byte extent overflows"))?;
        ensure!(
            lhs_entry.allocated_bytes >= lhs_bytes && rhs_entry.allocated_bytes >= rhs_bytes,
            "{operation_name}: physical integer input extent is shorter than its class"
        );
        let len = broadcast.len;
        if len > u32::MAX as usize {
            return Err(unsupported_provider_operation(
                gpu_dispatch_length_limit_error(operation_name, len).to_string(),
            ));
        }
        let launch = provider.linear_launch;
        let output_bytes = (len as u64)
            .checked_mul(provider.element_size as u64)
            .ok_or_else(|| anyhow!("{operation_name}: output byte extent overflows"))?;
        let chunk_capacity = check_kernel_limits(
            operation_name,
            &provider.adapter_limits,
            3,
            4,
            std::mem::size_of::<super::IntegerComparisonParams>() as u64,
            &[
                lhs_entry.allocated_bytes,
                rhs_entry.allocated_bytes,
                output_bytes,
            ],
            launch,
        )
        .map_err(|error| unsupported_provider_operation(error.to_string()))?;
        let shader = crate::backend::wgpu::shaders::integer::comparison_shader(
            provider.precision.as_str(),
            launch.workgroup(),
        );
        Ok(Self {
            lhs: lhs_entry,
            rhs: rhs_entry,
            broadcast,
            integer_type,
            launch,
            chunk_capacity,
            recipe: PipelineRecipe {
                shader,
                layout: crate::backend::wgpu::bindings::INTEGER_COMPARE_LAYOUT_TAG.into(),
                workgroup: launch.workgroup(),
            },
        })
    }

    pub(super) fn require_live_inputs(
        &self,
        provider: &WgpuProvider,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> Result<()> {
        let live_lhs = provider.get_entry_raw(lhs)?;
        let live_rhs = provider.get_entry_raw(rhs)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live_lhs.buffer, &self.lhs.buffer)
                && runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                    &live_rhs.buffer,
                    &self.rhs.buffer,
                )
                && live_lhs.integer_type() == Some(self.integer_type)
                && live_rhs.integer_type() == Some(self.integer_type),
            "integer comparison: borrowed operands changed during preparation"
        );
        Ok(())
    }
}

impl RoundDigitsFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
        digits: i32,
        significant: bool,
    ) -> Result<Self> {
        if significant && digits <= 0 {
            return Err(unsupported_provider_operation(
                "round_digits: significant count must be positive",
            ));
        }
        let entry = provider.get_entry(input)?;
        let storage = runmat_accelerate_api::handle_storage(input);
        if provider.allocations.requires_logical_encoding(input)?
            || runmat_accelerate_api::handle_is_logical(input)
            || storage != entry.storage
            || !matches!(
                entry.element_type,
                NumericElementType::F32 | NumericElementType::F64
            )
        {
            return Err(unsupported_provider_operation(
                "round_digits: numeric floating input required",
            ));
        }
        // BufferEntry.len is the physical scalar-lane count even when the
        // handle's logical shape counts interleaved complex values.
        let scalar_len = entry.len;
        if scalar_len > u32::MAX as usize {
            return Err(unsupported_provider_operation(
                gpu_dispatch_length_limit_error("round_digits", scalar_len).to_string(),
            ));
        }
        if scalar_len != 0 {
            provider.require_ieee_words(entry.precision)?;
        }
        let allocated_bytes = (scalar_len as u64)
            .checked_mul(provider.element_size as u64)
            .ok_or_else(|| anyhow!("round_digits: output byte extent overflows"))?;
        ensure!(
            entry.allocated_bytes >= allocated_bytes,
            "round_digits: physical input extent is shorter than its scalar lanes"
        );
        let launch = provider.linear_launch;
        let chunk_capacity = check_kernel_limits(
            "round_digits",
            &provider.adapter_limits,
            2,
            3,
            std::mem::size_of::<crate::backend::wgpu::params::RoundDigitsParams>() as u64,
            &[entry.allocated_bytes, allocated_bytes],
            launch,
        )
        .map_err(|error| unsupported_provider_operation(error.to_string()))?;
        let shader = round_digits_shader(provider.precision, significant, launch.workgroup());
        Ok(Self {
            input: entry,
            storage,
            scalar_len,
            allocated_bytes,
            launch,
            chunk_capacity,
            recipe: PipelineRecipe {
                shader,
                layout: if significant {
                    crate::backend::wgpu::bindings::ROUND_SIGNIFICANT_LAYOUT_TAG
                } else {
                    crate::backend::wgpu::bindings::ROUND_DECIMALS_LAYOUT_TAG
                }
                .into(),
                workgroup: launch.workgroup(),
            },
            significant,
        })
    }

    pub(super) fn require_live_input(
        &self,
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
    ) -> Result<()> {
        let live = provider.get_entry(input)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &self.input.buffer)
                && !provider.allocations.requires_logical_encoding(input)?
                && live.element_type == self.input.element_type
                && live.storage == self.storage,
            "round_digits: borrowed input changed during preparation"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "scoped_generic/tests.rs"]
mod tests;
