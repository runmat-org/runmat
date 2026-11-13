//! One request's authenticated floating-cast source and physical route.
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use runmat_accelerate_api::{unsupported_provider_operation, WgpuBufferOwner};

pub(super) struct FloatingCastFrame {
    pub(super) entry: BufferEntry,
    pub(super) target: NumericPrecision,
    pub(super) transpose: Option<runmat_accelerate_api::TransposeInfo>,
    pub(super) logical_producer: bool,
    pub(super) scalar_len: usize,
    pub(super) words: usize,
    pub(super) allocated_bytes: u64,
    pub(super) launch: Option<FrozenLinearLaunch>,
    pub(super) chunk_capacity: Option<usize>,
    pub(super) recipe: Option<super::super::pipeline_recipe::PipelineRecipe>,
}

fn source_allowed(
    storage: GpuTensorStorage,
    element_type: NumericElementType,
    precision: NumericPrecision,
    descriptor_storage: Option<GpuTensorStorage>,
    logical_view: bool,
    logical_producer: bool,
) -> bool {
    matches!(
        storage,
        GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
    ) && element_type
        == match precision {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        }
        && descriptor_storage == Some(storage)
        && (!logical_view || (logical_producer && storage == GpuTensorStorage::Real))
}

fn selected_launch(
    needs_pipeline: bool,
    workgroup: u32,
    max_groups: u32,
) -> Result<Option<(FrozenLinearLaunch, usize)>> {
    if !needs_pipeline {
        return Ok(None);
    }
    let launch = FrozenLinearLaunch::new(workgroup, max_groups).ok_or_else(|| {
        unsupported_provider_operation("floating cast has no enabled dispatch geometry")
    })?;
    let capacity = launch.chunk_capacity().ok_or_else(|| {
        unsupported_provider_operation("floating cast dispatch capacity overflows")
    })?;
    Ok(Some((launch, capacity)))
}

fn limits_allow(
    scalar_len: usize,
    input_bytes: u64,
    output_bytes: u64,
    launch: FrozenLinearLaunch,
    capacity: usize,
    limits: &wgpu::Limits,
) -> bool {
    scalar_len <= u32::MAX as usize
        && launch.workgroup() <= limits.max_compute_workgroup_size_x
        && launch.workgroup() <= limits.max_compute_invocations_per_workgroup
        && limits.max_uniform_buffers_per_shader_stage >= 1
        && limits.max_uniform_buffer_binding_size >= std::mem::size_of::<Params>() as u32
        && [input_bytes, output_bytes].into_iter().all(|bytes| {
            bytes <= limits.max_buffer_size
                && bytes <= u64::from(limits.max_storage_buffer_binding_size)
        })
        && capacity != 0
        && launch.dispatch_groups(scalar_len.min(capacity)).is_some()
}

fn require_output_extent(bytes: u64, limits: &wgpu::Limits) -> Result<()> {
    if bytes <= limits.max_buffer_size {
        Ok(())
    } else {
        Err(unsupported_provider_operation(
            "floating cast output exceeds enabled buffer extent",
        ))
    }
}

fn require_pipeline_limits(
    scalar_len: usize,
    input_bytes: u64,
    output_bytes: u64,
    launch: FrozenLinearLaunch,
    capacity: usize,
    limits: &wgpu::Limits,
) -> Result<()> {
    validate_compute_binding_counts("floating cast", 2, 3, limits)
        .map_err(|error| unsupported_provider_operation(error.to_string()))?;
    if limits_allow(
        scalar_len,
        input_bytes,
        output_bytes,
        launch,
        capacity,
        limits,
    ) {
        Ok(())
    } else {
        Err(unsupported_provider_operation(
            "floating cast selected physical route exceeds enabled limits",
        ))
    }
}

impl FloatingCastFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> Result<Self> {
        input.allocation_lease().validate_live()?;
        let entry = provider.get_entry_raw(input)?;
        let lanes = match entry.storage {
            GpuTensorStorage::Real => 1,
            GpuTensorStorage::ComplexInterleaved => 2,
        };
        // BufferEntry counts native scalar lanes; shape counts logical values.
        // A complex value owns two lanes, not two independent logical values.
        let scalar_len = product_checked(&entry.shape)
            .and_then(|logical_len| logical_len.checked_mul(lanes))
            .ok_or_else(|| anyhow!("floating cast scalar count overflow"))?;
        ensure!(
            input.shape() == entry.shape && scalar_len == entry.len,
            "floating cast source view does not match its allocation"
        );
        let transpose = runmat_accelerate_api::handle_transpose_info(input);
        if transpose.is_some() {
            ensure!(
                entry.shape.len() == 2,
                "floating cast transpose requires a matrix view"
            );
            super::super::backend_shared::build_matrix_operand_view(input, &entry)?;
        }
        let logical_producer = provider.allocations.requires_logical_encoding(input)?;
        ensure!(
            source_allowed(
                entry.storage,
                entry.element_type,
                entry.precision,
                input.descriptor().storage,
                runmat_accelerate_api::handle_is_logical(input),
                logical_producer
            ),
            "floating cast requires authentic real or interleaved complex floating storage"
        );
        if provider.precision == NumericPrecision::F32
            && (entry.precision == NumericPrecision::F64 || target == NumericPrecision::F64)
        {
            return Err(unsupported_provider_operation(
                "floating cast requires shader-f64 support",
            ));
        }
        let needs_pipeline = entry.precision != target && scalar_len != 0;
        if needs_pipeline && scalar_len > u32::MAX as usize {
            return Err(unsupported_provider_operation(
                "floating cast scalar count exceeds its u32 uniform ABI",
            ));
        }
        let input_bytes = (scalar_len as u64)
            .checked_mul(if entry.precision == NumericPrecision::F64 {
                8
            } else {
                4
            })
            .ok_or_else(|| anyhow!("floating cast input byte count overflow"))?;
        ensure!(
            entry.allocated_bytes >= input_bytes && entry.buffer.size() >= input_bytes,
            "floating cast source extent is shorter than its physical precision"
        );
        let words = scalar_len
            .checked_mul(if target == NumericPrecision::F64 {
                2
            } else {
                1
            })
            .ok_or_else(|| anyhow!("floating cast output word count overflow"))?;
        let allocated_bytes = crate::backend::wgpu_storage::raw_word_requested_bytes(words as u64)
            .ok_or_else(|| anyhow!("floating cast output byte count overflow"))?;
        let selected = selected_launch(
            needs_pipeline,
            provider.workgroup_config.scalar,
            provider.adapter_limits.max_compute_workgroups_per_dimension,
        )?;
        if entry.precision != target || logical_producer {
            require_output_extent(allocated_bytes, &provider.adapter_limits)?;
        }
        if needs_pipeline {
            let limits = &provider.adapter_limits;
            let (launch, capacity) = selected.expect("selected dispatch geometry");
            require_pipeline_limits(
                scalar_len,
                input_bytes,
                allocated_bytes,
                launch,
                capacity,
                limits,
            )?;
        }
        let recipe = selected.map(|(launch, _)| {
            super::super::pipeline_recipe::PipelineRecipe::floating_cast(
                entry.precision,
                target,
                launch.workgroup(),
            )
        });
        Ok(Self {
            entry,
            target,
            transpose,
            logical_producer,
            scalar_len,
            words,
            allocated_bytes,
            launch: selected.map(|(launch, _)| launch),
            chunk_capacity: selected.map(|(_, capacity)| capacity),
            recipe,
        })
    }

    pub(super) fn require_live(
        &self,
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
    ) -> Result<()> {
        input.allocation_lease().validate_live()?;
        let live = provider.get_entry_raw(input)?;
        ensure!(
            WgpuBufferOwner::ptr_eq(&live.buffer, &self.entry.buffer)
                && live.element_type == self.entry.element_type
                && live.precision == self.entry.precision
                && live.storage == self.entry.storage
                && live.shape == self.entry.shape
                && live.len == self.entry.len
                && live.allocated_bytes == self.entry.allocated_bytes
                && input.shape() == self.entry.shape
                && provider.allocations.requires_logical_encoding(input)? == self.logical_producer,
            "floating cast source changed while preparing its pipeline"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "frame/tests.rs"]
mod tests;
