//! Frozen physical admission for a resident integer cast.
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use runmat_accelerate_api::{unsupported_provider_operation, NumericElementType, WgpuBufferOwner};

pub(super) struct IntegerCastFrame {
    pub(super) entry: BufferEntry,
    pub(super) target: IntegerElementType,
    pub(super) source_type: u32,
    pub(super) source_precision: NumericPrecision,
    pub(super) logical_producer: bool,
    pub(super) raw_words: usize,
    pub(super) output_bytes: u64,
    pub(super) launch: Option<FrozenLinearLaunch>,
    pub(super) chunk_capacity: Option<usize>,
    pub(super) recipe: Option<super::super::pipeline_recipe::PipelineRecipe>,
}

fn logical_source_allowed(logical_view: bool, logical_producer: bool) -> bool {
    !logical_view || logical_producer
}

fn limits_allow(
    len: usize,
    input_bytes: u64,
    output_bytes: u64,
    launch: Option<FrozenLinearLaunch>,
    limits: &wgpu::Limits,
) -> bool {
    if output_bytes > limits.max_buffer_size {
        return false;
    }
    if len == 0 {
        return true;
    }
    let Some(launch) = launch else {
        return false;
    };
    let Some(capacity) = launch.chunk_capacity() else {
        return false;
    };
    len <= u32::MAX as usize
        && launch.workgroup() <= limits.max_compute_workgroup_size_x
        && launch.workgroup() <= limits.max_compute_invocations_per_workgroup
        && limits.max_uniform_buffers_per_shader_stage >= 1
        && limits.max_uniform_buffer_binding_size >= 16
        && [input_bytes, output_bytes].into_iter().all(|bytes| {
            bytes <= limits.max_buffer_size
                && bytes <= u64::from(limits.max_storage_buffer_binding_size)
        })
        && capacity != 0
        && launch.dispatch_groups(len.min(capacity)).is_some()
}

fn require_selected_limits(
    len: usize,
    input_bytes: u64,
    output_bytes: u64,
    launch: Option<FrozenLinearLaunch>,
    limits: &wgpu::Limits,
) -> Result<()> {
    if len != 0 {
        validate_compute_binding_counts("integer cast", 2, 3, limits)
            .map_err(|error| unsupported_provider_operation(error.to_string()))?;
    }
    if limits_allow(len, input_bytes, output_bytes, launch, limits) {
        Ok(())
    } else {
        Err(unsupported_provider_operation(
            "integer cast selected physical route exceeds enabled limits",
        ))
    }
}

impl IntegerCastFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
        target: IntegerElementType,
    ) -> Result<Self> {
        input.allocation_lease().validate_live()?;
        let entry = provider.get_entry_raw(input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && input.descriptor().storage == Some(entry.storage)
                && input.shape() == entry.shape,
            "integer cast requires an authenticated real source view"
        );
        ensure!(
            product_checked(&entry.shape) == Some(entry.len),
            "integer cast source shape does not match its allocation"
        );
        let logical_producer = provider.allocations.requires_logical_encoding(input)?;
        ensure!(
            logical_source_allowed(
                runmat_accelerate_api::handle_is_logical(input),
                logical_producer
            ),
            "integer cast logical view has no physical producer"
        );
        let source_precision = if entry.integer_type().is_some() {
            NumericPrecision::F32 // Word arithmetic does not read floating lanes.
        } else {
            ensure!(
                matches!(
                    entry.element_type,
                    NumericElementType::F32 | NumericElementType::F64
                ),
                "integer cast requires native integer or real floating storage"
            );
            ensure!(
                entry.element_type
                    == match entry.precision {
                        NumericPrecision::F32 => NumericElementType::F32,
                        NumericPrecision::F64 => NumericElementType::F64,
                    },
                "integer cast physical class and precision disagree"
            );
            if entry.precision == NumericPrecision::F64
                && provider.precision == NumericPrecision::F32
            {
                return Err(unsupported_provider_operation(
                    "integer cast requires shader-f64 support",
                ));
            }
            entry.precision
        };
        if entry.len != 0 {
            provider.require_ieee_words(source_precision)?;
        }
        let input_words = match entry.integer_type() {
            Some(kind) => integer_word_count(kind, entry.len)?,
            None => entry
                .len
                .checked_mul(if entry.precision == NumericPrecision::F64 {
                    2
                } else {
                    1
                })
                .ok_or_else(|| anyhow!("integer cast source word count overflow"))?,
        };
        let input_bytes = u64::try_from(input_words)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| anyhow!("integer cast source byte count overflow"))?;
        ensure!(
            entry.allocated_bytes >= input_bytes && entry.buffer.size() >= input_bytes,
            "integer cast source storage is shorter than its physical class"
        );
        let raw_words = integer_word_count(target, entry.len)?;
        let output_bytes = crate::backend::wgpu_storage::raw_word_requested_bytes(raw_words as u64)
            .ok_or_else(|| anyhow!("integer cast output byte count overflow"))?;
        let source_type = match entry.integer_type() {
            Some(kind) => integer_type_code(kind),
            None => {
                if source_precision == NumericPrecision::F64 {
                    9
                } else {
                    8
                }
            }
        };
        let launch = if entry.len == 0 {
            None
        } else {
            Some(
                FrozenLinearLaunch::new(
                    provider.workgroup_config.scalar,
                    provider.adapter_limits.max_compute_workgroups_per_dimension,
                )
                .ok_or_else(|| {
                    unsupported_provider_operation("integer cast has no enabled dispatch geometry")
                })?,
            )
        };
        let chunk_capacity = launch
            .map(|launch| {
                launch.chunk_capacity().ok_or_else(|| {
                    unsupported_provider_operation("integer cast dispatch capacity overflows")
                })
            })
            .transpose()?;
        require_selected_limits(
            entry.len,
            input_bytes,
            output_bytes,
            launch,
            &provider.adapter_limits,
        )?;
        Ok(Self {
            entry,
            target,
            source_type,
            source_precision,
            logical_producer,
            raw_words,
            output_bytes,
            launch,
            chunk_capacity,
            recipe: launch.map(|launch| {
                super::super::pipeline_recipe::PipelineRecipe::integer_cast(
                    source_precision,
                    launch.workgroup(),
                )
            }),
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
                && live.storage == self.entry.storage
                && live.shape == self.entry.shape
                && live.len == self.entry.len
                && live.allocated_bytes == self.entry.allocated_bytes
                && live.precision == self.entry.precision
                && input.shape() == self.entry.shape,
            "integer cast source changed while preparing its pipeline"
        );
        ensure!(
            provider.allocations.requires_logical_encoding(input)? == self.logical_producer,
            "integer cast logical producer changed while preparing its pipeline"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "cast_frame/tests.rs"]
mod tests;
