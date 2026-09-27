//! Resident floating casts use target-typed storage, not provider-native unary allocation.
use super::*;
use runmat_accelerate_api::NumericElementType;

#[cfg(test)]
#[path = "floating_cast/tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    len: u32,
    offset: u32,
    padding: [u32; 2],
}

impl WgpuProvider {
    pub(crate) fn floating_cast_exec(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(input)?;
        // Casting is elementwise in physical order. Preserve the validated
        // logical view rather than materializing or reinterpreting its order.
        ensure!(
            input.shape == entry.shape,
            "floating cast view shape mismatch"
        );
        ensure!(
            product_checked(&entry.shape) == Some(entry.len),
            "floating cast allocation shape mismatch"
        );
        let transpose = runmat_accelerate_api::handle_transpose_info(input);
        if transpose.is_some() {
            ensure!(
                entry.shape.len() == 2,
                "floating cast transpose requires a matrix view"
            );
            super::backend_shared::build_matrix_operand_view(input, &entry)?;
        }
        ensure!(
            entry.storage == GpuTensorStorage::Real && entry.integer_type().is_none(),
            "floating cast requires real floating storage"
        );
        if self.precision == NumericPrecision::F32
            && (entry.precision == NumericPrecision::F64 || target == NumericPrecision::F64)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "floating cast requires shader-f64 support",
            ));
        }
        let target_type = match target {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        if entry.precision == target {
            let output = self.register_numeric_buffer(
                entry.buffer,
                NumericBufferRegistration {
                    shape: entry.shape,
                    len: entry.len,
                    physical_element_type: target_type,
                    storage: GpuTensorStorage::Real,
                    allocated_bytes: entry.allocated_bytes,
                    usage: entry.usage,
                },
            );
            return self.finish_floating_cast_view(output, transpose);
        }
        let len = entry.len;
        if len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error("floating cast", len));
        }
        let words_per_element = match target {
            NumericPrecision::F32 => 1,
            NumericPrecision::F64 => 2,
        };
        let words = len
            .checked_mul(words_per_element)
            .ok_or_else(|| anyhow!("floating cast allocation overflow"))?;
        let allocated_bytes = (words as u64)
            .checked_mul(4)
            .ok_or_else(|| anyhow!("floating cast byte count overflow"))?;
        // The existing checked raw-word allocator is independent of provider
        // float precision; typed registration below owns the interpretation.
        let output = self.create_integer_word_buffer(words, "runmat-floating-cast")?;
        if len != 0 {
            let workgroup_size = self.workgroup_config.scalar;
            let recipe = super::pipeline_recipe::PipelineRecipe::floating_cast(
                entry.precision,
                target,
                workgroup_size,
            );
            let bindings = self.cached_bind_group_layout(&recipe.layout, |device| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runmat-floating-cast-bgl"),
                    entries: &[
                        crate::backend::wgpu::bindings::storage_read_entry(0),
                        crate::backend::wgpu::bindings::storage_read_write_entry(1),
                        crate::backend::wgpu::bindings::uniform_entry(2),
                    ],
                })
            });
            let pipeline = self.recipe_pipeline(&recipe, &bindings, "runmat-floating-cast");
            let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup_size)
                .ok_or_else(|| anyhow!("floating cast has invalid workgroup geometry"))?
                as usize;
            let mut offset = 0;
            while offset < len {
                let count = (len - offset).min(capacity);
                let params = self.uniform_buffer(
                    &Params {
                        len: count as u32,
                        offset: offset as u32,
                        padding: [0; 2],
                    },
                    "runmat-floating-cast-params",
                );
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-floating-cast-bg"),
                        layout: &bindings,
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
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline,
                    &bind_group,
                    groups,
                );
                offset += count;
            }
        }
        let output = self.register_numeric_buffer(
            output,
            NumericBufferRegistration {
                shape: entry.shape,
                len,
                physical_element_type: target_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        );
        self.finish_floating_cast_view(output, transpose)
    }

    fn finish_floating_cast_view(
        &self,
        output: GpuTensorHandle,
        transpose: Option<runmat_accelerate_api::TransposeInfo>,
    ) -> Result<GpuTensorHandle> {
        if let Some(info) = transpose {
            runmat_accelerate_api::record_handle_transpose(&output, info.base_rows, info.base_cols);
            if runmat_accelerate_api::handle_transpose_info(&output) != Some(info) {
                // A fresh registration owns this output, including same-type
                // aliases. Never free the borrowed input registration.
                self.free(&output)?;
                return Err(anyhow!(
                    "floating cast could not publish transpose metadata"
                ));
            }
        }
        Ok(output)
    }
}
