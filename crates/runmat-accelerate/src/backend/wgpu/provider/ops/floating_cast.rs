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
            return Ok(self.register_numeric_buffer(
                entry.buffer,
                NumericBufferRegistration {
                    shape: entry.shape,
                    len: entry.len,
                    physical_element_type: target_type,
                    storage: GpuTensorStorage::Real,
                    allocated_bytes: entry.allocated_bytes,
                    usage: entry.usage,
                },
            ));
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
            let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
            let source_type = match entry.precision {
                NumericPrecision::F32 => "f32",
                NumericPrecision::F64 => "f64",
            };
            let target_type = match target {
                NumericPrecision::F32 => "f32",
                NumericPrecision::F64 => "f64",
            };
            let shader = format!(
                "struct Params {{ len: u32, offset: u32, padding: vec2<u32> }};
                 @group(0) @binding(0) var<storage, read> input: array<{source_type}>;
                 @group(0) @binding(1) var<storage, read_write> output: array<{target_type}>;
                 @group(0) @binding(2) var<uniform> params: Params;
                 @compute @workgroup_size({workgroup_size})
                 fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                     if (id.x >= params.len) {{ return; }}
                     let index = params.offset + id.x;
                     output[index] = {target_type}(input[index]);
                 }}"
            );
            let bindings =
                self.device_ref()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-floating-cast-bgl"),
                        entries: &[
                            crate::backend::wgpu::bindings::storage_read_entry(0),
                            crate::backend::wgpu::bindings::storage_read_write_entry(1),
                            crate::backend::wgpu::bindings::uniform_entry(2),
                        ],
                    });
            let layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
                self.device_ref(),
                "runmat-floating-cast-pl",
                &bindings,
            );
            let module = crate::backend::wgpu::pipelines::create_shader_module(
                self.device_ref(),
                "runmat-floating-cast-module",
                &shader,
            );
            let key = self.compute_pipeline_hash_bytes(
                shader.as_bytes(),
                "runmat-floating-cast-bgl",
                Some(workgroup_size),
            );
            let pipeline = self.get_or_create_pipeline(
                key,
                &layout,
                &module,
                "runmat-floating-cast",
                Some(shader.as_bytes()),
                Some("runmat-floating-cast-bgl"),
                Some(workgroup_size),
            );
            let capacity = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize
                * workgroup_size as usize;
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
        Ok(self.register_numeric_buffer(
            output,
            NumericBufferRegistration {
                shape: entry.shape,
                len,
                physical_element_type: target_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        ))
    }
}
