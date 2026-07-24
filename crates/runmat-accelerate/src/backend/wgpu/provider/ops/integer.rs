use super::*;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, IntegerElementType};

pub(crate) fn integer_type_code(element_type: IntegerElementType) -> u32 {
    match element_type {
        IntegerElementType::I8 => 0,
        IntegerElementType::I16 => 1,
        IntegerElementType::I32 => 2,
        IntegerElementType::I64 => 3,
        IntegerElementType::U8 => 4,
        IntegerElementType::U16 => 5,
        IntegerElementType::U32 => 6,
        IntegerElementType::U64 => 7,
    }
}

fn integer_word_count(element_type: IntegerElementType, len: usize) -> Result<usize> {
    let lanes = match element_type {
        IntegerElementType::I64 | IntegerElementType::U64 => 2,
        _ => 1,
    };
    len.checked_mul(lanes)
        .ok_or_else(|| anyhow!("integer gpuArray word count overflow"))
}

impl WgpuProvider {
    pub(crate) fn integer_arithmetic_exec(
        &self,
        op: u32,
        operation_name: &str,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            op <= 2,
            "{operation_name}: unsupported integer arithmetic opcode"
        );
        let entry_a = self.get_entry_raw(a)?;
        let entry_b = self.get_entry_raw(b)?;
        let integer_type = entry_a
            .integer_type
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry_b.integer_type == Some(integer_type),
            "{operation_name}: integer operands must have the same class"
        );
        ensure!(
            entry_a.storage == GpuTensorStorage::Real && entry_b.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray arithmetic is not supported"
        );
        ensure!(
            entry_a.shape == entry_b.shape,
            "{operation_name}: shape mismatch between inputs"
        );
        ensure!(
            entry_a.len == entry_b.len,
            "{operation_name}: logical element count mismatch between inputs"
        );
        let len = entry_a.len;
        let raw_len = integer_word_count(integer_type, len)?;
        let bytes = (raw_len as u64).saturating_mul(4);
        if len == 0 {
            return Ok(self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-arithmetic-empty"),
                entry_a.shape,
                0,
                integer_type,
                0,
            ));
        }
        if len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(operation_name, len));
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            op: u32,
            offset: u32,
            total: u32,
            integer_type: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
        let shader = crate::backend::wgpu::shaders::integer::arithmetic_shader(workgroup_size);
        let layout = self
            .device_ref()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-integer-arithmetic-bgl"),
                entries: &[
                    crate::backend::wgpu::bindings::storage_read_entry(0),
                    crate::backend::wgpu::bindings::storage_read_entry(1),
                    crate::backend::wgpu::bindings::storage_read_write_entry(2),
                    crate::backend::wgpu::bindings::uniform_entry(3),
                ],
            });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-arithmetic-pl",
            &layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-arithmetic-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-arithmetic-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-arithmetic",
            Some(shader.as_bytes()),
            Some("runmat-integer-arithmetic-bgl"),
            Some(workgroup_size),
        );
        let out = self.create_storage_buffer_checked(raw_len, "runmat-integer-arithmetic-out")?;
        let capacity = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize
            * workgroup_size as usize;
        let mut offset = 0;
        while offset < len {
            let chunk = (len - offset).min(capacity);
            let params = Params {
                len: chunk as u32,
                op,
                offset: offset as u32,
                total: len as u32,
                integer_type: integer_type_code(integer_type),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let uniform = self.uniform_buffer(&params, "runmat-integer-arithmetic-params");
            let group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-integer-arithmetic-bg"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_b.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
            let groups =
                crate::backend::wgpu::dispatch::common::dispatch_size(chunk as u32, workgroup_size);
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &group,
                groups,
            );
            offset += chunk;
        }
        Ok(self.register_integer_buffer(out, entry_a.shape, len, integer_type, bytes))
    }

    pub(crate) fn integer_minmax_exec(
        &self,
        select_min: bool,
        operation_name: &str,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry_raw(a)?;
        let entry_b = self.get_entry_raw(b)?;
        let integer_type = entry_a
            .integer_type
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry_b.integer_type == Some(integer_type),
            "{operation_name}: integer operands must have the same class"
        );
        ensure!(
            entry_a.storage == GpuTensorStorage::Real && entry_b.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray selection is not supported"
        );
        ensure!(
            entry_a.shape == entry_b.shape,
            "{operation_name}: shape mismatch between inputs"
        );
        ensure!(
            entry_a.len == entry_b.len,
            "{operation_name}: logical element count mismatch between inputs"
        );
        let len = entry_a.len;
        let raw_len = integer_word_count(integer_type, len)?;
        let allocated_bytes = (raw_len as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-integer-minmax-empty");
            return Ok(self.register_integer_buffer(out, entry_a.shape, 0, integer_type, 0));
        }
        if len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(operation_name, len));
        }

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            select_min: u32,
            offset: u32,
            total: u32,
            integer_type: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
        let shader = crate::backend::wgpu::shaders::integer::minmax_shader(workgroup_size);
        let bind_group_layout =
            self.device_ref()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runmat-integer-minmax-bgl"),
                    entries: &[
                        crate::backend::wgpu::bindings::storage_read_entry(0),
                        crate::backend::wgpu::bindings::storage_read_entry(1),
                        crate::backend::wgpu::bindings::storage_read_write_entry(2),
                        crate::backend::wgpu::bindings::uniform_entry(3),
                    ],
                });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-minmax-pl",
            &bind_group_layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-minmax-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-minmax-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-minmax",
            Some(shader.as_bytes()),
            Some("runmat-integer-minmax-bgl"),
            Some(workgroup_size),
        );
        let out_buffer =
            self.create_storage_buffer_checked(raw_len, "runmat-integer-minmax-out")?;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * workgroup_size as usize;
        let mut offset = 0usize;
        while offset < len {
            let chunk_len = (len - offset).min(chunk_capacity);
            let params = Params {
                len: chunk_len as u32,
                select_min: u32::from(select_min),
                offset: offset as u32,
                total: len as u32,
                integer_type: integer_type_code(integer_type),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-integer-minmax-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-integer-minmax-bg"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_b.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                workgroup_size,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        Ok(self.register_integer_buffer(
            out_buffer,
            entry_a.shape,
            len,
            integer_type,
            allocated_bytes,
        ))
    }
}
