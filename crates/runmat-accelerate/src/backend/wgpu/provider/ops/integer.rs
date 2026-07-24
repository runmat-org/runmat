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

pub(crate) struct IntegerBroadcastPlan {
    pub output_shape: Vec<usize>,
    pub len: usize,
    pub rank: u32,
    pub out_shape:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    pub a_shape:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    pub b_shape:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    pub a_strides:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    pub b_strides:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
}

pub(crate) fn integer_broadcast_plan(
    operation_name: &str,
    input_shape_a: &[usize],
    input_len_a: usize,
    input_shape_b: &[usize],
    input_len_b: usize,
) -> Result<IntegerBroadcastPlan> {
    use crate::backend::wgpu::params::{AlignedU32, BCAST_MAX_RANK};

    let same_shape = input_shape_a == input_shape_b;
    let rank = input_shape_a.len().max(input_shape_b.len());
    ensure!(
        rank <= BCAST_MAX_RANK,
        "{operation_name}: broadcast rank exceeds limit"
    );
    let mut shape_a = vec![1usize; rank - input_shape_a.len()];
    shape_a.extend_from_slice(input_shape_a);
    let mut shape_b = vec![1usize; rank - input_shape_b.len()];
    shape_b.extend_from_slice(input_shape_b);
    let mut output_shape = vec![1usize; rank];
    for index in 0..rank {
        output_shape[index] = match (shape_a[index], shape_b[index]) {
            (left, right) if left == right => left,
            (1, right) => right,
            (left, 1) => left,
            _ => return Err(anyhow!("{operation_name}: shape mismatch between inputs")),
        };
    }
    let checked_product = |shape: &[usize], description: &str| {
        shape.iter().try_fold(1usize, |product, &dimension| {
            product
                .checked_mul(dimension)
                .ok_or_else(|| anyhow!("{operation_name}: {description} length overflow"))
        })
    };
    let len = checked_product(&output_shape, "broadcast output")?;
    ensure!(
        input_len_a == checked_product(&shape_a, "broadcast lhs")?,
        "{operation_name}: logical lhs length mismatch"
    );
    ensure!(
        input_len_b == checked_product(&shape_b, "broadcast rhs")?,
        "{operation_name}: logical rhs length mismatch"
    );
    let mut a_strides = [AlignedU32::new(0); BCAST_MAX_RANK];
    let mut b_strides = [AlignedU32::new(0); BCAST_MAX_RANK];
    let mut a_stride = 1usize;
    let mut b_stride = 1usize;
    for index in 0..rank {
        a_strides[index] = AlignedU32::new(
            u32::try_from(if shape_a[index] == 1 { 0 } else { a_stride })
                .map_err(|_| anyhow!("{operation_name}: lhs broadcast stride exceeds u32"))?,
        );
        b_strides[index] = AlignedU32::new(
            u32::try_from(if shape_b[index] == 1 { 0 } else { b_stride })
                .map_err(|_| anyhow!("{operation_name}: rhs broadcast stride exceeds u32"))?,
        );
        a_stride = a_stride
            .checked_mul(shape_a[index])
            .ok_or_else(|| anyhow!("{operation_name}: lhs broadcast stride overflow"))?;
        b_stride = b_stride
            .checked_mul(shape_b[index])
            .ok_or_else(|| anyhow!("{operation_name}: rhs broadcast stride overflow"))?;
    }
    let mut out_shape = [AlignedU32::new(0); BCAST_MAX_RANK];
    let mut a_shape = [AlignedU32::new(0); BCAST_MAX_RANK];
    let mut b_shape = [AlignedU32::new(0); BCAST_MAX_RANK];
    for index in 0..rank {
        out_shape[index] = AlignedU32::new(
            u32::try_from(output_shape[index])
                .map_err(|_| anyhow!("{operation_name}: broadcast dimension exceeds u32"))?,
        );
        a_shape[index] = AlignedU32::new(
            u32::try_from(shape_a[index])
                .map_err(|_| anyhow!("{operation_name}: lhs broadcast dimension exceeds u32"))?,
        );
        b_shape[index] = AlignedU32::new(
            u32::try_from(shape_b[index])
                .map_err(|_| anyhow!("{operation_name}: rhs broadcast dimension exceeds u32"))?,
        );
    }
    Ok(IntegerBroadcastPlan {
        output_shape,
        len,
        rank: if same_shape { 0 } else { rank as u32 },
        out_shape,
        a_shape,
        b_shape,
        a_strides,
        b_strides,
    })
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
        use crate::backend::wgpu::params::{AlignedU32, BCAST_MAX_RANK};

        let same_shape = entry_a.shape == entry_b.shape;
        let mut shape_a = entry_a.shape.clone();
        let mut shape_b = entry_b.shape.clone();
        let rank = shape_a.len().max(shape_b.len());
        ensure!(
            rank <= BCAST_MAX_RANK,
            "{operation_name}: broadcast rank exceeds limit"
        );
        if shape_a.len() < rank {
            let mut padded = vec![1usize; rank - shape_a.len()];
            padded.extend_from_slice(&shape_a);
            shape_a = padded;
        }
        if shape_b.len() < rank {
            let mut padded = vec![1usize; rank - shape_b.len()];
            padded.extend_from_slice(&shape_b);
            shape_b = padded;
        }
        let mut output_shape = vec![1usize; rank];
        for index in 0..rank {
            output_shape[index] = match (shape_a[index], shape_b[index]) {
                (left, right) if left == right => left,
                (1, right) => right,
                (left, 1) => left,
                _ => return Err(anyhow!("{operation_name}: shape mismatch between inputs")),
            };
        }
        let len = output_shape.iter().try_fold(1usize, |product, &dim| {
            product
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("{operation_name}: broadcast output length overflow"))
        })?;
        let a_len = shape_a.iter().try_fold(1usize, |product, &dim| {
            product
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("{operation_name}: broadcast lhs length overflow"))
        })?;
        let b_len = shape_b.iter().try_fold(1usize, |product, &dim| {
            product
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("{operation_name}: broadcast rhs length overflow"))
        })?;
        ensure!(
            entry_a.len == a_len,
            "{operation_name}: logical lhs length mismatch"
        );
        ensure!(
            entry_b.len == b_len,
            "{operation_name}: logical rhs length mismatch"
        );
        let raw_len = integer_word_count(integer_type, len)?;
        let bytes = (raw_len as u64).saturating_mul(4);
        if len == 0 {
            return Ok(self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-arithmetic-empty"),
                output_shape,
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
            rank: u32,
            _pad0: u32,
            _pad1: u32,
            out_shape: [AlignedU32; BCAST_MAX_RANK],
            a_shape: [AlignedU32; BCAST_MAX_RANK],
            b_shape: [AlignedU32; BCAST_MAX_RANK],
            a_strides: [AlignedU32; BCAST_MAX_RANK],
            b_strides: [AlignedU32; BCAST_MAX_RANK],
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
        let mut a_strides = [AlignedU32::new(0); BCAST_MAX_RANK];
        let mut b_strides = [AlignedU32::new(0); BCAST_MAX_RANK];
        let mut a_stride = 1usize;
        let mut b_stride = 1usize;
        for index in 0..rank {
            a_strides[index] = AlignedU32::new(
                u32::try_from(if shape_a[index] == 1 { 0 } else { a_stride })
                    .map_err(|_| anyhow!("{operation_name}: lhs broadcast stride exceeds u32"))?,
            );
            b_strides[index] = AlignedU32::new(
                u32::try_from(if shape_b[index] == 1 { 0 } else { b_stride })
                    .map_err(|_| anyhow!("{operation_name}: rhs broadcast stride exceeds u32"))?,
            );
            a_stride = a_stride
                .checked_mul(shape_a[index])
                .ok_or_else(|| anyhow!("{operation_name}: lhs broadcast stride overflow"))?;
            b_stride = b_stride
                .checked_mul(shape_b[index])
                .ok_or_else(|| anyhow!("{operation_name}: rhs broadcast stride overflow"))?;
        }
        let mut out_shape_params = [AlignedU32::new(0); BCAST_MAX_RANK];
        let mut a_shape_params = [AlignedU32::new(0); BCAST_MAX_RANK];
        let mut b_shape_params = [AlignedU32::new(0); BCAST_MAX_RANK];
        for index in 0..rank {
            out_shape_params[index] = AlignedU32::new(
                u32::try_from(output_shape[index])
                    .map_err(|_| anyhow!("{operation_name}: broadcast dimension exceeds u32"))?,
            );
            a_shape_params[index] =
                AlignedU32::new(u32::try_from(shape_a[index]).map_err(|_| {
                    anyhow!("{operation_name}: lhs broadcast dimension exceeds u32")
                })?);
            b_shape_params[index] =
                AlignedU32::new(u32::try_from(shape_b[index]).map_err(|_| {
                    anyhow!("{operation_name}: rhs broadcast dimension exceeds u32")
                })?);
        }
        while offset < len {
            let chunk = (len - offset).min(capacity);
            let params = Params {
                len: chunk as u32,
                op,
                offset: offset as u32,
                total: u32::try_from(len).expect("integer dispatch length was checked"),
                integer_type: integer_type_code(integer_type),
                rank: if same_shape { 0 } else { rank as u32 },
                _pad0: 0,
                _pad1: 0,
                out_shape: out_shape_params,
                a_shape: a_shape_params,
                b_shape: b_shape_params,
                a_strides,
                b_strides,
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
        Ok(self.register_integer_buffer(out, output_shape, len, integer_type, bytes))
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
        let broadcast = integer_broadcast_plan(
            operation_name,
            &entry_a.shape,
            entry_a.len,
            &entry_b.shape,
            entry_b.len,
        )?;
        let len = broadcast.len;
        let raw_len = integer_word_count(integer_type, len)?;
        let allocated_bytes = (raw_len as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-integer-minmax-empty");
            return Ok(self.register_integer_buffer(
                out,
                broadcast.output_shape,
                0,
                integer_type,
                0,
            ));
        }
        if len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(operation_name, len));
        }
        use crate::backend::wgpu::params::{AlignedU32, BCAST_MAX_RANK};

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            select_min: u32,
            offset: u32,
            total: u32,
            integer_type: u32,
            rank: u32,
            _pad0: u32,
            _pad1: u32,
            out_shape: [AlignedU32; BCAST_MAX_RANK],
            a_shape: [AlignedU32; BCAST_MAX_RANK],
            b_shape: [AlignedU32; BCAST_MAX_RANK],
            a_strides: [AlignedU32; BCAST_MAX_RANK],
            b_strides: [AlignedU32; BCAST_MAX_RANK],
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
                rank: broadcast.rank,
                _pad0: 0,
                _pad1: 0,
                out_shape: broadcast.out_shape,
                a_shape: broadcast.a_shape,
                b_shape: broadcast.b_shape,
                a_strides: broadcast.a_strides,
                b_strides: broadcast.b_strides,
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
            broadcast.output_shape,
            len,
            integer_type,
            allocated_bytes,
        ))
    }
}
