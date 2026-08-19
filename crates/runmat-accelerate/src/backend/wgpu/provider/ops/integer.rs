use super::*;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostIntegerDataView, HostIntegerTensorView,
    IntegerElementType,
};

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

pub(crate) fn identity_integer_buffer(
    provider: &WgpuProvider,
    element_type: IntegerElementType,
    len: usize,
    shape: &[usize],
    is_product: bool,
) -> Result<GpuTensorHandle> {
    match element_type {
        IntegerElementType::I8 => {
            let data = vec![if is_product { 1_i8 } else { 0_i8 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I8(&data),
                shape,
            })
        }
        IntegerElementType::I16 => {
            let data = vec![if is_product { 1_i16 } else { 0_i16 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I16(&data),
                shape,
            })
        }
        IntegerElementType::I32 => {
            let data = vec![if is_product { 1_i32 } else { 0_i32 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I32(&data),
                shape,
            })
        }
        IntegerElementType::I64 => {
            let data = vec![if is_product { 1_i64 } else { 0_i64 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&data),
                shape,
            })
        }
        IntegerElementType::U8 => {
            let data = vec![if is_product { 1_u8 } else { 0_u8 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&data),
                shape,
            })
        }
        IntegerElementType::U16 => {
            let data = vec![if is_product { 1_u16 } else { 0_u16 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&data),
                shape,
            })
        }
        IntegerElementType::U32 => {
            let data = vec![if is_product { 1_u32 } else { 0_u32 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U32(&data),
                shape,
            })
        }
        IntegerElementType::U64 => {
            let data = vec![if is_product { 1_u64 } else { 0_u64 }; len];
            provider.upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&data),
                shape,
            })
        }
    }
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
    let (shape_a, shape_b, output_shape) = super::backend_shared::matlab_broadcast_shapes(
        operation_name,
        input_shape_a,
        input_shape_b,
        BCAST_MAX_RANK,
    )?;
    let rank = output_shape.len();
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
    pub(crate) fn integer_cumulative_scan_exec(
        &self,
        op: u32,
        operation_name: &str,
        a: &GpuTensorHandle,
        dim: usize,
        direction: runmat_accelerate_api::ProviderScanDirection,
    ) -> Result<(GpuTensorHandle, Option<GpuTensorHandle>)> {
        let entry = self.get_entry_raw(a)?;
        let integer_type = entry
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray scan is not supported"
        );
        let rank = entry.shape.len();
        ensure!(rank > 0, "{operation_name}: rank must be greater than zero");
        ensure!(dim < rank, "{operation_name}: dimension out of bounds");
        ensure!(op <= 3, "{operation_name}: unsupported integer scan opcode");
        if entry.len == 0 {
            let values = self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-cumulative-empty"),
                entry.shape.clone(),
                0,
                integer_type,
                0,
            );
            let indices = if op >= 2 {
                Some(self.register_existing_buffer(
                    self.create_storage_buffer(0, "runmat-integer-cumulative-indices-empty"),
                    entry.shape.clone(),
                    0,
                ))
            } else {
                None
            };
            return Ok((values, indices));
        }

        let segment_len = entry.shape[dim];
        if segment_len == 0 {
            let values = self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-cumulative-empty"),
                entry.shape.clone(),
                0,
                integer_type,
                0,
            );
            let indices = if op >= 2 {
                Some(self.register_existing_buffer(
                    self.create_storage_buffer(0, "runmat-integer-cumulative-indices-empty"),
                    entry.shape.clone(),
                    0,
                ))
            } else {
                None
            };
            return Ok((values, indices));
        }

        let stride_before = entry.shape[..dim].iter().try_fold(1usize, |acc, &value| {
            acc.checked_mul(value.max(1))
                .ok_or_else(|| anyhow!("{operation_name}: stride_before overflow"))
        })?;
        let stride_after = entry.shape[dim + 1..]
            .iter()
            .try_fold(1usize, |acc, &value| {
                acc.checked_mul(value.max(1))
                    .ok_or_else(|| anyhow!("{operation_name}: stride_after overflow"))
            })?;
        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("{operation_name}: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("{operation_name}: segment stride exceeds GPU limits"))?;
        ensure!(
            segment_len <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && segments <= u32::MAX as usize
                && block <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "{operation_name}: tensor too large for GPU kernel"
        );

        let words = integer_word_count(integer_type, entry.len)?;
        let value_bytes = (words as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        let values_buffer =
            self.create_storage_buffer_checked(words, "runmat-integer-cumulative-values")?;
        let indices_buffer = if op >= 2 {
            self.create_storage_buffer(entry.len, "runmat-integer-cumulative-indices")
        } else {
            values_buffer.clone()
        };

        let scalar_type = match self.precision {
            crate::backend::wgpu::types::NumericPrecision::F64 => "f64",
            crate::backend::wgpu::types::NumericPrecision::F32 => "f32",
        };
        let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
        let shader = crate::backend::wgpu::shaders::integer::cumulative_scan_shader(
            scalar_type,
            workgroup_size,
        );
        let layout = self
            .device_ref()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-integer-cumulative-bgl"),
                entries: &[
                    crate::backend::wgpu::bindings::storage_read_entry(0),
                    crate::backend::wgpu::bindings::storage_read_write_entry(1),
                    crate::backend::wgpu::bindings::storage_read_write_entry(2),
                    crate::backend::wgpu::bindings::uniform_entry(3),
                ],
            });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-cumulative-pl",
            &layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-cumulative-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-cumulative-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-cumulative",
            Some(shader.as_bytes()),
            Some("runmat-integer-cumulative-bgl"),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            segment_len: u32,
            segments: u32,
            stride_before: u32,
            block: u32,
            total_len: u32,
            integer_type: u32,
            op: u32,
            direction: u32,
        }
        let params = Params {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            total_len: entry.len as u32,
            integer_type: integer_type_code(integer_type),
            op,
            direction: u32::from(matches!(
                direction,
                runmat_accelerate_api::ProviderScanDirection::Reverse
            )),
        };
        let uniform = self.uniform_buffer(&params, "runmat-integer-cumulative-params");
        let group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-integer-cumulative-bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(segments as u32, workgroup_size);
        crate::backend::wgpu::dispatch::elementwise::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &group,
            groups,
        );

        let values = self.register_integer_buffer(
            values_buffer,
            entry.shape.clone(),
            entry.len,
            integer_type,
            value_bytes,
        );
        let indices = if op >= 2 {
            Some(self.register_existing_buffer(indices_buffer, entry.shape, entry.len))
        } else {
            None
        };
        Ok((values, indices))
    }

    pub(crate) fn integer_reduce_sum_prod_dim_exec(
        &self,
        is_product: bool,
        dim: usize,
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(a)?;
        let integer_type = entry
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray reduction is not supported"
        );
        let rank = entry.shape.len();
        ensure!(rank > 0, "{operation_name}: rank must be greater than zero");
        ensure!(dim < rank, "{operation_name}: dimension out of bounds");
        ensure!(
            rank <= crate::backend::wgpu::params::BCAST_MAX_RANK,
            "{operation_name}: rank exceeds GPU kernel limit"
        );
        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (index, slot) in strides.iter_mut().enumerate() {
            *slot = stride;
            stride = stride
                .checked_mul(entry.shape[index].max(1))
                .ok_or_else(|| anyhow!("{operation_name}: shape strides overflow"))?;
        }
        let kept: Vec<usize> = (0..rank).filter(|&index| index != dim).collect();
        let rows = entry.shape[dim];
        let cols = kept.iter().try_fold(1usize, |product, &index| {
            product
                .checked_mul(entry.shape[index])
                .ok_or_else(|| anyhow!("{operation_name}: output length overflow"))
        })?;
        let mut out_shape = entry.shape.clone();
        out_shape[dim] = 1;
        let out_len = cols;
        if rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(
                operation_name,
                rows.max(cols),
            ));
        }
        if strides.iter().any(|&value| value > u32::MAX as usize) {
            return Err(anyhow!("{operation_name}: strides exceed GPU kernel limit"));
        }
        let raw_words = integer_word_count(integer_type, out_len)?;
        let allocated_bytes = (raw_words as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        if out_len == 0 {
            return Ok(self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-reduce-empty"),
                out_shape,
                0,
                integer_type,
                0,
            ));
        }
        if rows == 0 {
            return identity_integer_buffer(self, integer_type, out_len, &out_shape, is_product);
        }
        let out = self.create_storage_buffer_checked(raw_words, "runmat-integer-reduce-out")?;
        let workgroup_size = crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE;
        let shader = crate::backend::wgpu::shaders::integer::reduce_dim_shader(workgroup_size);
        let layout = self
            .device_ref()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-integer-reduce-dim-bgl"),
                entries: &[
                    crate::backend::wgpu::bindings::storage_read_entry(0),
                    crate::backend::wgpu::bindings::storage_read_write_entry(1),
                    crate::backend::wgpu::bindings::uniform_entry(2),
                ],
            });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-reduce-dim-pl",
            &layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-reduce-dim-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-reduce-dim-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-reduce-dim",
            Some(shader.as_bytes()),
            Some("runmat-integer-reduce-dim-bgl"),
            Some(workgroup_size),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            rank: u32,
            kept_count: u32,
            reduce_count: u32,
            op: u32,
            rows: u32,
            cols: u32,
            integer_type: u32,
            slice_offset: u32,
            kept_sizes: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
            reduce_sizes: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
            kept_strides: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
            reduce_strides: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
        }
        let mut kept_sizes = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        let mut reduce_sizes = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        let mut kept_strides = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        let mut reduce_strides = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        for (position, &index) in kept.iter().enumerate() {
            kept_sizes[position] = crate::backend::wgpu::params::AlignedU32::new(
                u32::try_from(entry.shape[index])
                    .map_err(|_| anyhow!("{operation_name}: shape exceeds GPU kernel limit"))?,
            );
            kept_strides[position] = crate::backend::wgpu::params::AlignedU32::new(
                u32::try_from(strides[index])
                    .map_err(|_| anyhow!("{operation_name}: stride exceeds GPU kernel limit"))?,
            );
        }
        reduce_sizes[0] = crate::backend::wgpu::params::AlignedU32::new(
            u32::try_from(rows)
                .map_err(|_| anyhow!("{operation_name}: reduce size exceeds GPU kernel limit"))?,
        );
        reduce_strides[0] = crate::backend::wgpu::params::AlignedU32::new(
            u32::try_from(strides[dim])
                .map_err(|_| anyhow!("{operation_name}: reduce stride exceeds GPU kernel limit"))?,
        );
        let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
        let mut offset = 0usize;
        while offset < out_len {
            let chunk = (out_len - offset).min(max_groups);
            let params = Params {
                rank: rank as u32,
                kept_count: kept.len() as u32,
                reduce_count: 1,
                op: u32::from(is_product),
                rows: rows as u32,
                cols: cols as u32,
                integer_type: integer_type_code(integer_type),
                slice_offset: offset as u32,
                kept_sizes,
                reduce_sizes,
                kept_strides,
                reduce_strides,
            };
            let uniform = self.uniform_buffer(&params, "runmat-integer-reduce-dim-params");
            let group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-integer-reduce-dim-bg"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &group,
                chunk as u32,
            );
            offset += chunk;
        }
        Ok(self.register_integer_buffer(out, out_shape, out_len, integer_type, allocated_bytes))
    }

    pub(crate) fn integer_reduce_sum_prod_global_exec(
        &self,
        is_product: bool,
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let rank = a.shape.len();
        if rank == 0 {
            return Err(anyhow!("{operation_name}: rank must be greater than zero"));
        }
        let mut current = a.clone();
        let mut owned = false;
        for dim in 0..rank {
            match self.integer_reduce_sum_prod_dim_exec(is_product, dim, operation_name, &current) {
                Ok(next) => {
                    if owned {
                        let _ = self.free_exec(&current);
                    }
                    current = next;
                    owned = true;
                }
                Err(error) => {
                    if owned {
                        let _ = self.free_exec(&current);
                    }
                    return Err(error);
                }
            }
        }
        Ok(current)
    }

    pub(crate) fn integer_reduce_mean_dim_exec(
        &self,
        dim: usize,
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.integer_reduce_mean_dims_exec(&[dim], operation_name, a)
    }

    pub(crate) fn integer_reduce_mean_global_exec(
        &self,
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let rank = a.shape.len();
        if rank == 0 {
            return Err(anyhow!("{operation_name}: rank must be greater than zero"));
        }
        let dims: Vec<usize> = (0..rank).collect();
        self.integer_reduce_mean_dims_exec(&dims, operation_name, a)
    }

    pub(crate) fn integer_reduce_mean_dims_exec(
        &self,
        dims: &[usize],
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(a)?;
        let integer_type = entry
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray mean is not supported"
        );
        let rank = entry.shape.len();
        ensure!(rank > 0, "{operation_name}: rank must be greater than zero");
        ensure!(
            rank <= crate::backend::wgpu::params::BCAST_MAX_RANK,
            "{operation_name}: rank exceeds GPU kernel limit"
        );
        let mut reduced = vec![false; rank];
        for &dim in dims {
            ensure!(dim < rank, "{operation_name}: dimension out of bounds");
            reduced[dim] = true;
        }
        if !reduced.iter().any(|&value| value) {
            return Ok(a.clone());
        }
        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (index, slot) in strides.iter_mut().enumerate() {
            *slot = stride;
            stride = stride
                .checked_mul(entry.shape[index].max(1))
                .ok_or_else(|| anyhow!("{operation_name}: shape strides overflow"))?;
        }
        let kept: Vec<usize> = (0..rank).filter(|&index| !reduced[index]).collect();
        let reduce_dims: Vec<usize> = (0..rank).filter(|&index| reduced[index]).collect();
        let rows = reduce_dims.iter().try_fold(1usize, |product, &index| {
            product
                .checked_mul(entry.shape[index])
                .ok_or_else(|| anyhow!("{operation_name}: reduce length overflow"))
        })?;
        let cols = kept.iter().try_fold(1usize, |product, &index| {
            product
                .checked_mul(entry.shape[index])
                .ok_or_else(|| anyhow!("{operation_name}: output length overflow"))
        })?;
        let mut out_shape = entry.shape.clone();
        for &dim in &reduce_dims {
            out_shape[dim] = 1;
        }
        if rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(
                operation_name,
                rows.max(cols),
            ));
        }
        if strides.iter().any(|&value| value > u32::MAX as usize) {
            return Err(anyhow!("{operation_name}: strides exceed GPU kernel limit"));
        }
        if cols == 0 {
            return Ok(self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-mean-empty"),
                out_shape,
                0,
                integer_type,
                0,
            ));
        }
        if rows == 0 {
            return identity_integer_buffer(self, integer_type, cols, &out_shape, false);
        }
        let raw_words = integer_word_count(integer_type, cols)?;
        let allocated_bytes = (raw_words as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        let out = self.create_storage_buffer_checked(raw_words, "runmat-integer-mean-out")?;
        let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
        let shader = crate::backend::wgpu::shaders::integer::mean_dim_shader(workgroup_size);
        let layout = self
            .device_ref()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-integer-mean-dim-bgl"),
                entries: &[
                    crate::backend::wgpu::bindings::storage_read_entry(0),
                    crate::backend::wgpu::bindings::storage_read_write_entry(1),
                    crate::backend::wgpu::bindings::uniform_entry(2),
                ],
            });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-mean-dim-pl",
            &layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-mean-dim-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-mean-dim-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-mean-dim",
            Some(shader.as_bytes()),
            Some("runmat-integer-mean-dim-bgl"),
            Some(workgroup_size),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            rank: u32,
            kept_count: u32,
            reduce_count: u32,
            rows: u32,
            cols: u32,
            integer_type: u32,
            slice_offset: u32,
            _pad0: u32,
            kept_sizes: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
            reduce_sizes: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
            kept_strides: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
            reduce_strides: [crate::backend::wgpu::params::AlignedU32;
                crate::backend::wgpu::params::BCAST_MAX_RANK],
        }
        let mut kept_sizes = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        let mut reduce_sizes = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        let mut kept_strides = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        let mut reduce_strides = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::BCAST_MAX_RANK];
        for (position, &index) in kept.iter().enumerate() {
            kept_sizes[position] = crate::backend::wgpu::params::AlignedU32::new(
                u32::try_from(entry.shape[index])
                    .map_err(|_| anyhow!("{operation_name}: shape exceeds GPU kernel limit"))?,
            );
            kept_strides[position] = crate::backend::wgpu::params::AlignedU32::new(
                u32::try_from(strides[index])
                    .map_err(|_| anyhow!("{operation_name}: stride exceeds GPU kernel limit"))?,
            );
        }
        for (position, &index) in reduce_dims.iter().enumerate() {
            reduce_sizes[position] = crate::backend::wgpu::params::AlignedU32::new(
                u32::try_from(entry.shape[index])
                    .map_err(|_| anyhow!("{operation_name}: shape exceeds GPU kernel limit"))?,
            );
            reduce_strides[position] = crate::backend::wgpu::params::AlignedU32::new(
                u32::try_from(strides[index])
                    .map_err(|_| anyhow!("{operation_name}: stride exceeds GPU kernel limit"))?,
            );
        }
        let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
        let mut offset = 0usize;
        while offset < cols {
            let chunk = (cols - offset).min(max_groups);
            let params = Params {
                rank: rank as u32,
                kept_count: kept.len() as u32,
                reduce_count: reduce_dims.len() as u32,
                rows: rows as u32,
                cols: cols as u32,
                integer_type: integer_type_code(integer_type),
                slice_offset: offset as u32,
                _pad0: 0,
                kept_sizes,
                reduce_sizes,
                kept_strides,
                reduce_strides,
            };
            let uniform = self.uniform_buffer(&params, "runmat-integer-mean-dim-params");
            let group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-integer-mean-dim-bg"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &group,
                chunk as u32,
            );
            offset += chunk;
        }
        Ok(self.register_integer_buffer(out, out_shape, cols, integer_type, allocated_bytes))
    }

    pub(crate) fn integer_reduce_extrema_dim_exec(
        &self,
        select_min: bool,
        dim: usize,
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<runmat_accelerate_api::ReduceDimResult> {
        let entry = self.get_entry_raw(a)?;
        let integer_type = entry
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray reduction is not supported"
        );
        ensure!(
            entry.shape.len() == 2,
            "{operation_name}: only 2D tensors are supported"
        );
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let (out_len, out_shape) = match dim {
            0 => (cols, vec![1, cols]),
            1 => (rows, vec![rows, 1]),
            _ => return Err(anyhow!("{operation_name}: only dims 0 or 1 are supported")),
        };
        if rows == 0 || cols == 0 {
            // Keep empty extrema device-resident rather than routing through
            // the f64 host gather fallback, which cannot reconstruct wide
            // integers.
            let values = self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-reduce-empty-extrema-values"),
                out_shape.clone(),
                0,
                integer_type,
                0,
            );
            let indices = self.register_existing_buffer(
                self.create_storage_buffer(0, "runmat-integer-reduce-empty-extrema-indices"),
                out_shape,
                0,
            );
            return Ok(runmat_accelerate_api::ReduceDimResult { values, indices });
        }
        if rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(
                operation_name,
                rows.max(cols),
            ));
        }
        let words = integer_word_count(integer_type, out_len)?;
        let value_bytes = (words as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        let values_buffer =
            self.create_storage_buffer_checked(words, "runmat-integer-reduce-dim-extrema-values")?;
        let indices_buffer = self
            .create_storage_buffer_checked(out_len, "runmat-integer-reduce-dim-extrema-indices")?;
        let scalar_type = match self.precision {
            crate::backend::wgpu::types::NumericPrecision::F64 => "f64",
            crate::backend::wgpu::types::NumericPrecision::F32 => "f32",
        };
        let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
        let shader =
            crate::backend::wgpu::shaders::integer::extrema_dim_shader(scalar_type, workgroup_size);
        let layout = self
            .device_ref()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-integer-reduce-dim-extrema-bgl"),
                entries: &[
                    crate::backend::wgpu::bindings::storage_read_entry(0),
                    crate::backend::wgpu::bindings::storage_read_write_entry(1),
                    crate::backend::wgpu::bindings::storage_read_write_entry(2),
                    crate::backend::wgpu::bindings::uniform_entry(3),
                ],
            });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-reduce-dim-extrema-pl",
            &layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-reduce-dim-extrema-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-reduce-dim-extrema-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-reduce-dim-extrema",
            Some(shader.as_bytes()),
            Some("runmat-integer-reduce-dim-extrema-bgl"),
            Some(workgroup_size),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            rows: u32,
            cols: u32,
            dim: u32,
            select_min: u32,
            integer_type: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
            dim: dim as u32,
            select_min: u32::from(select_min),
            integer_type: integer_type_code(integer_type),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform = self.uniform_buffer(&params, "runmat-integer-reduce-dim-extrema-params");
        let group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-integer-reduce-dim-extrema-bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(out_len as u32, workgroup_size);
        crate::backend::wgpu::dispatch::elementwise::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &group,
            groups,
        );
        Ok(runmat_accelerate_api::ReduceDimResult {
            values: self.register_integer_buffer(
                values_buffer,
                out_shape.clone(),
                out_len,
                integer_type,
                value_bytes,
            ),
            indices: self.register_existing_buffer(indices_buffer, out_shape, out_len),
        })
    }

    pub(crate) fn integer_arithmetic_exec(
        &self,
        op: u32,
        operation_name: &str,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            op <= 6,
            "{operation_name}: unsupported integer arithmetic opcode"
        );
        let entry_a = self.get_entry_raw(a)?;
        let entry_b = self.get_entry_raw(b)?;
        let integer_type = entry_a
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry_b.integer_type() == Some(integer_type),
            "{operation_name}: integer operands must have the same class"
        );
        ensure!(
            entry_a.storage == GpuTensorStorage::Real && entry_b.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray arithmetic is not supported"
        );
        use crate::backend::wgpu::params::{AlignedU32, BCAST_MAX_RANK};
        let broadcast = integer_broadcast_plan(
            operation_name,
            &entry_a.shape,
            entry_a.len,
            &entry_b.shape,
            entry_b.len,
        )?;
        let len = broadcast.len;
        let raw_len = integer_word_count(integer_type, len)?;
        let bytes = (raw_len as u64).saturating_mul(4);
        if len == 0 {
            return Ok(self.register_integer_buffer(
                self.create_storage_buffer(0, "runmat-integer-arithmetic-empty"),
                broadcast.output_shape,
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
        while offset < len {
            let chunk = (len - offset).min(capacity);
            let params = Params {
                len: chunk as u32,
                op,
                offset: offset as u32,
                total: u32::try_from(len).expect("integer dispatch length was checked"),
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
        Ok(self.register_integer_buffer(out, broadcast.output_shape, len, integer_type, bytes))
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
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            entry_b.integer_type() == Some(integer_type),
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

    pub(crate) fn cast_to_integer_exec(
        &self,
        a: &GpuTensorHandle,
        target_type: IntegerElementType,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(a)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "integer cast: complex gpuArray inputs are not supported by the resident integer cast kernel"
        );
        let len = entry.len;
        let raw_len = integer_word_count(target_type, len)?;
        let allocated_bytes = (raw_len as u64).saturating_mul(std::mem::size_of::<u32>() as u64);
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-integer-cast-empty");
            return Ok(self.register_integer_buffer(out, entry.shape, 0, target_type, 0));
        }
        if len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error("integer cast", len));
        }
        let source_type = if let Some(integer_type) = entry.integer_type() {
            integer_type_code(integer_type)
        } else {
            match entry.precision {
                NumericPrecision::F32 => 8,
                NumericPrecision::F64 => 9,
            }
        };
        let raw_out = self.create_storage_buffer_checked(raw_len, "runmat-integer-cast-out")?;
        let workgroup_size = crate::backend::wgpu::config::effective_workgroup_size();
        let scalar_type = match entry.precision {
            NumericPrecision::F32 => "f32",
            NumericPrecision::F64 => "f64",
        };
        let shader =
            crate::backend::wgpu::shaders::integer::cast_shader(scalar_type, workgroup_size);
        let bind_group_layout =
            self.device_ref()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runmat-integer-cast-bgl"),
                    entries: &[
                        crate::backend::wgpu::bindings::storage_read_entry(0),
                        crate::backend::wgpu::bindings::storage_read_write_entry(1),
                        crate::backend::wgpu::bindings::uniform_entry(2),
                    ],
                });
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-integer-cast-pl",
            &bind_group_layout,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-integer-cast-module",
            &shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-integer-cast-bgl",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pipeline_layout,
            &module,
            "runmat-integer-cast",
            Some(shader.as_bytes()),
            Some("runmat-integer-cast-bgl"),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            offset: u32,
            source_type: u32,
            target_type: u32,
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * workgroup_size as usize;
        let mut offset = 0usize;
        while offset < len {
            let chunk_len = (len - offset).min(chunk_capacity);
            let params = Params {
                len: chunk_len as u32,
                offset: offset as u32,
                source_type,
                target_type: integer_type_code(target_type),
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-integer-cast-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-integer-cast-bg"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: raw_out.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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
        Ok(self.register_integer_buffer(raw_out, entry.shape, len, target_type, allocated_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
        HostTensorView, IntegerElementType,
    };

    fn register_wgpu_provider_for_test() -> Option<&'static WgpuProvider> {
        match crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => Some(provider),
            Err(error) if error.to_string() == "wgpu: no compatible adapter found" => None,
            Err(error) => panic!("register wgpu provider failed: {error:?}"),
        }
    }

    #[test]
    fn integer_broadcast_plan_appends_singletons_and_enforces_zero_rules() {
        let plan = integer_broadcast_plan("plus", &[2, 3], 6, &[2, 3, 4], 24)
            .expect("missing dimensions are trailing singletons");
        assert_eq!(
            plan.a_shape[..3]
                .iter()
                .map(|dimension| dimension.value)
                .collect::<Vec<_>>(),
            vec![2, 3, 1]
        );
        assert_eq!(
            plan.b_shape[..3]
                .iter()
                .map(|dimension| dimension.value)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert_eq!(plan.output_shape, vec![2, 3, 4]);

        assert!(integer_broadcast_plan("plus", &[2, 3], 6, &[1, 2, 3], 6).is_err());

        let empty = integer_broadcast_plan("plus", &[0, 3], 0, &[1, 3], 3)
            .expect("zero is compatible with singleton");
        assert_eq!(empty.output_shape, vec![0, 3]);
        assert!(integer_broadcast_plan("plus", &[0, 3], 0, &[2, 3], 6).is_err());

        let row_shorthand = integer_broadcast_plan("plus", &[3], 3, &[1, 3, 2], 6)
            .expect("one-dimensional tensors represent MATLAB row vectors");
        assert_eq!(row_shorthand.output_shape, vec![1, 3, 2]);
    }

    #[test]
    fn wgpu_native_integer_extrema_preserve_all_classes_and_indices() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $values:expr, $min_dim0:expr, $max_dim0:expr, $min_dim1:expr, $max_dim1:expr) => {{
                let values = $values;
                let input = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&values),
                        shape: &[3, 2],
                    })
                    .expect("upload exact integer extrema input");
                let min_dim0 = provider
                    .integer_reduce_extrema_dim_exec(true, 0, "min", &input)
                    .expect("min along rows");
                assert_eq!(
                    block_on(provider.download_integer_exec(&min_dim0.values))
                        .expect("download native min dim 0")
                        .data,
                    HostIntegerDataOwned::$owned($min_dim0.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_exec(&min_dim0.indices))
                        .expect("download min indices dim 0")
                        .data,
                    vec![2.0, 2.0]
                );
                let max_dim0 = provider
                    .integer_reduce_extrema_dim_exec(false, 0, "max", &input)
                    .expect("max along rows");
                assert_eq!(
                    block_on(provider.download_integer_exec(&max_dim0.values))
                        .expect("download native max dim 0")
                        .data,
                    HostIntegerDataOwned::$owned($max_dim0.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_exec(&max_dim0.indices))
                        .expect("download max indices dim 0")
                        .data,
                    vec![1.0, 1.0]
                );
                let min_dim1 = provider
                    .integer_reduce_extrema_dim_exec(true, 1, "min", &input)
                    .expect("min along columns");
                assert_eq!(
                    block_on(provider.download_integer_exec(&min_dim1.values))
                        .expect("download native min dim 1")
                        .data,
                    HostIntegerDataOwned::$owned($min_dim1.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_exec(&min_dim1.indices))
                        .expect("download min indices dim 1")
                        .data,
                    vec![1.0, 2.0, 2.0]
                );
                let max_dim1 = provider
                    .integer_reduce_extrema_dim_exec(false, 1, "max", &input)
                    .expect("max along columns");
                assert_eq!(
                    block_on(provider.download_integer_exec(&max_dim1.values))
                        .expect("download native max dim 1")
                        .data,
                    HostIntegerDataOwned::$owned($max_dim1.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_exec(&max_dim1.indices))
                        .expect("download max indices dim 1")
                        .data,
                    vec![2.0, 1.0, 1.0]
                );
            }};
        }
        check!(
            I8,
            I8,
            [3_i8, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            I16,
            I16,
            [3_i16, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            I32,
            I32,
            [3_i32, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            I64,
            I64,
            [3_i64, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            U8,
            U8,
            [3_u8, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            U16,
            U16,
            [3_u16, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            U32,
            U32,
            [3_u32, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
        check!(
            U64,
            U64,
            [3_u64, 2, 2, 5, 1, 1],
            [2, 1],
            [3, 5],
            [3, 1, 1],
            [5, 2, 2]
        );
    }

    #[test]
    fn wgpu_native_integer_extrema_public_hooks_keep_values_resident() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[1_u64 << 63, u64::MAX, 5, 4]),
                shape: &[2, 2],
            })
            .expect("upload packed u64 extrema input");
        let min = block_on(provider.reduce_min_dim(&input, 0)).expect("public min hook");
        let max = block_on(provider.reduce_max_dim(&input, 1)).expect("public max hook");
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&min.values),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&max.values),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&min.values))
                .expect("download min values")
                .data,
            HostIntegerDataOwned::U64(vec![1_u64 << 63, 4])
        );
        assert_eq!(
            block_on(provider.download_exec(&min.indices))
                .expect("download min indices")
                .data,
            vec![1.0, 2.0]
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&max.values))
                .expect("download max values")
                .data,
            HostIntegerDataOwned::U64(vec![1_u64 << 63, u64::MAX])
        );
        assert_eq!(
            block_on(provider.download_exec(&max.indices))
                .expect("download max indices")
                .data,
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn wgpu_native_integer_empty_extrema_keep_wide_values_and_indices_resident() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[]),
                shape: &[0, 2],
            })
            .expect("upload empty packed u64 extrema input");
        let min = block_on(provider.reduce_min_dim(&input, 0)).expect("public empty min hook");
        let max = block_on(provider.reduce_max_dim(&input, 1)).expect("public empty max hook");

        for (result, expected_shape) in [(&min, vec![1, 2]), (&max, vec![0, 1])] {
            assert_eq!(result.values.shape, expected_shape);
            assert_eq!(result.indices.shape, expected_shape);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&result.values),
                Some(IntegerElementType::U64)
            );
            assert_eq!(
                block_on(provider.download_integer_exec(&result.values))
                    .expect("download empty native extrema values")
                    .data,
                HostIntegerDataOwned::U64(Vec::new())
            );
            assert!(block_on(provider.download_exec(&result.indices))
                .expect("download empty extrema indices")
                .data
                .is_empty());
        }
    }

    #[test]
    fn wgpu_native_integer_sum_prod_reduce_all_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $values:expr, $sum_dim0:expr, $sum_dim1:expr, $prod_dim0:expr, $prod_dim1:expr, $sum_global:expr, $prod_global:expr) => {{
                let values = $values;
                let input = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&values),
                        shape: &[3, 2],
                    })
                    .expect("upload exact integer reduction input");
                let sum_dim0 = block_on(provider.reduce_integer_sum_native_dim(&input, 0))
                    .expect("sum native dim 0");
                let sum_dim1 = block_on(provider.reduce_integer_sum_native_dim(&input, 1))
                    .expect("sum native dim 1");
                let prod_dim0 = block_on(provider.reduce_integer_prod_native_dim(&input, 0))
                    .expect("prod native dim 0");
                let prod_dim1 = block_on(provider.reduce_integer_prod_native_dim(&input, 1))
                    .expect("prod native dim 1");
                let sum_global =
                    block_on(provider.reduce_integer_sum_native(&input)).expect("sum native all");
                let prod_global =
                    block_on(provider.reduce_integer_prod_native(&input)).expect("prod native all");
                assert_eq!(
                    block_on(provider.download_integer_exec(&sum_dim0))
                        .expect("download sum dim 0")
                        .data,
                    HostIntegerDataOwned::$owned($sum_dim0.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&sum_dim1))
                        .expect("download sum dim 1")
                        .data,
                    HostIntegerDataOwned::$owned($sum_dim1.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&prod_dim0))
                        .expect("download prod dim 0")
                        .data,
                    HostIntegerDataOwned::$owned($prod_dim0.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&prod_dim1))
                        .expect("download prod dim 1")
                        .data,
                    HostIntegerDataOwned::$owned($prod_dim1.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&sum_global))
                        .expect("download global sum")
                        .data,
                    HostIntegerDataOwned::$owned($sum_global.to_vec())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&prod_global))
                        .expect("download global prod")
                        .data,
                    HostIntegerDataOwned::$owned($prod_global.to_vec())
                );
            }};
        }
        check!(
            I8,
            I8,
            [2_i8, -4, 8, 3, 5, -2],
            [6, 6],
            [5, 1, 6],
            [-64, -30],
            [6, -20, -16],
            [12],
            [127]
        );
        check!(
            I16,
            I16,
            [2_i16, -4, 8, 3, 5, -2],
            [6, 6],
            [5, 1, 6],
            [-64, -30],
            [6, -20, -16],
            [12],
            [7680]
        );
        check!(
            I32,
            I32,
            [2_i32, -4, 8, 3, 5, -2],
            [6, 6],
            [5, 1, 6],
            [-64, -30],
            [6, -20, -16],
            [12],
            [7680]
        );
        check!(
            I64,
            I64,
            [2_i64, -4, 8, 3, 5, -2],
            [6, 6],
            [5, 1, 6],
            [-64, -30],
            [6, -20, -16],
            [12],
            [7680]
        );
        check!(
            U8,
            U8,
            [2_u8, 4, 8, 3, 5, 2],
            [14, 10],
            [5, 9, 10],
            [64, 30],
            [6, 20, 16],
            [24],
            [255]
        );
        check!(
            U16,
            U16,
            [2_u16, 4, 8, 3, 5, 2],
            [14, 10],
            [5, 9, 10],
            [64, 30],
            [6, 20, 16],
            [24],
            [7680]
        );
        check!(
            U32,
            U32,
            [2_u32, 4, 8, 3, 5, 2],
            [14, 10],
            [5, 9, 10],
            [64, 30],
            [6, 20, 16],
            [24],
            [7680]
        );
        check!(
            U64,
            U64,
            [2_u64, 4, 8, 3, 5, 2],
            [14, 10],
            [5, 9, 10],
            [64, 30],
            [6, 20, 16],
            [24],
            [7680]
        );
    }

    #[test]
    fn wgpu_native_integer_sum_prod_saturate_packed_extremes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let signed = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[i64::MAX, 1, -2, 3]),
                shape: &[2, 2],
            })
            .expect("upload signed packed integers");
        let unsigned = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[u64::MAX, 2, 3, 4]),
                shape: &[2, 2],
            })
            .expect("upload unsigned packed integers");
        let signed_sum = block_on(provider.reduce_integer_sum_native_dim(&signed, 0))
            .expect("signed packed sum");
        let unsigned_sum = block_on(provider.reduce_integer_sum_native_dim(&unsigned, 0))
            .expect("unsigned packed sum");
        let unsigned_prod = block_on(provider.reduce_integer_prod_native_dim(&unsigned, 0))
            .expect("unsigned packed product");
        assert_eq!(
            block_on(provider.download_integer_exec(&signed_sum))
                .expect("download signed sum")
                .data,
            HostIntegerDataOwned::I64(vec![i64::MAX, 1])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&unsigned_sum))
                .expect("download unsigned sum")
                .data,
            HostIntegerDataOwned::U64(vec![u64::MAX, 7])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&unsigned_prod))
                .expect("download unsigned product")
                .data,
            HostIntegerDataOwned::U64(vec![u64::MAX, 12])
        );
    }

    #[test]
    fn wgpu_native_integer_sum_prod_empty_dimension_identities() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U32(&[]),
                shape: &[0, 3],
            })
            .expect("upload empty native integer gpuArray");
        let sum = block_on(provider.reduce_integer_sum_native_dim(&input, 0))
            .expect("sum native empty dimension");
        let prod = block_on(provider.reduce_integer_prod_native_dim(&input, 0))
            .expect("prod native empty dimension");
        assert_eq!(
            block_on(provider.download_integer_exec(&sum))
                .expect("download empty sum")
                .data,
            HostIntegerDataOwned::U32(vec![0, 0, 0])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&prod))
                .expect("download empty prod")
                .data,
            HostIntegerDataOwned::U32(vec![1, 1, 1])
        );
    }

    #[test]
    fn wgpu_native_integer_sum_prod_reduce_nd_shapes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let values: Vec<u16> = (1..=12).collect();
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&values),
                shape: &[2, 3, 2],
            })
            .expect("upload native integer ndarray");
        let sum_pages = block_on(provider.reduce_integer_sum_native_dim(&input, 2))
            .expect("sum across third dimension");
        let sum_cols = block_on(provider.reduce_integer_sum_native_dim(&input, 1))
            .expect("sum across second dimension");
        let sum_all = block_on(provider.reduce_integer_sum_native(&input)).expect("sum all dims");
        let prod_all =
            block_on(provider.reduce_integer_prod_native(&input)).expect("prod all dims");
        assert_eq!(
            block_on(provider.download_integer_exec(&sum_pages))
                .expect("download page sum")
                .data,
            HostIntegerDataOwned::U16(vec![8, 10, 12, 14, 16, 18])
        );
        assert_eq!(sum_pages.shape, vec![2, 3, 1]);
        assert_eq!(
            block_on(provider.download_integer_exec(&sum_cols))
                .expect("download column sum")
                .data,
            HostIntegerDataOwned::U16(vec![9, 12, 27, 30])
        );
        assert_eq!(sum_cols.shape, vec![2, 1, 2]);
        assert_eq!(
            block_on(provider.download_integer_exec(&sum_all))
                .expect("download global sum")
                .data,
            HostIntegerDataOwned::U16(vec![78])
        );
        assert_eq!(sum_all.shape, vec![1, 1, 1]);
        assert_eq!(
            block_on(provider.download_integer_exec(&prod_all))
                .expect("download global product")
                .data,
            HostIntegerDataOwned::U16(vec![u16::MAX])
        );
        assert_eq!(prod_all.shape, vec![1, 1, 1]);
    }

    #[test]
    fn wgpu_native_integer_cumulative_scans_preserve_class_and_direction() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I8(&[120, 10, 2, 4, -5, 3]),
                shape: &[3, 2],
            })
            .expect("upload native integer scan input");
        let cumsum = provider
            .integer_cumsum_scan(
                &input,
                0,
                runmat_accelerate_api::ProviderScanDirection::Forward,
            )
            .expect("native integer cumsum");
        let cumprod = provider
            .integer_cumprod_scan(
                &input,
                0,
                runmat_accelerate_api::ProviderScanDirection::Reverse,
            )
            .expect("native integer cumprod reverse");
        assert_eq!(
            block_on(provider.download_integer_exec(&cumsum))
                .expect("download cumsum")
                .data,
            HostIntegerDataOwned::I8(vec![120, 127, 127, 4, -1, 2])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&cumprod))
                .expect("download cumprod")
                .data,
            HostIntegerDataOwned::I8(vec![127, 20, 2, -60, -15, 3])
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&cumsum),
            Some(IntegerElementType::I8)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&cumprod),
            Some(IntegerElementType::I8)
        );
    }

    #[test]
    fn wgpu_native_integer_cumulative_extrema_return_indices() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[4, 2, 2, 7, 3, 7, 9, 9, 1, 8, 8, 0]),
                shape: &[2, 3, 2],
            })
            .expect("upload packed native integer scan input");
        let mins = provider
            .integer_cummin_scan(
                &input,
                1,
                runmat_accelerate_api::ProviderScanDirection::Forward,
            )
            .expect("native integer cummin");
        let maxes = provider
            .integer_cummax_scan(
                &input,
                2,
                runmat_accelerate_api::ProviderScanDirection::Reverse,
            )
            .expect("native integer cummax reverse");
        assert_eq!(
            block_on(provider.download_integer_exec(&mins.values))
                .expect("download cummin")
                .data,
            HostIntegerDataOwned::U64(vec![4, 2, 2, 2, 2, 2, 9, 9, 1, 8, 1, 0])
        );
        assert_eq!(
            block_on(provider.download_exec(&mins.indices))
                .expect("download cummin indices")
                .data,
            vec![1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&maxes.values))
                .expect("download cummax")
                .data,
            HostIntegerDataOwned::U64(vec![9, 9, 2, 8, 8, 7, 9, 9, 1, 8, 8, 0])
        );
        assert_eq!(
            block_on(provider.download_exec(&maxes.indices))
                .expect("download cummax indices")
                .data,
            vec![2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&mins.values),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&maxes.values),
            Some(IntegerElementType::U64)
        );
    }

    #[test]
    fn wgpu_native_integer_cumulative_scans_cover_all_classes_and_ties() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $ty:ty, $element_type:expr) => {{
                let scan_values: [$ty; 4] = [1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty];
                let scan = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&scan_values),
                        shape: &[2, 2],
                    })
                    .expect("upload scan values");
                let cumsum = provider
                    .integer_cumsum_scan(
                        &scan,
                        0,
                        runmat_accelerate_api::ProviderScanDirection::Forward,
                    )
                    .expect("cumsum all-class");
                let cumprod = provider
                    .integer_cumprod_scan(
                        &scan,
                        1,
                        runmat_accelerate_api::ProviderScanDirection::Reverse,
                    )
                    .expect("cumprod all-class");
                assert_eq!(
                    block_on(provider.download_integer_exec(&cumsum))
                        .expect("download cumsum")
                        .data,
                    HostIntegerDataOwned::$owned(vec![1 as $ty, 3 as $ty, 3 as $ty, 7 as $ty])
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&cumprod))
                        .expect("download cumprod")
                        .data,
                    HostIntegerDataOwned::$owned(vec![3 as $ty, 8 as $ty, 3 as $ty, 4 as $ty])
                );
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&cumsum),
                    Some($element_type)
                );
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&cumprod),
                    Some($element_type)
                );

                let tie_values: [$ty; 4] = [5 as $ty, 5 as $ty, 4 as $ty, 6 as $ty];
                let ties = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&tie_values),
                        shape: &[2, 2],
                    })
                    .expect("upload tie values");
                let cummax = provider
                    .integer_cummax_scan(
                        &ties,
                        0,
                        runmat_accelerate_api::ProviderScanDirection::Forward,
                    )
                    .expect("cummax ties");
                let cummin = provider
                    .integer_cummin_scan(
                        &ties,
                        1,
                        runmat_accelerate_api::ProviderScanDirection::Reverse,
                    )
                    .expect("cummin ties");
                assert_eq!(
                    block_on(provider.download_integer_exec(&cummax.values))
                        .expect("download cummax values")
                        .data,
                    HostIntegerDataOwned::$owned(vec![5 as $ty, 5 as $ty, 4 as $ty, 6 as $ty])
                );
                assert_eq!(
                    block_on(provider.download_exec(&cummax.indices))
                        .expect("download cummax indices")
                        .data,
                    vec![1.0, 1.0, 1.0, 2.0]
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&cummin.values))
                        .expect("download cummin values")
                        .data,
                    HostIntegerDataOwned::$owned(vec![4 as $ty, 5 as $ty, 4 as $ty, 6 as $ty])
                );
                assert_eq!(
                    block_on(provider.download_exec(&cummin.indices))
                        .expect("download cummin indices")
                        .data,
                    vec![2.0, 1.0, 2.0, 2.0]
                );
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&cummax.values),
                    Some($element_type)
                );
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&cummin.values),
                    Some($element_type)
                );
            }};
        }

        check!(I8, I8, i8, IntegerElementType::I8);
        check!(I16, I16, i16, IntegerElementType::I16);
        check!(I32, I32, i32, IntegerElementType::I32);
        check!(I64, I64, i64, IntegerElementType::I64);
        check!(U8, U8, u8, IntegerElementType::U8);
        check!(U16, U16, u16, IntegerElementType::U16);
        check!(U32, U32, u32, IntegerElementType::U32);
        check!(U64, U64, u64, IntegerElementType::U64);
    }

    #[test]
    fn wgpu_native_integer_cumulative_scans_cover_empty_all_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $ty:ty) => {{
                let values: [$ty; 0] = [];
                let input = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&values),
                        shape: &[0, 2],
                    })
                    .expect("upload empty integer scan values");
                let cumsum = provider
                    .integer_cumsum_scan(
                        &input,
                        0,
                        runmat_accelerate_api::ProviderScanDirection::Forward,
                    )
                    .expect("empty cumsum");
                let cumprod = provider
                    .integer_cumprod_scan(
                        &input,
                        0,
                        runmat_accelerate_api::ProviderScanDirection::Reverse,
                    )
                    .expect("empty cumprod");
                let cummin = provider
                    .integer_cummin_scan(
                        &input,
                        0,
                        runmat_accelerate_api::ProviderScanDirection::Forward,
                    )
                    .expect("empty cummin");
                let cummax = provider
                    .integer_cummax_scan(
                        &input,
                        0,
                        runmat_accelerate_api::ProviderScanDirection::Reverse,
                    )
                    .expect("empty cummax");
                assert_eq!(cumsum.shape, vec![0, 2]);
                assert_eq!(cumprod.shape, vec![0, 2]);
                assert_eq!(cummin.values.shape, vec![0, 2]);
                assert_eq!(cummin.indices.shape, vec![0, 2]);
                assert_eq!(cummax.values.shape, vec![0, 2]);
                assert_eq!(cummax.indices.shape, vec![0, 2]);
                assert_eq!(
                    block_on(provider.download_integer_exec(&cumsum))
                        .expect("download empty cumsum")
                        .data,
                    HostIntegerDataOwned::$owned(Vec::<$ty>::new())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&cumprod))
                        .expect("download empty cumprod")
                        .data,
                    HostIntegerDataOwned::$owned(Vec::<$ty>::new())
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&cummin.values))
                        .expect("download empty cummin")
                        .data,
                    HostIntegerDataOwned::$owned(Vec::<$ty>::new())
                );
                assert_eq!(
                    block_on(provider.download_exec(&cummin.indices))
                        .expect("download empty cummin indices")
                        .data,
                    Vec::<f64>::new()
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&cummax.values))
                        .expect("download empty cummax")
                        .data,
                    HostIntegerDataOwned::$owned(Vec::<$ty>::new())
                );
                assert_eq!(
                    block_on(provider.download_exec(&cummax.indices))
                        .expect("download empty cummax indices")
                        .data,
                    Vec::<f64>::new()
                );
            }};
        }

        check!(I8, I8, i8);
        check!(I16, I16, i16);
        check!(I32, I32, i32);
        check!(I64, I64, i64);
        check!(U8, U8, u8);
        check!(U16, U16, u16);
        check!(U32, U32, u32);
        check!(U64, U64, u64);
    }

    #[test]
    fn wgpu_native_integer_mean_preserves_64bit_precision_and_rounding() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let large = 1_u64 << 63;
        let unsigned = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[large + 1, large + 3, 7, 9]),
                shape: &[2, 2],
            })
            .expect("upload uint64 mean input");
        let signed = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[-2, -3, 10, 13]),
                shape: &[2, 2],
            })
            .expect("upload int64 mean input");
        let unsigned_dim0 = block_on(provider.reduce_integer_mean_native_dim(&unsigned, 0))
            .expect("uint64 mean dim 0");
        let signed_dim0 = block_on(provider.reduce_integer_mean_native_dim(&signed, 0))
            .expect("int64 mean dim 0");
        assert_eq!(
            block_on(provider.download_integer_exec(&unsigned_dim0))
                .expect("download uint64 mean")
                .data,
            HostIntegerDataOwned::U64(vec![large + 2, 8])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&signed_dim0))
                .expect("download int64 mean")
                .data,
            HostIntegerDataOwned::I64(vec![-3, 12])
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&unsigned_dim0),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&signed_dim0),
            Some(IntegerElementType::I64)
        );
    }

    #[test]
    fn wgpu_native_integer_mean_vecdim_rounds_once() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I16(&[1, 1, 1, 3]),
                shape: &[2, 2],
            })
            .expect("upload int16 mean input");
        let vecdim = block_on(provider.reduce_integer_mean_native_dims(&input, &[0, 1]))
            .expect("int16 mean vecdim");
        assert_eq!(
            block_on(provider.download_integer_exec(&vecdim))
                .expect("download vecdim mean")
                .data,
            HostIntegerDataOwned::I16(vec![2])
        );
        assert_eq!(vecdim.shape, vec![1, 1]);
    }

    #[test]
    fn wgpu_native_integer_mean_covers_all_classes_and_empty_reductions() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $ty:ty, $element_type:expr) => {{
                let values: [$ty; 4] = [1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty];
                let input = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&values),
                        shape: &[2, 2],
                    })
                    .expect("upload integer mean values");
                let dim0 = block_on(provider.reduce_integer_mean_native_dim(&input, 0))
                    .expect("mean dim 0");
                let dim1 = block_on(provider.reduce_integer_mean_native_dim(&input, 1))
                    .expect("mean dim 1");
                let global =
                    block_on(provider.reduce_integer_mean_native(&input)).expect("mean global");
                assert_eq!(
                    block_on(provider.download_integer_exec(&dim0))
                        .expect("download mean dim 0")
                        .data,
                    HostIntegerDataOwned::$owned(vec![2 as $ty, 4 as $ty])
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&dim1))
                        .expect("download mean dim 1")
                        .data,
                    HostIntegerDataOwned::$owned(vec![2 as $ty, 3 as $ty])
                );
                assert_eq!(
                    block_on(provider.download_integer_exec(&global))
                        .expect("download global mean")
                        .data,
                    HostIntegerDataOwned::$owned(vec![3 as $ty])
                );
                assert_eq!(dim0.shape, vec![1, 2]);
                assert_eq!(dim1.shape, vec![2, 1]);
                assert_eq!(global.shape, vec![1, 1]);
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&dim0),
                    Some($element_type)
                );
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&global),
                    Some($element_type)
                );

                let empty_values: [$ty; 0] = [];
                let empty = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&empty_values),
                        shape: &[0, 3],
                    })
                    .expect("upload empty integer mean values");
                let empty_dim0 = block_on(provider.reduce_integer_mean_native_dim(&empty, 0))
                    .expect("mean empty dim 0");
                assert_eq!(empty_dim0.shape, vec![1, 3]);
                assert_eq!(
                    block_on(provider.download_integer_exec(&empty_dim0))
                        .expect("download empty mean")
                        .data,
                    HostIntegerDataOwned::$owned(vec![0 as $ty, 0 as $ty, 0 as $ty])
                );
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&empty_dim0),
                    Some($element_type)
                );
            }};
        }

        check!(I8, I8, i8, IntegerElementType::I8);
        check!(I16, I16, i16, IntegerElementType::I16);
        check!(I32, I32, i32, IntegerElementType::I32);
        check!(I64, I64, i64, IntegerElementType::I64);
        check!(U8, U8, u8, IntegerElementType::U8);
        check!(U16, U16, u16, IntegerElementType::U16);
        check!(U32, U32, u32, IntegerElementType::U32);
        check!(U64, U64, u64, IntegerElementType::U64);
    }

    #[test]
    fn wgpu_public_reductions_route_native_wide_integers_without_f64_readback() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let signed = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[i64::MAX, 1, i64::MIN, -1]),
                shape: &[2, 2],
            })
            .expect("upload int64");
        let unsigned = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[(1_u64 << 63) + 1, (1_u64 << 63) + 3, 7, 9]),
                shape: &[2, 2],
            })
            .expect("upload uint64");

        let sum = block_on(provider.reduce_sum(&signed)).expect("public int64 sum");
        let prod = block_on(provider.reduce_prod(&unsigned)).expect("public uint64 prod");
        let mean = block_on(provider.reduce_mean_dim(&unsigned, 0)).expect("public uint64 mean");

        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&sum),
            Some(IntegerElementType::I64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&prod),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&mean),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&sum))
                .expect("download int64 sum")
                .data,
            HostIntegerDataOwned::I64(vec![0])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&prod))
                .expect("download uint64 prod")
                .data,
            HostIntegerDataOwned::U64(vec![u64::MAX])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&mean))
                .expect("download uint64 mean")
                .data,
            HostIntegerDataOwned::U64(vec![(1_u64 << 63) + 2, 8])
        );
    }

    #[test]
    fn wgpu_integer_cast_from_real_gpuarray_stays_resident() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let values = [-129.6, -1.0, 4.4, 5.6, f64::NAN, f64::INFINITY];
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[2, 3],
            })
            .expect("upload real cast input");
        let int8 = block_on(provider.cast_to_integer(&input, IntegerElementType::I8))
            .expect("cast to int8");
        let uint32 = block_on(provider.cast_to_integer(&input, IntegerElementType::U32))
            .expect("cast to uint32");
        assert_eq!(int8.shape, vec![2, 3]);
        assert_eq!(uint32.shape, vec![2, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&int8),
            Some(IntegerElementType::I8)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&uint32),
            Some(IntegerElementType::U32)
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&int8))
                .expect("download int8 cast")
                .data,
            HostIntegerDataOwned::I8(vec![i8::MIN, -1, 4, 6, 0, i8::MAX])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&uint32))
                .expect("download uint32 cast")
                .data,
            HostIntegerDataOwned::U32(vec![0, 0, 4, 6, 0, u32::MAX])
        );
    }

    #[test]
    fn wgpu_integer_cast_between_native_classes_is_exact() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let signed = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[-1, 0, i64::MAX]),
                shape: &[1, 3],
            })
            .expect("upload signed integer cast input");
        let unsigned = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0, 1_u64 << 63, u64::MAX]),
                shape: &[1, 3],
            })
            .expect("upload unsigned integer cast input");
        let to_u64 = block_on(provider.cast_to_integer(&signed, IntegerElementType::U64))
            .expect("cast int64 to uint64");
        let to_i64 = block_on(provider.cast_to_integer(&unsigned, IntegerElementType::I64))
            .expect("cast uint64 to int64");
        assert_eq!(
            block_on(provider.download_integer_exec(&to_u64))
                .expect("download uint64 cast")
                .data,
            HostIntegerDataOwned::U64(vec![0, 0, i64::MAX as u64])
        );
        assert_eq!(
            block_on(provider.download_integer_exec(&to_i64))
                .expect("download int64 cast")
                .data,
            HostIntegerDataOwned::I64(vec![0, i64::MAX, i64::MAX])
        );
    }
}
