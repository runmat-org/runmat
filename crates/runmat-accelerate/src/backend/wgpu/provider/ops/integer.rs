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
    pub(crate) fn integer_reduce_extrema_dim_exec(
        &self,
        select_min: bool,
        dim: usize,
        operation_name: &str,
        a: &GpuTensorHandle,
    ) -> Result<runmat_accelerate_api::ReduceDimResult> {
        let entry = self.get_entry_raw(a)?;
        let integer_type = entry
            .integer_type
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
            return Err(anyhow!(
                "{operation_name}: empty native integer extrema requires host empty-shape handling"
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
        IntegerElementType,
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
}
