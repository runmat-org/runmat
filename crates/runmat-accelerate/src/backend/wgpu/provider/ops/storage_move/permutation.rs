//! Existing device permutation kernel, independent of tensor registration.
//! An unregistered result is owned by its Arc and never enters a reusable pool.
use super::*;

impl WgpuProvider {
    /// Route-owned real floating permutation demand: one physical output and
    /// one immutable params uniform per dispatched chunk. A lazy inverse
    /// transpose is a view and does not call this kernel.
    pub(crate) fn real_permute_buffer_demand(&self, elements: u64) -> Option<u64> {
        let output = elements.max(1).checked_mul(self.element_size as u64)?;
        if elements == 0 {
            return Some(output);
        }
        let capacity =
            crate::backend::wgpu::config::elementwise_chunk_capacity(self.workgroup_config.scalar)?;
        output.checked_add(
            elements
                .div_ceil(capacity)
                .checked_mul(
                    std::mem::size_of::<crate::backend::wgpu::params::PermuteParams>() as u64,
                )?,
        )
    }

    pub(in super::super) fn permute_buffer_optional_exec(
        &self,
        entry: &BufferEntry,
        order: &[usize],
        allocated: &std::cell::Cell<bool>,
        on_buffer_created: &dyn Fn(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<Option<BufferEntry>> {
        ensure!(!order.is_empty(), "permute: order must not be empty");
        let logical_rank = order.len();
        let integer_type = entry.integer_type();
        let word_copy = integer_type.is_some() || entry.precision != self.precision;
        let word_factor = match (word_copy, entry.element_type) {
            (
                true,
                runmat_accelerate_api::NumericElementType::F64
                | runmat_accelerate_api::NumericElementType::I64
                | runmat_accelerate_api::NumericElementType::U64,
            ) => 2usize,
            _ => 1,
        };
        let storage = entry.storage;
        let component_factor = match storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let lane_factor = word_factor * component_factor;
        let kernel_rank = logical_rank + usize::from(lane_factor > 1);
        if kernel_rank > crate::backend::wgpu::params::PERMUTE_MAX_RANK {
            return Ok(None);
        }

        ensure!(
            entry.shape.len() <= logical_rank,
            "permute: order length ({}) must be at least ndims(A) ({})",
            logical_rank,
            entry.shape.len()
        );

        let mut src_shape = entry.shape.clone();
        if src_shape.len() < logical_rank {
            src_shape.extend(std::iter::repeat_n(1usize, logical_rank - src_shape.len()));
        }
        if src_shape
            .iter()
            .any(|&dimension| dimension > u32::MAX as usize)
        {
            return Ok(None);
        }

        let logical_total = if src_shape.contains(&0) {
            0
        } else {
            src_shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(|| anyhow!("permute: tensor storage length exceeds GPU limits"))?
        };
        let storage_total = logical_total
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("permute: tensor storage length exceeds GPU limits"))?;
        let registered_storage_len = if integer_type.is_some() {
            logical_total
        } else {
            logical_total
                .checked_mul(component_factor)
                .ok_or_else(|| anyhow!("permute: component count overflows"))?
        };
        ensure!(
            registered_storage_len == entry.len || (registered_storage_len == 0 && entry.len == 0),
            "permute: shape/storage mismatch ({} vs {})",
            registered_storage_len,
            entry.len
        );
        if storage_total > u32::MAX as usize {
            return Ok(None);
        }
        let submission_geometry = crate::backend::wgpu_geometry::permutation_submission_geometry(
            u64::try_from(storage_total).map_err(|_| anyhow!("permute: storage size overflow"))?,
            self.workgroup_config.scalar,
        )
        .ok_or_else(|| anyhow!("permute: invalid provider workgroup or submission geometry"))?;

        let mut dst_shape = vec![0usize; logical_rank];
        let mut seen = vec![false; logical_rank];
        for (dst_dim, &src_dim) in order.iter().enumerate() {
            ensure!(
                src_dim < logical_rank,
                "permute: invalid dimension index {}",
                src_dim + 1
            );
            ensure!(
                !seen[src_dim],
                "permute: duplicate dimension index {}",
                src_dim + 1
            );
            seen[src_dim] = true;
            dst_shape[dst_dim] = src_shape[src_dim];
        }
        ensure!(
            seen.iter().all(|&flag| flag),
            "permute: order must include every dimension exactly once"
        );

        if src_shape.iter().any(|&d| d > u32::MAX as usize)
            || dst_shape.iter().any(|&d| d > u32::MAX as usize)
        {
            return Ok(None);
        }

        let mut kernel_src_shape = Vec::with_capacity(kernel_rank);
        let mut kernel_dst_shape = Vec::with_capacity(kernel_rank);
        let mut kernel_order = Vec::with_capacity(kernel_rank);
        if lane_factor > 1 {
            kernel_src_shape.push(lane_factor);
            kernel_dst_shape.push(lane_factor);
            kernel_order.push(0usize);
            kernel_src_shape.extend(src_shape.iter().copied());
            kernel_dst_shape.extend(dst_shape.iter().copied());
            kernel_order.extend(order.iter().map(|&dim| dim + 1));
        } else {
            kernel_src_shape.extend(src_shape.iter().copied());
            kernel_dst_shape.extend(dst_shape.iter().copied());
            kernel_order.extend(order.iter().copied());
        }

        let mut src_strides = vec![0usize; kernel_rank];
        let mut stride = 1usize;
        if storage_total != 0 {
            for (idx, &dim) in kernel_src_shape.iter().enumerate() {
                src_strides[idx] = stride;
                stride = stride
                    .checked_mul(dim)
                    .ok_or_else(|| anyhow!("permute: dimension product exceeds GPU limits"))?;
            }
        }

        ensure!(
            product_checked(&kernel_dst_shape)
                .ok_or_else(|| anyhow!("permute: output dimension product exceeds GPU limits"))?
                == storage_total,
            "permute: output shape/product mismatch"
        );

        let mut src_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut dst_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut order_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        for i in 0..kernel_rank {
            src_shape_arr[i] =
                crate::backend::wgpu::params::AlignedU32::new(kernel_src_shape[i] as u32);
            dst_shape_arr[i] =
                crate::backend::wgpu::params::AlignedU32::new(kernel_dst_shape[i] as u32);
            order_arr[i] = crate::backend::wgpu::params::AlignedU32::new(kernel_order[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(src_strides[i] as u32);
        }

        let bundle = if word_copy {
            &self.pipelines.integer_permute
        } else {
            &self.pipelines.permute
        };
        // This operation owns two synchronous phases separated by the existing
        // progress boundary. Callers must not already hold an emission permit.
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer = if word_copy {
                self.create_integer_word_buffer(
                    emission,
                    storage_total,
                    "runmat-bitwise-permute-out",
                )?
            } else {
                self.create_storage_buffer(emission, storage_total, "runmat-permute-out")?
            };
            allocated.set(true);
            on_buffer_created(&out_buffer)?;
            if storage_total > 0 {
                let pipeline = bundle.get(emission)?;
                let mut enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-permute-noop"),
                    },
                );
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-permute-noop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                drop(pass);
                self.submit(emission, pending, enc)?;
            }
            Ok(out_buffer)
        })??;
        let out_shape = dst_shape;
        if storage_total == 0 {
            return Ok(Some(BufferEntry {
                logical: entry.logical,
                allocated_bytes: out_buffer.size(),
                buffer: out_buffer,
                shape: out_shape,
                len: 0,
                usage: BufferUsageClass::Generic,
                last_submission_id: None,
                ..entry.clone()
            }));
        }

        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            let pipeline = bundle.get(emission)?;
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-permute-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

            let chunk_capacity = submission_geometry.chunk_capacity as usize;
            let mut offset = 0usize;
            while offset < storage_total {
                let remaining = storage_total - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let params = crate::backend::wgpu::params::PermuteParams {
                    len: chunk_len as u32,
                    offset: offset as u32,
                    rank: kernel_rank as u32,
                    _pad: 0,
                    src_shape: src_shape_arr,
                    dst_shape: dst_shape_arr,
                    order: order_arr,
                    src_strides: strides_arr,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-permute-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-permute-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    self.workgroup_config.scalar,
                );
                crate::backend::wgpu::dispatch::permute::run(
                    emission,
                    pending,
                    &pipeline.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset += chunk_len;
            }

            Ok(())
        })??;
        Ok(Some(BufferEntry {
            logical: entry.logical,
            allocated_bytes: out_buffer.size(),
            buffer: out_buffer,
            shape: out_shape,
            usage: BufferUsageClass::Generic,
            last_submission_id: None,
            ..entry.clone()
        }))
    }
}
