use super::*;

impl WgpuProvider {
    pub(crate) fn repmat_exec(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(
            !reps.is_empty(),
            "repmat: replication factors must be specified"
        );
        let entry = self.get_entry(handle)?;
        let orig_rank = if entry.shape.is_empty() {
            1
        } else {
            entry.shape.len()
        };
        let rank = if reps.len() == 1 {
            orig_rank.max(2)
        } else {
            orig_rank.max(reps.len())
        };
        if rank > crate::backend::wgpu::params::REPMAT_MAX_RANK {
            return Err(anyhow!(
                "repmat: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::REPMAT_MAX_RANK
            ));
        }

        let mut base_shape = vec![1usize; rank];
        for (idx, &dim) in entry.shape.iter().enumerate() {
            if idx < rank {
                base_shape[idx] = dim;
            }
        }

        let mut factors = vec![1usize; rank];
        if reps.len() == 1 {
            factors.fill(reps[0]);
        } else {
            for (idx, &factor) in reps.iter().enumerate() {
                if idx < rank {
                    factors[idx] = factor;
                }
            }
        }

        let mut new_shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let new_dim = base_shape[i]
                .checked_mul(factors[i])
                .ok_or_else(|| anyhow!("repmat: requested output exceeds GPU limits"))?;
            new_shape.push(new_dim);
        }

        let orig_total = base_shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow!("repmat: dimension product exceeds GPU limits"))
        })?;

        ensure!(
            orig_total == entry.len || (orig_total == 0 && entry.len == 0),
            "repmat: internal shape mismatch"
        );

        let new_total = new_shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow!("repmat: requested output exceeds GPU limits"))
        })?;

        if new_total > u32::MAX as usize {
            return Err(anyhow!("repmat: tensor too large for GPU kernel"));
        }

        if base_shape.iter().any(|&d| d > u32::MAX as usize)
            || new_shape.iter().any(|&d| d > u32::MAX as usize)
        {
            return Err(anyhow!(
                "repmat: dimensions exceed GPU kernel coordinate precision"
            ));
        }

        let mut base_strides = vec![0usize; rank];
        let mut stride = 1usize;
        for i in 0..rank {
            base_strides[i] = stride;
            stride = stride
                .checked_mul(base_shape[i].max(1))
                .ok_or_else(|| anyhow!("repmat: stride computation exceeds GPU limits"))?;
        }

        if base_strides.iter().any(|&s| s > u32::MAX as usize) {
            return Err(anyhow!(
                "repmat: source strides exceed GPU kernel coordinate precision"
            ));
        }

        let mut base_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        let mut new_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        for i in 0..rank {
            base_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(base_shape[i] as u32);
            new_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(new_shape[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(base_strides[i] as u32);
        }

        // Use checked allocation so we fail with a clear error instead of
        // creating an invalid WebGPU buffer (which later triggers a validation error).
        let out_buffer = self.create_storage_buffer_checked(new_total, "runmat-repmat-out")?;
        let out_shape = new_shape.clone();
        if new_total == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-repmat-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-repmat-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.repmat.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-repmat-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < new_total {
            let remaining = new_total - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::RepmatParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                base_shape: base_shape_arr,
                new_shape: new_shape_arr,
                base_strides: strides_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-repmat-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-repmat-bind"),
                    layout: &self.pipelines.repmat.layout,
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
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::repmat::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.repmat.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, new_total))
    }
    pub(crate) fn cat_exec(
        &self,
        dim: usize,
        inputs: &[GpuTensorHandle],
    ) -> Result<GpuTensorHandle> {
        ensure!(
            inputs.len() >= 2,
            "cat: at least two input arrays are required"
        );
        ensure!(dim >= 1, "cat: dimension must be >= 1");
        let dim_zero = dim - 1;

        let mut entries = Vec::with_capacity(inputs.len());
        for handle in inputs {
            entries.push(self.get_entry(handle)?);
        }

        let precision = entries[0].precision;
        for entry in &entries {
            ensure!(
                entry.precision == precision,
                "cat: input precision mismatch"
            );
        }

        let mut shapes: Vec<Vec<usize>> = entries.iter().map(|e| e.shape.clone()).collect();
        let mut rank = shapes
            .iter()
            .map(|s| if s.is_empty() { 0 } else { s.len() })
            .max()
            .unwrap_or(1);
        rank = rank.max(dim_zero + 1);
        if rank == 0 {
            rank = 1;
        }

        for shape in &mut shapes {
            if shape.is_empty() {
                shape.push(1);
            }
            while shape.len() < rank {
                shape.push(1);
            }
        }

        for (idx, shape) in shapes.iter().enumerate() {
            let expected = product_checked(shape)
                .ok_or_else(|| anyhow!("cat: input {} exceeds GPU limits", idx + 1))?;
            ensure!(
                expected == entries[idx].len,
                "cat: input {} has {} elements but the shape multiplies to {}",
                idx + 1,
                entries[idx].len,
                expected
            );
        }

        for axis in 0..rank {
            if axis == dim_zero {
                continue;
            }
            let reference = shapes[0][axis];
            for (idx, shape) in shapes.iter().enumerate().skip(1) {
                ensure!(
                    shape[axis] == reference,
                    "cat: dimension {} mismatch between input 1 (size {}) and input {} (size {})",
                    axis + 1,
                    reference,
                    idx + 1,
                    shape[axis]
                );
            }
        }

        let mut output_shape = shapes[0].clone();
        let mut concat_dim = 0usize;
        for shape in &shapes {
            concat_dim = concat_dim
                .checked_add(shape[dim_zero])
                .ok_or_else(|| anyhow!("cat: concatenated dimension exceeds GPU limits"))?;
        }
        output_shape[dim_zero] = concat_dim;

        let total_len = product_checked(&output_shape)
            .ok_or_else(|| anyhow!("cat: resulting array exceeds GPU limits"))?;

        let normalized_shape = normalize_concat_shape(output_shape.clone(), dim_zero);

        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-cat-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized_shape, 0));
        }

        let inner = product_checked(&output_shape[..dim_zero])
            .ok_or_else(|| anyhow!("cat: internal dimension overflow"))?;
        let outer = product_checked(&output_shape[dim_zero + 1..])
            .ok_or_else(|| anyhow!("cat: internal dimension overflow"))?;

        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-cat-encoder"),
                });

        let mut dst_offset_elems = 0usize;
        for outer_idx in 0..outer {
            for (entry, shape) in entries.iter().zip(shapes.iter()) {
                let mid = shape[dim_zero];
                let chunk = mid
                    .checked_mul(inner)
                    .ok_or_else(|| anyhow!("cat: chunk size overflow"))?;
                if chunk == 0 {
                    continue;
                }
                let src_offset = outer_idx
                    .checked_mul(chunk)
                    .ok_or_else(|| anyhow!("cat: source offset overflow"))?;
                let bytes = chunk
                    .checked_mul(self.element_size)
                    .ok_or_else(|| anyhow!("cat: copy size overflow"))?;
                let src_bytes = src_offset
                    .checked_mul(self.element_size)
                    .ok_or_else(|| anyhow!("cat: source offset overflow"))?;
                let dst_bytes = dst_offset_elems
                    .checked_mul(self.element_size)
                    .ok_or_else(|| anyhow!("cat: destination offset overflow"))?;
                encoder.copy_buffer_to_buffer(
                    entry.buffer.as_ref(),
                    src_bytes as u64,
                    out_buffer.as_ref(),
                    dst_bytes as u64,
                    bytes as u64,
                );
                dst_offset_elems = dst_offset_elems
                    .checked_add(chunk)
                    .ok_or_else(|| anyhow!("cat: destination offset overflow"))?;
            }
        }

        debug_assert_eq!(dst_offset_elems, total_len);

        self.submit(encoder);

        Ok(self.register_existing_buffer(out_buffer, normalized_shape, total_len))
    }
    pub(crate) fn kron_exec(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(left)?;
        let entry_b = self.get_entry(right)?;

        let rank = entry_a.shape.len().max(entry_b.shape.len()).max(1);
        ensure!(
            rank <= crate::backend::wgpu::params::KRON_MAX_RANK,
            "kron: rank {} exceeds GPU support (max {})",
            rank,
            crate::backend::wgpu::params::KRON_MAX_RANK
        );

        let mut shape_a = vec![1usize; rank];
        for (idx, &dim) in entry_a.shape.iter().enumerate() {
            if idx < rank {
                shape_a[idx] = dim;
            }
        }
        let mut shape_b = vec![1usize; rank];
        for (idx, &dim) in entry_b.shape.iter().enumerate() {
            if idx < rank {
                shape_b[idx] = dim;
            }
        }

        let mut shape_out = Vec::with_capacity(rank);
        for i in 0..rank {
            let dim = shape_a[i]
                .checked_mul(shape_b[i])
                .ok_or_else(|| anyhow!("kron: requested output exceeds GPU limits"))?;
            shape_out.push(dim);
        }

        let len_a = product_checked(&shape_a)
            .ok_or_else(|| anyhow!("kron: left operand size exceeds GPU limits"))?;
        let len_b = product_checked(&shape_b)
            .ok_or_else(|| anyhow!("kron: right operand size exceeds GPU limits"))?;
        let len_out = product_checked(&shape_out)
            .ok_or_else(|| anyhow!("kron: output size exceeds GPU limits"))?;

        ensure!(
            len_a == entry_a.len || (len_a == 0 && entry_a.len == 0),
            "kron: left operand shape mismatch"
        );
        ensure!(
            len_b == entry_b.len || (len_b == 0 && entry_b.len == 0),
            "kron: right operand shape mismatch"
        );

        if len_out == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-kron-out");
            return Ok(self.register_existing_buffer(out_buffer, shape_out, 0));
        }

        if len_out > u32::MAX as usize {
            return Err(anyhow!("kron: tensor too large for GPU kernel"));
        }

        for &dim in &shape_out {
            if dim > u32::MAX as usize {
                return Err(anyhow!(
                    "kron: dimensions exceed GPU kernel coordinate precision"
                ));
            }
        }

        let mut strides_a = vec![0usize; rank];
        let mut stride = 1usize;
        for i in 0..rank {
            strides_a[i] = stride;
            stride = stride
                .checked_mul(shape_a[i].max(1))
                .ok_or_else(|| anyhow!("kron: left stride overflow"))?;
        }

        let mut strides_b = vec![0usize; rank];
        stride = 1usize;
        for i in 0..rank {
            strides_b[i] = stride;
            stride = stride
                .checked_mul(shape_b[i].max(1))
                .ok_or_else(|| anyhow!("kron: right stride overflow"))?;
        }

        for &value in &strides_a {
            if value > u32::MAX as usize {
                return Err(anyhow!(
                    "kron: left strides exceed GPU kernel coordinate precision"
                ));
            }
        }
        for &value in &strides_b {
            if value > u32::MAX as usize {
                return Err(anyhow!(
                    "kron: right strides exceed GPU kernel coordinate precision"
                ));
            }
        }

        let mut shape_a_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut shape_b_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut shape_out_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut stride_a_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut stride_b_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        for i in 0..rank {
            shape_a_arr[i] = crate::backend::wgpu::params::AlignedU32::new(shape_a[i] as u32);
            shape_b_arr[i] = crate::backend::wgpu::params::AlignedU32::new(shape_b[i] as u32);
            shape_out_arr[i] = crate::backend::wgpu::params::AlignedU32::new(shape_out[i] as u32);
            stride_a_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides_a[i] as u32);
            stride_b_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides_b[i] as u32);
        }

        let out_buffer = self.create_storage_buffer_checked(len_out, "runmat-kron-out")?;
        let out_shape = shape_out.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-kron-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-kron-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.kron.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-kron-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len_out {
            let remaining = len_out - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::KronParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                shape_a: shape_a_arr,
                shape_b: shape_b_arr,
                shape_out: shape_out_arr,
                stride_a: stride_a_arr,
                stride_b: stride_b_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-kron-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-kron-bind"),
                    layout: &self.pipelines.kron.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::kron::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.kron.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, len_out))
    }

    pub(crate) fn transpose_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("transpose: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let len = entry.len;

        if let Some(info) = runmat_accelerate_api::handle_transpose_info(a) {
            let base_rows = info.base_rows;
            let base_cols = info.base_cols;
            let shape = vec![base_rows, base_cols];
            let handle = self.register_existing_buffer(entry.buffer.clone(), shape, len);
            runmat_accelerate_api::clear_handle_transpose(&handle);
            return Ok(handle);
        }

        let shape = vec![cols, rows];
        let handle = self.register_existing_buffer(entry.buffer.clone(), shape, len);
        runmat_accelerate_api::record_handle_transpose(&handle, rows, cols);
        Ok(handle)
    }

    pub(crate) fn permute_exec(
        &self,
        handle: &GpuTensorHandle,
        order: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(!order.is_empty(), "permute: order must not be empty");
        let rank = order.len();
        if rank > crate::backend::wgpu::params::PERMUTE_MAX_RANK {
            return Err(anyhow!(
                "permute: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::PERMUTE_MAX_RANK
            ));
        }

        let entry = self.get_entry(handle)?;
        ensure!(
            entry.shape.len() <= rank,
            "permute: order length ({}) must be at least ndims(A) ({})",
            rank,
            entry.shape.len()
        );

        let mut src_shape = entry.shape.clone();
        if src_shape.len() < rank {
            src_shape.extend(std::iter::repeat_n(1usize, rank - src_shape.len()));
        }

        let total: usize = src_shape.iter().copied().product();
        ensure!(
            total == entry.len,
            "permute: shape/product mismatch ({} vs {})",
            total,
            entry.len
        );
        if entry.len > u32::MAX as usize {
            return Err(anyhow!("permute: tensor too large for GPU kernel"));
        }

        let mut dst_shape = vec![0usize; rank];
        let mut seen = vec![false; rank];
        for (dst_dim, &src_dim) in order.iter().enumerate() {
            ensure!(
                src_dim < rank,
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
            return Err(anyhow!("permute: dimensions exceed GPU kernel limits"));
        }

        let mut src_strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (idx, &dim) in src_shape.iter().enumerate() {
            src_strides[idx] = stride;
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("permute: dimension product exceeds GPU limits"))?;
        }

        ensure!(
            dst_shape.iter().copied().product::<usize>() == entry.len,
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
        for i in 0..rank {
            src_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(src_shape[i] as u32);
            dst_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(dst_shape[i] as u32);
            order_arr[i] = crate::backend::wgpu::params::AlignedU32::new(order[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(src_strides[i] as u32);
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-permute-out");
        let out_shape = dst_shape;
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-permute-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-permute-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.permute.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-permute-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < entry.len {
            let remaining = entry.len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::PermuteParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                src_shape: src_shape_arr,
                dst_shape: dst_shape_arr,
                order: order_arr,
                src_strides: strides_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-permute-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-permute-bind"),
                    layout: &self.pipelines.permute.layout,
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
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::permute::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.permute.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
    }
    pub(crate) fn circshift_exec(
        &self,
        handle: &GpuTensorHandle,
        shifts: &[isize],
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let mut ext_shape = if entry.shape.is_empty() {
            vec![1usize]
        } else {
            entry.shape.clone()
        };

        if shifts.len() > ext_shape.len() {
            ext_shape.extend(std::iter::repeat_n(1usize, shifts.len() - ext_shape.len()));
        }

        let rank = ext_shape.len();
        if rank == 0 {
            return Ok(handle.clone());
        }
        if rank > crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK {
            return Err(anyhow!(
                "circshift: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK
            ));
        }

        let total = product_checked(&ext_shape)
            .ok_or_else(|| anyhow!("circshift: dimension product exceeds GPU limits"))?;
        ensure!(
            total == entry.len || (total == 0 && entry.len == 0),
            "circshift: shape/product mismatch ({} vs {})",
            total,
            entry.len
        );

        ensure!(
            entry.len <= u32::MAX as usize,
            "circshift: tensor too large for GPU kernel"
        );
        ensure!(
            ext_shape.iter().all(|&d| d <= u32::MAX as usize),
            "circshift: dimensions exceed GPU kernel limits"
        );

        let mut normalized = vec![0usize; rank];
        let mut has_effect = false;
        for axis in 0..rank {
            let size = ext_shape[axis];
            let shift = if axis < shifts.len() { shifts[axis] } else { 0 };
            if size <= 1 {
                normalized[axis] = 0;
                continue;
            }
            let size_isize = size as isize;
            let mut norm = shift % size_isize;
            if norm < 0 {
                norm += size_isize;
            }
            let norm_usize = norm as usize;
            normalized[axis] = norm_usize;
            if norm_usize != 0 {
                has_effect = true;
            }
        }

        if !has_effect {
            return Ok(handle.clone());
        }

        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for axis in 0..rank {
            strides[axis] = stride;
            stride = stride
                .checked_mul(ext_shape[axis].max(1))
                .ok_or_else(|| anyhow!("circshift: stride computation exceeds GPU limits"))?;
        }

        ensure!(
            strides.iter().all(|&s| s <= u32::MAX as usize),
            "circshift: strides exceed GPU kernel limits"
        );

        let mut shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        let mut shifts_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        for axis in 0..rank {
            shape_arr[axis] = crate::backend::wgpu::params::AlignedU32::new(ext_shape[axis] as u32);
            strides_arr[axis] = crate::backend::wgpu::params::AlignedU32::new(strides[axis] as u32);
            let denom = ext_shape[axis].max(1);
            shifts_arr[axis] =
                crate::backend::wgpu::params::AlignedU32::new((normalized[axis] % denom) as u32);
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-circshift-out");
        let out_shape = entry.shape.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-circshift-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-circshift-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.circshift.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-circshift-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < entry.len {
            let remaining = entry.len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::CircshiftParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                shape: shape_arr,
                strides: strides_arr,
                shifts: shifts_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-circshift-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-circshift-bind"),
                    layout: &self.pipelines.circshift.layout,
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
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::circshift::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.circshift.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }

    pub(crate) async fn tril_exec(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let rows = entry.shape.first().copied().unwrap_or(1);
        let cols = entry.shape.get(1).copied().unwrap_or(1);
        let plane = rows.saturating_mul(cols);
        if plane == 0 {
            return Ok(handle.clone());
        }
        if plane > entry.len {
            return self.tril_exec_fallback(handle, offset).await;
        }
        if entry.len % plane != 0 {
            return self.tril_exec_fallback(handle, offset).await;
        }
        let pages = entry.len / plane;
        let max_u32 = u32::MAX as usize;
        if rows > max_u32
            || cols > max_u32
            || plane > max_u32
            || entry.len > max_u32
            || pages > max_u32
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return self.tril_exec_fallback(handle, offset).await;
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -i32::MAX
        } else {
            offset as i32
        };

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-tril-out");
        let out_shape = entry.shape.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-tril-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-tril-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.tril.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-tril-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let plane_u32 = plane as u32;
        let mut offset_idx = 0usize;
        while offset_idx < entry.len {
            let remaining = entry.len - offset_idx;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::TrilParams {
                len: chunk_len as u32,
                start: offset_idx as u32,
                rows: rows_u32,
                cols: cols_u32,
                plane: plane_u32,
                diag_offset,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-tril-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-tril-bind"),
                    layout: &self.pipelines.tril.layout,
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
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::tril::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.tril.pipeline,
                &bind_group,
                workgroups,
            );
            offset_idx += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }

    async fn tril_exec_fallback(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned {
            mut data, shape, ..
        } = <Self as AccelProvider>::download(self, handle).await?;
        apply_tril_mask_host(&mut data, &shape, offset)?;
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        <Self as AccelProvider>::upload(self, &view)
    }

    pub(crate) async fn triu_exec(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let rows = entry.shape.first().copied().unwrap_or(1);
        let cols = entry.shape.get(1).copied().unwrap_or(1);
        let plane = rows.saturating_mul(cols);
        if plane == 0 {
            return Ok(handle.clone());
        }
        if plane > entry.len || entry.len % plane != 0 {
            return self.triu_exec_fallback(handle, offset).await;
        }
        let pages = entry.len / plane;
        let max_u32 = u32::MAX as usize;
        if rows > max_u32
            || cols > max_u32
            || plane > max_u32
            || entry.len > max_u32
            || pages > max_u32
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return self.triu_exec_fallback(handle, offset).await;
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -i32::MAX
        } else {
            offset as i32
        };

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-triu-out");
        let out_shape = entry.shape.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-triu-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-triu-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.triu.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-triu-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let plane_u32 = plane as u32;
        let mut offset_idx = 0usize;
        while offset_idx < entry.len {
            let remaining = entry.len - offset_idx;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::TriuParams {
                len: chunk_len as u32,
                start: offset_idx as u32,
                rows: rows_u32,
                cols: cols_u32,
                plane: plane_u32,
                diag_offset,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-triu-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-triu-bind"),
                    layout: &self.pipelines.triu.layout,
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
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::triu::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.triu.pipeline,
                &bind_group,
                workgroups,
            );
            offset_idx += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }

    async fn triu_exec_fallback(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned {
            mut data, shape, ..
        } = <Self as AccelProvider>::download(self, handle).await?;
        apply_triu_mask_host(&mut data, &shape, offset)?;
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        <Self as AccelProvider>::upload(self, &view)
    }
    pub(crate) fn flip_exec(
        &self,
        handle: &GpuTensorHandle,
        axes: &[usize],
    ) -> Result<GpuTensorHandle> {
        if axes.is_empty() {
            return Ok(handle.clone());
        }

        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let mut ext_shape = if entry.shape.is_empty() {
            vec![1usize]
        } else {
            entry.shape.clone()
        };

        if let Some(&max_axis) = axes.iter().max() {
            let needed = max_axis + 1;
            if needed > ext_shape.len() {
                ext_shape.extend(std::iter::repeat_n(1usize, needed - ext_shape.len()));
            }
        }

        let rank = ext_shape.len();
        if rank == 0 {
            return Ok(handle.clone());
        }
        if rank > crate::backend::wgpu::params::FLIP_MAX_RANK {
            return Err(anyhow!(
                "flip: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::FLIP_MAX_RANK
            ));
        }

        let total = product_checked(&ext_shape)
            .ok_or_else(|| anyhow!("flip: dimension product exceeds GPU limits"))?;
        ensure!(
            total == entry.len || (total == 0 && entry.len == 0),
            "flip: shape/product mismatch ({} vs {})",
            total,
            entry.len
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "flip: tensor too large for GPU kernel"
        );
        ensure!(
            ext_shape.iter().all(|&d| d <= u32::MAX as usize),
            "flip: dimensions exceed GPU kernel limits"
        );

        let mut flags = vec![false; rank];
        for &axis in axes {
            if axis < rank {
                flags[axis] = !flags[axis];
            }
        }
        let has_effect = flags
            .iter()
            .enumerate()
            .any(|(idx, flag)| *flag && ext_shape[idx] > 1);
        if !has_effect {
            return Ok(handle.clone());
        }

        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (idx, &dim) in ext_shape.iter().enumerate() {
            strides[idx] = stride;
            stride = stride
                .checked_mul(dim.max(1))
                .ok_or_else(|| anyhow!("flip: stride computation exceeds GPU limits"))?;
        }
        ensure!(
            strides.iter().all(|&s| s <= u32::MAX as usize),
            "flip: strides exceed GPU kernel limits"
        );

        let mut shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut flags_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        for i in 0..rank {
            shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(ext_shape[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides[i] as u32);
            flags_arr[i] =
                crate::backend::wgpu::params::AlignedU32::new(if flags[i] { 1 } else { 0 });
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-flip-out");
        let out_shape = entry.shape.clone();
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-flip-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-flip-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.flip.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-flip-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < entry.len {
            let remaining = entry.len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::FlipParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                shape: shape_arr,
                strides: strides_arr,
                flags: flags_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-flip-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-flip-bind"),
                    layout: &self.pipelines.flip.layout,
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
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::flip::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.flip.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }
}
