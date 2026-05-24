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

}
