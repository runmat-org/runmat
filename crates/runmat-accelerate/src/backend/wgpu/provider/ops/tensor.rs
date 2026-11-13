use super::*;

impl WgpuProvider {
    pub(crate) async fn sort_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        order: SortOrder,
        comparison: SortComparison,
    ) -> runmat_accelerate_api::ProviderExecutionResult<SortResult> {
        let host = self.download_exec(handle).await?;
        let shape = host.shape.clone();
        let (values, indices) = sort_host_tensor(&host.data, &host.shape, dim, order, comparison)?;
        Ok(SortResult {
            values: HostTensorOwned {
                data: values,
                shape: shape.clone(),
                storage: GpuTensorStorage::Real,
            },
            indices: HostTensorOwned {
                data: indices,
                shape,
                storage: GpuTensorStorage::Real,
            },
        })
    }

    pub(crate) async fn sort_rows_exec(
        &self,
        handle: &GpuTensorHandle,
        columns: &[SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> runmat_accelerate_api::ProviderExecutionResult<SortResult> {
        let host = self.download_exec(handle).await?;
        let SortRowsHostOutputs {
            values,
            indices,
            indices_shape,
        } = sort_rows_host(&host.data, &host.shape, columns, comparison)?;
        Ok(SortResult {
            values: HostTensorOwned {
                data: values,
                shape: host.shape.clone(),
                storage: GpuTensorStorage::Real,
            },
            indices: HostTensorOwned {
                data: indices,
                shape: indices_shape,
                storage: GpuTensorStorage::Real,
            },
        })
    }

    pub(crate) async fn unique_exec(
        &self,
        handle: &GpuTensorHandle,
        options: &UniqueOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<UniqueResult> {
        let host = self.download_exec(handle).await?;
        let HostTensorOwned { data, shape, .. } = host;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("unique: {e}"))?;
        let eval =
            runmat_runtime::builtins::array::sorting_sets::unique::unique_numeric_from_tensor(
                tensor, options,
            )
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
        eval.into_numeric_unique_result()
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
    }

    pub(crate) async fn ismember_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &IsMemberOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<IsMemberResult> {
        if !(runmat_accelerate_api::handle_integer_type(a).is_none()
            && runmat_accelerate_api::handle_integer_type(b).is_none())
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ismember: resident integer inputs require exact runtime fallback",
            )
            .into());
        }
        let host_a = self.download_exec(a).await?;
        let host_b = self.download_exec(b).await?;
        let tensor_a =
            Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("ismember: {e}"))?;
        let tensor_b =
            Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("ismember: {e}"))?;
        let eval =
            runmat_runtime::builtins::array::sorting_sets::ismember::ismember_numeric_from_tensors(
                tensor_a,
                tensor_b,
                options.rows,
            )
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
        eval.into_numeric_ismember_result()
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
    }

    pub(crate) async fn union_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &UnionOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<UnionResult> {
        if !(runmat_accelerate_api::handle_integer_type(a).is_none()
            && runmat_accelerate_api::handle_integer_type(b).is_none())
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "union: resident integer inputs require exact runtime fallback",
            )
            .into());
        }
        let host_a = self.download_exec(a).await?;
        let host_b = self.download_exec(b).await?;
        let tensor_a = Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("union: {e}"))?;
        let tensor_b = Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("union: {e}"))?;
        let eval =
            runmat_runtime::builtins::array::sorting_sets::union::union_numeric_from_tensors(
                tensor_a, tensor_b, options,
            )
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
        eval.into_numeric_union_result()
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
    }

    pub(crate) async fn setdiff_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &SetdiffOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<SetdiffResult> {
        if !(runmat_accelerate_api::handle_integer_type(a).is_none()
            && runmat_accelerate_api::handle_integer_type(b).is_none())
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "setdiff: resident integer inputs require exact runtime fallback",
            )
            .into());
        }
        let host_a = self.download_exec(a).await?;
        let host_b = self.download_exec(b).await?;
        let tensor_a =
            Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
        let tensor_b =
            Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
        let eval =
            runmat_runtime::builtins::array::sorting_sets::setdiff::setdiff_numeric_from_tensors(
                tensor_a, tensor_b, options,
            )
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
        eval.into_numeric_setdiff_result()
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
    }

    pub(crate) fn reshape_exec(
        &self,
        handle: &GpuTensorHandle,
        new_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let new_len = if new_shape.is_empty() {
            1
        } else {
            product_checked(new_shape)
                .ok_or_else(|| anyhow!("reshape: dimension product exceeds GPU limits"))?
        };
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
            let entry = self.get_entry_for_bitwise_move(handle, "reshape")?;
            ensure!(
                entry.shape.len() == 2
                    && entry.shape[0] == info.base_cols
                    && entry.shape[1] == info.base_rows,
                "reshape: transpose metadata mismatch for buffer {}",
                handle.buffer_id()
            );
            let storage = if runmat_accelerate_api::handle_storage(handle)
                == GpuTensorStorage::ComplexInterleaved
            {
                GpuTensorStorage::ComplexInterleaved
            } else {
                entry.storage
            };
            let lane_factor = match storage {
                GpuTensorStorage::Real => 1usize,
                GpuTensorStorage::ComplexInterleaved => 2usize,
            };
            let expected_len = new_len
                .checked_mul(lane_factor)
                .ok_or_else(|| anyhow!("reshape: dimension product exceeds GPU limits"))?;
            ensure!(
                entry.len == expected_len,
                "reshape: product of dimensions ({}) must equal original tensor length ({})",
                new_len,
                entry.len / lane_factor
            );
            ensure!(
                info.base_rows
                    .checked_mul(info.base_cols)
                    .and_then(|len| len.checked_mul(lane_factor))
                    == Some(entry.len),
                "reshape: transpose metadata product mismatch for buffer {}",
                handle.buffer_id()
            );

            let canonical = handle.with_view(
                handle
                    .view()
                    .clone()
                    .with_shape(vec![info.base_rows, info.base_cols])
                    .with_layout(runmat_accelerate_api::allocation::GpuTensorLayout::Contiguous),
            )?;
            let materialized = self.permute_exec(&canonical, &[1, 0])?;
            return self.reshape_exec(&materialized, new_shape);
        }
        let entry = self.get_entry_raw(handle)?;
        let storage = if runmat_accelerate_api::handle_storage(handle)
            == GpuTensorStorage::ComplexInterleaved
        {
            GpuTensorStorage::ComplexInterleaved
        } else {
            entry.storage
        };
        let lane_factor = match storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let expected_len = new_len
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("reshape: dimension product exceeds GPU limits"))?;
        ensure!(
            entry.len == expected_len,
            "reshape: product of dimensions ({}) must equal original tensor length ({})",
            new_len,
            entry.len / lane_factor
        );
        Ok(handle.with_view(handle.view().clone().with_shape(new_shape.to_vec()))?)
    }

    #[cfg(test)]
    pub(crate) fn repmat_exec(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.repmat_optional_exec(handle, reps, &std::cell::Cell::new(false), &|_| Ok(()))?
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "repmat: unsupported physical request",
                )
            })
    }

    pub(crate) fn repmat_optional_exec(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
        allocated: &std::cell::Cell<bool>,
        on_buffer_created: &dyn Fn(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<Option<GpuTensorHandle>> {
        ensure!(
            !reps.is_empty(),
            "repmat: replication factors must be specified"
        );
        if handle.shape().contains(&0) && handle.physical().element_capacity() != 0 {
            self.allocations.validate_empty_view(handle)?;
            return Ok(None);
        }
        let entry = self.get_entry_raw(handle)?;
        ensure!(
            handle.physical().element_type() == entry.element_type
                && handle.physical().storage() == entry.storage,
            "repmat: physical descriptor mismatch"
        );
        if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
            || handle.shape() != entry.shape
            || (entry.integer_type().is_none() && entry.precision != self.precision)
            || (entry.integer_type().is_some()
                && entry.storage == GpuTensorStorage::ComplexInterleaved)
        {
            return Ok(None);
        }
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
            return Ok(None);
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

        if base_shape
            .iter()
            .chain(&new_shape)
            .any(|&dimension| dimension > u32::MAX as usize)
        {
            return Ok(None);
        }
        let orig_total = if base_shape.contains(&0) {
            0
        } else {
            base_shape.iter().try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| anyhow!("repmat: dimension product exceeds GPU limits"))
            })?
        };

        let storage = if runmat_accelerate_api::handle_storage(handle)
            == GpuTensorStorage::ComplexInterleaved
            || entry.storage == GpuTensorStorage::ComplexInterleaved
        {
            GpuTensorStorage::ComplexInterleaved
        } else {
            GpuTensorStorage::Real
        };
        let storage_factor = match storage {
            runmat_accelerate_api::GpuTensorStorage::Real => 1usize,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let integer_type = entry.integer_type();
        let word_factor = integer_type.map_or(storage_factor, |element_type| match element_type {
            runmat_accelerate_api::IntegerElementType::I64
            | runmat_accelerate_api::IntegerElementType::U64 => 2,
            _ => 1,
        });
        let expected_input_len = orig_total
            .checked_mul(if integer_type.is_some() {
                1
            } else {
                storage_factor
            })
            .ok_or_else(|| anyhow!("repmat: input storage length exceeds GPU limits"))?;
        ensure!(
            expected_input_len == entry.len || (orig_total == 0 && entry.len == 0),
            "repmat: internal shape mismatch"
        );

        let new_total = if new_shape.contains(&0) {
            0
        } else {
            new_shape.iter().try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| anyhow!("repmat: requested output exceeds GPU limits"))
            })?
        };

        let storage_total = new_total
            .checked_mul(word_factor)
            .ok_or_else(|| anyhow!("repmat: requested output exceeds GPU limits"))?;

        if storage_total > u32::MAX as usize {
            return Ok(None);
        }

        if base_shape.iter().any(|&d| d > u32::MAX as usize)
            || new_shape.iter().any(|&d| d > u32::MAX as usize)
        {
            return Ok(None);
        }

        let mut base_strides = vec![0usize; rank];
        let mut stride = 1usize;
        if storage_total != 0 {
            for i in 0..rank {
                base_strides[i] = stride;
                stride = stride
                    .checked_mul(base_shape[i].max(1))
                    .ok_or_else(|| anyhow!("repmat: stride computation exceeds GPU limits"))?;
            }
        }

        if base_strides.iter().any(|&s| s > u32::MAX as usize) {
            return Ok(None);
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
        let pipeline_bundle = if integer_type.is_some() {
            &self.pipelines.integer_repmat
        } else {
            &self.pipelines.repmat
        };
        let launch = pipeline_bundle
            .linear_launch()
            .ok_or_else(|| anyhow!("repmat pipeline has no linear launch"))?;
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("repmat chunk capacity overflows"))?;

        // Use checked allocation so we fail with a clear error instead of
        // creating an invalid WebGPU buffer (which later triggers a validation error).
        let out_buffer = if integer_type.is_some() {
            self.with_emission(|emission, _| {
                let output = self.create_integer_word_buffer(
                    emission,
                    storage_total,
                    "runmat-integer-repmat-out",
                )?;
                allocated.set(true);
                on_buffer_created(&output)?;
                Ok::<_, anyhow::Error>(output)
            })??
        } else {
            self.with_emission(|emission, _| {
                let output = self.create_storage_buffer_checked(
                    emission,
                    storage_total,
                    "runmat-repmat-out",
                )?;
                allocated.set(true);
                on_buffer_created(&output)?;
                Ok::<_, anyhow::Error>(output)
            })??
        };
        let out_shape = new_shape.clone();
        if storage_total == 0 {
            return self
                .register_storage_move_buffer(
                    out_buffer.clone(),
                    &entry,
                    out_shape,
                    0,
                    out_buffer.size(),
                )
                .map(Some);
        }

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-repmat-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-repmat-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_bundle.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-repmat-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

            let mut offset = 0usize;
            while offset < storage_total {
                let remaining = storage_total - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let params = crate::backend::wgpu::params::RepmatParams {
                    len: chunk_len as u32,
                    offset: offset as u32,
                    rank: rank as u32,
                    storage_factor: word_factor as u32,
                    base_shape: base_shape_arr,
                    new_shape: new_shape_arr,
                    base_strides: strides_arr,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-repmat-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-repmat-bind"),
                        layout: &pipeline_bundle.get(emission)?.layout,
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
                let workgroups = launch
                    .dispatch_groups(chunk_len)
                    .ok_or_else(|| anyhow!("repmat launch exceeds enabled limits"))?;
                crate::backend::wgpu::dispatch::repmat::run(
                    emission,
                    pending,
                    &pipeline_bundle.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset += chunk_len;
            }

            Ok(())
        })??;

        self.register_storage_move_buffer(
            out_buffer.clone(),
            &entry,
            out_shape,
            if integer_type.is_some() {
                new_total
            } else {
                storage_total
            },
            out_buffer.size(),
        )
        .map(Some)
    }
    pub(crate) fn cat_exec(
        &self,
        dim: usize,
        inputs: &[GpuTensorHandle],
    ) -> Result<GpuTensorHandle> {
        self.cat_exec_inner(dim, inputs, &std::cell::Cell::new(false), None)
    }

    pub(crate) async fn cat_optional_exec(
        &self,
        dim: usize,
        inputs: &[GpuTensorHandle],
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let effected = std::cell::Cell::new(false);
        self.cat_exec_inner(dim, inputs, &effected, None)
            .map_err(|error| structural_execution_error(error, effected.get()))
    }

    fn cat_exec_inner(
        &self,
        dim: usize,
        inputs: &[GpuTensorHandle],
        effected: &std::cell::Cell<bool>,
        post_buffer_fault: Option<anyhow::Error>,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            inputs.len() >= 2,
            "cat: at least two input arrays are required"
        );
        ensure!(dim >= 1, "cat: dimension must be >= 1");
        let dim_zero = dim - 1;

        let mut entries = Vec::with_capacity(inputs.len());
        for handle in inputs {
            if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "cat: native copy requires contiguous input layout",
                ));
            }
            let entry = self.get_entry_for_bitwise_move(handle, "cat")?;
            if entry.storage != GpuTensorStorage::Real
                || entry.integer_type().is_some()
                || entry.logical
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "cat: provider copy requires real floating storage",
                ));
            }
            entries.push(entry);
        }

        let element_type = entries[0].element_type;
        for entry in &entries {
            if entry.element_type != element_type {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "cat: provider copy requires matching physical element types",
                ));
            }
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

        effected.set(true);
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer_checked_with_element_size(
                emission,
                total_len,
                "runmat-cat-out",
                crate::backend::wgpu::residency::BufferUsageClass::Generic,
                element_type.element_size(),
            )
        })??;
        if let Some(error) = post_buffer_fault {
            return Err(error);
        }
        if total_len == 0 {
            return self.register_storage_move_buffer(
                out_buffer.clone(),
                &entries[0],
                normalized_shape,
                0,
                out_buffer.size(),
            );
        }

        let inner = product_checked(&output_shape[..dim_zero])
            .ok_or_else(|| anyhow!("cat: internal dimension overflow"))?;
        let outer = product_checked(&output_shape[dim_zero + 1..])
            .ok_or_else(|| anyhow!("cat: internal dimension overflow"))?;

        self.with_emission(|emission, pending| -> Result<()> {
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-cat-encoder"),
                },
            );

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
                        .checked_mul(element_type.element_size())
                        .ok_or_else(|| anyhow!("cat: copy size overflow"))?;
                    let src_bytes = src_offset
                        .checked_mul(element_type.element_size())
                        .ok_or_else(|| anyhow!("cat: source offset overflow"))?;
                    let dst_bytes = dst_offset_elems
                        .checked_mul(element_type.element_size())
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

            self.submit(emission, pending, encoder)?;
            Ok(())
        })??;

        self.register_storage_move_buffer(
            out_buffer.clone(),
            &entries[0],
            normalized_shape,
            total_len,
            out_buffer.size(),
        )
    }
    pub(crate) async fn kron_exec(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let effected = std::cell::Cell::new(false);
        self.kron_exec_inner(left, right, &effected, None)
            .map_err(|error| structural_execution_error(error, effected.get()))
    }

    fn kron_exec_inner(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
        effected: &std::cell::Cell<bool>,
        mut post_buffer_fault: Option<anyhow::Error>,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.real_floating_entry(left)?;
        let entry_b = self.real_floating_entry(right)?;
        if entry_a.precision != entry_b.precision {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "kron: operands require matching physical floating precision",
            ));
        }
        let selected_precision = entry_a.precision;
        let pipeline = if selected_precision == self.precision {
            &self.pipelines.kron
        } else {
            &self
                .f32_real_floating
                .as_ref()
                .ok_or_else(|| {
                    runmat_accelerate_api::unsupported_provider_operation(
                        "kron: F32 physical kernel is unavailable",
                    )
                })?
                .kron
        };
        pipeline.preflight()?;

        let rank = entry_a.shape.len().max(entry_b.shape.len()).max(1);
        if rank > crate::backend::wgpu::params::KRON_MAX_RANK {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "kron: rank exceeds the provider kernel limit",
            ));
        }

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
            effected.set(true);
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked_with_element_size(
                    emission,
                    0,
                    "runmat-kron-out",
                    crate::backend::wgpu::residency::BufferUsageClass::Generic,
                    entry_a.element_type.element_size(),
                )
            })??;
            return self.register_real_floating_output(
                out_buffer,
                shape_out,
                0,
                selected_precision,
            );
        }

        if len_out > u32::MAX as usize {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "kron: tensor exceeds the provider kernel limit",
            ));
        }

        for &dim in &shape_out {
            if dim > u32::MAX as usize {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "kron: dimension exceeds the provider kernel coordinate limit",
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
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "kron: left stride exceeds the provider kernel coordinate limit",
                ));
            }
        }
        for &value in &strides_b {
            if value > u32::MAX as usize {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "kron: right stride exceeds the provider kernel coordinate limit",
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
        let launch = pipeline
            .linear_launch()
            .ok_or_else(|| anyhow!("kron pipeline has no linear launch"))?;
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("kron chunk capacity overflows"))?;

        effected.set(true);
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer_checked_with_element_size(
                emission,
                len_out,
                "runmat-kron-out",
                crate::backend::wgpu::residency::BufferUsageClass::Generic,
                entry_a.element_type.element_size(),
            )
        })??;
        if let Some(error) = post_buffer_fault.take() {
            return Err(error);
        }
        let out_shape = shape_out.clone();

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-kron-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-kron-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-kron-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

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
                let params_buffer = self.uniform_buffer(emission, &params, "runmat-kron-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-kron-bind"),
                        layout: &pipeline.get(emission)?.layout,
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
                let workgroups = launch
                    .dispatch_groups(chunk_len)
                    .ok_or_else(|| anyhow!("kron launch exceeds enabled limits"))?;
                crate::backend::wgpu::dispatch::kron::run(
                    emission,
                    pending,
                    &pipeline.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset += chunk_len;
            }

            Ok(())
        })??;

        self.register_real_floating_output(out_buffer, out_shape, len_out, selected_precision)
    }

    pub(crate) fn transpose_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_for_bitwise_move(a, "transpose")?;
        if entry.shape.len() != 2 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "transpose: only 2D tensors supported",
            ));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let storage =
            if runmat_accelerate_api::handle_storage(a) == GpuTensorStorage::ComplexInterleaved {
                GpuTensorStorage::ComplexInterleaved
            } else {
                entry.storage
            };

        if storage == GpuTensorStorage::ComplexInterleaved {
            return self.permute_exec(a, &[1, 0]);
        }

        if let Some(info) = runmat_accelerate_api::handle_transpose_info(a) {
            let base_rows = info.base_rows;
            let base_cols = info.base_cols;
            let shape = vec![base_rows, base_cols];
            return Ok(a.with_view(
                a.view()
                    .clone()
                    .with_shape(shape)
                    .with_layout(runmat_accelerate_api::allocation::GpuTensorLayout::Contiguous),
            )?);
        }

        let shape = vec![cols, rows];
        Ok(a.with_view(a.view().clone().with_shape(shape).with_layout(
            runmat_accelerate_api::allocation::GpuTensorLayout::Transpose(
                runmat_accelerate_api::TransposeInfo {
                    base_rows: rows,
                    base_cols: cols,
                },
            ),
        ))?)
    }

    pub(crate) fn permute_exec(
        &self,
        handle: &GpuTensorHandle,
        order: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.permute_optional_exec(handle, order, &std::cell::Cell::new(false), &|_| Ok(()))?
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "permute: unsupported physical request",
                )
            })
    }

    pub(crate) fn permute_optional_exec(
        &self,
        handle: &GpuTensorHandle,
        order: &[usize],
        allocated: &std::cell::Cell<bool>,
        on_buffer_created: &dyn Fn(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<Option<GpuTensorHandle>> {
        ensure!(!order.is_empty(), "permute: order must not be empty");
        if handle.shape().contains(&0) && handle.physical().element_capacity() != 0 {
            self.allocations.validate_empty_view(handle)?;
            return Ok(None);
        }
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
            // A 2-D lazy transpose already describes the permuted coordinates.
            // Its inverse restores the original contiguous interpretation of
            // the same authenticated allocation; feeding its logical shape to
            // the raw permutation shader would permute the underlying bytes
            // a second time.
            self.get_entry_raw(handle)?;
            if order == [0, 1] {
                return Ok(Some(handle.clone()));
            }
            if order == [1, 0] {
                return Ok(Some(
                    handle.with_view(
                        handle
                            .view()
                            .clone()
                            .with_shape(vec![info.base_rows, info.base_cols])
                            .with_layout(runmat_accelerate_api::GpuTensorLayout::Contiguous),
                    )?,
                ));
            }
        }
        if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous {
            return Ok(None);
        }
        let entry = self.get_entry_for_bitwise_move(handle, "permute")?;
        let Some(output) =
            self.permute_buffer_optional_exec(&entry, order, allocated, on_buffer_created)?
        else {
            return Ok(None);
        };
        self.register_storage_move_buffer(
            output.buffer.clone(),
            &output,
            output.shape.clone(),
            output.len,
            output.allocated_bytes,
        )
        .map(Some)
    }
    pub(crate) fn circshift_exec(
        &self,
        handle: &GpuTensorHandle,
        shifts: &[isize],
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_for_storage_move(handle, "circshift")?;
        let integer_type = entry.integer_type();
        let word_factor = integer_type.map_or(1usize, |element_type| match element_type {
            runmat_accelerate_api::IntegerElementType::I64
            | runmat_accelerate_api::IntegerElementType::U64 => 2,
            _ => 1,
        });
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

        let storage_len = entry
            .len
            .checked_mul(word_factor)
            .ok_or_else(|| anyhow!("circshift: tensor storage length exceeds GPU limits"))?;
        let pipeline_bundle = if integer_type.is_some() {
            &self.pipelines.integer_circshift
        } else {
            &self.pipelines.circshift
        };
        let launch = pipeline_bundle
            .linear_launch()
            .ok_or_else(|| anyhow!("circshift pipeline has no linear launch"))?;
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("circshift chunk capacity overflows"))?;
        let out_buffer = if integer_type.is_some() {
            self.with_emission(|emission, _| {
                self.create_integer_word_buffer(
                    emission,
                    storage_len,
                    "runmat-integer-circshift-out",
                )
            })??
        } else {
            self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, entry.len, "runmat-circshift-out")
            })??
        };
        let out_shape = ext_shape.clone();

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-circshift-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-circshift-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_bundle.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-circshift-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

            let mut offset = 0usize;
            while offset < entry.len {
                let remaining = entry.len - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let params = crate::backend::wgpu::params::CircshiftParams {
                    len: chunk_len as u32,
                    offset: offset as u32,
                    rank: rank as u32,
                    _pad: word_factor as u32,
                    shape: shape_arr,
                    strides: strides_arr,
                    shifts: shifts_arr,
                };
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-circshift-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-circshift-bind"),
                        layout: &pipeline_bundle.get(emission)?.layout,
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
                let workgroups = launch
                    .dispatch_groups(chunk_len)
                    .ok_or_else(|| anyhow!("circshift launch exceeds enabled limits"))?;
                crate::backend::wgpu::dispatch::circshift::run(
                    emission,
                    pending,
                    &pipeline_bundle.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset += chunk_len;
            }

            Ok(())
        })??;

        if let Some(element_type) = integer_type {
            self.register_integer_buffer(
                out_buffer,
                out_shape,
                entry.len,
                element_type,
                (storage_len as u64) * std::mem::size_of::<u32>() as u64,
            )
        } else {
            self.register_existing_buffer(out_buffer, out_shape, entry.len)
        }
    }

    pub(crate) async fn tril_exec(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let effected = std::cell::Cell::new(false);
        self.tril_exec_inner(handle, offset, &effected, None)
            .await
            .map_err(|error| structural_execution_error(error, effected.get()))
    }

    async fn tril_exec_inner(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
        effected: &std::cell::Cell<bool>,
        mut post_buffer_fault: Option<anyhow::Error>,
    ) -> Result<GpuTensorHandle> {
        if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "tril: provider kernel requires real floating storage",
            ));
        }
        let entry = self.real_floating_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }
        let selected_precision = entry.precision;
        let pipeline = if selected_precision == self.precision {
            &self.pipelines.tril
        } else {
            &self
                .f32_real_floating
                .as_ref()
                .ok_or_else(|| {
                    runmat_accelerate_api::unsupported_provider_operation(
                        "tril: F32 physical kernel is unavailable",
                    )
                })?
                .tril
        };
        pipeline.preflight()?;

        let rows = entry.shape.first().copied().unwrap_or(1);
        let cols = entry.shape.get(1).copied().unwrap_or(1);
        let plane = rows.saturating_mul(cols);
        if plane == 0 {
            return Ok(handle.clone());
        }
        if plane > entry.len {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "tril: unsupported provider plane geometry",
            ));
        }
        if entry.len % plane != 0 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "tril: unsupported provider plane geometry",
            ));
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
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "tril: dimensions exceed the provider kernel limit",
            ));
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -i32::MAX
        } else {
            offset as i32
        };
        let launch = pipeline
            .linear_launch()
            .ok_or_else(|| anyhow!("tril pipeline has no linear launch"))?;
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("tril chunk capacity overflows"))?;

        let out_buffer = self.with_emission(|emission, _| {
            effected.set(true);
            self.create_storage_buffer_checked_with_element_size(
                emission,
                entry.len,
                "runmat-tril-out",
                crate::backend::wgpu::residency::BufferUsageClass::Generic,
                entry.element_type.element_size(),
            )
        })??;
        if let Some(error) = post_buffer_fault.take() {
            return Err(error);
        }
        let out_shape = entry.shape.clone();

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-tril-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-tril-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-tril-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

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
                let params_buffer = self.uniform_buffer(emission, &params, "runmat-tril-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-tril-bind"),
                        layout: &pipeline.get(emission)?.layout,
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
                let workgroups = launch
                    .dispatch_groups(chunk_len)
                    .ok_or_else(|| anyhow!("tril launch exceeds enabled limits"))?;
                crate::backend::wgpu::dispatch::tril::run(
                    emission,
                    pending,
                    &pipeline.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset_idx += chunk_len;
            }

            Ok(())
        })??;

        let handle = self.register_real_floating_output(
            out_buffer,
            out_shape,
            entry.len,
            selected_precision,
        )?;

        Ok(handle)
    }

    pub(crate) async fn triu_exec(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let effected = std::cell::Cell::new(false);
        self.triu_exec_inner(handle, offset, &effected, None)
            .await
            .map_err(|error| structural_execution_error(error, effected.get()))
    }

    async fn triu_exec_inner(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
        effected: &std::cell::Cell<bool>,
        mut post_buffer_fault: Option<anyhow::Error>,
    ) -> Result<GpuTensorHandle> {
        if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "triu: provider kernel requires real floating storage",
            ));
        }
        let entry = self.real_floating_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }
        let selected_precision = entry.precision;
        let pipeline = if selected_precision == self.precision {
            &self.pipelines.triu
        } else {
            &self
                .f32_real_floating
                .as_ref()
                .ok_or_else(|| {
                    runmat_accelerate_api::unsupported_provider_operation(
                        "triu: F32 physical kernel is unavailable",
                    )
                })?
                .triu
        };
        pipeline.preflight()?;

        let rows = entry.shape.first().copied().unwrap_or(1);
        let cols = entry.shape.get(1).copied().unwrap_or(1);
        let plane = rows.saturating_mul(cols);
        if plane == 0 {
            return Ok(handle.clone());
        }
        if plane > entry.len || entry.len % plane != 0 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "triu: unsupported provider plane geometry",
            ));
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
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "triu: dimensions exceed the provider kernel limit",
            ));
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -i32::MAX
        } else {
            offset as i32
        };
        let launch = pipeline
            .linear_launch()
            .ok_or_else(|| anyhow!("triu pipeline has no linear launch"))?;
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("triu chunk capacity overflows"))?;

        let out_buffer = self.with_emission(|emission, _| {
            effected.set(true);
            self.create_storage_buffer_checked_with_element_size(
                emission,
                entry.len,
                "runmat-triu-out",
                crate::backend::wgpu::residency::BufferUsageClass::Generic,
                entry.element_type.element_size(),
            )
        })??;
        if let Some(error) = post_buffer_fault.take() {
            return Err(error);
        }
        let out_shape = entry.shape.clone();

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-triu-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-triu-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-triu-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

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
                let params_buffer = self.uniform_buffer(emission, &params, "runmat-triu-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-triu-bind"),
                        layout: &pipeline.get(emission)?.layout,
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
                let workgroups = launch
                    .dispatch_groups(chunk_len)
                    .ok_or_else(|| anyhow!("triu launch exceeds enabled limits"))?;
                crate::backend::wgpu::dispatch::triu::run(
                    emission,
                    pending,
                    &pipeline.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset_idx += chunk_len;
            }

            Ok(())
        })??;

        let handle = self.register_real_floating_output(
            out_buffer,
            out_shape,
            entry.len,
            selected_precision,
        )?;

        Ok(handle)
    }
}

fn structural_execution_error(
    error: anyhow::Error,
    after_effect: bool,
) -> runmat_accelerate_api::ProviderExecutionError {
    let error = runmat_accelerate_api::ProviderExecutionError::from(error);
    if after_effect {
        error.after_effect()
    } else {
        error
    }
}

#[cfg(test)]
#[path = "tensor/preparation_geometry_tests.rs"]
mod preparation_geometry_tests;

#[cfg(test)]
#[path = "tensor/linear_launch_tests.rs"]
mod linear_launch_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::provider::{
        register_retained_test_wgpu_provider, register_test_wgpu_provider, WgpuProviderOptions,
    };
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires a live WGPU adapter; run explicitly for triangular effect-phase qualification"]
    fn triangular_futures_are_lazy_and_post_buffer_failures_are_terminal() {
        let session = register_retained_test_wgpu_provider(WgpuProviderOptions::default())
            .expect("live WGPU adapter required for triangular effect-phase qualification");
        let provider = session.provider();
        let values_f32 = [1.0_f32, 2.0, 3.0, 4.0];
        let values_f64 = [1.0_f64, 2.0, 3.0, 4.0];
        let data = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => HostNumericDataView::F32(&values_f32),
            runmat_accelerate_api::ProviderPrecision::F64 => HostNumericDataView::F64(&values_f64),
        };
        let input = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .expect("input upload");
        let retained = provider.allocations.retained_bytes();

        // Constructing and abandoning the actual primitive future performs no work.
        drop(provider.tril_exec(&input, 0));
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(provider.allocations.retained_bytes(), retained);

        for lower in [true, false] {
            let unsupported = runmat_accelerate_api::unsupported_provider_operation(
                "injected triangular failure after actual output allocation",
            )
            .downcast::<runmat_accelerate_api::UnsupportedProviderOperation>()
            .expect("canonical unsupported source");
            let fault = anyhow::Error::new(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                unsupported,
            ));
            let effected = std::cell::Cell::new(false);
            let failure = if lower {
                block_on(provider.tril_exec_inner(&input, 0, &effected, Some(fault)))
            } else {
                block_on(provider.triu_exec_inner(&input, 0, &effected, Some(fault)))
            }
            .map_err(|error| structural_execution_error(error, effected.get()))
            .unwrap_err();
            assert!(failure.is_after_effect());
            assert!(!failure.is_unsupported());
            let mut source: &(dyn std::error::Error + 'static) = &failure;
            let mut broken_pipe = false;
            let mut unsupported = false;
            loop {
                if let Some(error) = source.downcast_ref::<std::io::Error>() {
                    broken_pipe |= error.kind() == std::io::ErrorKind::BrokenPipe;
                    unsupported |= error.get_ref().is_some_and(|cause| {
                        cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>()
                    });
                }
                unsupported |= source.is::<runmat_accelerate_api::UnsupportedProviderOperation>();
                let Some(next) = source.source() else { break };
                source = next;
            }
            assert!(broken_pipe && unsupported, "nested provider cause survives");
            input.allocation_lease().validate_live().unwrap();
            assert_eq!(provider.allocations.retained_bytes(), retained);
        }

        drop(provider.kron_exec(&input, &input));
        drop(provider.cat_optional_exec(2, &[input.clone(), input.clone()]));
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(provider.allocations.retained_bytes(), retained);

        let fault = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            runmat_accelerate_api::unsupported_provider_operation(
                "injected cat failure after actual output allocation",
            ),
        ));
        let effected = std::cell::Cell::new(false);
        let failure = provider
            .cat_exec_inner(2, &[input.clone(), input.clone()], &effected, Some(fault))
            .map_err(|error| structural_execution_error(error, effected.get()))
            .unwrap_err();
        assert!(failure.is_after_effect());
        assert!(!failure.is_unsupported());
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(provider.allocations.retained_bytes(), retained);
        let fault = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "injected kron failure after actual output allocation",
        ));
        let effected = std::cell::Cell::new(false);
        let failure = provider
            .kron_exec_inner(&input, &input, &effected, Some(fault))
            .map_err(|error| structural_execution_error(error, effected.get()))
            .unwrap_err();
        assert!(failure.is_after_effect());
        assert!(!failure.is_unsupported());
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(provider.allocations.retained_bytes(), retained);

        for selected in [
            runmat_accelerate_api::ProviderPrecision::F32,
            runmat_accelerate_api::ProviderPrecision::F64,
        ] {
            if selected == runmat_accelerate_api::ProviderPrecision::F64
                && provider.precision() != selected
            {
                continue;
            }
            let data = match selected {
                runmat_accelerate_api::ProviderPrecision::F32 => {
                    HostNumericDataView::F32(&values_f32)
                }
                runmat_accelerate_api::ProviderPrecision::F64 => {
                    HostNumericDataView::F64(&values_f64)
                }
            };
            let source = provider
                .upload_numeric_exec(&HostNumericTensorView {
                    data,
                    shape: &[2, 2],
                    storage: GpuTensorStorage::Real,
                })
                .expect("native floating source upload");
            assert_eq!(
                runmat_accelerate_api::handle_precision(&source),
                Some(selected)
            );
            let lower = block_on(provider.tril_exec(&source, 0)).expect("selected tril");
            let upper = block_on(provider.triu_exec(&source, 0)).expect("selected triu");
            let product = block_on(provider.kron_exec(&source, &source)).expect("selected kron");
            let concat = block_on(provider.cat_optional_exec(2, &[source.clone(), source.clone()]))
                .expect("selected cat");
            assert_eq!(
                runmat_accelerate_api::handle_precision(&concat),
                Some(selected)
            );
            assert_eq!(concat.shape(), &[2, 4]);
            for (handle, shape, expected) in [
                (&lower, vec![2, 2], vec![1.0, 2.0, 0.0, 4.0]),
                (&upper, vec![2, 2], vec![1.0, 0.0, 3.0, 4.0]),
                (
                    &concat,
                    vec![2, 4],
                    vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                ),
                (
                    &product,
                    vec![4, 4],
                    (0..4)
                        .flat_map(|col| {
                            (0..4).map(move |row| {
                                values_f64[(col / 2) * 2 + row / 2]
                                    * values_f64[(col % 2) * 2 + row % 2]
                            })
                        })
                        .collect(),
                ),
            ] {
                assert_eq!(
                    runmat_accelerate_api::handle_precision(handle),
                    Some(selected)
                );
                assert_eq!(handle.shape(), shape);
                let downloaded =
                    block_on(provider.download_exec(handle)).expect("selected download");
                assert_eq!(downloaded.shape, shape);
                assert_eq!(downloaded.data, expected);
            }
        }
    }

    #[test]
    fn transpose_complex_interleaved_downloads_by_logical_element() {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data = vec![
            1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0, -8.0, 9.0, 10.0, 11.0, -12.0,
        ];
        let handle = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &[2, 3],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("upload");

        let transposed = provider.transpose_exec(&handle).expect("transpose");
        assert_eq!(transposed.shape(), vec![3, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&transposed),
            GpuTensorStorage::ComplexInterleaved
        );
        assert!(runmat_accelerate_api::handle_transpose_info(&transposed).is_none());

        let host = block_on(provider.download_exec(&transposed)).expect("download");
        assert_eq!(host.shape, vec![3, 2]);
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(
            host.data,
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 3.0, -4.0, 7.0, -8.0, 11.0, -12.0,]
        );
    }

    #[test]
    fn transpose_complex_interleaved_feeds_downstream_abs_in_transposed_order() {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data = vec![
            1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0, -8.0, 9.0, 10.0, 11.0, -12.0,
        ];
        let handle = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &[2, 3],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("upload");

        let transposed = provider.transpose_exec(&handle).expect("transpose");
        let magnitudes = provider.unary_abs_exec(&transposed).expect("abs");
        let host = block_on(provider.download_exec(&magnitudes)).expect("download");
        assert_eq!(host.shape, vec![3, 2]);
        assert_eq!(host.storage, GpuTensorStorage::Real);
        let expected = [
            (1.0f64 * 1.0 + 2.0 * 2.0).sqrt(),
            (5.0f64 * 5.0 + 6.0 * 6.0).sqrt(),
            (9.0f64 * 9.0 + 10.0 * 10.0).sqrt(),
            (3.0f64 * 3.0 + 4.0 * 4.0).sqrt(),
            (7.0f64 * 7.0 + 8.0 * 8.0).sqrt(),
            (11.0f64 * 11.0 + 12.0 * 12.0).sqrt(),
        ];
        for (idx, (actual, expected)) in host.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "magnitude mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn reshape_materializes_lazy_transpose_before_metadata_update() {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let handle = provider
            .upload_exec(&HostTensorView {
                data: &data,
                shape: &[2, 3],
            })
            .expect("upload");

        let transposed = provider.transpose_exec(&handle).expect("transpose");
        assert_eq!(transposed.shape(), vec![3, 2]);
        assert!(runmat_accelerate_api::handle_transpose_info(&transposed).is_some());

        let reshaped = provider
            .reshape_exec(&transposed, &[2, 3])
            .expect("reshape lazy transpose");
        assert_eq!(reshaped.shape(), vec![2, 3]);
        assert!(runmat_accelerate_api::handle_transpose_info(&reshaped).is_none());
        let reshaped_host = block_on(provider.download_exec(&reshaped)).expect("download reshape");
        assert_eq!(reshaped_host.shape, vec![2, 3]);
        assert_eq!(reshaped_host.storage, GpuTensorStorage::Real);
        assert_eq!(reshaped_host.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

        let transposed_host =
            block_on(provider.download_exec(&transposed)).expect("download original transpose");
        assert_eq!(transposed_host.shape, vec![3, 2]);
        assert_eq!(transposed_host.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn permute_complex_interleaved_three_dimensional_stays_resident() {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data = vec![
            1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0, 8.0, -8.0,
        ];
        let handle = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &[2, 2, 2],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("upload");

        let permuted = provider.permute_exec(&handle, &[1, 0, 2]).expect("permute");
        assert_eq!(permuted.shape(), vec![2, 2, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&permuted),
            GpuTensorStorage::ComplexInterleaved
        );
        let host = block_on(provider.download_exec(&permuted)).expect("download");
        assert_eq!(host.shape, vec![2, 2, 2]);
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(
            host.data,
            vec![
                1.0, -1.0, 3.0, -3.0, 2.0, -2.0, 4.0, -4.0, 5.0, -5.0, 7.0, -7.0, 6.0, -6.0, 8.0,
                -8.0,
            ]
        );
    }

    #[test]
    fn permute_wide_uint64_uses_exact_integer_words_without_float_mirror() {
        use runmat_accelerate_api::{
            HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
        };

        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [u64::MAX, (1u64 << 63) + 17, (1u64 << 53) + 1, 19, 23, 29];
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[2, 3],
            })
            .expect("exact integer upload");
        let permuted = provider
            .permute_exec(&input, &[1, 0])
            .expect("integer permute");

        assert_ne!(
            input.buffer_id(),
            permuted.buffer_id(),
            "permute must materialize on device"
        );
        assert_eq!(permuted.shape(), vec![3, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&permuted),
            Some(IntegerElementType::U64)
        );
        let raw = provider
            .get_entry_raw(&permuted)
            .expect("resident integer output");
        assert_eq!(
            raw.allocated_bytes,
            6 * 2 * std::mem::size_of::<u32>() as u64
        );
        let gathered =
            block_on(provider.download_integer_exec(&permuted)).expect("exact integer download");
        assert_eq!(
            gathered.data,
            HostIntegerDataOwned::U64(vec![
                u64::MAX,
                (1u64 << 53) + 1,
                23,
                (1u64 << 63) + 17,
                19,
                29,
            ])
        );
    }

    #[test]
    fn flip_wide_int64_uses_exact_integer_words_without_float_mirror() {
        use runmat_accelerate_api::{
            HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
        };

        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [i64::MIN, -1, 0, i64::MAX, -2, 17];
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&values),
                shape: &[2, 3],
            })
            .expect("exact integer upload");
        let flipped = provider.flip_exec(&input, &[0]).expect("integer flip");

        assert_ne!(
            input.buffer_id(),
            flipped.buffer_id(),
            "flip must materialize on device"
        );
        assert_eq!(flipped.shape(), vec![2, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&flipped),
            Some(IntegerElementType::I64)
        );
        let raw = provider
            .get_entry_raw(&flipped)
            .expect("resident integer output");
        assert_eq!(
            raw.allocated_bytes,
            6 * 2 * std::mem::size_of::<u32>() as u64
        );
        let gathered =
            block_on(provider.download_integer_exec(&flipped)).expect("exact integer download");
        assert_eq!(
            gathered.data,
            HostIntegerDataOwned::I64(vec![-1, i64::MIN, i64::MAX, 0, 17, -2])
        );
    }

    #[test]
    fn repmat_wide_uint64_uses_exact_integer_words_without_float_mirror() {
        use runmat_accelerate_api::{
            HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
        };

        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [u64::MAX, (1u64 << 63) + 17, (1u64 << 53) + 1];
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[3, 1],
            })
            .expect("exact integer upload");
        let tiled = provider
            .repmat_exec(&input, &[1, 3])
            .expect("integer repmat");

        assert_ne!(
            input.buffer_id(),
            tiled.buffer_id(),
            "repmat must materialize on device"
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&tiled),
            Some(IntegerElementType::U64)
        );
        let raw = provider
            .get_entry_raw(&tiled)
            .expect("resident integer output");
        assert_eq!(
            raw.allocated_bytes,
            9 * 2 * std::mem::size_of::<u32>() as u64
        );
        assert_eq!(tiled.shape(), vec![3, 3]);
        let gathered =
            block_on(provider.download_integer_exec(&tiled)).expect("exact integer download");
        assert_eq!(
            gathered.data,
            HostIntegerDataOwned::U64(vec![
                u64::MAX,
                (1u64 << 63) + 17,
                (1u64 << 53) + 1,
                u64::MAX,
                (1u64 << 63) + 17,
                (1u64 << 53) + 1,
                u64::MAX,
                (1u64 << 63) + 17,
                (1u64 << 53) + 1
            ])
        );
    }

    #[test]
    fn circshift_wide_uint64_uses_exact_integer_words_without_float_mirror() {
        use runmat_accelerate_api::{
            HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
        };

        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [u64::MAX, (1u64 << 63) + 17, (1u64 << 53) + 1, 19, 23, 29];
        let input = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[2, 3],
            })
            .expect("exact integer upload");
        let shifted = provider
            .circshift_exec(&input, &[1, -1])
            .expect("integer circshift");

        assert_ne!(
            input.buffer_id(),
            shifted.buffer_id(),
            "circshift must materialize on device"
        );
        assert_eq!(shifted.shape(), vec![2, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&shifted),
            Some(IntegerElementType::U64)
        );
        let raw = provider
            .get_entry_raw(&shifted)
            .expect("resident integer output");
        assert_eq!(
            raw.allocated_bytes,
            6 * 2 * std::mem::size_of::<u32>() as u64
        );
        let gathered =
            block_on(provider.download_integer_exec(&shifted)).expect("exact integer download");
        assert_eq!(
            gathered.data,
            HostIntegerDataOwned::U64(vec![
                19,
                (1u64 << 53) + 1,
                29,
                23,
                (1u64 << 63) + 17,
                u64::MAX
            ])
        );
    }
}
