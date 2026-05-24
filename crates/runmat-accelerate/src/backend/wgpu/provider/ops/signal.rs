use super::*;

impl WgpuProvider {
    pub(crate) fn conv1d_exec(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        let entry_signal = self.get_entry(signal)?;
        let entry_kernel = self.get_entry(kernel)?;

        ensure!(
            entry_signal.precision == self.precision && entry_kernel.precision == self.precision,
            "conv1d: mixed precision tensors are not supported"
        );

        let signal_len = entry_signal.len;
        let kernel_len = entry_kernel.len;

        let (output_len, start_offset, _) = conv1d_window(signal_len, kernel_len, options.mode)?;

        if output_len == 0 {
            let out_shape = conv1d_output_shape(0, options.orientation);
            let out_buffer = self.create_storage_buffer(0, "runmat-conv1d-empty");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        ensure!(
            signal_len <= u32::MAX as usize
                && kernel_len <= u32::MAX as usize
                && output_len <= u32::MAX as usize
                && start_offset <= u32::MAX as usize,
            "conv1d: tensor exceeds GPU kernel limits"
        );

        let out_shape = conv1d_output_shape(output_len, options.orientation);
        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-conv1d-out")?;

        let params = Conv1dParams {
            signal_len: signal_len as u32,
            kernel_len: kernel_len as u32,
            output_len: output_len as u32,
            start_offset: start_offset as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-conv1d-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-conv1d-bind"),
                layout: &self.pipelines.conv1d.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry_signal.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: entry_kernel.buffer.as_ref().as_entire_binding(),
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
            output_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::conv::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.conv1d.pipeline,
            &bind_group,
            workgroups,
        );

        let handle = self.register_existing_buffer(out_buffer, out_shape, output_len);

        Ok(handle)
    }
    pub(crate) async fn iir_filter_exec(
        &self,
        b: &GpuTensorHandle,
        a: &GpuTensorHandle,
        x: &GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> Result<ProviderIirFilterResult> {
        let ProviderIirFilterOptions { dim, zi } = options;

        let entry_b = self.get_entry(b)?;
        let entry_a = self.get_entry(a)?;
        let entry_x = self.get_entry(x)?;

        ensure!(
            entry_b.precision == self.precision
                && entry_a.precision == self.precision
                && entry_x.precision == self.precision,
            "iir_filter: mixed precision tensors are not supported"
        );

        let nb = entry_b.len;
        let na = entry_a.len;
        ensure!(
            nb > 0,
            "iir_filter: numerator coefficients must not be empty"
        );
        ensure!(
            na > 0,
            "iir_filter: denominator coefficients must not be empty"
        );

        let b_host = <Self as AccelProvider>::download(self, b).await?;
        let a_host = <Self as AccelProvider>::download(self, a).await?;
        let a0 = *a_host
            .data
            .first()
            .ok_or_else(|| anyhow!("iir_filter: denominator coefficients cannot be empty"))?;
        ensure!(
            a0 != 0.0,
            "iir_filter: denominator coefficient a(1) must be non-zero"
        );

        let order = nb.max(na);
        ensure!(
            order <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );

        let mut b_norm = vec![0.0f64; order];
        let mut a_norm = vec![0.0f64; order];
        for i in 0..order {
            let b_coeff = if i < nb { b_host.data[i] } else { 0.0 };
            b_norm[i] = b_coeff / a0;
            if i == 0 {
                a_norm[0] = 1.0;
            } else {
                let a_coeff = if i < na { a_host.data[i] } else { 0.0 };
                a_norm[i] = a_coeff / a0;
            }
        }

        let state_len = order.saturating_sub(1);

        let mut shape_ext = entry_x.shape.clone();
        if dim >= shape_ext.len() {
            shape_ext.extend(std::iter::repeat_n(1, dim + 1 - shape_ext.len()));
        }
        ensure!(
            dim < shape_ext.len(),
            "iir_filter: dimension argument exceeds tensor rank"
        );
        let dim_idx = dim;
        let dim_len = shape_ext[dim_idx];

        let leading = if dim_idx == 0 {
            1usize
        } else {
            product_checked(&shape_ext[..dim_idx])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let trailing = if dim_idx + 1 >= shape_ext.len() {
            1usize
        } else {
            product_checked(&shape_ext[dim_idx + 1..])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let channel_count = leading
            .checked_mul(trailing)
            .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?;

        ensure!(
            shape_ext.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: tensors exceed supported rank for GPU kernel"
        );

        let state_shape = filter_state_shape(shape_ext.clone(), dim_idx, state_len);
        ensure!(
            state_shape.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: filter state rank exceeds GPU limits"
        );

        let state_total = if state_len == 0 {
            0usize
        } else {
            product_checked(&state_shape)
                .ok_or_else(|| anyhow!("iir_filter: filter state exceeds GPU limits"))?
        };

        if let Some(ref zi_handle) = zi {
            let zi_entry = self.get_entry(zi_handle)?;
            ensure!(
                zi_entry.precision == self.precision,
                "iir_filter: initial conditions use incompatible precision"
            );
            ensure!(
                shapes_compatible(&state_shape, &zi_entry.shape),
                "iir_filter: initial conditions are not compatible with the requested dimension"
            );
            let zi_dim = if dim_idx < zi_entry.shape.len() {
                zi_entry.shape[dim_idx]
            } else {
                1
            };
            ensure!(
                zi_dim == state_len,
                "iir_filter: initial conditions must have {} states along dimension {}",
                state_len,
                dim + 1
            );
            if state_total == 0 {
                ensure!(
                    zi_entry.len == 0,
                    "iir_filter: initial conditions have {} elements but zero were expected",
                    zi_entry.len
                );
            } else {
                ensure!(
                    zi_entry.len == state_total,
                    "iir_filter: initial state vector length mismatch (expected {}, found {})",
                    state_total,
                    zi_entry.len
                );
            }
        }

        ensure!(
            entry_x.len <= u32::MAX as usize,
            "iir_filter: signal length exceeds GPU limits"
        );
        ensure!(
            leading <= u32::MAX as usize
                && trailing <= u32::MAX as usize
                && channel_count <= u32::MAX as usize,
            "iir_filter: tensor exceeds GPU kernel limits"
        );
        ensure!(
            dim_len <= u32::MAX as usize,
            "iir_filter: dimension length exceeds GPU limits"
        );
        ensure!(
            state_len <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );
        ensure!(
            state_total <= u32::MAX as usize,
            "iir_filter: filter state size exceeds GPU limits"
        );

        let state_buffer_len = if state_len == 0 {
            0usize
        } else {
            state_len
                .checked_mul(channel_count)
                .ok_or_else(|| anyhow!("iir_filter: state buffer length overflow"))?
        };
        ensure!(
            state_buffer_len <= u32::MAX as usize,
            "iir_filter: state buffer length exceeds GPU limits"
        );

        let mut cleanup_handles: Vec<GpuTensorHandle> = Vec::new();
        let result = (|| -> Result<ProviderIirFilterResult> {
            let b_shape = [order, 1usize];
            let b_view = HostTensorView {
                data: &b_norm,
                shape: &b_shape,
            };
            let b_norm_handle = self.upload(&b_view)?;
            cleanup_handles.push(b_norm_handle.clone());

            let a_shape = [order, 1usize];
            let a_view = HostTensorView {
                data: &a_norm,
                shape: &a_shape,
            };
            let a_norm_handle = self.upload(&a_view)?;
            cleanup_handles.push(a_norm_handle.clone());

            let b_norm_entry = self.get_entry(&b_norm_handle)?;
            let a_norm_entry = self.get_entry(&a_norm_handle)?;

            let out_buffer = self.create_storage_buffer(entry_x.len, "runmat-iir-filter-out");
            let states_buffer =
                self.create_storage_buffer(state_buffer_len, "runmat-iir-filter-state");
            let final_state_buffer =
                self.create_storage_buffer(state_total, "runmat-iir-filter-final");

            let (zi_buffer, zi_present_flag) = if let Some(ref zi_handle) = zi {
                let zi_entry = self.get_entry(zi_handle)?;
                (zi_entry.buffer, 1u32)
            } else {
                (
                    self.create_storage_buffer(state_total, "runmat-iir-filter-zi"),
                    0u32,
                )
            };

            let mut signal_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in shape_ext.iter().enumerate() {
                signal_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }
            let mut state_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in state_shape.iter().enumerate() {
                state_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }

            let params = FilterParams {
                dim_len: dim_len as u32,
                leading: leading as u32,
                trailing: trailing as u32,
                order: order as u32,
                state_len: state_len as u32,
                signal_len: entry_x.len as u32,
                channel_count: channel_count as u32,
                zi_present: zi_present_flag,
                dim_idx: dim_idx as u32,
                rank: shape_ext.len() as u32,
                state_rank: state_shape.len() as u32,
                _pad: 0,
                signal_shape: signal_shape_arr,
                state_shape: state_shape_arr,
            };

            let params_buffer = self.uniform_buffer(&params, "runmat-iir-filter-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-iir-filter-bind"),
                    layout: &self.pipelines.filter.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_x.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_norm_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: a_norm_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: zi_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: states_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: final_state_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                channel_count as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::filter::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.filter.pipeline,
                &bind_group,
                workgroups,
            );

            let output_handle =
                self.register_existing_buffer(out_buffer, entry_x.shape.clone(), entry_x.len);
            let final_state_handle =
                self.register_existing_buffer(final_state_buffer, state_shape.clone(), state_total);

            Ok(ProviderIirFilterResult {
                output: output_handle,
                final_state: Some(final_state_handle),
            })
        })();

        for handle in cleanup_handles {
            let _ = self.free(&handle);
        }

        result
    }
    pub(crate) fn diff_once_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;

        let mut ext_shape = if entry.shape.is_empty() {
            vec![if entry.len == 0 { 1 } else { entry.len }]
        } else {
            entry.shape.clone()
        };
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];

        let mut out_shape = entry.shape.clone();
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        if len_dim <= 1 || entry.len == 0 {
            out_shape[dim] = out_shape[dim].saturating_sub(1);
            let out_len = product_checked(&out_shape).unwrap_or(0);
            let out_buffer = self.create_storage_buffer(out_len, "runmat-diff-empty");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim)
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("diff: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == entry.len,
            "diff: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            entry.len
        );

        let segment_out = len_dim - 1;
        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("diff: segment count exceeds GPU limits"))?;
        let out_len = segments
            .checked_mul(segment_out)
            .ok_or_else(|| anyhow!("diff: output size exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| anyhow!("diff: block size exceeds GPU limits"))?;

        ensure!(
            len_dim <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && stride_after <= u32::MAX as usize
                && segments <= u32::MAX as usize
                && block <= u32::MAX as usize
                && out_len <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "diff: tensor exceeds GPU kernel limits"
        );

        let out_buffer = self.create_storage_buffer(out_len, "runmat-diff-out");
        out_shape[dim] = len_dim - 1;

        let params = DiffParams {
            stride_before: stride_before as u32,
            segments: segments as u32,
            segment_len: len_dim as u32,
            segment_out: segment_out as u32,
            block: block as u32,
            total_out: out_len as u32,
            total_in: entry.len as u32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diff-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diff-bind"),
                layout: &self.pipelines.diff.layout,
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
            out_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diff::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diff.pipeline,
            &bind_group,
            workgroups,
        );

        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }

    pub(crate) fn diff_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        order: usize,
    ) -> Result<GpuTensorHandle> {
        if order == 0 {
            return Ok(handle.clone());
        }

        let mut current = handle.clone();
        let mut owns_current = false;
        for _ in 0..order {
            let next = self.diff_once_exec(&current, dim)?;
            if owns_current {
                let _ = self.free(&current);
            }
            current = next;
            owns_current = true;

            let entry = self.get_entry(&current)?;
            if entry.len == 0 {
                break;
            }
        }
        Ok(current)
    }

    pub(crate) fn gradient_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: f64,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;

        ensure!(
            entry.storage == GpuTensorStorage::Real,
            "gradient: complex GPU gradients are not implemented"
        );

        let mut ext_shape = normalize_gradient_shape(&entry.shape, entry.len);
        if ext_shape.is_empty() {
            ext_shape = vec![0, 0];
        }
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];
        let mut out_shape = normalize_gradient_shape(&entry.shape, entry.len);
        if out_shape.is_empty() {
            out_shape = vec![0, 0];
        }
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-gradient-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim.max(1))
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("gradient: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == entry.len,
            "gradient: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            entry.len
        );

        let block = stride_before
            .checked_mul(len_dim.max(1))
            .ok_or_else(|| anyhow!("gradient: block size exceeds GPU limits"))?;
        ensure!(
            len_dim <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && block <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "gradient: tensor exceeds GPU kernel limits"
        );

        let params_buffer = match self.precision {
            NumericPrecision::F64 => self.uniform_buffer(
                &GradientParamsF64 {
                    stride_before: stride_before as u32,
                    segment_len: len_dim as u32,
                    block: block as u32,
                    total: entry.len as u32,
                    spacing,
                    _pad0: 0.0,
                    _pad1: 0.0,
                    _pad2: 0.0,
                },
                "runmat-gradient-params",
            ),
            NumericPrecision::F32 => self.uniform_buffer(
                &GradientParamsF32 {
                    meta0: crate::backend::wgpu::params::PackedU32([
                        stride_before as u32,
                        len_dim as u32,
                        block as u32,
                        entry.len as u32,
                    ]),
                    meta1: crate::backend::wgpu::params::PackedF32([spacing as f32, 0.0, 0.0, 0.0]),
                },
                "runmat-gradient-params",
            ),
        };

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-gradient-bind"),
                layout: &self.pipelines.gradient.layout,
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
            entry.len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::gradient::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.gradient.pipeline,
            &bind_group,
            workgroups,
        );

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
    }

    pub(crate) fn cumsum_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumsum_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            let _ = self.free(&flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            let _ = self.free(&forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumsum: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cumsum-empty");
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumsum: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumsum: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumsum: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumsum: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumsum: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumsum: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumsum: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-cumsum-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumsumParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cumsum-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cumsum-bind"),
                layout: &self.pipelines.cumsum.layout,
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cumsum.pipeline,
            &bind_group,
            groups,
            "runmat-cumsum-encoder",
            "runmat-cumsum-pass",
        );
        Ok(self.register_existing_buffer(out_buffer, entry.shape, entry.len))
    }
    pub(crate) fn cumprod_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumprod_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            let _ = self.free(&flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            let _ = self.free(&forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumprod: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cumprod-empty");
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumprod: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumprod: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumprod: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumprod: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumprod: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumprod: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumprod: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-cumprod-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumprodParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cumprod-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cumprod-bind"),
                layout: &self.pipelines.cumprod.layout,
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cumprod.pipeline,
            &bind_group,
            groups,
            "runmat-cumprod-encoder",
            "runmat-cumprod-pass",
        );
        Ok(self.register_existing_buffer(out_buffer, entry.shape, entry.len))
    }
    pub(crate) fn cummin_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCumminResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummin-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummin-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCumminResult { values, indices });
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummin: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummin-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummin-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCumminResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummin: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummin: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummin: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummin: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummin: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummin: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummin: tensor too large for GPU kernel"
        );

        let values_buffer = self.create_storage_buffer(entry.len, "runmat-cummin-values");
        let indices_buffer = self.create_storage_buffer(entry.len, "runmat-cummin-indices");

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumminParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cummin-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cummin-bind"),
                layout: &self.pipelines.cummin.layout,
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
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cummin.pipeline,
            &bind_group,
            groups,
            "runmat-cummin-encoder",
            "runmat-cummin-pass",
        );

        let values = self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
        let indices = self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
        Ok(ProviderCumminResult { values, indices })
    }
    pub(crate) fn cummax_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCummaxResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummax-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummax-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCummaxResult { values, indices });
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummax: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummax-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummax-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCummaxResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummax: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummax: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummax: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummax: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummax: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummax: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummax: tensor too large for GPU kernel"
        );

        let values_buffer = self.create_storage_buffer(entry.len, "runmat-cummax-values");
        let indices_buffer = self.create_storage_buffer(entry.len, "runmat-cummax-indices");

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CummaxParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cummax-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cummax-bind"),
                layout: &self.pipelines.cummax.layout,
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
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cummax.pipeline,
            &bind_group,
            groups,
            "runmat-cummax-encoder",
            "runmat-cummax-pass",
        );

        let values = self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
        let indices = self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
        Ok(ProviderCummaxResult { values, indices })
    }
}
