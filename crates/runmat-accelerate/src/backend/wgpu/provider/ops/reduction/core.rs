use super::*;

impl WgpuProvider {
    pub(crate) fn reduce_any_exec(
        &self,
        a: &GpuTensorHandle,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AnyOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AnyInclude
        };
        let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
        match self.reduce_dim_sum_mean_exec(&first, 1, op) {
            Ok(handle) => {
                let _ = self.free_exec(&first);
                Ok(handle)
            }
            Err(err) => {
                let _ = self.free_exec(&first);
                Err(err)
            }
        }
    }

    pub(crate) fn reduce_all_exec(
        &self,
        a: &GpuTensorHandle,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AllOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AllInclude
        };
        let total_elems = if a.shape.is_empty() {
            1
        } else {
            product_checked(&a.shape)
                .ok_or_else(|| anyhow!("reduce_all: tensor size exceeds GPU limits"))?
        };
        if total_elems == 0 {
            return self.fill_exec(&[1usize, 1usize], f64::NAN);
        }
        if a.shape.len() <= 2 {
            let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
            match self.reduce_dim_sum_mean_exec(&first, 1, op) {
                Ok(handle) => {
                    let _ = self.free_exec(&first);
                    Ok(handle)
                }
                Err(err) => {
                    let _ = self.free_exec(&first);
                    Err(err)
                }
            }
        } else {
            let original_shape = a.shape.clone();
            let flattened_shape = vec![total_elems, 1usize];
            let flattened = self.reshape_exec(a, &flattened_shape)?;
            let result = self.reduce_dim_sum_mean_exec(&flattened, 0, op);
            let _ = self.reshape_exec(a, &original_shape);
            result
        }
    }

    pub(crate) async fn reduce_median_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let host = self.download_exec(a).await?;
        let median = median_from_slice(&host.data);
        let data = [median];
        let shape = [1usize, 1usize];
        self.upload_exec(&HostTensorView {
            data: &data,
            shape: &shape,
        })
    }

    pub(crate) async fn reduce_median_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let host = self.download_exec(a).await?;
        if host.shape.len() != 2 {
            return Err(anyhow!("reduce_median_dim: only 2D supported"));
        }
        let rows = host.shape[0];
        let cols = host.shape[1];
        let mut scratch = Vec::<f64>::with_capacity(rows.max(cols));
        let (out, shape) = if dim <= 1 {
            let mut values = vec![f64::NAN; cols];
            for (c, value) in values.iter_mut().enumerate().take(cols) {
                scratch.clear();
                let mut saw_nan = false;
                for r in 0..rows {
                    let v = host.data[r + c * rows];
                    if v.is_nan() {
                        saw_nan = true;
                        scratch.clear();
                        break;
                    }
                    scratch.push(v);
                }
                *value = if saw_nan || scratch.is_empty() {
                    f64::NAN
                } else {
                    compute_median_inplace(&mut scratch)
                };
            }
            (values, vec![1usize, cols])
        } else {
            let mut values = vec![f64::NAN; rows];
            for (r, value) in values.iter_mut().enumerate().take(rows) {
                scratch.clear();
                let mut saw_nan = false;
                for c in 0..cols {
                    let v = host.data[r + c * rows];
                    if v.is_nan() {
                        saw_nan = true;
                        scratch.clear();
                        break;
                    }
                    scratch.push(v);
                }
                *value = if saw_nan || scratch.is_empty() {
                    f64::NAN
                } else {
                    compute_median_inplace(&mut scratch)
                };
            }
            (values, vec![rows, 1usize])
        };
        self.upload_exec(&HostTensorView {
            data: &out,
            shape: &shape,
        })
    }

    pub(crate) fn reduce_mean_global_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let sum_handle =
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let total_elems: usize = self.get_entry(a)?.len.max(1);
        let scalar = 1.0 / (total_elems as f64);
        let out = self.scalar_op_exec(
            crate::backend::wgpu::types::ScalarOpCode::Mul,
            &sum_handle,
            scalar,
        )?;
        let _ = self.free_exec(&sum_handle);
        Ok(out)
    }

    pub(crate) fn reduce_any_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AnyOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AnyInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    pub(crate) fn reduce_all_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AllOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AllInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_global_exec(
        &self,
        a: &GpuTensorHandle,
        op: crate::backend::wgpu::types::GlobalReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            log::debug!(
                "[reduce-global] in ptr={:p} len={} op={}",
                entry.buffer.as_ref(),
                entry.len,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_GLOBAL").is_ok() {
            return Err(anyhow!(
                "reduce_global disabled via RUNMAT_DISABLE_REDUCE_GLOBAL"
            ));
        }
        if entry.len == 0 {
            let default = match op {
                crate::backend::wgpu::types::GlobalReduceOp::Sum => 0.0,
                crate::backend::wgpu::types::GlobalReduceOp::Prod => 1.0,
                crate::backend::wgpu::types::GlobalReduceOp::Min => f64::INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::Max => f64::NEG_INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::CountNonZero => 0.0,
            };
            let data = [default];
            let shape = [1usize, 1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            return self.upload_exec(&view);
        }
        let mut current = if std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-global-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-global-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(enc);
            snap
        } else {
            entry.buffer.clone()
        };
        let mut current_len = entry.len;
        while current_len > 1 {
            let wg = crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE as usize;
            let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
            let elems_per_group = 2 * wg;
            let max_input_per_pass = max_groups * elems_per_group;

            let output_len_total = current_len.div_ceil(elems_per_group).max(1);
            // Metal-safe (opt-in): snapshot input buffer for this pass to avoid any SR/SRW conflicts
            // Enabled only if RUNMAT_FORCE_REDUCE_SNAPSHOT is set to avoid perf impact by default.
            let mut input_for_pass = current.clone();
            if self.adapter_info.backend == wgpu::Backend::Metal
                && std::env::var("RUNMAT_FORCE_REDUCE_SNAPSHOT").is_ok()
            {
                let size_bytes = (current_len * self.element_size) as u64;
                let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-input-snapshot"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                let mut enc =
                    self.device_ref()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-reduce-pass-input-snapshot-copy"),
                        });
                enc.copy_buffer_to_buffer(current.as_ref(), 0, snap.as_ref(), 0, size_bytes);
                self.submit(enc);
                input_for_pass = snap;
            }
            let mut out_buffer =
                self.create_storage_buffer_checked(output_len_total, "runmat-reduce-pass")?;
            // Prevent aliasing: output buffer must not be the same as input buffer
            if std::ptr::eq(out_buffer.as_ref(), input_for_pass.as_ref()) {
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: alias detected; current_ptr={:p} out_ptr={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                    eprintln!(
                        "[reduction] alias current={:p} out={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                }
                let size_bytes = (output_len_total * self.element_size) as u64;
                if size_bytes > self.adapter_limits.max_buffer_size {
                    return Err(gpu_per_buffer_limit_error(
                        "runmat-reduce-pass-unique",
                        size_bytes,
                        self.adapter_limits.max_buffer_size,
                    ));
                }
                out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-unique"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            let mut in_offset_elems = 0usize;
            let mut _out_offset_elems = 0usize;
            while in_offset_elems < current_len {
                let remain = current_len - in_offset_elems;
                let chunk_in = remain.min(max_input_per_pass);
                let chunk_out = chunk_in.div_ceil(elems_per_group).max(1);

                let params = crate::backend::wgpu::params::ReduceGlobalParams {
                    len: chunk_in as u32,
                    op: op as u32,
                    offset: in_offset_elems as u32,
                    total: current_len as u32,
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-reduce-global-params");

                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduce-global-bind"),
                        layout: &self.pipelines.reduce_global.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_for_pass.as_ref().as_entire_binding(),
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
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size_reduce(
                    chunk_in as u32,
                    crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
                );
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: dispatch groups={} current_ptr={:p} out_ptr={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                    eprintln!(
                        "[reduction] dispatch groups={} in={:p} out={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                }
                crate::backend::wgpu::dispatch::reduction::run_single_pass(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_global.pipeline,
                    &bind_group,
                    groups,
                );
                in_offset_elems += chunk_in;
                _out_offset_elems += chunk_out;
            }

            current = out_buffer;
            current_len = output_len_total;
        }
        Ok(self.register_existing_buffer(current, vec![1, 1], 1))
    }
    pub(crate) fn reduce_dim_sum_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_DIM").is_ok() {
            return Err(anyhow!("reduce_dim disabled via RUNMAT_DISABLE_REDUCE_DIM"));
        }
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        // Optional snapshot of input
        let in_buf = if std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-dim-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(enc);
            snap
        } else {
            entry.buffer.clone()
        };
        let mut out_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-out")?;
        // Prevent aliasing: output must not be identical to input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                log::debug!(
                    "reduce_dim_sum_mean_exec: alias detected; in_ptr={:p} out_ptr={:p} rows={} cols={} out_len={}",
                    entry.buffer.as_ref(),
                    out_buffer.as_ref(),
                    rows,
                    cols,
                    out_len
                );
            }
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-out-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if out_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-dim-bind"),
                layout: &self.pipelines.reduce_dim_sum_mean.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_ref().as_entire_binding(),
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
            out_len as u32,
            crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
        );
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean] in ptr={:p} out ptr={:p} rows={} cols={} dim={} groups={}",
                entry.buffer.as_ref(),
                out_buffer.as_ref(),
                rows,
                cols,
                reduce_dim,
                groups
            );
        }
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.reduce_dim_sum_mean.pipeline,
            &bind_group,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }
    pub(crate) fn reduce_dim_minmax_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceExtrema,
    ) -> Result<ReduceDimResult> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        let mut values_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-ext-values")?;
        let mut indices_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-ext-indices")?;
        // Prevent aliasing either output with input buffer
        if std::ptr::eq(values_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-ext-values-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            values_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-ext-values-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if std::ptr::eq(indices_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-ext-indices-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-ext-indices-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if out_len == 0 {
            let values_handle =
                self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
            let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
            return Ok(ReduceDimResult {
                values: values_handle,
                indices: indices_handle,
            });
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-ext-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-dim-ext-bind"),
                layout: &self.pipelines.reduce_dim_minmax.layout,
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
            out_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.reduce_dim_minmax.pipeline,
            &bind_group,
            groups,
        );
        let values_handle =
            self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
        let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
        Ok(ReduceDimResult {
            values: values_handle,
            indices: indices_handle,
        })
    }

    fn finish_std_from_sums_exec(
        &self,
        sum_handle: GpuTensorHandle,
        sum_sq_handle: GpuTensorHandle,
        sample_count: usize,
        normalization: ProviderStdNormalization,
        nan_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let inv_len = 1.0 / (sample_count as f64);

        let sum_squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_handle,
            &sum_handle,
        )?;
        let _ = self.free_exec(&sum_handle);

        let sum_shape = self.get_entry(&sum_squared)?.shape.clone();
        let scale_tensor = self.fill_exec(&sum_shape, inv_len)?;
        let sum_squared_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_squared,
            &scale_tensor,
        )?;
        let _ = self.free_exec(&scale_tensor);
        let variance_numer = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &sum_sq_handle,
            &sum_squared_scaled,
        )?;
        let _ = self.free_exec(&sum_sq_handle);
        let _ = self.free_exec(&sum_squared);
        let _ = self.free_exec(&sum_squared_scaled);

        let denom = match normalization {
            ProviderStdNormalization::Sample => {
                if sample_count > 1 {
                    (sample_count - 1) as f64
                } else {
                    1.0
                }
            }
            ProviderStdNormalization::Population => sample_count as f64,
        };
        if denom == 0.0 {
            let _ = self.free_exec(&variance_numer);
            return self.fill_exec(nan_shape, f64::NAN);
        }

        let variance_shape = self.get_entry(&variance_numer)?.shape.clone();
        let denom_tensor = self.fill_exec(&variance_shape, denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &variance_numer,
            &denom_tensor,
        )?;
        let _ = self.free_exec(&denom_tensor);
        let _ = self.free_exec(&variance_numer);

        let abs_variance =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, &variance)?;
        let variance_plus_abs = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
        )?;
        let _ = self.free_exec(&abs_variance);
        let _ = self.free_exec(&variance);

        let half_shape = self.get_entry(&variance_plus_abs)?.shape.clone();
        let half_tensor = self.fill_exec(&half_shape, 0.5)?;
        let clamped = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
        )?;
        let _ = self.free_exec(&half_tensor);
        let _ = self.free_exec(&variance_plus_abs);

        let std_handle =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &clamped)?;
        let _ = self.free_exec(&clamped);
        Ok(std_handle)
    }

    pub(crate) fn reduce_std_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(anyhow!(
                "reduce_std_dim: omitnan is not supported by the wgpu provider"
            ));
        }
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce_std_dim: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_len = match dim {
            0 => rows,
            1 => cols,
            _ => return Err(anyhow!("reduce_std_dim: only dims 0 or 1 supported")),
        };
        let out_shape = if dim == 0 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        if reduce_len == 0 {
            return self.fill_exec(&out_shape, f64::NAN);
        }

        let sum_handle =
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)?;
        let squared = self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, a)?;
        let sum_sq_handle = self.reduce_dim_sum_mean_exec(
            &squared,
            dim,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let _ = self.free_exec(&squared);
        self.finish_std_from_sums_exec(
            sum_handle,
            sum_sq_handle,
            reduce_len,
            normalization,
            &out_shape,
        )
    }

    pub(crate) fn reduce_std_exec(
        &self,
        a: &GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(anyhow!(
                "reduce_std: omitnan is not supported by the wgpu provider"
            ));
        }
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let scalar_shape = [1usize, 1usize];
        if len == 0 {
            return self.fill_exec(&scalar_shape, f64::NAN);
        }

        let sum_handle =
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let squared = self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, a)?;
        let sum_sq_handle =
            self.reduce_global_exec(&squared, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let _ = self.free_exec(&squared);
        self.finish_std_from_sums_exec(sum_handle, sum_sq_handle, len, normalization, &scalar_shape)
    }
}
