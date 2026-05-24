    fn find(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        self.find_exec(a, limit, direction)
    }
    fn issymmetric(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderSymmetryKind,
        tolerance: f64,
    ) -> Result<bool> {
        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_symmetry_shape(&entry.shape).map_err(|e| anyhow!("issymmetric: {e}"))?;
        if rows != cols {
            return Ok(false);
        }
        if rows == 0 || cols == 0 {
            return Ok(true);
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("issymmetric: matrix dimensions too large"))?;
        if total > entry.len {
            return Err(anyhow!(
                "issymmetric: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("issymmetric: matrix exceeds GPU limits"));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "issymmetric: tolerance must be finite and non-negative"
            ));
        }

        let mode = match kind {
            ProviderSymmetryKind::Symmetric => 0u32,
            ProviderSymmetryKind::Skew => 1u32,
        };

        let output_init = [1u32];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runmat-issymmetric-output"),
                contents: cast_slice(&output_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let pipeline = &self.pipelines.symmetry;
        match entry.precision {
            NumericPrecision::F64 => {
                let params = SymmetryParamsF64 {
                    rows: rows as u32,
                    cols: cols as u32,
                    len: total as u32,
                    mode,
                    tolerance,
                    _pad: 0.0,
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-issymmetric-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-issymmetric-bind-group-f64"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline.pipeline,
                    &bind_group,
                    groups,
                );
            }
            NumericPrecision::F32 => {
                let tol32 = tolerance.min(f32::MAX as f64).max(0.0) as f32;
                let params = SymmetryParamsF32 {
                    rows: rows as u32,
                    cols: cols as u32,
                    len: total as u32,
                    mode,
                    tolerance: tol32,
                    _pad: [0.0; 3],
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-issymmetric-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-issymmetric-bind-group-f32"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline.pipeline,
                    &bind_group,
                    groups,
                );
            }
        }

        let staging_size = std::mem::size_of::<u32>() as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-issymmetric-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-issymmetric-copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
        self.submit(encoder);

        let bytes = self.map_readback_bytes_sync(staging, staging_size, "issymmetric")?;
        let words: &[u32] = cast_slice(&bytes);
        let flag = words.first().copied().unwrap_or(0);

        Ok(flag != 0)
    }

    fn ishermitian<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> AccelProviderFuture<'a, bool> {
        Box::pin(async move {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(anyhow!(
                    "ishermitian: tolerance must be finite and non-negative"
                ));
            }
            let host = <Self as AccelProvider>::download(self, matrix).await?;
            let skew = matches!(kind, ProviderHermitianKind::Skew);
            ishermitian_host_real_data(&host.shape, &host.data, skew, tolerance)
                .map_err(|e| anyhow!(e))
        })
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        self.bandwidth_exec(matrix)
    }

    fn sym_rcm<'a>(&'a self, matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, matrix).await?;
            symrcm_host_real_data(&host.shape, &host.data).map_err(|e| anyhow!(e))
        })
    }
    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> Result<f64> {
        let entry = self.get_entry(h)?;
        let elem_size = match entry.precision {
            NumericPrecision::F64 => std::mem::size_of::<f64>() as u64,
            NumericPrecision::F32 => std::mem::size_of::<f32>() as u64,
        };
        let total_bytes = (linear_index as u64)
            .checked_mul(elem_size)
            .ok_or_else(|| anyhow!("read_scalar: index overflow"))?;
        if (linear_index + 1) > entry.len {
            return Err(anyhow!(
                "read_scalar: index {} out of bounds (len {})",
                linear_index + 1,
                entry.len
            ));
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-read-scalar-staging"),
            size: elem_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-read-scalar-enc"),
            });
        encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), total_bytes, &staging, 0, elem_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, elem_size, "read_scalar")?;
        let value = match entry.precision {
            NumericPrecision::F64 => {
                let words: &[f64] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0)
            }
            NumericPrecision::F32 => {
                let words: &[f32] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0) as f64
            }
        };
        Ok(value)
    }

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_elementwise_exec(shader, inputs, output_shape, len);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    fn fused_elementwise_multi(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        num_outputs: usize,
    ) -> Result<Vec<GpuTensorHandle>> {
        let start = Instant::now();
        let result =
            self.fused_elementwise_multi_exec(shader, inputs, output_shape, len, num_outputs);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
                ("num_outputs", num_outputs as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise_multi", &shape, &tuning);
        }
        result
    }

    fn map_nan_to_zero(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-nan-to-zero-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F32,
        };
        self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)
    }
    fn not_nan_mask(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-not-nan-mask-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F32,
        };
        self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)
    }

    fn fused_reduction(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_reduction_exec(
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        );
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_reduction_duration(elapsed);
            let actual_wg = if workgroup_size == 0 {
                self.default_reduction_workgroup_size()
            } else {
                workgroup_size
            } as u64;
            let flavor_tag = match flavor {
                ReductionFlavor::Sum => 0,
                ReductionFlavor::Mean => 1,
                ReductionFlavor::CustomScale(_) => 2,
            };
            let shape = [
                ("reduce_len", reduce_len as u64),
                ("slices", num_slices as u64),
                ("rank", output_shape.len() as u64),
            ];
            let tuning = [("wg", actual_wg), ("flavor", flavor_tag)];
            self.record_kernel_launch_basic("fused_reduction", &shape, &tuning);
        }
        result
    }
    fn warmup(&self) {
        if std::env::var("RUNMAT_WGPU_SKIP_WARMUP")
            .ok()
            .and_then(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    None
                } else if trimmed.eq_ignore_ascii_case("1")
                    || trimmed.eq_ignore_ascii_case("true")
                    || trimmed.eq_ignore_ascii_case("yes")
                {
                    Some(true)
                } else if trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("no")
                {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(false)
        {
            log::info!("RunMat Accelerate: skipping wgpu warmup (RUNMAT_WGPU_SKIP_WARMUP=1)");
            return;
        }

        let start = Instant::now();
        self.warmup_from_disk();
        // Proactively warm common pipelines used by normalization and reduction chains
        let pl = &self.pipelines;
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary_broadcast.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.unary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.scalar.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_dim_sum_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_nd_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_global.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.fill.pipeline,
        );

        let ms = start.elapsed().as_millis() as u64;
        self.metrics.set_last_warmup_millis(ms);
    }
    fn fused_cache_counters(&self) -> (u64, u64) {
        self.metrics.counters()
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        Some(self.metrics.last_warmup_millis())
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        let (fusion_hits, fusion_misses) = self.metrics.counters();
        let (bind_hits, bind_misses) = self.bind_group_cache.counters();
        // Build per-layout telemetry by resolving layout pointers to tags
        let mut by_layout: Vec<runmat_accelerate_api::BindGroupLayoutTelemetry> = Vec::new();
        let per = self.bind_group_cache.per_layout_counters();
        if let Ok(tags) = self.bind_group_layout_tags.lock() {
            for (ptr, (h, m)) in per {
                let tag = tags
                    .get(&ptr)
                    .cloned()
                    .unwrap_or_else(|| format!("layout_ptr_{:#x}", ptr));
                by_layout.push(runmat_accelerate_api::BindGroupLayoutTelemetry {
                    tag,
                    hits: h,
                    misses: m,
                });
            }
        }
        self.telemetry.snapshot(
            fusion_hits,
            fusion_misses,
            bind_hits,
            bind_misses,
            Some(by_layout),
        )
    }

    fn reset_telemetry(&self) {
        self.telemetry.reset();
        self.metrics.reset();
        self.bind_group_cache.reset_counters();
    }

    fn default_reduction_workgroup_size(&self) -> u32 {
        self.reduction_workgroup_size_default
    }

    fn two_pass_threshold(&self) -> usize {
        self.reduction_two_pass_threshold
    }

    fn reduction_two_pass_mode(&self) -> ReductionTwoPassMode {
        self.reduction_two_pass_mode
    }

    fn scatter_column(
        &self,
        matrix: &GpuTensorHandle,
        col_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.scatter_column_exec(matrix, col_index, values)
    }
    fn scatter_row(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.scatter_row_exec(matrix, row_index, values)
    }

    fn sub2ind(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.sub2ind_exec(dims, strides, inputs, scalar_mask, len, output_shape)
    }

    fn supports_ind2sub(&self) -> bool {
        true
    }

    fn ind2sub(
        &self,
        dims: &[usize],
        strides: &[usize],
        indices: &GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &[usize],
    ) -> Result<Vec<GpuTensorHandle>> {
        self.ind2sub_exec(dims, strides, indices, total, len, output_shape)
    }

    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let _span = info_span!(
            "gpu.transfer.upload",
            shape = ?host.shape,
            len = host.data.len()
        )
        .entered();
        let len = host.data.len();
        let shape = host.shape.to_vec();
        let bytes = (len as u64).saturating_mul(self.element_size as u64);
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                "upload",
                bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        let buffer =
            if len == 0 {
                self.create_storage_buffer(0, "runmat-upload-empty")
            } else {
                match self.precision {
                    NumericPrecision::F64 => {
                        let contents = cast_slice(host.data);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                    NumericPrecision::F32 => {
                        let data_f32: Vec<f32> = host.data.iter().map(|v| *v as f32).collect();
                        let contents = cast_slice(&data_f32);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                }
            };
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_existing_buffer(buffer, shape, len))
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let span = info_span!(
                "gpu.transfer.download",
                shape = ?h.shape,
                buffer_id = h.buffer_id
            );
            let entry = {
                let _guard = span.enter();
                log::trace!("wgpu download id={} shape={:?}", h.buffer_id, &h.shape);
                self.get_entry(h)?
            };
            if let Some(last) = entry.last_submission_id {
                log::trace!(
                    "wgpu download id={} last_submission_id={}",
                    h.buffer_id,
                    last
                );
            } else {
                log::trace!("wgpu download id={} last_submission_id=<none>", h.buffer_id);
            }
            if entry.len == 0 {
                return Ok(HostTensorOwned {
                    data: Vec::new(),
                    shape: h.shape.clone(),
                    storage: runmat_accelerate_api::handle_storage(h),
                });
            }

            let size_bytes = (entry.len * self.element_size) as u64;

            // Shared post-map readback logic: decode mapped bytes, unmap, record telemetry,
            // apply transpose metadata, and return host tensor.
            let finish_readback =
                |staging: wgpu::Buffer, size_bytes: u64| -> Result<HostTensorOwned> {
                    let slice = staging.slice(..);
                    let data = slice.get_mapped_range();
                    log::trace!(
                        "wgpu download copying data id={} len={} bytes={}",
                        h.buffer_id,
                        entry.len,
                        size_bytes
                    );

                    let mut out = vec![0.0f64; entry.len];
                    match entry.precision {
                        NumericPrecision::F64 => out.copy_from_slice(cast_slice(&data)),
                        NumericPrecision::F32 => {
                            let f32_slice: &[f32] = cast_slice(&data);
                            for (dst, src) in out.iter_mut().zip(f32_slice.iter()) {
                                *dst = *src as f64;
                            }
                        }
                    }
                    drop(data);
                    staging.unmap();
                    log::trace!("wgpu download finished copy id={}", h.buffer_id);
                    self.telemetry.record_download_bytes(size_bytes);

                    let mut shape = h.shape.clone();
                    if let Some(info) = runmat_accelerate_api::handle_transpose_info(h) {
                        let base_rows = info.base_rows;
                        let base_cols = info.base_cols;
                        if base_rows * base_cols != out.len() {
                            return Err(anyhow!(
                                "download: transpose metadata mismatch for buffer {}",
                                h.buffer_id
                            ));
                        }
                        if shape.len() == 2 {
                            let rows_t = base_cols;
                            let cols_t = base_rows;
                            let mut transposed = vec![0.0f64; out.len()];
                            for col in 0..base_cols {
                                for row in 0..base_rows {
                                    let src_idx = row + col * base_rows;
                                    let dst_idx = col + row * base_cols;
                                    transposed[dst_idx] = out[src_idx];
                                }
                            }
                            out = transposed;
                            shape[0] = rows_t;
                            shape[1] = cols_t;
                        }
                    }

                    log::trace!(
                        "wgpu download complete id={} final_shape={:?}",
                        h.buffer_id,
                        shape
                    );

                    Ok(HostTensorOwned {
                        data: out,
                        shape,
                        storage: runmat_accelerate_api::handle_storage(h),
                    })
                };

            log::trace!(
                "wgpu download creating staging buffer id={} bytes={}",
                h.buffer_id,
                size_bytes
            );
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-download-staging"),
                size: size_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-download-encoder"),
                });
            encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, &staging, 0, size_bytes);
            self.submit(encoder);
            let slice = staging.slice(..);
            let (tx, rx) = oneshot::channel();

            let map_buffer_id = h.buffer_id;
            slice.map_async(wgpu::MapMode::Read, move |res| {
                log::trace!(
                    "wgpu download map_async callback id={} status={:?}",
                    map_buffer_id,
                    res
                );
                let _ = tx.send(res);
            });
            log::trace!(
                "wgpu download awaiting map_async completion id={} bytes={}",
                h.buffer_id,
                size_bytes
            );
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.device.poll(wgpu::Maintain::Wait);
            }
            let map_result = rx
                .await
                .map_err(|_| anyhow!("map_async callback dropped for buffer {}", h.buffer_id))?;

            log::trace!("wgpu download map_async success id={}", h.buffer_id);
            map_result.map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
            finish_readback(staging, size_bytes)
        })
    }
    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        // Remove from handle table and return buffer to pool for reuse
        log::trace!("wgpu free id={}", h.buffer_id);
        let entry = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .remove(&h.buffer_id);
        if let Some(entry) = entry {
            if entry.len > 0 {
                let size_bytes = (entry.len as u64).saturating_mul(self.element_size as u64);
                let poolable_by_size = self.buffer_residency_max_poolable_bytes > 0
                    && size_bytes <= self.buffer_residency_max_poolable_bytes;
                let buffer_ptr = entry.buffer.as_ref() as *const wgpu::Buffer as usize;
                // Always invalidate bind-group cache first so cache-held references
                // do not pin dropped buffers across loop iterations.
                self.bind_group_cache.invalidate_buffer(buffer_ptr);
                let strong_count = Arc::strong_count(&entry.buffer);
                if poolable_by_size && strong_count == 1 {
                    self.buffer_residency
                        .release(entry.usage, entry.len, entry.buffer.clone());
                } else {
                    log::trace!(
                        "buffer_residency: not pooling buffer id={} len={} bytes={} strong_count={} poolable_by_size={}",
                        h.buffer_id,
                        entry.len,
                        size_bytes,
                        strong_count,
                        poolable_by_size
                    );
                }
            }
        }
        self.kernel_resources.clear_matmul_source(h.buffer_id);
        runmat_accelerate_api::clear_handle_logical(h);
        runmat_accelerate_api::clear_handle_storage(h);
        runmat_accelerate_api::clear_handle_transpose(h);
        Ok(())
    }

    fn device_info(&self) -> String {
        format!(
            "{} ({:?})",
            self.adapter_info.name, self.adapter_info.backend
        )
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        let backend = format!("{:?}", self.adapter_info.backend).to_ascii_lowercase();
        let memory_bytes = if self.adapter_limits.max_buffer_size > 0 {
            Some(self.adapter_limits.max_buffer_size)
        } else {
            None
        };
        ApiDeviceInfo {
            device_id: self.runtime_device_id,
            name: self.adapter_info.name.clone(),
            vendor: canonical_vendor_name(&self.adapter_info),
            memory_bytes,
            backend: Some(backend),
        }
    }
}
