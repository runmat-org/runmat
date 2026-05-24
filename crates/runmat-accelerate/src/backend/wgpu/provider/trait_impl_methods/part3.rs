    fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        let res = self.kernel_resources.take_matmul_sources(product);
        log::debug!(
            "take_matmul_sources: product={} found={} active_fusion={:?}",
            product.buffer_id,
            res.is_some(),
            active_fusion()
        );
        res
    }

    fn qr_power_iter<'a>(
        &'a self,
        product: &'a GpuTensorHandle,
        product_lhs: Option<&'a GpuTensorHandle>,
        q_handle: &'a GpuTensorHandle,
        options: &'a ProviderQrOptions,
    ) -> AccelProviderFuture<'a, Option<ProviderQrPowerIterResult>> {
        Box::pin(async move {
            let debug_qr = std::env::var("RUNMAT_DEBUG_QR").is_ok();
            if !options.economy {
                return Ok(None);
            }

            let product_entry = self.get_entry(product)?;
            if product_entry.shape.len() != 2 {
                return Ok(None);
            }
            let rows = product_entry.shape[0];
            let cols = product_entry.shape[1];
            if rows == 0 || cols == 0 {
                return Ok(None);
            }
            if cols > QR_DEVICE_MAX_COLS {
                if debug_qr {
                    log::debug!(
                        "qr_power_iter: column count {} exceeds device kernel limit {}; falling back",
                        cols,
                        QR_DEVICE_MAX_COLS
                    );
                }
                return Ok(None);
            }
            if self.precision() != ProviderPrecision::F32 {
                if debug_qr {
                    log::debug!(
                        "qr_power_iter: precision {:?} unsupported for device QR kernel; falling back",
                        self.precision()
                    );
                }
                return Ok(None);
            }
            let q_entry = self.get_entry(q_handle)?;
            if q_entry.shape != product_entry.shape {
                return Ok(None);
            }
            let k = cols;

            let mut pre_product_max = match <Self as AccelProvider>::download(self, product).await {
                Ok(host) => Some(
                    host.data
                        .iter()
                        .fold(0.0f64, |acc, value| acc.max(value.abs())),
                ),
                Err(err) => {
                    log::warn!("qr_power_iter pre-download failed: {err}");
                    None
                }
            };

            let pre_q_max = match <Self as AccelProvider>::download(self, q_handle).await {
                Ok(host) => Some(
                    host.data
                        .iter()
                        .fold(0.0f64, |acc, value| acc.max(value.abs())),
                ),
                Err(err) => {
                    log::warn!("qr_power_iter q-handle pre-download failed: {err}");
                    None
                }
            };

            const PRODUCT_EPS: f64 = 1.0e-12;
            const Q_EPS: f64 = 1.0e-6;
            if pre_product_max.unwrap_or(0.0) <= PRODUCT_EPS && pre_q_max.unwrap_or(0.0) > Q_EPS {
                let debug_zero_host = std::env::var("RUNMAT_DEBUG_QR_ZEROHOST").is_ok();
                if debug_zero_host {
                    if let Some(lhs_handle) = product_lhs {
                        let lhs_download =
                            <Self as AccelProvider>::download(self, lhs_handle).await;
                        let q_download = <Self as AccelProvider>::download(self, q_handle).await;
                        match (lhs_download, q_download) {
                            (Ok(lhs_host), Ok(q_host)) => {
                                let lhs_rows = lhs_host.shape.first().copied().unwrap_or(0);
                                let lhs_cols = lhs_host.shape.get(1).copied().unwrap_or(0);
                                let q_rows = q_host.shape.first().copied().unwrap_or(0);
                                let q_cols = q_host.shape.get(1).copied().unwrap_or(0);
                                if lhs_rows == q_rows
                                    && lhs_cols == q_rows
                                    && q_rows == rows
                                    && q_cols == cols
                                {
                                    let mut max_host_product = 0.0f64;
                                    for col in 0..cols {
                                        for row in 0..rows {
                                            let mut sum = 0.0f64;
                                            for k_idx in 0..lhs_cols {
                                                let lhs_idx = row + k_idx * lhs_rows;
                                                let q_idx = k_idx + col * q_rows;
                                                sum += lhs_host.data[lhs_idx] * q_host.data[q_idx];
                                            }
                                            max_host_product = max_host_product.max(sum.abs());
                                        }
                                    }
                                    log::info!(
                                    "qr_power_iter host check: rows={} cols={} host_max_product={:.6e}",
                                    rows,
                                    cols,
                                    max_host_product
                                );
                                } else {
                                    log::info!(
                                    "qr_power_iter host check skipped: lhs_shape={:?} q_shape={:?} rows={} cols={}",
                                    lhs_host.shape,
                                    q_host.shape,
                                    rows,
                                    cols
                                );
                                }
                            }
                            (lhs_res, q_res) => {
                                log::info!(
                                    "qr_power_iter host check download failed: lhs={:?} q={:?} product_id={}",
                                    lhs_res.err(),
                                    q_res.err(),
                                    product.buffer_id
                                );
                            }
                        }
                    } else {
                        log::info!(
                            "qr_power_iter host check skipped: product_lhs unavailable (product_id={})",
                            product.buffer_id
                        );
                    }
                }
                if let Some(lhs_handle) = product_lhs {
                    log::warn!(
                        "qr_power_iter: detected zero matmul product (product_id={} max_product_abs_pre={:?} max_q_abs_pre={:?}); recomputing",
                        product.buffer_id,
                        pre_product_max,
                        pre_q_max
                    );
                    if let Ok(lhs_entry) = self.get_entry(lhs_handle) {
                        if let Ok(rhs_entry) = self.get_entry(q_handle) {
                            let lhs_view = build_matrix_operand_view(lhs_handle, &lhs_entry)
                                .unwrap_or(MatrixOperandView {
                                    rows: 0,
                                    cols: 0,
                                    lda: 0,
                                    transpose: false,
                                });
                            let rhs_view = build_matrix_operand_view(q_handle, &rhs_entry)
                                .unwrap_or(MatrixOperandView {
                                    rows: 0,
                                    cols: 0,
                                    lda: 0,
                                    transpose: false,
                                });
                            log::info!(
                                "qr_power_iter recompute operands: product_id={} lhs_shape={:?} rhs_shape={:?} lhs_view={{rows:{} cols:{} lda:{} transpose:{}}} rhs_view={{rows:{} cols:{} lda:{} transpose:{}}}",
                                product.buffer_id,
                                lhs_entry.shape,
                                rhs_entry.shape,
                                lhs_view.rows,
                                lhs_view.cols,
                                lhs_view.lda,
                                lhs_view.transpose,
                                rhs_view.rows,
                                rhs_view.cols,
                                rhs_view.lda,
                                rhs_view.transpose
                            );
                            log::info!(
                                "qr_power_iter recompute buffers: product_id={} lhs_ptr={:p} rhs_ptr={:p}",
                                product.buffer_id,
                                lhs_entry.buffer.as_ref(),
                                rhs_entry.buffer.as_ref()
                            );
                        }
                    }
                    let recomputed = self.matmul_exec_with_usage(
                        lhs_handle,
                        q_handle,
                        BufferUsageClass::FusionOut,
                    )?;
                    let mut recomputed_max: Option<f64> = None;
                    if debug_zero_host {
                        match <Self as AccelProvider>::download(self, &recomputed).await {
                            Ok(host) => {
                                let max_recomputed = host
                                    .data
                                    .iter()
                                    .fold(0.0f64, |acc, value| acc.max(value.abs()));
                                log::info!(
                                    "qr_power_iter recompute check: product_id={} max_recomputed_abs={:.6e}",
                                    product.buffer_id,
                                    max_recomputed
                                );
                                recomputed_max = Some(max_recomputed);
                            }
                            Err(err) => {
                                log::info!(
                                    "qr_power_iter recompute check failed: product_id={} err={}",
                                    product.buffer_id,
                                    err
                                );
                            }
                        }
                    }
                    let recomputed_entry = self.get_entry(&recomputed)?;
                    log::info!(
                        "qr_power_iter recompute start: product_id={} original_len={} recomputed_len={}",
                        product.buffer_id,
                        product_entry.len,
                        recomputed_entry.len
                    );
                    let bytes = (recomputed_entry.len as u64) * self.element_size as u64;
                    log::info!(
                        "qr_power_iter recompute copy detail: product_id={} product_ptr={:p} recomputed_ptr={:p}",
                        product.buffer_id,
                        product_entry.buffer.as_ref(),
                        recomputed_entry.buffer.as_ref()
                    );
                    if bytes > 0 {
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("runmat-qr-product-recompute"),
                                });
                        encoder.copy_buffer_to_buffer(
                            recomputed_entry.buffer.as_ref(),
                            0,
                            product_entry.buffer.as_ref(),
                            0,
                            bytes,
                        );
                        self.submit(encoder);
                    }

                    let max_val = if let Some(val) = recomputed_max {
                        val
                    } else {
                        match <Self as AccelProvider>::download(self, product).await {
                            Ok(host) => host
                                .data
                                .iter()
                                .fold(0.0f64, |acc, value| acc.max(value.abs())),
                            Err(err) => {
                                log::warn!("qr_power_iter recompute verification failed: {err}");
                                0.0
                            }
                        }
                    };
                    log::info!(
                        "qr_power_iter recompute copy: product_id={} bytes={} post_max={:.6e}",
                        product.buffer_id,
                        bytes,
                        max_val
                    );
                    if max_val == 0.0 {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let q_download =
                                <Self as AccelProvider>::download(self, q_handle).await;
                            if let Ok(lhs_dump) =
                                <Self as AccelProvider>::download(self, lhs_handle).await
                            {
                                if let Ok(ref q_dump) = q_download {
                                    let dump_dir = Path::new("target/matmul_zero");
                                    let _ = fs::create_dir_all(dump_dir);
                                    let lhs_path = dump_dir.join(format!(
                                        "lhs_{}_{}.bin",
                                        product.buffer_id,
                                        lhs_dump.data.len()
                                    ));
                                    let rhs_path = dump_dir.join(format!(
                                        "rhs_{}_{}.bin",
                                        product.buffer_id,
                                        q_dump.data.len()
                                    ));
                                    let _ =
                                        fs::write(&lhs_path, cast_slice(lhs_dump.data.as_slice()));
                                    let _ =
                                        fs::write(&rhs_path, cast_slice(q_dump.data.as_slice()));
                                    log::warn!(
                                        "qr_power_iter dump written: product_id={} lhs_path={} rhs_path={}",
                                        product.buffer_id,
                                        lhs_path.display(),
                                        rhs_path.display()
                                    );
                                }
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            log::warn!("qr_power_iter: skipping matmul dump because filesystem APIs are unavailable on wasm");
                        }
                        log::warn!(
                            "qr_power_iter: recomputed product is still zero; falling back to host QR"
                        );
                        let _ = self.free(&recomputed);
                        if let Some(handle) = self.qr_power_iter_host(product, options).await? {
                            return Ok(Some(handle));
                        }
                        return Ok(None);
                    }
                    pre_product_max = Some(max_val);

                    let _ = self.free(&recomputed);
                } else {
                    log::warn!(
                        "qr_power_iter: zero product detected for buffer {} without lhs handle; proceeding with existing data",
                        product.buffer_id
                    );
                }
            }

            let (q_result, r_handle, mut r_inv_opt) = self.qr_factor_device(
                product,
                rows,
                cols,
                Some(q_handle),
                "runmat-qr-power",
                true,
            )?;

            let mut fallback_needed = false;
            if let Ok(host_r) = <Self as AccelProvider>::download(self, &r_handle).await {
                for col in 0..cols {
                    let diag = host_r.data[col + col * cols];
                    if !diag.is_finite() || diag.abs() <= 1.0e-12 {
                        fallback_needed = true;
                        break;
                    }
                }
            }

            if fallback_needed {
                if let Some(handle) = r_inv_opt.take() {
                    let _ = self.free(&handle);
                }
                let _ = self.free(&q_result);
                let _ = self.free(&r_handle);
                return self.qr_power_iter_host(product, options).await;
            }

            if pre_product_max.unwrap_or(0.0) <= 1.0e-8 {
                if let Some(handle) = r_inv_opt.take() {
                    let _ = self.free(&handle);
                }
                let _ = self.free(&q_result);
                let _ = self.free(&r_handle);
                return self.qr_power_iter_host(product, options).await;
            }

            if debug_qr {
                if let Err(err) = self
                    .debug_qr_power_iter(
                        product,
                        &product_entry,
                        pre_product_max,
                        pre_q_max,
                        &q_result,
                        &r_handle,
                        r_inv_opt
                            .as_ref()
                            .expect("retain_r_inv=true must provide inverse handle"),
                        None::<&runmat_accelerate_api::HostTensorOwned>,
                        rows,
                        cols,
                    )
                    .await
                {
                    log::warn!("qr_power_iter debug failed: {err}");
                }
            }

            if let Some(handle) = r_inv_opt.take() {
                let _ = self.free(&handle);
            }

            let mut perm_matrix = vec![0.0f64; k * k];
            for i in 0..k {
                perm_matrix[i + i * k] = 1.0;
            }
            let perm_vector: Vec<f64> = (1..=k).map(|v| v as f64).collect();

            let perm_matrix_shape = [k, k];
            let perm_matrix_handle = self.upload(&HostTensorView {
                data: &perm_matrix,
                shape: &perm_matrix_shape,
            })?;
            let perm_vector_shape = vec![k, 1];
            let perm_vector_handle = self.upload(&HostTensorView {
                data: &perm_vector,
                shape: &perm_vector_shape,
            })?;

            let _ = self.free(product);

            Ok(Some(ProviderQrPowerIterResult {
                q: q_result,
                r: r_handle,
                perm_matrix: perm_matrix_handle,
                perm_vector: perm_vector_handle,
            }))
        })
    }
    fn matmul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_exec(a, b) })
    }

    fn syrk(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.syrk_exec(a)
    }
    fn matmul_epilogue<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        ep: &'a runmat_accelerate_api::MatmulEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            use runmat_accelerate_api::ProviderPrecision;
            let entry_a = self.get_entry(a)?;
            let entry_b = self.get_entry(b)?;
            if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
                return Err(anyhow!("matmul_epilogue: only 2D tensors supported"));
            }
            let view_a = build_matrix_operand_view(a, &entry_a)
                .map_err(|e| anyhow!("matmul_epilogue: {e}"))?;
            let view_b = build_matrix_operand_view(b, &entry_b)
                .map_err(|e| anyhow!("matmul_epilogue: {e}"))?;

            if view_a.cols != view_b.rows {
                return Err(anyhow!("matmul_epilogue: inner dimensions must match"));
            }
            let m = view_a.rows;
            let n = view_b.cols;
            let k = view_a.cols;

            let out_shape = vec![m, n];
            let len = m * n;
            let out_buffer =
                self.create_storage_buffer_checked(len, "runmat-matmul-epilogue-out")?;
            if len == 0 {
                return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
            }

            let start = Instant::now();

            let m_u32 =
                u32::try_from(m).map_err(|_| anyhow!("matmul_epilogue: m exceeds GPU limits"))?;
            let n_u32 =
                u32::try_from(n).map_err(|_| anyhow!("matmul_epilogue: n exceeds GPU limits"))?;
            let k_u32 =
                u32::try_from(k).map_err(|_| anyhow!("matmul_epilogue: k exceeds GPU limits"))?;
            let mut flags = 0u32;
            if view_a.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
            }
            if view_b.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
            }

            let params = crate::backend::wgpu::params::MatmulParams {
                m: m_u32,
                n: n_u32,
                k: k_u32,
                lda: view_a.lda,
                ldb: view_b.lda,
                ldc: m_u32,
                offset_a: 0,
                offset_b: 0,
                offset_out: 0,
                flags,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-matmul-epilogue-params");

            // Resolve optional scales and epilogue params by precision
            use crate::backend::wgpu::params::{
                MATMUL_EPILOGUE_FLAG_CLAMP_MAX, MATMUL_EPILOGUE_FLAG_CLAMP_MIN,
                MATMUL_EPILOGUE_FLAG_COL_DIV, MATMUL_EPILOGUE_FLAG_COL_SCALE,
                MATMUL_EPILOGUE_FLAG_DIAG_WRITE, MATMUL_EPILOGUE_FLAG_POW,
                MATMUL_EPILOGUE_FLAG_ROW_DIV, MATMUL_EPILOGUE_FLAG_ROW_SCALE,
            };
            let has_row = ep.row_scale.is_some();
            let has_col = ep.col_scale.is_some();
            let dummy_rowcol = self.create_storage_buffer(1, "runmat-matmul-epilogue-dummy-scale");
            let row_buf = match &ep.row_scale {
                Some(h) => self.get_entry(h)?.buffer.clone(),
                None => dummy_rowcol.clone(),
            };
            let col_buf = match &ep.col_scale {
                Some(h) => self.get_entry(h)?.buffer.clone(),
                None => dummy_rowcol.clone(),
            };

            let (diag_rows, diag_stride, diag_offset, has_diag) = match &ep.diag_output {
                Some(_) => {
                    return Err(anyhow!(
                        "matmul_epilogue: diag_output is not supported by the WGPU provider yet"
                    ));
                }
                None => (0u32, 1u32, 0u32, false),
            };

            let mut flags: u32 = 0;
            if has_row {
                flags |= MATMUL_EPILOGUE_FLAG_ROW_SCALE;
                if matches!(ep.row_op, runmat_accelerate_api::ScaleOp::Divide) {
                    flags |= MATMUL_EPILOGUE_FLAG_ROW_DIV;
                }
            }
            if has_col {
                flags |= MATMUL_EPILOGUE_FLAG_COL_SCALE;
                if matches!(ep.col_op, runmat_accelerate_api::ScaleOp::Divide) {
                    flags |= MATMUL_EPILOGUE_FLAG_COL_DIV;
                }
            }

            let mut clamp_min = 0.0f64;
            if let Some(v) = ep.clamp_min {
                clamp_min = v;
                flags |= MATMUL_EPILOGUE_FLAG_CLAMP_MIN;
            }
            let mut clamp_max = 0.0f64;
            if let Some(v) = ep.clamp_max {
                clamp_max = v;
                flags |= MATMUL_EPILOGUE_FLAG_CLAMP_MAX;
            }
            let mut pow_exponent = 1.0f64;
            if let Some(v) = ep.pow_exponent {
                pow_exponent = v;
                flags |= MATMUL_EPILOGUE_FLAG_POW;
            }
            if has_diag {
                flags |= MATMUL_EPILOGUE_FLAG_DIAG_WRITE;
            }

            let tile = crate::backend::wgpu::config::effective_matmul_tile();
            let groups_x =
                crate::backend::wgpu::dispatch::common::dispatch_size_dim(n as u32, tile);
            let groups_y =
                crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile);

            // Build a layout tag incorporating the epilogue mask for cache keying
            let layout_tag = format!("runmat-matmul-epilogue-layout-flags-{flags:08x}");

            // Create module from the static WGSL (token substitution handled inside)
            let (shader_src, ep_buf, pipeline_layout) = match self.precision() {
                ProviderPrecision::F64 => {
                    let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF64 {
                        alpha: ep.alpha,
                        beta: ep.beta,
                        clamp_min,
                        clamp_max,
                        pow_exponent,
                        flags,
                        diag_offset,
                        diag_stride,
                        diag_rows,
                        _pad: 0,
                        _pad2: 0,
                    };
                    let ep_buf = self.uniform_buffer(&ep_params, "runmat-matmul-epilogue-uniform");
                    let pl = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                        self.device_ref(),
                        "runmat-matmul-epilogue-pl",
                        &self.pipelines.matmul_epilogue.layout,
                    );
                    (
                        crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F64,
                        ep_buf,
                        pl,
                    )
                }
                ProviderPrecision::F32 => {
                    let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF32 {
                        alpha: ep.alpha as f32,
                        beta: ep.beta as f32,
                        clamp_min: clamp_min as f32,
                        clamp_max: clamp_max as f32,
                        pow_exponent: pow_exponent as f32,
                        flags,
                        diag_offset,
                        diag_stride,
                        diag_rows,
                        _pad: 0,
                    };
                    let ep_buf = self.uniform_buffer(&ep_params, "runmat-matmul-epilogue-uniform");
                    let pl = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                        self.device_ref(),
                        "runmat-matmul-epilogue-pl",
                        &self.pipelines.matmul_epilogue.layout,
                    );
                    (
                        crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F32,
                        ep_buf,
                        pl,
                    )
                }
            };

            let module = crate::backend::wgpu::pipelines::create_shader_module(
                self.device_ref(),
                "runmat-matmul-epilogue-module",
                shader_src,
            );
            let key =
                self.compute_pipeline_hash_bytes(shader_src.as_bytes(), &layout_tag, Some(tile));
            let pipeline = self.get_or_create_pipeline(
                key,
                &pipeline_layout,
                &module,
                "runmat-matmul-epilogue",
                Some(shader_src.as_bytes()),
                Some(&layout_tag),
                Some(tile),
            );

            let bg = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-matmul-epilogue-bind"),
                    layout: &self.pipelines.matmul_epilogue.layout,
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
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: row_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: col_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: ep_buf.as_entire_binding(),
                        },
                    ],
                });
            crate::backend::wgpu::dispatch::matmul::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &bg,
                groups_x,
                groups_y,
            );
            let handle = self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                len,
                BufferUsageClass::FusionOut,
            );

            self.telemetry.record_matmul_duration(start.elapsed());

            Ok(handle)
        })
    }
