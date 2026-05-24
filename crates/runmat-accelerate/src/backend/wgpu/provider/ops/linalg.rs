use super::*;

impl WgpuProvider {
    pub(crate) fn take_matmul_sources_exec(
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

    pub(crate) async fn qr_power_iter_exec(
        &self,
        product: &GpuTensorHandle,
        product_lhs: Option<&GpuTensorHandle>,
        q_handle: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> Result<Option<ProviderQrPowerIterResult>> {
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
                    let lhs_download = <Self as AccelProvider>::download(self, lhs_handle).await;
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
                        let lhs_view = build_matrix_operand_view(lhs_handle, &lhs_entry).unwrap_or(
                            MatrixOperandView {
                                rows: 0,
                                cols: 0,
                                lda: 0,
                                transpose: false,
                            },
                        );
                        let rhs_view = build_matrix_operand_view(q_handle, &rhs_entry).unwrap_or(
                            MatrixOperandView {
                                rows: 0,
                                cols: 0,
                                lda: 0,
                                transpose: false,
                            },
                        );
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
                let recomputed =
                    self.matmul_exec_with_usage(lhs_handle, q_handle, BufferUsageClass::FusionOut)?;
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
                        let q_download = <Self as AccelProvider>::download(self, q_handle).await;
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
                                let _ = fs::write(&lhs_path, cast_slice(lhs_dump.data.as_slice()));
                                let _ = fs::write(&rhs_path, cast_slice(q_dump.data.as_slice()));
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

        let (q_result, r_handle, mut r_inv_opt) =
            self.qr_factor_device(product, rows, cols, Some(q_handle), "runmat-qr-power", true)?;

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
    }

    pub(crate) async fn matmul_epilogue_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        ep: &runmat_accelerate_api::MatmulEpilogue,
    ) -> Result<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderPrecision;
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
            return Err(anyhow!("matmul_epilogue: only 2D tensors supported"));
        }
        let view_a =
            build_matrix_operand_view(a, &entry_a).map_err(|e| anyhow!("matmul_epilogue: {e}"))?;
        let view_b =
            build_matrix_operand_view(b, &entry_b).map_err(|e| anyhow!("matmul_epilogue: {e}"))?;

        if view_a.cols != view_b.rows {
            return Err(anyhow!("matmul_epilogue: inner dimensions must match"));
        }
        let m = view_a.rows;
        let n = view_b.cols;
        let k = view_a.cols;

        let out_shape = vec![m, n];
        let len = m * n;
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-matmul-epilogue-out")?;
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
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n as u32, tile);
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile);

        let layout_tag = format!("runmat-matmul-epilogue-layout-flags-{flags:08x}");
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
        let key = self.compute_pipeline_hash_bytes(shader_src.as_bytes(), &layout_tag, Some(tile));
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
    }

    pub(crate) async fn image_normalize_exec(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(input)?;
        ensure!(
            entry.shape.len() == 3,
            "image_normalize: expected 3-D tensor, got {:?}",
            entry.shape
        );
        ensure!(
            entry.shape[0] == desc.batch
                && entry.shape[1] == desc.height
                && entry.shape[2] == desc.width,
            "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
            (desc.batch, desc.height, desc.width),
            entry.shape
        );

        if entry.len == 0 {
            return self.image_normalize_cpu_fallback(input, desc).await;
        }

        match self.precision {
            NumericPrecision::F64 => self.image_normalize_cpu_fallback(input, desc).await,
            NumericPrecision::F32 => {
                ensure!(
                    desc.epsilon.is_finite(),
                    "image_normalize: epsilon must be finite"
                );
                ensure!(
                    desc.epsilon >= 0.0,
                    "image_normalize: epsilon must be non-negative"
                );

                let batches = entry.shape[0];
                let height = entry.shape[1];
                let width = entry.shape[2];
                let plane = height
                    .checked_mul(width)
                    .ok_or_else(|| anyhow!("image_normalize: height*width overflow"))?;
                ensure!(
                    entry.len == plane * batches,
                    "image_normalize: inconsistent tensor length {} vs dims {:?}",
                    entry.len,
                    entry.shape
                );

                let stride_h = batches;
                let stride_w = batches
                    .checked_mul(height)
                    .ok_or_else(|| anyhow!("image_normalize: stride overflow"))?;

                let batches_u32 = u32::try_from(batches)
                    .map_err(|_| anyhow!("image_normalize: batch size too large"))?;
                let height_u32 = u32::try_from(height)
                    .map_err(|_| anyhow!("image_normalize: height too large"))?;
                let width_u32 = u32::try_from(width)
                    .map_err(|_| anyhow!("image_normalize: width too large"))?;
                let plane_u32 = u32::try_from(plane)
                    .map_err(|_| anyhow!("image_normalize: plane size too large"))?;
                let stride_h_u32 = u32::try_from(stride_h)
                    .map_err(|_| anyhow!("image_normalize: stride_h too large"))?;
                let stride_w_u32 = u32::try_from(stride_w)
                    .map_err(|_| anyhow!("image_normalize: stride_w too large"))?;
                let (tuning, cache_hit) =
                    self.resolve_image_normalize_tuning(batches_u32, plane_u32);
                log::debug!(
                    "image_normalize tuning batches={} plane={} lane={} spatial={} values/thread={} cache_hit={}",
                    batches_u32,
                    plane_u32,
                    tuning.lane_count,
                    tuning.spatial_tile,
                    tuning.values_per_thread,
                    cache_hit
                );
                let pipeline = self.image_normalize_pipeline(&tuning)?;

                let mut flags = 0u32;
                if desc.gain.is_some() {
                    flags |= IMAGE_NORMALIZE_FLAG_GAIN;
                }
                if desc.bias.is_some() {
                    flags |= IMAGE_NORMALIZE_FLAG_BIAS;
                }
                if desc.gamma.is_some() {
                    flags |= IMAGE_NORMALIZE_FLAG_GAMMA;
                }

                let mut uniforms = ImageNormalizeUniforms {
                    batch_count: 0,
                    height: height_u32,
                    width: width_u32,
                    plane: plane_u32,
                    stride_h: stride_h_u32,
                    stride_w: stride_w_u32,
                    flags,
                    batch_stride: batches_u32,
                    batch_offset: 0,
                    _pad0: 0,
                    epsilon: desc.epsilon as f32,
                    gain: desc.gain.unwrap_or(1.0) as f32,
                    bias: desc.bias.unwrap_or(0.0) as f32,
                    gamma: desc.gamma.unwrap_or(1.0) as f32,
                    _pad1: 0,
                };

                let out_buffer = self.create_storage_buffer_checked_with_usage(
                    entry.len,
                    "runmat-image-normalize-out",
                    BufferUsageClass::FusionOut,
                )?;
                let uniform_buf = self.kernel_resources.uniform_buffer(
                    self.device_ref(),
                    UniformBufferKey::ImageNormalizeUniforms,
                    std::mem::size_of::<ImageNormalizeUniforms>() as u64,
                    "runmat-image-normalize-uniform",
                );
                let stream_hot_cap = self
                    .image_normalize_hot_stream_cap(plane_u32, batches_u32)
                    .max(1);
                let cold_cap = stream_hot_cap.min((Self::IMAGE_NORMALIZE_STREAM_COLD_CAP).max(1));
                let chunk_limit = if cache_hit {
                    stream_hot_cap
                } else {
                    cold_cap.max(1)
                };

                let bind_entries = [
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
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                ];
                let layout = &self.pipelines.image_normalize.layout;
                let bind_group =
                    self.bind_group_cache
                        .get_or_create(layout, &bind_entries, || {
                            Arc::new(self.device_ref().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-image-normalize-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        });

                let mut offset = 0u32;
                while offset < batches_u32 {
                    let remaining = batches_u32 - offset;
                    let chunk = remaining.min(chunk_limit).max(1);
                    uniforms.batch_count = chunk;
                    uniforms.batch_offset = offset;
                    self.queue
                        .write_buffer(uniform_buf.as_ref(), 0, bytes_of(&uniforms));
                    crate::backend::wgpu::dispatch::image_normalize::run(
                        self.device_ref(),
                        self.queue_ref(),
                        pipeline.as_ref(),
                        bind_group.as_ref(),
                        chunk,
                        tuning.batch_tile,
                    );
                    offset += chunk;
                }

                Ok(self.register_existing_buffer_with_usage(
                    out_buffer,
                    entry.shape.clone(),
                    entry.len,
                    BufferUsageClass::FusionOut,
                ))
            }
        }
    }

    pub(crate) async fn matmul_power_step_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        epilogue: &runmat_accelerate_api::PowerStepEpilogue,
    ) -> Result<GpuTensorHandle> {
        let rhs_entry = self.get_entry(rhs)?;
        let product = self.matmul_exec(lhs, rhs)?;
        let squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &product,
            &product,
        )?;
        let mut sum_sq = self.reduce_dim_sum_mean_exec(
            &squared,
            0,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let _ = self.free(&squared);
        if epilogue.epsilon != 0.0 {
            let eps = self.fill_exec(&sum_sq.shape, epilogue.epsilon)?;
            let adjusted = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Add,
                &sum_sq,
                &eps,
            )?;
            let _ = self.free(&sum_sq);
            let _ = self.free(&eps);
            sum_sq = adjusted;
        }
        let norms = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
        let _ = self.free(&sum_sq);
        let normalized = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &product,
            &norms,
        )?;
        let _ = self.free(&product);
        let _ = self.free(&norms);

        let mut reused = false;
        let rhs_shape_match = rhs_entry.shape == normalized.shape;
        let rhs_transposed = runmat_accelerate_api::handle_transpose_info(rhs).is_some();
        let rhs_ref_count = Arc::strong_count(&rhs_entry.buffer);
        if rhs_shape_match && !rhs_transposed && rhs_entry.len > 0 && rhs_ref_count <= 2 {
            if let Ok(normalized_entry) = self.get_entry(&normalized) {
                let bytes = (rhs_entry.len as u64) * self.element_size as u64;
                if bytes > 0 {
                    let mut encoder =
                        self.device_ref()
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-power-step-copy"),
                            });
                    encoder.copy_buffer_to_buffer(
                        normalized_entry.buffer.as_ref(),
                        0,
                        rhs_entry.buffer.as_ref(),
                        0,
                        bytes,
                    );
                    self.submit(encoder);
                }
                let _ = self.free(&normalized);
                self.mark_buffer_usage(rhs, BufferUsageClass::FusionOut);
                log::debug!(
                    "matmul_power_step: reused rhs buffer {} for normalized output (len={})",
                    rhs.buffer_id,
                    rhs_entry.len
                );
                reused = true;
            }
        }

        if reused {
            Ok(rhs.clone())
        } else {
            log::debug!(
                "matmul_power_step: fallback reuse (shape_match={} transpose={} len={} ref_count={})",
                rhs_shape_match,
                rhs_transposed,
                rhs_entry.len,
                rhs_ref_count
            );
            Ok(normalized)
        }
    }

    pub(crate) async fn covariance_with_optional_exec(
        &self,
        matrix: &GpuTensorHandle,
        second: Option<&GpuTensorHandle>,
        weights: Option<&GpuTensorHandle>,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CovRows::All {
            return Err(anyhow!(
                "covariance: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }
        if options.has_weight_vector || weights.is_some() {
            return Err(anyhow!(
                "covariance: weight vectors are not supported by WGPU provider"
            ));
        }

        let combined = if let Some(rhs) = second {
            let left_entry = self.get_entry(matrix)?;
            let right_entry = self.get_entry(rhs)?;

            let rows_left = match left_entry.shape.len() {
                0 => 1usize,
                1 => left_entry.shape[0],
                2 => left_entry.shape[0],
                _ => {
                    return Err(anyhow!(
                        "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                        left_entry.shape
                    ))
                }
            };
            let rows_right = match right_entry.shape.len() {
                0 => 1usize,
                1 => right_entry.shape[0],
                2 => right_entry.shape[0],
                _ => {
                    return Err(anyhow!(
                        "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                        right_entry.shape
                    ))
                }
            };

            ensure!(
                rows_left == rows_right,
                "covariance: inputs must have the same number of rows (got {} and {})",
                rows_left,
                rows_right
            );

            let cat_inputs = vec![matrix.clone(), rhs.clone()];
            Some(self.cat_exec(2, &cat_inputs)?)
        } else {
            None
        };

        let result = {
            let source = combined.as_ref().unwrap_or(matrix);
            self.covariance_exec(source, options).await
        };

        if let Some(handle) = combined {
            let _ = self.free(&handle);
        }

        result
    }

    pub(crate) async fn eig_exec(
        &self,
        handle: &GpuTensorHandle,
        compute_left: bool,
    ) -> Result<ProviderEigResult> {
        let host = <Self as AccelProvider>::download(self, handle).await?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("eig: {e}"))?;
        let eval = runmat_runtime::builtins::math::linalg::factor::eig::evaluate(
            Value::Tensor(tensor),
            &[],
            compute_left,
        )
        .await
        .map_err(|err| runtime_flow_to_anyhow("eig", err))?;

        let eigenvalues_tensor = host_tensor_from_value("eig", eval.eigenvalues())?;
        let diagonal_tensor = host_tensor_from_value("eig", eval.diagonal_matrix())?;
        let right_tensor = host_tensor_from_value("eig", eval.right())?;

        let left_value = if compute_left {
            Some(
                eval.left()
                    .map_err(|err| runtime_flow_to_anyhow("eig", err))?,
            )
        } else {
            None
        };

        let left_tensor = match left_value {
            Some(value) => Some(host_tensor_from_value("eig", value)?),
            None => None,
        };

        let eigenvalues = self.upload(&HostTensorView {
            data: &eigenvalues_tensor.data,
            shape: &eigenvalues_tensor.shape,
        })?;
        let diagonal = self.upload(&HostTensorView {
            data: &diagonal_tensor.data,
            shape: &diagonal_tensor.shape,
        })?;
        let right = self.upload(&HostTensorView {
            data: &right_tensor.data,
            shape: &right_tensor.shape,
        })?;
        let left = match left_tensor {
            Some(tensor) => Some(self.upload(&HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            })?),
            None => None,
        };

        if compute_left && left.is_none() {
            return Err(anyhow!(
                "eig: left eigenvectors are not available for the requested matrix"
            ));
        }

        Ok(ProviderEigResult {
            eigenvalues,
            diagonal,
            right,
            left,
        })
    }

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
                let _ = self.free(&first);
                Ok(handle)
            }
            Err(err) => {
                let _ = self.free(&first);
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
            return self.fill(&[1usize, 1usize], f64::NAN);
        }
        if a.shape.len() <= 2 {
            let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
            match self.reduce_dim_sum_mean_exec(&first, 1, op) {
                Ok(handle) => {
                    let _ = self.free(&first);
                    Ok(handle)
                }
                Err(err) => {
                    let _ = self.free(&first);
                    Err(err)
                }
            }
        } else {
            let original_shape = a.shape.clone();
            let flattened_shape = vec![total_elems, 1usize];
            let flattened = self.reshape(a, &flattened_shape)?;
            let result = self.reduce_dim_sum_mean_exec(&flattened, 0, op);
            let _ = self.reshape(a, &original_shape);
            result
        }
    }

    pub(crate) async fn reduce_median_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let host = <Self as AccelProvider>::download(self, a).await?;
        let median = median_from_slice(&host.data);
        let data = [median];
        let shape = [1usize, 1usize];
        self.upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })
    }

    pub(crate) async fn reduce_median_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let host = <Self as AccelProvider>::download(self, a).await?;
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
        self.upload(&HostTensorView {
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
        let _ = self.free(&sum_handle);
        Ok(out)
    }

    pub(crate) fn issymmetric_exec(
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

    pub(crate) async fn ishermitian_exec(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "ishermitian: tolerance must be finite and non-negative"
            ));
        }
        let host = <Self as AccelProvider>::download(self, matrix).await?;
        let skew = matches!(kind, ProviderHermitianKind::Skew);
        ishermitian_host_real_data(&host.shape, &host.data, skew, tolerance).map_err(|e| anyhow!(e))
    }

    pub(crate) async fn sym_rcm_exec(&self, matrix: &GpuTensorHandle) -> Result<Vec<usize>> {
        let host = <Self as AccelProvider>::download(self, matrix).await?;
        symrcm_host_real_data(&host.shape, &host.data).map_err(|e| anyhow!(e))
    }

    pub(crate) fn bandwidth_exec(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_bandwidth_shape(&entry.shape).map_err(|e| anyhow!("bandwidth: {e}"))?;
        if rows == 0 || cols == 0 {
            return Ok(ProviderBandwidth { lower: 0, upper: 0 });
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("bandwidth: matrix dimensions too large"))?;
        if total == 0 {
            return Ok(ProviderBandwidth { lower: 0, upper: 0 });
        }
        if total > entry.len {
            return Err(anyhow!(
                "bandwidth: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("bandwidth: matrix exceeds GPU limits"));
        }

        let pipeline = &self.pipelines.bandwidth;
        let output_init = [0u32, 0u32];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runmat-bandwidth-output"),
                contents: cast_slice(&output_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let params = BandwidthParams {
            rows: rows as u32,
            cols: cols as u32,
            len: total as u32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-bandwidth-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-bandwidth-bind-group"),
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

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
        crate::backend::wgpu::dispatch::elementwise::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline.pipeline,
            &bind_group,
            groups,
        );

        let staging_size = (std::mem::size_of::<u32>() * 2) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-bandwidth-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-bandwidth-copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
        self.submit(encoder);

        let bytes = self.map_readback_bytes_sync(staging, staging_size, "bandwidth")?;
        let words: &[u32] = cast_slice(&bytes);
        let lower = words.first().copied().unwrap_or(0);
        let upper = words.get(1).copied().unwrap_or(0);

        Ok(ProviderBandwidth { lower, upper })
    }
    pub(crate) fn syrk_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("syrk: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let out_shape = vec![cols, cols];
        let len = cols
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("syrk: output size overflow"))?;

        let out_bytes = (len as u64) * (self.element_size as u64);
        let out_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::SyrkOut,
            out_bytes,
            "runmat-syrk-out-scratch",
        );
        if len == 0 {
            return Ok(self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                0,
                BufferUsageClass::SyrkOut,
            ));
        }

        let rows_u32 =
            u32::try_from(rows).map_err(|_| anyhow!("syrk: row count exceeds GPU limits"))?;
        let cols_u32 =
            u32::try_from(cols).map_err(|_| anyhow!("syrk: column count exceeds GPU limits"))?;
        let lda_u32 = rows_u32;
        let ldc_u32 = cols_u32;

        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(cols_u32, tile);
        let groups_y = groups_x;

        const SYRK_ROW_CHUNK: usize = 32768;
        let mut offset = 0usize;
        let mut first_chunk = true;
        while offset < rows {
            let remaining = rows - offset;
            let chunk_rows = remaining.min(SYRK_ROW_CHUNK.max(1));
            let chunk_rows_u32 = u32::try_from(chunk_rows)
                .map_err(|_| anyhow!("syrk: chunk rows exceed GPU limits"))?;
            let row_offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("syrk: row offset exceeds GPU limits"))?;

            let mut flags = SYRK_FLAG_FILL_BOTH;
            if !first_chunk {
                flags |= SYRK_FLAG_ACCUMULATE;
            }

            let params = SyrkParams {
                rows_total: rows_u32,
                cols: cols_u32,
                lda: lda_u32,
                ldc: ldc_u32,
                row_offset: row_offset_u32,
                chunk_rows: chunk_rows_u32,
                flags,
                offset_out: 0,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::SyrkParams,
                std::mem::size_of::<crate::backend::wgpu::params::SyrkParams>() as u64,
                "runmat-syrk-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let bind_entries = [
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
            ];
            let layout = &self.pipelines.syrk.layout;
            let bind_group = self
                .bind_group_cache
                .get_or_create(layout, &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-syrk-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                });

            crate::backend::wgpu::dispatch::syrk::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.syrk.pipeline,
                bind_group.as_ref(),
                groups_x,
                groups_y,
            );

            offset += chunk_rows;
            first_chunk = false;
        }

        Ok(self.register_existing_buffer_with_usage(
            out_buffer,
            out_shape,
            len,
            BufferUsageClass::SyrkOut,
        ))
    }

    pub(crate) fn matmul_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.matmul_exec_with_usage(a, b, BufferUsageClass::MatmulOut)
    }
    pub(super) fn matmul_exec_with_usage(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        out_usage: BufferUsageClass,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
            return Err(anyhow!("matmul: only 2D tensors supported"));
        }

        let view_a = build_matrix_operand_view(a, &entry_a).map_err(|e| anyhow!("matmul: {e}"))?;
        let view_b = build_matrix_operand_view(b, &entry_b).map_err(|e| anyhow!("matmul: {e}"))?;

        if view_a.cols != view_b.rows {
            return Err(anyhow!("matmul: inner dimensions must match"));
        }

        let m = view_a.rows;
        let n = view_b.cols;
        let k = view_a.cols;

        let debug_matmul = std::env::var("RUNMAT_DEBUG_MATMUL").is_ok();
        if debug_matmul {
            log::debug!(
                "[matmul_debug] ptr_a={:p} ptr_b={:p}",
                entry_a.buffer.as_ref(),
                entry_b.buffer.as_ref()
            );
            log::debug!(
                "[matmul_debug] m={} n={} k={} lda={} ldb={} transpose_a={} transpose_b={}",
                m,
                n,
                k,
                view_a.lda,
                view_b.lda,
                view_a.transpose,
                view_b.transpose
            );
        }

        let out_shape = vec![m, n];
        let len = m * n;
        if len == 0 {
            let (out_buffer, _) =
                self.create_storage_buffer_for_usage(out_usage, 0, "runmat-matmul-out");
            return Ok(
                self.register_existing_buffer_with_usage(out_buffer, out_shape, 0, out_usage)
            );
        }

        let m_u32 = u32::try_from(m).map_err(|_| anyhow!("matmul: m exceeds GPU limits"))?;
        let n_u32 = u32::try_from(n).map_err(|_| anyhow!("matmul: n exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k).map_err(|_| anyhow!("matmul: k exceeds GPU limits"))?;

        const K_CHUNK: usize = 8192;
        const K_CHUNK_SWITCH: usize = 65536; // only chunk for very large k to avoid regressions

        let can_vec4 = self.precision == NumericPrecision::F32
            && !view_a.transpose
            && !view_b.transpose
            && m % 4 == 0
            && m >= 4
            && n > 0;
        let disable_vec4 = std::env::var("RUNMAT_DISABLE_VEC4")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True"))
            .unwrap_or(false);
        let use_vec4 = can_vec4 && k < K_CHUNK_SWITCH && !disable_vec4;
        let enable_chunk = !view_a.transpose && !view_b.transpose && k >= K_CHUNK_SWITCH;
        if debug_matmul {
            log::debug!(
                "[matmul_debug] can_vec4={} use_vec4={} enable_chunk={} usage={:?}",
                can_vec4,
                use_vec4,
                enable_chunk,
                out_usage
            );
        }

        let start = Instant::now();

        if enable_chunk {
            self.prepare_matmul_pipeline();
            self.device_ref().poll(wgpu::Maintain::Poll);
            let lda_u32 = view_a.lda;
            let ldb_u32 = view_b.lda;
            // Accumulator handle across chunks
            let mut acc: Option<GpuTensorHandle> = None;
            let mut k_off: usize = 0;
            let partial_storage = self.create_storage_buffer_checked_with_usage(
                len,
                "runmat-matmul-partial",
                BufferUsageClass::MatmulPartial,
            )?;
            while k_off < k {
                let k_sub = std::cmp::min(K_CHUNK, k - k_off);
                // Create partial output buffer and bind group
                let partial_buffer = partial_storage.clone();
                let offset_a_elems = k_off
                    .checked_mul(view_a.rows)
                    .ok_or_else(|| anyhow!("matmul: offset overflow"))?;
                let offset_a_u32 = u32::try_from(offset_a_elems)
                    .map_err(|_| anyhow!("matmul: A offset exceeds GPU limits"))?;
                let offset_b_u32 = u32::try_from(k_off)
                    .map_err(|_| anyhow!("matmul: B offset exceeds GPU limits"))?;
                let params = crate::backend::wgpu::params::MatmulParams {
                    m: m_u32,
                    n: n_u32,
                    k: k_sub as u32,
                    lda: lda_u32,
                    ldb: ldb_u32,
                    ldc: m_u32,
                    offset_a: offset_a_u32,
                    offset_b: offset_b_u32,
                    offset_out: 0,
                    flags: 0,
                };
                let params_buffer = self.kernel_resources.uniform_buffer(
                    self.device_ref(),
                    UniformBufferKey::MatmulParams,
                    std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
                    "runmat-matmul-params",
                );
                self.queue
                    .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
                let bind_entries = [
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
                        resource: partial_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ];
                let layout = &self.pipelines.matmul.layout;
                let bg =
                    self.bind_group_cache
                        .get_or_create(layout, &bind_entries, || {
                            Arc::new(self.device_ref().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-matmul-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        });
                let tile = crate::backend::wgpu::config::effective_matmul_tile();
                let groups_x =
                    crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
                let groups_y =
                    crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);
                crate::backend::wgpu::dispatch::matmul::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.matmul.pipeline,
                    bg.as_ref(),
                    groups_x,
                    groups_y,
                );
                // Wrap partial buffer into handle
                let partial = self.register_existing_buffer_with_usage(
                    partial_buffer,
                    out_shape.clone(),
                    len,
                    BufferUsageClass::MatmulPartial,
                );
                acc = match acc {
                    None => Some(partial),
                    Some(prev) => {
                        let sum = self.binary_op_exec(
                            crate::backend::wgpu::types::BinaryOpCode::Add,
                            &prev,
                            &partial,
                        )?;
                        self.free(&prev).ok();
                        self.free(&partial).ok();
                        Some(sum)
                    }
                };
                k_off += k_sub;
            }
            let handle = acc.expect("matmul chunking produced no output");
            self.remember_matmul_sources(&handle, a, b);
            self.mark_buffer_usage(&handle, out_usage);
            self.telemetry.record_matmul_duration(start.elapsed());
            self.record_matmul_kernel_launch(m, n, k, use_vec4, true);
            return Ok(handle);
        }

        // Default single-dispatch path
        let out_buffer =
            self.create_storage_buffer_checked_with_usage(len, "runmat-matmul-out", out_usage)?;
        if use_vec4 {
            self.prepare_matmul_vec4_pipeline();
        } else {
            self.prepare_matmul_pipeline();
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
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
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::MatmulParams,
            std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
            "runmat-matmul-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
        let layout = if use_vec4 {
            &self.pipelines.matmul_vec4.layout
        } else {
            &self.pipelines.matmul.layout
        };
        let pipeline = if use_vec4 {
            &self.pipelines.matmul_vec4.pipeline
        } else {
            &self.pipelines.matmul.pipeline
        };
        let bind_entries = [
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
        ];
        let bg = if out_usage == BufferUsageClass::MatmulOut {
            Arc::new(
                self.device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-matmul-bind"),
                        layout,
                        entries: &bind_entries,
                    }),
            )
        } else {
            self.bind_group_cache
                .get_or_create(layout, &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-matmul-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                })
        };
        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
        let groups_y = if use_vec4 {
            let rows_vec = (m as u32) / 4;
            crate::backend::wgpu::dispatch::common::dispatch_size_dim(rows_vec, tile)
        } else {
            crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile)
        };
        crate::backend::wgpu::dispatch::matmul::run(
            self.device_ref(),
            self.queue_ref(),
            pipeline,
            bg.as_ref(),
            groups_x,
            groups_y,
        );
        let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer;
        let handle =
            self.register_existing_buffer_with_usage(out_buffer, out_shape, len, out_usage);
        if debug_matmul {
            log::debug!("[matmul_debug] out_ptr={:p} len={}", out_ptr, len);
        }
        self.remember_matmul_sources(&handle, a, b);
        self.telemetry.record_matmul_duration(start.elapsed());
        self.record_matmul_kernel_launch(m, n, k, use_vec4, false);
        Ok(handle)
    }
    pub(crate) fn pagefun_exec(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        match request.op {
            PagefunOp::Mtimes => self.pagefun_mtimes_exec(request),
        }
    }
    fn pagefun_mtimes_exec(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        ensure!(
            request.inputs.len() == 2,
            "pagefun: @mtimes expects exactly two inputs"
        );
        ensure!(
            request.input_page_dims.len() == request.inputs.len(),
            "pagefun: input metadata mismatch"
        );

        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let entry_a = self.get_entry(lhs)?;
        let entry_b = self.get_entry(rhs)?;

        let view_a = build_matrix_operand_view(lhs, &entry_a)
            .map_err(|e| anyhow!("pagefun @mtimes: {e}"))?;
        let view_b = build_matrix_operand_view(rhs, &entry_b)
            .map_err(|e| anyhow!("pagefun @mtimes: {e}"))?;

        let canonical_a = canonical_matrix_shape(&entry_a.shape);
        let canonical_b = canonical_matrix_shape(&entry_b.shape);
        ensure!(
            canonical_a.len() >= 2 && canonical_b.len() >= 2,
            "pagefun: @mtimes operands must be at least 2-D"
        );

        let rows = view_a.rows;
        let k_a = view_a.cols;
        let k_b = view_b.rows;
        let cols = view_b.cols;
        ensure!(
            k_a == k_b,
            "pagefun: inner matrix dimensions must agree ({} vs {})",
            k_a,
            k_b
        );

        let rank = request.page_dims.len();
        let lhs_dims = pad_dims(request.input_page_dims[0].clone(), rank);
        let rhs_dims = pad_dims(request.input_page_dims[1].clone(), rank);
        let lhs_strides = compute_page_strides(&lhs_dims);
        let rhs_strides = compute_page_strides(&rhs_dims);

        let lhs_page_size = rows
            .checked_mul(k_a)
            .ok_or_else(|| anyhow!("pagefun: lhs page size overflow"))?;
        let rhs_page_size = k_b
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("pagefun: rhs page size overflow"))?;
        let out_page_size = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("pagefun: output page size overflow"))?;

        let page_volume = if rank == 0 {
            1
        } else {
            product_checked(&request.page_dims)
                .ok_or_else(|| anyhow!("pagefun: page dimensions overflow"))?
        };

        let total_len = out_page_size
            .checked_mul(page_volume)
            .ok_or_else(|| anyhow!("pagefun: output size overflow"))?;
        let out_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-pagefun-mtimes-out")?;

        if total_len == 0 {
            return Ok(self.register_existing_buffer(
                out_buffer,
                request.output_shape.clone(),
                total_len,
            ));
        }

        let m_u32 = u32::try_from(rows)
            .map_err(|_| anyhow!("pagefun: matrix row count exceeds GPU limits"))?;
        let n_u32 = u32::try_from(cols)
            .map_err(|_| anyhow!("pagefun: matrix column count exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k_a)
            .map_err(|_| anyhow!("pagefun: shared dimension exceeds GPU limits"))?;

        let lda = view_a.lda;
        let ldb = view_b.lda;
        let ldc = m_u32;

        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);

        self.prepare_matmul_pipeline();
        self.device_ref().poll(wgpu::Maintain::Poll);

        let start = Instant::now();

        let mut multi_index = vec![0usize; rank];
        for page_idx in 0..page_volume {
            if rank > 0 {
                decode_multi_index(page_idx, &request.page_dims, &mut multi_index);
            }

            let lhs_linear = broadcast_linear_index(&lhs_dims, &lhs_strides, &multi_index);
            let rhs_linear = broadcast_linear_index(&rhs_dims, &rhs_strides, &multi_index);

            let lhs_offset_elements = lhs_linear
                .checked_mul(lhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: lhs offset overflow"))?;
            let rhs_offset_elements = rhs_linear
                .checked_mul(rhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: rhs offset overflow"))?;
            let out_offset_elements = page_idx
                .checked_mul(out_page_size)
                .ok_or_else(|| anyhow!("pagefun: output offset overflow"))?;

            let lhs_end = lhs_offset_elements
                .checked_add(lhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: lhs offset overflow"))?;
            let rhs_end = rhs_offset_elements
                .checked_add(rhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: rhs offset overflow"))?;
            let out_end = out_offset_elements
                .checked_add(out_page_size)
                .ok_or_else(|| anyhow!("pagefun: output offset overflow"))?;

            ensure!(
                lhs_end <= entry_a.len,
                "pagefun: lhs page out of bounds (page {})",
                page_idx
            );
            ensure!(
                rhs_end <= entry_b.len,
                "pagefun: rhs page out of bounds (page {})",
                page_idx
            );
            ensure!(
                out_end <= total_len,
                "pagefun: output page out of bounds (page {})",
                page_idx
            );

            let offset_a_u32 = u32::try_from(lhs_offset_elements)
                .map_err(|_| anyhow!("pagefun: lhs offset exceeds GPU limits"))?;
            let offset_b_u32 = u32::try_from(rhs_offset_elements)
                .map_err(|_| anyhow!("pagefun: rhs offset exceeds GPU limits"))?;
            let offset_out_u32 = u32::try_from(out_offset_elements)
                .map_err(|_| anyhow!("pagefun: output offset exceeds GPU limits"))?;

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
                lda,
                ldb,
                ldc,
                offset_a: offset_a_u32,
                offset_b: offset_b_u32,
                offset_out: offset_out_u32,
                flags,
            };

            let params_buffer = self.uniform_buffer(&params, "runmat-pagefun-mtimes-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-pagefun-mtimes-bind"),
                    layout: &self.pipelines.matmul.layout,
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

            crate::backend::wgpu::dispatch::matmul::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.matmul.pipeline,
                &bind_group,
                groups_x,
                groups_y,
            );
        }

        self.telemetry.record_matmul_duration(start.elapsed());

        let handle =
            self.register_existing_buffer(out_buffer, request.output_shape.clone(), total_len);

        Ok(handle)
    }
    async fn centered_gram_exec_kernel(
        &self,
        matrix: &GpuTensorHandle,
        matrix_entry: &BufferEntry,
        means: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        denom: f64,
    ) -> Result<GpuTensorHandle> {
        let rows_f64 = rows as f64;
        let means_entry = self.get_entry(means)?;
        let mut means_used = means.clone();
        let mut casted_means = false;
        if means_entry.precision != matrix_entry.precision {
            means_used = self
                .cast_tensor_precision(means, matrix_entry.precision)
                .await?;
            casted_means = true;
        }

        // Compute X^T * X using the SYRK pipeline (no explicit transpose required).
        let xtx = self.syrk_exec(matrix)?;

        // Form n * μ μᵀ without materialising a centered copy of X.
        let means_scaled = self.scalar_mul(&means_used, rows_f64)?;
        let means_col = self
            .reshape(&means_scaled, &[cols, 1])
            .map_err(|e| anyhow!("centered_gram: reshape means col failed: {e}"))?;
        let means_row_scaled = self
            .reshape(&means_scaled, &[1, cols])
            .map_err(|e| anyhow!("centered_gram: reshape means row failed: {e}"))?;

        let outer_scaled = self.matmul_exec_with_usage(
            &means_col,
            &means_row_scaled,
            BufferUsageClass::FusionOut,
        )?;
        let outer = self.scalar_mul(&outer_scaled, 1.0 / rows_f64)?;

        let _ = self.free(&means_col);
        let _ = self.free(&means_row_scaled);
        let _ = self.free(&outer_scaled);

        let centered =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &xtx, &outer)?;

        let _ = self.free(&xtx);
        let _ = self.free(&outer);
        let _ = self.free(&means_scaled);

        let handle = self.scalar_mul(&centered, 1.0 / denom)?;
        let _ = self.free(&centered);

        self.mark_buffer_usage(&handle, BufferUsageClass::FusionOut);

        if std::env::var("RUNMAT_DEBUG_CENTERED_GRAM").is_ok() {
            if let Err(err) = self
                .debug_centered_gram(
                    matrix,
                    matrix_entry.precision,
                    &means_used,
                    &handle,
                    rows,
                    cols,
                    denom,
                )
                .await
            {
                log::warn!("centered_gram debug instrumentation failed: {err}");
            }
        }

        if casted_means {
            let _ = self.free(&means_used);
        }

        Ok(handle)
    }
    #[allow(clippy::too_many_arguments)]
    async fn debug_centered_gram(
        &self,
        matrix: &GpuTensorHandle,
        precision: NumericPrecision,
        means: &GpuTensorHandle,
        output: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        denom: f64,
    ) -> Result<()> {
        let matrix_host = <Self as AccelProvider>::download(self, matrix).await?;
        let means_gpu = <Self as AccelProvider>::download(self, means).await?;
        let output_gpu = <Self as AccelProvider>::download(self, output).await?;
        if matrix_host.data.len() != rows * cols {
            return Err(anyhow!(
                "centered_gram debug: matrix download length mismatch ({} vs {})",
                matrix_host.data.len(),
                rows * cols
            ));
        }

        let mut mean_ref = vec![0.0f64; cols];
        for (col, mean_slot) in mean_ref.iter_mut().enumerate().take(cols) {
            let mut sum = 0.0f64;
            let base = col * rows;
            for row in 0..rows {
                sum += matrix_host.data[base + row];
            }
            *mean_slot = sum / (rows as f64);
        }

        let mut max_mean_diff = 0.0f64;
        for (mean, gpu_mean) in mean_ref.iter().zip(means_gpu.data.iter()) {
            let diff = (*mean - *gpu_mean).abs();
            if diff > max_mean_diff {
                max_mean_diff = diff;
            }
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..cols).collect();
        indices.shuffle(&mut rng);
        indices.truncate(cols.min(32));
        indices.sort_unstable();

        let mut max_abs_err = 0.0f64;
        let mut max_abs_idx = (0usize, 0usize);
        let mut max_rel_err = 0.0f64;
        let mut max_rel_idx = (0usize, 0usize);
        let mut max_diag_neg = 0.0f64;
        let mut max_diag_idx = 0usize;

        for &j in &indices {
            for &i in &indices {
                let mut sum = 0.0f64;
                let base_i = i * rows;
                let base_j = j * rows;
                for row in 0..rows {
                    let centered_i = matrix_host.data[base_i + row] - mean_ref[i];
                    let centered_j = matrix_host.data[base_j + row] - mean_ref[j];
                    sum += centered_i * centered_j;
                }
                sum /= denom;

                let gpu_val = output_gpu.data[i + j * cols];
                let abs_err = (gpu_val - sum).abs();
                if i == j && std::env::var("RUNMAT_DEBUG_CENTERED_GRAM_TRACE").is_ok() {
                    log::info!(
                        "centered_gram diag sample col={} gpu={:.6e} ref={:.6e}",
                        i,
                        gpu_val,
                        sum
                    );
                }
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                    max_abs_idx = (i, j);
                }
                if sum.abs() > 0.0 {
                    let rel_err = abs_err / sum.abs();
                    if rel_err > max_rel_err {
                        max_rel_err = rel_err;
                        max_rel_idx = (i, j);
                    }
                }
                if i == j && gpu_val < 0.0 {
                    let neg = gpu_val.abs();
                    if neg > max_diag_neg {
                        max_diag_neg = neg;
                        max_diag_idx = i;
                    }
                }
            }
        }

        let sample_preview: Vec<usize> = indices.iter().copied().take(16).collect();
        let rows_out = output_gpu.shape.first().copied().unwrap_or(cols);
        let diag_len = cols.min(rows_out);
        let mut trace = 0.0f64;
        for d in 0..diag_len {
            let idx = d + d * rows_out;
            if let Some(val) = output_gpu.data.get(idx) {
                trace += *val;
            }
        }
        log::info!(
            "centered_gram debug [{}]: rows={} cols={} sample_cols={} trace={:.6e} max_mean_diff={:.3e} max_abs_err={:.3e} at ({}, {}) max_rel_err={:.3e} at ({}, {}) max_diag_neg={:.3e} at ({}) samples={:?}",
            match precision {
                NumericPrecision::F32 => "f32",
                NumericPrecision::F64 => "f64",
            },
            rows,
            cols,
            indices.len(),
            trace,
            max_mean_diff,
            max_abs_err,
            max_abs_idx.0,
            max_abs_idx.1,
            max_rel_err,
            max_rel_idx.0,
            max_rel_idx.1,
            max_diag_neg,
            max_diag_idx,
            sample_preview
        );

        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn debug_qr_power_iter(
        &self,
        product: &GpuTensorHandle,
        product_entry: &BufferEntry,
        pre_product_max: Option<f64>,
        pre_q_max: Option<f64>,
        q_result: &GpuTensorHandle,
        r_handle: &GpuTensorHandle,
        r_inv_handle: &GpuTensorHandle,
        gram_host: Option<&HostTensorOwned>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let product_host = <Self as AccelProvider>::download(self, product).await?;
        let q_gpu_host = <Self as AccelProvider>::download(self, q_result).await?;
        let r_gpu_host = <Self as AccelProvider>::download(self, r_handle).await?;
        let r_inv_gpu_host = <Self as AccelProvider>::download(self, r_inv_handle).await?;
        let max_r_inv_abs = r_inv_gpu_host
            .data
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));

        if product_host.data.len() != rows * cols
            || q_gpu_host.data.len() != rows * cols
            || r_gpu_host.data.len() != cols * cols
            || r_inv_gpu_host.data.len() != cols * cols
        {
            return Err(anyhow!(
                "qr_power_iter debug: length mismatch (rows={}, cols={})",
                rows,
                cols
            ));
        }

        let gram_cow: Cow<'_, HostTensorOwned> = if let Some(host) = gram_host {
            Cow::Borrowed(host)
        } else {
            let product_t_tmp = self.transpose_exec(product)?;
            let gram_tmp =
                self.matmul_exec_with_usage(&product_t_tmp, product, BufferUsageClass::FusionOut)?;
            let _ = self.free(&product_t_tmp);
            let owned = <Self as AccelProvider>::download(self, &gram_tmp).await?;
            let _ = self.free(&gram_tmp);
            Cow::Owned(owned)
        };
        let gram_view: &HostTensorOwned = gram_cow.as_ref();

        if gram_view.data.len() != cols * cols {
            return Err(anyhow!(
                "qr_power_iter debug: Gram data mismatch (cols={})",
                cols
            ));
        }

        let mut min_r_diag = f64::MAX;
        let mut max_r_diag = f64::MIN;
        for i in 0..cols {
            let diag = r_gpu_host.data[i + i * cols];
            min_r_diag = min_r_diag.min(diag);
            max_r_diag = max_r_diag.max(diag);
        }

        let mut min_gram_diag = f64::MAX;
        let mut max_gram_diag = f64::MIN;
        for i in 0..cols {
            let diag = gram_view.data[i + i * cols];
            min_gram_diag = min_gram_diag.min(diag);
            max_gram_diag = max_gram_diag.max(diag);
        }

        let mut q_ref = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    sum += product_host.data[row + k * rows] * r_inv_gpu_host.data[k + col * cols];
                }
                q_ref[row + col * rows] = sum;
            }
        }

        let mut max_q_diff = 0.0f64;
        let mut max_q_diff_idx = 0usize;
        let mut max_q_abs = 0.0f64;
        let mut min_q_abs = f64::MAX;
        let mut non_zero_q = false;
        for (idx, (val, ref_val)) in q_gpu_host
            .data
            .iter()
            .zip(q_ref.iter())
            .enumerate()
            .take(rows * cols)
        {
            let diff = (*val - *ref_val).abs();
            if diff > max_q_diff {
                max_q_diff = diff;
                max_q_diff_idx = idx;
            }
            let abs_val = val.abs();
            if abs_val > max_q_abs {
                max_q_abs = abs_val;
            }
            if abs_val < min_q_abs {
                min_q_abs = abs_val;
            }
            if abs_val > 1.0e-12 {
                non_zero_q = true;
            }
        }
        if min_q_abs == f64::MAX {
            min_q_abs = 0.0;
        }

        let mut max_qtq_diag = 0.0f64;
        let mut max_qtq_diag_idx = 0usize;
        let mut max_qtq_off = 0.0f64;
        let mut max_qtq_off_idx = (0usize, 0usize);
        let mut min_diag_val = f64::MAX;
        let mut max_diag_val = f64::MIN;
        for j in 0..cols {
            for i in 0..cols {
                let mut sum = 0.0f64;
                for row in 0..rows {
                    sum += q_gpu_host.data[row + i * rows] * q_gpu_host.data[row + j * rows];
                }
                if i == j {
                    let err = (sum - 1.0).abs();
                    if err > max_qtq_diag {
                        max_qtq_diag = err;
                        max_qtq_diag_idx = i;
                    }
                    if sum < min_diag_val {
                        min_diag_val = sum;
                    }
                    if sum > max_diag_val {
                        max_diag_val = sum;
                    }
                } else {
                    let err = sum.abs();
                    if err > max_qtq_off {
                        max_qtq_off = err;
                        max_qtq_off_idx = (i, j);
                    }
                }
            }
        }

        let mut max_residual = 0.0f64;
        let mut max_residual_idx = (0usize, 0usize);
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    sum += q_gpu_host.data[row + k * rows] * r_gpu_host.data[k + col * cols];
                }
                let diff = (sum - product_host.data[row + col * rows]).abs();
                if diff > max_residual {
                    max_residual = diff;
                    max_residual_idx = (row, col);
                }
            }
        }

        let mut gq_gpu = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for l in 0..cols {
                    sum += gram_view.data[l + col * cols] * q_gpu_host.data[row + l * rows];
                }
                gq_gpu[row + col * rows] = sum;
            }
        }
        let mut gq_ref = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for l in 0..cols {
                    sum += gram_view.data[l + col * cols] * q_ref[row + l * rows];
                }
                gq_ref[row + col * rows] = sum;
            }
        }

        let mut gpu_topk = 0.0f64;
        let mut ref_topk = 0.0f64;
        for col in 0..cols {
            let mut diag_gpu = 0.0f64;
            let mut diag_ref = 0.0f64;
            for row in 0..rows {
                diag_gpu += q_gpu_host.data[row + col * rows] * gq_gpu[row + col * rows];
                diag_ref += q_ref[row + col * rows] * gq_ref[row + col * rows];
            }
            gpu_topk += diag_gpu;
            ref_topk += diag_ref;
        }
        let topk_diff = gpu_topk - ref_topk;
        let max_product_abs = product_host
            .data
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));

        log::info!(
            "qr_power_iter debug: rows={} cols={} max_q_diff={:.3e} at idx={} max_q_abs={:.3e} min_q_abs={:.3e} non_zero_q={} max_qtq_diag_err={:.3e} at col={} max_qtq_off={:.3e} at ({}, {}) min_diag={:.3e} max_diag={:.3e} max_residual={:.3e} at ({}, {}) max_product_abs_pre={:?} max_product_abs={:.3e} max_q_abs_pre={:?} max_r_inv_abs={:.3e} min_r_diag={:.3e} max_r_diag={:.3e} min_gram_diag={:.3e} max_gram_diag={:.3e} gpu_topk={:.6e} ref_topk={:.6e} diff={:.3e}",
            rows,
            cols,
            max_q_diff,
            max_q_diff_idx,
            max_q_abs,
            min_q_abs,
            non_zero_q,
            max_qtq_diag,
            max_qtq_diag_idx,
            max_qtq_off,
            max_qtq_off_idx.0,
            max_qtq_off_idx.1,
            min_diag_val,
            max_diag_val,
            max_residual,
            max_residual_idx.0,
            max_residual_idx.1,
            pre_product_max,
            max_product_abs,
            pre_q_max,
            max_r_inv_abs,
            min_r_diag,
            max_r_diag,
            min_gram_diag,
            max_gram_diag,
            gpu_topk,
            ref_topk,
            topk_diff
        );

        if !non_zero_q || max_product_abs <= 1.0e-12 {
            let active = active_fusion();
            let plan = active_group_plan_clone();
            log::warn!(
                "qr_power_iter zero-data alert: product={} len={} non_zero_q={} max_product_abs_pre={:?} max_product_abs={:.3e} max_q_abs_pre={:?} active={:?} plan_inputs={:?} stack_pattern={:?}",
                product.buffer_id,
                product_entry.len,
                non_zero_q,
                pre_product_max,
                max_product_abs,
                pre_q_max,
                active,
                plan.as_ref().map(|p| p.inputs.clone()),
                plan.as_ref().map(|p| p.stack_pattern.clone())
            );
        }

        Ok(())
    }
    pub(crate) async fn covariance_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CovRows::All {
            return Err(anyhow!(
                "covariance: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }
        if options.has_weight_vector {
            return Err(anyhow!(
                "covariance: weight vectors are not supported by WGPU provider"
            ));
        }

        let entry = self.get_entry(matrix)?;
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                    shape
                ))
            }
        };

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cov-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let denom = match options.normalization {
            CovNormalization::Unbiased => (rows as f64) - 1.0,
            CovNormalization::Biased => rows as f64,
        };

        if denom <= 0.0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let means = self.reduce_dim_sum_mean_exec(
            matrix,
            0,
            crate::backend::wgpu::types::DimReduceOp::Mean,
        )?;
        let result = self
            .centered_gram_exec_kernel(matrix, &entry, &means, rows, cols, denom)
            .await;
        let _ = self.free(&means);
        result
    }
    pub(crate) async fn corrcoef_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: &CorrcoefOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CorrcoefRows::All {
            return Err(anyhow!(
                "corrcoef: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }

        let entry = self.get_entry(matrix)?;
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "corrcoef: inputs must be 2-D matrices or vectors (got shape {:?})",
                    shape
                ))
            }
        };

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-corrcoef-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let denom = match options.normalization {
            CorrcoefNormalization::Unbiased => (rows as f64) - 1.0,
            CorrcoefNormalization::Biased => rows as f64,
        };

        if denom <= 0.0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let means = self.reduce_dim_sum_mean_exec(
            matrix,
            0,
            crate::backend::wgpu::types::DimReduceOp::Mean,
        )?;
        let ones = self.fill_exec(&[rows, 1], 1.0)?;
        let means_full = self.matmul_exec(&ones, &means)?;
        let centered = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            matrix,
            &means_full,
        )?;
        let squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &centered,
            &centered,
        )?;
        let centered_t = self.transpose_exec(&centered)?;
        let covariance = self.matmul_exec(&centered_t, &centered)?;
        let inv_denom = 1.0 / denom;
        let inv_cov = self.fill_exec(&covariance.shape, inv_denom)?;
        let covariance_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &covariance,
            &inv_cov,
        )?;

        let variance_sum = self.reduce_dim_sum_mean_exec(
            &squared,
            0,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let inv_var = self.fill_exec(&variance_sum.shape, inv_denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_sum,
            &inv_var,
        )?;

        // Clamp tiny negative variances to zero to stabilise sqrt
        let mut host_variance = <Self as AccelProvider>::download(self, &variance).await?;
        for value in host_variance.data.iter_mut() {
            if *value < 0.0 && *value > -1.0e-12 {
                *value = 0.0;
            }
        }
        let view = HostTensorView {
            data: &host_variance.data,
            shape: &host_variance.shape,
        };
        let variance_adjusted = self.upload(&view)?;
        self.free(&variance)?;

        let std = self.unary_op_exec(
            crate::backend::wgpu::types::UnaryOpCode::Sqrt,
            &variance_adjusted,
        )?;
        let std_t = self.transpose_exec(&std)?;
        let std_outer = self.matmul_exec(&std_t, &std)?;
        let correlation = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &covariance_scaled,
            &std_outer,
        )?;

        // Free temporaries
        let _ = self.free(&means);
        let _ = self.free(&ones);
        let _ = self.free(&means_full);
        let _ = self.free(&centered);
        let _ = self.free(&centered_t);
        let _ = self.free(&covariance);
        let _ = self.free(&inv_cov);
        let _ = self.free(&covariance_scaled);
        let _ = self.free(&squared);
        let _ = self.free(&variance_sum);
        let _ = self.free(&inv_var);
        let _ = self.free(&variance_adjusted);
        let _ = self.free(&std);
        let _ = self.free(&std_t);
        let _ = self.free(&std_outer);

        Ok(correlation)
    }
    async fn cast_tensor_precision(
        &self,
        tensor: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(tensor)?;
        if entry.precision == target {
            return Ok(tensor.clone());
        }

        let mut host = <Self as AccelProvider>::download(self, tensor).await?;
        if matches!(target, NumericPrecision::F32) {
            for value in host.data.iter_mut() {
                *value = (*value as f32) as f64;
            }
        }

        let view = HostTensorView {
            data: host.data.as_slice(),
            shape: host.shape.as_slice(),
        };
        self.upload(&view)
    }

    pub(crate) fn dot_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry_lhs = self.get_entry(lhs)?;
        let entry_rhs = self.get_entry(rhs)?;
        ensure!(
            entry_lhs.shape == entry_rhs.shape,
            "dot: shape mismatch between inputs"
        );
        if entry_lhs.shape.is_empty() {
            return self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs);
        }
        if entry_lhs.shape.len() != 2 {
            return Err(anyhow!(
                "dot: only 2D tensors are currently supported by the WGPU provider"
            ));
        }

        let shape = entry_lhs.shape.clone();
        let default_dim = shape
            .iter()
            .position(|&extent| extent != 1)
            .map(|idx| idx + 1)
            .unwrap_or(1);
        let target_dim = dim.unwrap_or(default_dim);
        let dim_index = target_dim.saturating_sub(1);

        if dim_index >= shape.len() {
            return self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs);
        }
        if dim_index > 1 {
            return Err(anyhow!(
                "dot: unsupported dimension {} for WGPU provider",
                target_dim
            ));
        }

        let product =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs)?;

        let reduce = self.reduce_dim_sum_mean_exec(
            &product,
            dim_index,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        );
        match reduce {
            Ok(handle) => {
                let _ = self.free(&product);
                Ok(handle)
            }
            Err(err) => {
                let _ = self.free(&product);
                Err(err)
            }
        }
    }

    pub(crate) fn cross_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry_lhs = self.get_entry(lhs)?;
        let entry_rhs = self.get_entry(rhs)?;
        ensure!(
            entry_lhs.shape == entry_rhs.shape,
            "cross: shape mismatch between inputs"
        );

        let shape = if entry_lhs.shape.is_empty() {
            vec![1, 1]
        } else {
            entry_lhs.shape.clone()
        };
        let rank = shape.len();
        let target_dim = match dim {
            Some(target_dim) => {
                ensure!(
                    target_dim >= 1 && target_dim <= rank,
                    "cross: dimension {} exceeds the number of array dimensions ({})",
                    target_dim,
                    rank
                );
                ensure!(
                    shape[target_dim - 1] == 3,
                    "cross: dimension {} must have length 3",
                    target_dim
                );
                target_dim
            }
            None => shape
                .iter()
                .position(|&extent| extent == 3)
                .map(|idx| idx + 1)
                .ok_or_else(|| anyhow!("cross: inputs must have a dimension of length 3"))?,
        };
        let dim_index = target_dim - 1;
        let total_len = entry_lhs.len;
        if total_len == 0 {
            return self.zeros_exec(&shape);
        }

        let stride_before = product_checked(&shape[..dim_index])
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let stride_after = product_checked(&shape[dim_index + 1..])
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let slice_stride = stride_before
            .checked_mul(3)
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let slice_count = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;

        let mut comp1 = Vec::with_capacity(slice_count);
        let mut comp2 = Vec::with_capacity(slice_count);
        let mut comp3 = Vec::with_capacity(slice_count);
        for after in 0..stride_after {
            let slice_base = after
                .checked_mul(slice_stride)
                .ok_or_else(|| anyhow!("cross: internal index overflow"))?;
            for before in 0..stride_before {
                let idx1 = slice_base + before;
                let idx2 = idx1 + stride_before;
                let idx3 = idx2 + stride_before;
                comp1.push(
                    u32::try_from(idx1).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
                comp2.push(
                    u32::try_from(idx2).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
                comp3.push(
                    u32::try_from(idx3).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
            }
        }

        let mut reduced_shape = shape.clone();
        reduced_shape[dim_index] = 1;

        // Track every intermediate handle outside the computation closure so that
        // handles allocated before a failing `?` are still freed on error.
        let mut to_free: Vec<GpuTensorHandle> = Vec::with_capacity(15);

        let compute_result: Result<GpuTensorHandle> = (|| {
            let a1 = self.gather_linear_exec(lhs, &comp1, &reduced_shape)?;
            to_free.push(a1.clone());
            let a2 = self.gather_linear_exec(lhs, &comp2, &reduced_shape)?;
            to_free.push(a2.clone());
            let a3 = self.gather_linear_exec(lhs, &comp3, &reduced_shape)?;
            to_free.push(a3.clone());
            let b1 = self.gather_linear_exec(rhs, &comp1, &reduced_shape)?;
            to_free.push(b1.clone());
            let b2 = self.gather_linear_exec(rhs, &comp2, &reduced_shape)?;
            to_free.push(b2.clone());
            let b3 = self.gather_linear_exec(rhs, &comp3, &reduced_shape)?;
            to_free.push(b3.clone());

            let a2b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b3)?;
            to_free.push(a2b3.clone());
            let a3b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b2)?;
            to_free.push(a3b2.clone());
            let c1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a2b3, &a3b2)?;
            to_free.push(c1.clone());

            let a3b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b1)?;
            to_free.push(a3b1.clone());
            let a1b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b3)?;
            to_free.push(a1b3.clone());
            let c2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a3b1, &a1b3)?;
            to_free.push(c2.clone());

            let a1b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b2)?;
            to_free.push(a1b2.clone());
            let a2b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b1)?;
            to_free.push(a2b1.clone());
            let c3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a1b2, &a2b1)?;
            to_free.push(c3.clone());

            let out = self.zeros_exec(&shape)?;
            let scatter_result = (|| -> Result<()> {
                self.scatter_linear_exec(&out, &comp1, &c1)?;
                self.scatter_linear_exec(&out, &comp2, &c2)?;
                self.scatter_linear_exec(&out, &comp3, &c3)?;
                Ok(())
            })();

            match scatter_result {
                Ok(()) => Ok(out),
                Err(err) => {
                    let _ = self.free(&out);
                    Err(err)
                }
            }
        })();

        for h in &to_free {
            let _ = self.free(h);
        }

        compute_result
    }

    pub(super) fn qr_factor_device(
        &self,
        matrix: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        reuse_q: Option<&GpuTensorHandle>,
        label: &str,
        retain_r_inv: bool,
    ) -> Result<(GpuTensorHandle, GpuTensorHandle, Option<GpuTensorHandle>)> {
        ensure!(rows >= cols, "qr: rows must be >= cols for device path");
        ensure!(
            cols > 0,
            "qr: zero-column input not supported for device path"
        );

        let gram_handle = self.syrk_exec(matrix)?;

        let gram_entry = self.get_entry(&gram_handle)?;
        let gram_len = cols * cols;
        ensure!(
            gram_entry.len == gram_len,
            "qr: gram len mismatch (expected {}, got {})",
            gram_len,
            gram_entry.len
        );
        let gram_bytes = (gram_len as u64) * (self.element_size as u64);
        let gram_scratch = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrGram,
            gram_bytes,
            "runmat-qr-gram-scratch",
        );
        if gram_bytes > 0 {
            let gram_copy_label = format!("{label}-gram-copy");
            let mut encoder =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(gram_copy_label.as_str()),
                    });
            encoder.copy_buffer_to_buffer(
                gram_entry.buffer.as_ref(),
                0,
                gram_scratch.as_ref(),
                0,
                gram_bytes,
            );
            self.submit(encoder);
        }

        let len_out = cols * cols;
        let r_bytes = (len_out as u64) * (self.element_size as u64);
        let r_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrR,
            r_bytes,
            "runmat-qr-r-scratch",
        );
        let r_inv_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrRInv,
            r_bytes,
            "runmat-qr-rinv-scratch",
        );

        let params = QrPowerIterParams {
            cols: cols as u32,
            stride: cols as u32,
            _pad0: [0, 0],
        };
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::QrPowerIterParams,
            std::mem::size_of::<QrPowerIterParams>() as u64,
            "runmat-qr-power-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));

        let layout = &self.pipelines.qr_power_iter.layout;
        let bind_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gram_scratch.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: r_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: r_inv_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bind_group = self
            .bind_group_cache
            .get_or_create(layout, &bind_entries, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-qr-power-bind"),
                            layout,
                            entries: &bind_entries,
                        }),
                )
            });
        crate::backend::wgpu::dispatch::qr_power_iter::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.qr_power_iter.pipeline,
            bind_group.as_ref(),
        );

        let _ = self.free(&gram_handle);

        let r_shape = vec![cols, cols];
        let r_handle = self.register_existing_buffer_with_usage(
            r_buffer.clone(),
            r_shape.clone(),
            len_out,
            BufferUsageClass::FusionOut,
        );
        self.mark_buffer_usage(&r_handle, BufferUsageClass::FusionOut);

        let r_inv_handle = self.register_existing_buffer_with_usage(
            r_inv_buffer.clone(),
            r_shape,
            len_out,
            BufferUsageClass::FusionOut,
        );
        self.mark_buffer_usage(&r_inv_handle, BufferUsageClass::FusionOut);

        let q_temp =
            self.matmul_exec_with_usage(matrix, &r_inv_handle, BufferUsageClass::FusionOut)?;

        let q_temp_entry = self.get_entry(&q_temp)?;
        let q_result = if let Some(target) = reuse_q {
            let target_entry = self.get_entry(target)?;
            if Arc::strong_count(&target_entry.buffer) <= 2 && target_entry.len == q_temp_entry.len
            {
                let bytes = (target_entry.len as u64) * self.element_size as u64;
                if bytes > 0 {
                    let copy_label = format!("{label}-reuse-copy");
                    let mut encoder =
                        self.device_ref()
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some(copy_label.as_str()),
                            });
                    encoder.copy_buffer_to_buffer(
                        q_temp_entry.buffer.as_ref(),
                        0,
                        target_entry.buffer.as_ref(),
                        0,
                        bytes,
                    );
                    self.submit(encoder);
                }
                let _ = self.free(&q_temp);
                self.mark_buffer_usage(target, BufferUsageClass::FusionOut);
                target.clone()
            } else {
                q_temp
            }
        } else {
            q_temp
        };

        let r_inv_result = if retain_r_inv {
            Some(r_inv_handle)
        } else {
            let _ = self.free(&r_inv_handle);
            None
        };

        Ok((q_result, r_handle, r_inv_result))
    }

    pub(super) async fn qr_power_iter_host(
        &self,
        product: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> Result<Option<ProviderQrPowerIterResult>> {
        let host_product = <Self as AccelProvider>::download(self, product).await?;
        let tensor =
            Tensor::new(host_product.data.clone(), host_product.shape.clone()).map_err(|e| {
                anyhow!("qr_power_iter: failed to construct host tensor for fallback: {e}")
            })?;
        let host_result = self.qr_host_result(tensor, options).await?;
        let _ = self.free(product);
        Ok(Some(ProviderQrPowerIterResult {
            q: host_result.q,
            r: host_result.r,
            perm_matrix: host_result.perm_matrix,
            perm_vector: host_result.perm_vector,
        }))
    }

    pub(super) fn try_qr_device(
        &self,
        matrix: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> Result<Option<ProviderQrResult>> {
        if !options.economy {
            return Ok(None);
        }
        if options.pivot != ProviderQrPivot::Matrix {
            return Ok(None);
        }
        if self.precision() != ProviderPrecision::F32 {
            return Ok(None);
        }
        let entry = self.get_entry(matrix)?;
        if entry.shape.len() != 2 {
            return Ok(None);
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if rows < cols || cols == 0 {
            return Ok(None);
        }
        if cols > QR_DEVICE_MAX_COLS {
            return Ok(None);
        }
        if rows
            .checked_mul(cols)
            .map(|v| v > QR_DEVICE_MAX_ELEMS)
            .unwrap_or(true)
        {
            return Ok(None);
        }

        let (q_handle, r_handle, _) =
            self.qr_factor_device(matrix, rows, cols, None, "runmat-qr-direct", false)?;

        let mut perm_matrix = vec![0.0f64; cols * cols];
        for i in 0..cols {
            perm_matrix[i + i * cols] = 1.0;
        }
        let perm_vector: Vec<f64> = (1..=cols).map(|v| v as f64).collect();

        let perm_matrix_shape = [cols, cols];
        let perm_matrix_handle = self.upload(&HostTensorView {
            data: &perm_matrix,
            shape: &perm_matrix_shape,
        })?;
        let perm_vector_shape = vec![cols, 1];
        let perm_vector_handle = self.upload(&HostTensorView {
            data: &perm_vector,
            shape: &perm_vector_shape,
        })?;

        Ok(Some(ProviderQrResult {
            q: q_handle,
            r: r_handle,
            perm_matrix: perm_matrix_handle,
            perm_vector: perm_vector_handle,
        }))
    }

    pub(super) async fn qr_host_result(
        &self,
        tensor: Tensor,
        options: &ProviderQrOptions,
    ) -> Result<ProviderQrResult> {
        let mut args = Vec::new();
        if options.economy {
            args.push(Value::Num(0.0));
        }
        if matches!(options.pivot, ProviderQrPivot::Vector) {
            args.push(Value::from("vector"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::qr::evaluate(
            Value::Tensor(tensor),
            &args,
        )
        .await
        .map_err(|err| runtime_flow_to_anyhow("qr", err))?;

        let q_tensor = host_tensor_from_value("qr", eval.q())?;
        let r_tensor = host_tensor_from_value("qr", eval.r())?;
        let perm_matrix_tensor = host_tensor_from_value("qr", eval.permutation_matrix())?;
        let perm_vector_tensor = host_tensor_from_value("qr", eval.permutation_vector())?;

        let q = self.upload(&HostTensorView {
            data: &q_tensor.data,
            shape: &q_tensor.shape,
        })?;
        let r = self.upload(&HostTensorView {
            data: &r_tensor.data,
            shape: &r_tensor.shape,
        })?;
        let perm_matrix = self.upload(&HostTensorView {
            data: &perm_matrix_tensor.data,
            shape: &perm_matrix_tensor.shape,
        })?;
        let perm_vector = self.upload(&HostTensorView {
            data: &perm_vector_tensor.data,
            shape: &perm_vector_tensor.shape,
        })?;

        Ok(ProviderQrResult {
            q,
            r,
            perm_matrix,
            perm_vector,
        })
    }

    pub(crate) async fn lu_exec(&self, handle: &GpuTensorHandle) -> Result<ProviderLuResult> {
        let host = <Self as AccelProvider>::download(self, handle).await?;
        let LuHostFactors {
            combined,
            lower,
            upper,
            perm_matrix,
            pivot_vector,
            combined_shape,
            lower_shape,
            upper_shape,
            perm_shape,
            pivot_shape,
        } = lu_factor_host(&host.data, &host.shape)?;
        let combined = self.upload(&HostTensorView {
            data: &combined,
            shape: &combined_shape,
        })?;
        let lower = self.upload(&HostTensorView {
            data: &lower,
            shape: &lower_shape,
        })?;
        let upper = self.upload(&HostTensorView {
            data: &upper,
            shape: &upper_shape,
        })?;
        let perm_matrix = self.upload(&HostTensorView {
            data: &perm_matrix,
            shape: &perm_shape,
        })?;
        let perm_vector = self.upload(&HostTensorView {
            data: &pivot_vector,
            shape: &pivot_shape,
        })?;
        Ok(ProviderLuResult {
            combined,
            lower,
            upper,
            perm_matrix,
            perm_vector,
        })
    }

    pub(crate) async fn chol_exec(
        &self,
        handle: &GpuTensorHandle,
        lower: bool,
    ) -> Result<ProviderCholResult> {
        let host = <Self as AccelProvider>::download(self, handle).await?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("chol: {e}"))?;
        let mut args = Vec::new();
        if lower {
            args.push(Value::from("lower"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
            Value::Tensor(tensor),
            &args,
        )
        .await
        .map_err(|err| runtime_flow_to_anyhow("chol", err))?;
        let factor_tensor = host_tensor_from_value("chol", eval.factor())?;
        let factor = self.upload(&HostTensorView {
            data: &factor_tensor.data,
            shape: &factor_tensor.shape,
        })?;
        Ok(ProviderCholResult {
            factor,
            info: eval.flag_index() as u32,
        })
    }

    pub(crate) async fn qr_exec(
        &self,
        handle: &GpuTensorHandle,
        options: ProviderQrOptions,
    ) -> Result<ProviderQrResult> {
        if let Some(result) = self.try_qr_device(handle, &options)? {
            return Ok(result);
        }
        let host = <Self as AccelProvider>::download(self, handle).await?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("qr: {e}"))?;
        self.qr_host_result(tensor, &options).await
    }
}
