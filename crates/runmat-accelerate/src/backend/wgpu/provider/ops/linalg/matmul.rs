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

    fn tensor_max_abs_scalar_exec(&self, tensor: &GpuTensorHandle) -> Result<f64> {
        let abs = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, tensor)?;
        let max =
            self.reduce_global_exec(&abs, crate::backend::wgpu::types::GlobalReduceOp::Max)?;
        let value = self.read_scalar_exec(&max, 0);
        let _ = self.free_exec(&max);
        let _ = self.free_exec(&abs);
        value
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
        if self.provider_precision_exec() != ProviderPrecision::F32 {
            if debug_qr {
                log::debug!(
                    "qr_power_iter: precision {:?} unsupported for device QR kernel; falling back",
                    self.provider_precision_exec()
                );
            }
            return Ok(None);
        }
        let q_entry = self.get_entry(q_handle)?;
        if q_entry.shape != product_entry.shape {
            return Ok(None);
        }
        let k = cols;

        let mut pre_product_max = self.tensor_max_abs_scalar_exec(product).ok();
        let pre_q_max = self.tensor_max_abs_scalar_exec(q_handle).ok();

        const PRODUCT_EPS: f64 = 1.0e-12;
        const Q_EPS: f64 = 1.0e-6;
        if pre_product_max.unwrap_or(0.0) <= PRODUCT_EPS && pre_q_max.unwrap_or(0.0) > Q_EPS {
            let debug_zero_host = std::env::var("RUNMAT_DEBUG_QR_ZEROHOST").is_ok();
            if debug_zero_host {
                if let Some(lhs_handle) = product_lhs {
                    let lhs_download = self.download_exec(lhs_handle).await;
                    let q_download = self.download_exec(q_handle).await;
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
                let recomputed_max = self.tensor_max_abs_scalar_exec(&recomputed).ok();
                if debug_zero_host {
                    if let Some(max_recomputed) = recomputed_max {
                        log::info!(
                            "qr_power_iter recompute check: product_id={} max_recomputed_abs={:.6e}",
                            product.buffer_id,
                            max_recomputed
                        );
                    } else {
                        log::info!(
                            "qr_power_iter recompute check failed: product_id={}",
                            product.buffer_id
                        );
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

                let max_val = recomputed_max.unwrap_or(0.0);
                log::info!(
                    "qr_power_iter recompute copy: product_id={} bytes={} post_max={:.6e}",
                    product.buffer_id,
                    bytes,
                    max_val
                );
                if max_val == 0.0 {
                    log::warn!(
                        "qr_power_iter: recomputed product is still zero; falling back to host QR"
                    );
                    let _ = self.free_exec(&recomputed);
                    if let Some(handle) = self.qr_power_iter_host(product, options).await? {
                        return Ok(Some(handle));
                    }
                    return Ok(None);
                }
                pre_product_max = Some(max_val);

                let _ = self.free_exec(&recomputed);
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
        if let Some(r_inv_handle) = r_inv_opt.as_ref() {
            if let Ok(max_r_inv_abs) = self.tensor_max_abs_scalar_exec(r_inv_handle) {
                if !max_r_inv_abs.is_finite() {
                    fallback_needed = true;
                }
            }
        }

        if fallback_needed {
            if let Some(handle) = r_inv_opt.take() {
                let _ = self.free_exec(&handle);
            }
            let _ = self.free_exec(&q_result);
            let _ = self.free_exec(&r_handle);
            return self.qr_power_iter_host(product, options).await;
        }

        if pre_product_max.unwrap_or(0.0) <= 1.0e-8 {
            if let Some(handle) = r_inv_opt.take() {
                let _ = self.free_exec(&handle);
            }
            let _ = self.free_exec(&q_result);
            let _ = self.free_exec(&r_handle);
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
            let _ = self.free_exec(&handle);
        }

        let mut perm_matrix = vec![0.0f64; k * k];
        for i in 0..k {
            perm_matrix[i + i * k] = 1.0;
        }
        let perm_vector: Vec<f64> = (1..=k).map(|v| v as f64).collect();

        let perm_matrix_shape = [k, k];
        let perm_matrix_handle = self.upload_exec(&HostTensorView {
            data: &perm_matrix,
            shape: &perm_matrix_shape,
        })?;
        let perm_vector_shape = vec![k, 1];
        let perm_vector_handle = self.upload_exec(&HostTensorView {
            data: &perm_vector,
            shape: &perm_vector_shape,
        })?;

        let _ = self.free_exec(product);

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
        let (shader_src, ep_buf, pipeline_layout) = match self.provider_precision_exec() {
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
        let _ = self.free_exec(&squared);
        if epilogue.epsilon != 0.0 {
            let eps = self.fill_exec(&sum_sq.shape, epilogue.epsilon)?;
            let adjusted = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Add,
                &sum_sq,
                &eps,
            )?;
            let _ = self.free_exec(&sum_sq);
            let _ = self.free_exec(&eps);
            sum_sq = adjusted;
        }
        let norms = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
        let _ = self.free_exec(&sum_sq);
        let normalized = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &product,
            &norms,
        )?;
        let _ = self.free_exec(&product);
        let _ = self.free_exec(&norms);

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
                let _ = self.free_exec(&normalized);
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
    pub(in super::super) fn matmul_exec_with_usage(
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
                        self.free_exec(&prev).ok();
                        self.free_exec(&partial).ok();
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
}
