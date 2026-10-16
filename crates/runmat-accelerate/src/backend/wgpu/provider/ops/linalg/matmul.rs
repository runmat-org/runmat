use super::*;

const K_CHUNK: usize = 8192;
const K_CHUNK_SWITCH: usize = 65536;

#[path = "matmul/selection.rs"]
mod selection;
#[path = "matmul/validated_epilogue.rs"]
mod validated_epilogue;
pub(in crate::backend::wgpu::provider) use selection::{
    selected_matmul_pipeline, MatmulPipelineRoute,
};

#[cfg(test)]
#[path = "matmul/ownership_tests.rs"]
mod ownership_tests;

impl WgpuProvider {
    /// Upper bound for the native real matmul buffer path, including the
    /// possible chunked accumulation when both operands are contiguous.
    pub(crate) fn matmul_buffer_demand(&self, m: u64, n: u64, k: u64) -> Option<u64> {
        let len = m.checked_mul(n)?;
        let output = len.max(1).checked_mul(self.element_size as u64)?;
        if len == 0 {
            return Some(output);
        }
        let params_size = (std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>()
            as u64)
            .max(1)
            .div_ceil(256)
            .checked_mul(256)?;
        let ordinary = output.checked_add(params_size)?;
        if k < K_CHUNK_SWITCH as u64 {
            return Some(ordinary);
        }
        let chunks = k.div_ceil(K_CHUNK as u64);
        let accumulation = self.binary_same_shape_buffer_demand(len)?;
        let chunked = output
            .checked_mul(2)?
            .checked_add(chunks.checked_mul(params_size)?)?
            .checked_add((chunks - 1).checked_mul(accumulation)?)?;
        Some(ordinary.max(chunked))
    }

    pub(crate) fn take_matmul_sources_exec(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        self.get_entry_raw(product).ok()?;
        let res = self.kernel_resources.take_matmul_sources(product);
        log::debug!(
            "take_matmul_sources: product={} found={} active_fusion={:?}",
            product.buffer_id(),
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
        drop(max);
        drop(abs);
        value
    }

    pub(crate) async fn qr_power_iter_exec(
        &self,
        product: &GpuTensorHandle,
        product_lhs: Option<&GpuTensorHandle>,
        q_handle: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<ProviderQrPowerIterResult>> {
        let debug_qr = std::env::var("RUNMAT_DEBUG_QR").is_ok();
        if !options.economy
            || !matches!(options.column_order, ProviderQrColumnOrder::Unpivoted)
            || !matches!(
                options.outputs,
                ProviderQrOutputs::Qr | ProviderQrOutputs::FullEnvelope
            )
        {
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
        let mut repaired_product = None;
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
                                product.buffer_id()
                            );
                        }
                    }
                } else {
                    log::info!(
                        "qr_power_iter host check skipped: product_lhs unavailable (product_id={})",
                        product.buffer_id()
                    );
                }
            }
            if let Some(lhs_handle) = product_lhs {
                log::warn!(
                    "qr_power_iter: detected zero matmul product (product_id={} max_product_abs_pre={:?} max_q_abs_pre={:?}); recomputing",
                    product.buffer_id(),
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
                            product.buffer_id(),
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
                            product.buffer_id(),
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
                            product.buffer_id(),
                            max_recomputed
                        );
                    } else {
                        log::info!(
                            "qr_power_iter recompute check failed: product_id={}",
                            product.buffer_id()
                        );
                    }
                }
                let recomputed_entry = self.get_entry(&recomputed)?;
                log::info!(
                    "qr_power_iter recompute start: product_id={} original_len={} recomputed_len={}",
                    product.buffer_id(),
                    product_entry.len,
                    recomputed_entry.len
                );
                let max_val = recomputed_max.unwrap_or(0.0);
                log::info!(
                    "qr_power_iter recomputed privately: product_id={} post_max={:.6e}",
                    product.buffer_id(),
                    max_val
                );
                if max_val == 0.0 {
                    log::warn!(
                        "qr_power_iter: recomputed product is still zero; falling back to host QR"
                    );
                    if let Some(handle) = self.qr_power_iter_host(&recomputed, options).await? {
                        return Ok(Some(handle));
                    }
                    return Ok(None);
                }
                pre_product_max = Some(max_val);

                repaired_product = Some(recomputed);
            } else {
                log::warn!(
                    "qr_power_iter: zero product detected for buffer {} without lhs handle; proceeding with existing data",
                    product.buffer_id()
                );
            }
        }

        let product = repaired_product.as_ref().unwrap_or(product);
        let product_entry = self.get_entry(product)?;
        let (q_result, r_handle, mut r_inv_opt) =
            self.qr_factor_device(product, rows, cols, "runmat-qr-power", true)?;

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
                drop(handle);
            }
            drop(q_result);
            drop(r_handle);
            return self.qr_power_iter_host(product, options).await;
        }

        if pre_product_max.unwrap_or(0.0) <= 1.0e-8 {
            if let Some(handle) = r_inv_opt.take() {
                drop(handle);
            }
            drop(q_result);
            drop(r_handle);
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
            drop(handle);
        }

        if matches!(options.outputs, ProviderQrOutputs::Qr) {
            return Ok(Some(ProviderQrResult::Qr {
                q: q_result,
                r: r_handle,
            }));
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

        Ok(Some(ProviderQrResult::FullEnvelope {
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
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderExecutionError;
        use runmat_accelerate_api::ProviderPrecision;
        let frame = validated_epilogue::MatmulEpilogueFrame::plan(self, a, b, ep)?;
        let out_shape = vec![frame.m, frame.n];
        if frame.len == 0 {
            let out_buffer = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer_checked(emission, 0, "runmat-matmul-epilogue-out")
                })
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            return self
                .register_existing_buffer(out_buffer, out_shape, 0)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects());
        }
        let validated = self
            .prepare_validated_matmul_epilogue_pipeline(&frame.recipe)
            .await?;
        validated
            .require_matmul_epilogue(&frame.recipe)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        frame
            .require_live_inputs(self, a, b, ep)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let start = Instant::now();
        let out_buffer = self
            .with_emission(|emission, pending| -> Result<_> {
                frame.require_live_inputs(self, a, b, ep)?;
                let out_buffer = self.create_storage_buffer_checked(
                    emission,
                    frame.len,
                    "runmat-matmul-epilogue-out",
                )?;
                let params_buffer =
                    self.uniform_buffer(emission, &frame.params, "runmat-matmul-epilogue-params")?;

                use crate::backend::wgpu::params::{
                    MATMUL_EPILOGUE_FLAG_CLAMP_MAX, MATMUL_EPILOGUE_FLAG_CLAMP_MIN,
                    MATMUL_EPILOGUE_FLAG_COL_DIV, MATMUL_EPILOGUE_FLAG_COL_SCALE,
                    MATMUL_EPILOGUE_FLAG_POW, MATMUL_EPILOGUE_FLAG_ROW_DIV,
                    MATMUL_EPILOGUE_FLAG_ROW_SCALE,
                };
                let has_row = ep.row_scale.is_some();
                let has_col = ep.col_scale.is_some();
                let dummy_rowcol =
                    self.create_storage_buffer(emission, 1, "runmat-matmul-epilogue-dummy-scale")?;
                let row_buf = match &frame.row {
                    Some(entry) => entry.buffer.clone(),
                    None => dummy_rowcol.clone(),
                };
                let col_buf = match &frame.column {
                    Some(entry) => entry.buffer.clone(),
                    None => dummy_rowcol.clone(),
                };

                let (mut flags, alpha, beta) =
                    crate::backend::wgpu::params::matmul_affine_params(ep);
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
                let ep_buf = match self.provider_precision_exec() {
                    ProviderPrecision::F64 => {
                        let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF64 {
                            alpha,
                            beta,
                            clamp_min,
                            clamp_max,
                            pow_exponent,
                            flags,
                            diag_offset: 0,
                            diag_stride: 1,
                            diag_rows: 0,
                            _pad: 0,
                            _pad2: 0,
                        };
                        self.uniform_buffer(emission, &ep_params, "runmat-matmul-epilogue-uniform")?
                    }
                    ProviderPrecision::F32 => {
                        let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF32 {
                            alpha: alpha as f32,
                            beta: beta as f32,
                            clamp_min: clamp_min as f32,
                            clamp_max: clamp_max as f32,
                            pow_exponent: pow_exponent as f32,
                            flags,
                            diag_offset: 0,
                            diag_stride: 1,
                            diag_rows: 0,
                            _pad: 0,
                        };
                        self.uniform_buffer(emission, &ep_params, "runmat-matmul-epilogue-uniform")?
                    }
                };

                let bg = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-matmul-epilogue-bind"),
                        layout: validated.layout.as_ref(),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: frame.a.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: frame.b.buffer.as_ref().as_entire_binding(),
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
                    emission,
                    pending,
                    validated.pipeline.as_ref(),
                    &bg,
                    frame.groups[0],
                    frame.groups[1],
                )?;
                Ok(out_buffer)
            })
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let handle = self
            .register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                frame.len,
                BufferUsageClass::FusionOut,
            )
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;

        self.telemetry.record_matmul_duration(start.elapsed());

        Ok(handle)
    }

    pub(crate) async fn matmul_power_step_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        epilogue: &runmat_accelerate_api::PowerStepEpilogue,
    ) -> Result<GpuTensorHandle> {
        // The selected chain reaches an F64 reduction after matmul and
        // elementwise work. Reject an unavailable IEEE-word kernel before
        // any of those earlier provider effects can occur.
        self.require_ieee_words(self.precision)?;
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
        drop(squared);
        if epilogue.epsilon != 0.0 {
            let eps = self.fill_exec(sum_sq.shape(), epilogue.epsilon)?;
            let adjusted = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Add,
                &sum_sq,
                &eps,
            )?;
            drop(sum_sq);
            drop(eps);
            sum_sq = adjusted;
        }
        let norms = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
        drop(sum_sq);
        let normalized = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &product,
            &norms,
        )?;
        drop(product);
        drop(norms);

        // A borrowed RHS may have other views sharing its physical lease.
        // Buffer Arc counts do not establish exclusive tensor ownership.
        Ok(normalized)
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

        let out_buffer = self.with_emission(|emission, _| {
            // This buffer escapes as a live result. Reusable kernel scratch
            // cannot be its physical owner while earlier results remain live.
            self.create_storage_buffer_checked_with_usage(
                emission,
                len,
                "runmat-syrk-out",
                BufferUsageClass::SyrkOut,
            )
        })??;
        if len == 0 {
            return self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                0,
                BufferUsageClass::SyrkOut,
            );
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

        self.with_emission(|emission, pending| -> Result<()> {
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
                    emission,
                    UniformBufferKey::SyrkParams,
                    std::mem::size_of::<crate::backend::wgpu::params::SyrkParams>() as u64,
                    "runmat-syrk-params",
                )?;
                emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
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
                let layout = &self.pipelines.syrk.get(emission)?.layout;
                let bind_group =
                    self.bind_group_cache.get_or_create(
                        emission,
                        layout,
                        &bind_entries,
                        |emission| {
                            Arc::new(emission.device().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-syrk-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        },
                    )?;

                crate::backend::wgpu::dispatch::syrk::run(
                    emission,
                    pending,
                    &self.pipelines.syrk.get(emission)?.pipeline,
                    bind_group.as_ref(),
                    groups_x,
                    groups_y,
                )?;

                offset += chunk_rows;
                first_chunk = false;
            }

            Ok(())
        })??;

        self.register_existing_buffer_with_usage(
            out_buffer,
            out_shape,
            len,
            BufferUsageClass::SyrkOut,
        )
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
            let (out_buffer, _) = self.with_emission(|emission, _| {
                self.create_storage_buffer_for_usage(emission, out_usage, 0, "runmat-matmul-out")
            })??;
            return self.register_existing_buffer_with_usage(out_buffer, out_shape, 0, out_usage);
        }

        let m_u32 = u32::try_from(m).map_err(|_| anyhow!("matmul: m exceeds GPU limits"))?;
        let n_u32 = u32::try_from(n).map_err(|_| anyhow!("matmul: n exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k).map_err(|_| anyhow!("matmul: k exceeds GPU limits"))?;

        let selected_pipeline =
            selected_matmul_pipeline(self.precision, m, n, k, view_a.transpose, view_b.transpose);
        let use_vec4 = selected_pipeline == MatmulPipelineRoute::Vec4;
        let enable_chunk = selected_pipeline == MatmulPipelineRoute::Chunked;
        if debug_matmul {
            log::debug!(
                "[matmul_debug] selected_pipeline={:?} use_vec4={} enable_chunk={} usage={:?}",
                selected_pipeline,
                use_vec4,
                enable_chunk,
                out_usage
            );
        }

        let start = Instant::now();

        if enable_chunk {
            self.with_emission(|emission, pending| {
                self.prepare_matmul_pipeline(emission, pending)
            })??;
            self.access.poll(wgpu::Maintain::Poll);
            let lda_u32 = view_a.lda;
            let ldb_u32 = view_b.lda;
            // Accumulator handle across chunks
            let mut acc: Option<GpuTensorHandle> = None;
            let mut k_off: usize = 0;
            let partial_storage = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked_with_usage(
                    emission,
                    len,
                    "runmat-matmul-partial",
                    BufferUsageClass::MatmulPartial,
                )
            })??;
            while k_off < k {
                let k_sub = std::cmp::min(K_CHUNK, k - k_off);
                // Create partial output buffer and bind group
                // The first partial becomes the accumulator itself. It must
                // not alias reusable storage overwritten by the next chunk.
                // Subsequent partials are consumed by a distinct binary sum
                // before this same-queue scratch is reused.
                let partial_buffer = if k_off == 0 {
                    self.with_emission(|emission, _| {
                        self.create_storage_buffer_checked_with_usage(
                            emission,
                            len,
                            "runmat-matmul-first-partial",
                            BufferUsageClass::MatmulPartial,
                        )
                    })??
                } else {
                    partial_storage.clone()
                };
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
                self.with_emission(|emission, pending| -> Result<()> {
                    let params_buffer = self.kernel_resources.uniform_buffer(
                        emission,
                        UniformBufferKey::MatmulParams,
                        std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
                        "runmat-matmul-params",
                    )?;
                    emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
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
                    let layout = &self.pipelines.matmul.get(emission)?.layout;
                    let bg = self.bind_group_cache.get_or_create(
                        emission,
                        layout,
                        &bind_entries,
                        |emission| {
                            Arc::new(emission.device().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-matmul-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        },
                    )?;
                    let tile = crate::backend::wgpu::config::effective_matmul_tile();
                    let groups_x =
                        crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
                    let groups_y =
                        crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);
                    crate::backend::wgpu::dispatch::matmul::run(
                        emission,
                        pending,
                        &self.pipelines.matmul.get(emission)?.pipeline,
                        bg.as_ref(),
                        groups_x,
                        groups_y,
                    )?;
                    Ok(())
                })??;
                // Wrap partial buffer into handle
                let partial = self.register_existing_buffer_with_usage(
                    partial_buffer,
                    out_shape.clone(),
                    len,
                    BufferUsageClass::MatmulPartial,
                )?;
                acc = match acc {
                    None => Some(partial),
                    Some(prev) => {
                        let sum = self.binary_op_exec(
                            crate::backend::wgpu::types::BinaryOpCode::Add,
                            &prev,
                            &partial,
                        )?;
                        drop(prev);
                        drop(partial);
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
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer_checked_with_usage(
                emission,
                len,
                "runmat-matmul-out",
                out_usage,
            )
        })??;
        if use_vec4 {
            self.with_emission(|emission, pending| {
                self.prepare_matmul_vec4_pipeline(emission, pending)
            })??;
        } else {
            self.with_emission(|emission, pending| {
                self.prepare_matmul_pipeline(emission, pending)
            })??;
        }
        self.access.poll(wgpu::Maintain::Poll);
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
        self.with_emission(|emission, pending| -> Result<()> {
            let params_buffer = self.kernel_resources.uniform_buffer(
                emission,
                UniformBufferKey::MatmulParams,
                std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
                "runmat-matmul-params",
            )?;
            emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let layout = if use_vec4 {
                &self.pipelines.matmul_vec4.get(emission)?.layout
            } else {
                &self.pipelines.matmul.get(emission)?.layout
            };
            let pipeline = if use_vec4 {
                &self.pipelines.matmul_vec4.get(emission)?.pipeline
            } else {
                &self.pipelines.matmul.get(emission)?.pipeline
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
            let bg =
                if out_usage == BufferUsageClass::MatmulOut {
                    Arc::new(
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-matmul-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                } else {
                    self.bind_group_cache.get_or_create(
                        emission,
                        layout,
                        &bind_entries,
                        |emission| {
                            Arc::new(emission.device().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-matmul-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        },
                    )?
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
                emission,
                pending,
                pipeline,
                bg.as_ref(),
                groups_x,
                groups_y,
            )?;
            Ok(())
        })??;
        let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer;
        let handle =
            self.register_existing_buffer_with_usage(out_buffer, out_shape, len, out_usage)?;
        if debug_matmul {
            log::debug!("[matmul_debug] out_ptr={:p} len={}", out_ptr, len);
        }
        self.remember_matmul_sources(&handle, a, b);
        self.telemetry.record_matmul_duration(start.elapsed());
        self.record_matmul_kernel_launch(m, n, k, use_vec4, false);
        Ok(handle)
    }
}
