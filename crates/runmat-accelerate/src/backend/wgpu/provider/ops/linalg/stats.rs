use super::*;

impl WgpuProvider {
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
            let _ = self.free_exec(&handle);
        }

        result
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

        let _ = self.free_exec(&means_col);
        let _ = self.free_exec(&means_row_scaled);
        let _ = self.free_exec(&outer_scaled);

        let centered =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &xtx, &outer)?;

        let _ = self.free_exec(&xtx);
        let _ = self.free_exec(&outer);
        let _ = self.free_exec(&means_scaled);

        let handle = self.scalar_mul(&centered, 1.0 / denom)?;
        let _ = self.free_exec(&centered);

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
            let _ = self.free_exec(&means_used);
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
        let matrix_host = self.download_exec(matrix).await?;
        let means_gpu = self.download_exec(means).await?;
        let output_gpu = self.download_exec(output).await?;
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

        let product_host = self.download_exec(product).await?;
        let q_gpu_host = self.download_exec(q_result).await?;
        let r_gpu_host = self.download_exec(r_handle).await?;
        let r_inv_gpu_host = self.download_exec(r_inv_handle).await?;
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
            let _ = self.free_exec(&product_t_tmp);
            let owned = self.download_exec(&gram_tmp).await?;
            let _ = self.free_exec(&gram_tmp);
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
        let _ = self.free_exec(&means);
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
        let mut host_variance = self.download_exec(&variance).await?;
        for value in host_variance.data.iter_mut() {
            if *value < 0.0 && *value > -1.0e-12 {
                *value = 0.0;
            }
        }
        let view = HostTensorView {
            data: &host_variance.data,
            shape: &host_variance.shape,
        };
        let variance_adjusted = self.upload_exec(&view)?;
        self.free_exec(&variance)?;

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
        let _ = self.free_exec(&means);
        let _ = self.free_exec(&ones);
        let _ = self.free_exec(&means_full);
        let _ = self.free_exec(&centered);
        let _ = self.free_exec(&centered_t);
        let _ = self.free_exec(&covariance);
        let _ = self.free_exec(&inv_cov);
        let _ = self.free_exec(&covariance_scaled);
        let _ = self.free_exec(&squared);
        let _ = self.free_exec(&variance_sum);
        let _ = self.free_exec(&inv_var);
        let _ = self.free_exec(&variance_adjusted);
        let _ = self.free_exec(&std);
        let _ = self.free_exec(&std_t);
        let _ = self.free_exec(&std_outer);

        Ok(correlation)
    }
}
