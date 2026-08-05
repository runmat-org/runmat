use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CovarianceToCorrelationParams {
    n: u32,
    total: u32,
    offset: u32,
    chunk: u32,
}

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

        let combined = if let Some(rhs) = second {
            let left_entry = self.get_entry(matrix)?;
            let right_entry = self.get_entry(rhs)?;

            let left_is_vector = match left_entry.shape.len() {
                0 => true,
                1 => true,
                2 => left_entry.shape[0] == 1 || left_entry.shape[1] == 1,
                _ => {
                    return Err(anyhow!(
                        "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                        left_entry.shape
                    ))
                }
            };
            let right_is_vector = match right_entry.shape.len() {
                0 => true,
                1 => true,
                2 => right_entry.shape[0] == 1 || right_entry.shape[1] == 1,
                _ => {
                    return Err(anyhow!(
                        "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                        right_entry.shape
                    ))
                }
            };

            let compatible = if left_is_vector && right_is_vector {
                left_entry.len == right_entry.len
            } else {
                left_entry.shape == right_entry.shape
            };
            ensure!(
                compatible,
                "covariance: paired inputs must have the same size"
            );

            let left_column = self.reshape(matrix, &[left_entry.len, 1])?;
            let right_column = self.reshape(rhs, &[right_entry.len, 1])?;
            let cat_inputs = vec![left_column.clone(), right_column.clone()];
            let result = self.cat_exec(2, &cat_inputs);
            let _ = self.free_exec(&left_column);
            let _ = self.free_exec(&right_column);
            Some(result?)
        } else {
            None
        };

        let normalized_single = if combined.is_none() {
            let entry = self.get_entry(matrix)?;
            match entry.shape.as_slice() {
                [1, _] => Some(self.reshape(matrix, &[entry.len, 1])?),
                _ => None,
            }
        } else {
            None
        };
        let result = {
            let source = combined
                .as_ref()
                .or(normalized_single.as_ref())
                .unwrap_or(matrix);
            self.covariance_exec(source, weights, options).await
        };

        if let Some(handle) = combined {
            let _ = self.free_exec(&handle);
        }
        if let Some(handle) = normalized_single {
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
    fn covariance_weight_column_view(
        &self,
        weights: &GpuTensorHandle,
        rows: usize,
    ) -> Result<(GpuTensorHandle, bool)> {
        let entry = self.get_entry(weights)?;
        ensure!(
            entry.storage != GpuTensorStorage::ComplexInterleaved,
            "covariance: complex weight vectors are not supported"
        );
        ensure!(
            entry.len == rows,
            "covariance: weight vector length must match input rows"
        );
        let (weight_rows, weight_cols) = match entry.shape.len() {
            0 => (1usize, 1usize),
            1 => (entry.shape[0], 1usize),
            2 => (entry.shape[0], entry.shape[1]),
            _ => {
                return Err(anyhow!(
                    "covariance: weight vector must be one-dimensional (got shape {:?})",
                    entry.shape
                ))
            }
        };
        ensure!(
            weight_rows == 1 || weight_cols == 1,
            "covariance: weight vector must be one-dimensional"
        );
        ensure!(
            weight_rows == rows || weight_cols == rows,
            "covariance: weight vector length must match input rows"
        );
        if entry.shape.as_slice() == [rows, 1] {
            return Ok((weights.clone(), false));
        }
        let handle = self.register_existing_buffer_with_storage(
            entry.buffer.clone(),
            vec![rows, 1],
            entry.len,
            entry.storage,
        );
        Ok((handle, true))
    }

    async fn validate_covariance_weights(&self, weights: &GpuTensorHandle) -> Result<f64> {
        let finite_mask = self.logical_isfinite_exec(weights)?;
        let all_finite = match self.reduce_all_exec(&finite_mask, false) {
            Ok(handle) => handle,
            Err(err) => {
                let _ = self.free_exec(&finite_mask);
                return Err(err);
            }
        };
        let all_finite_value = self.read_scalar_exec(&all_finite, 0);
        let _ = self.free_exec(&finite_mask);
        let _ = self.free_exec(&all_finite);
        ensure!(
            all_finite_value? != 0.0,
            "covariance: weights must be non-negative finite values"
        );

        let min_weight =
            self.reduce_global_exec(weights, crate::backend::wgpu::types::GlobalReduceOp::Min)?;
        let min_value = self.read_scalar_exec(&min_weight, 0);
        let _ = self.free_exec(&min_weight);
        ensure!(
            min_value? >= 0.0,
            "covariance: weights must be non-negative finite values"
        );

        let sum_weight =
            self.reduce_global_exec(weights, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let sum_value = self.read_scalar_exec(&sum_weight, 0);
        let _ = self.free_exec(&sum_weight);
        sum_value
    }

    async fn weighted_covariance_exec(
        &self,
        matrix: &GpuTensorHandle,
        matrix_entry: &BufferEntry,
        weights: &GpuTensorHandle,
        rows: usize,
        cols: usize,
    ) -> Result<GpuTensorHandle> {
        let (weights_column, weights_alias) = self.covariance_weight_column_view(weights, rows)?;
        let sum_w = match self.validate_covariance_weights(&weights_column).await {
            Ok(value) => value,
            Err(err) => {
                if weights_alias {
                    let _ = self.free_exec(&weights_column);
                }
                return Err(err);
            }
        };
        let denom = sum_w - 1.0;
        if sum_w <= 0.0 || denom <= 0.0 {
            if weights_alias {
                let _ = self.free_exec(&weights_column);
            }
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let weights_entry = self.get_entry(&weights_column)?;
        let mut weights_used = weights_column.clone();
        let mut casted_weights = false;
        if weights_entry.precision != matrix_entry.precision {
            weights_used = match self
                .cast_tensor_precision(&weights_column, matrix_entry.precision)
                .await
            {
                Ok(handle) => handle,
                Err(err) => {
                    if weights_alias {
                        let _ = self.free_exec(&weights_column);
                    }
                    return Err(err);
                }
            };
            casted_weights = true;
        }

        let weighted = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            matrix,
            &weights_used,
        )?;
        let weighted_sum = match self.reduce_dim_sum_mean_exec(
            &weighted,
            0,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        ) {
            Ok(handle) => handle,
            Err(err) => {
                let _ = self.free_exec(&weighted);
                if casted_weights {
                    let _ = self.free_exec(&weights_used);
                }
                if weights_alias {
                    let _ = self.free_exec(&weights_column);
                }
                return Err(err);
            }
        };
        let weighted_means = self.scalar_mul(&weighted_sum, 1.0 / sum_w)?;
        let ones = self.fill_exec(&[rows, 1], 1.0)?;
        let means_full = self.matmul_exec(&ones, &weighted_means)?;
        let centered = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            matrix,
            &means_full,
        )?;
        let weighted_centered = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &centered,
            &weights_used,
        )?;
        let centered_t = self.transpose_exec(&centered)?;
        let covariance = self.matmul_exec(&centered_t, &weighted_centered)?;
        let result = self.scalar_mul(&covariance, 1.0 / denom);

        let _ = self.free_exec(&weighted);
        let _ = self.free_exec(&weighted_sum);
        let _ = self.free_exec(&weighted_means);
        let _ = self.free_exec(&ones);
        let _ = self.free_exec(&means_full);
        let _ = self.free_exec(&centered);
        let _ = self.free_exec(&weighted_centered);
        let _ = self.free_exec(&centered_t);
        let _ = self.free_exec(&covariance);
        if casted_weights {
            let _ = self.free_exec(&weights_used);
        }
        if weights_alias {
            let _ = self.free_exec(&weights_column);
        }

        result
    }

    pub(crate) async fn covariance_exec(
        &self,
        matrix: &GpuTensorHandle,
        weights: Option<&GpuTensorHandle>,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CovRows::All {
            return Err(anyhow!(
                "covariance: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }
        if options.has_weight_vector && weights.is_none() {
            return Err(anyhow!("covariance: weight vector handle is required"));
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

        if entry.len == 0 && cols == 0 {
            return self.fill_exec(&[1, 1], f64::NAN);
        }

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cov-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        if let Some(weights) = weights {
            return self
                .weighted_covariance_exec(matrix, &entry, weights, rows, cols)
                .await;
        }

        let denom = match options.normalization {
            CovNormalization::Unbiased => ((rows as f64) - 1.0).max(1.0),
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
        let entry = self.get_entry(matrix)?;
        let normalized = match entry.shape.as_slice() {
            [1, _] => Some(self.register_existing_buffer_with_storage(
                entry.buffer.clone(),
                vec![entry.len, 1],
                entry.len,
                entry.storage,
            )),
            _ => None,
        };
        let source = normalized.as_ref().unwrap_or(matrix);
        let result = self.corrcoef_matrix_exec(source, options).await;
        if let Some(handle) = normalized {
            let _ = self.free_exec(&handle);
        }
        result
    }

    async fn corrcoef_matrix_exec(
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

        // Clamp negatives to zero on-device before sqrt:
        // max(x, 0) = 0.5 * (x + |x|)
        let abs_variance =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, &variance)?;
        let variance_plus_abs = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
        )?;
        let _ = self.free_exec(&abs_variance);
        let half_tensor = self.fill_exec(&self.get_entry(&variance_plus_abs)?.shape, 0.5)?;
        let variance_adjusted = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
        )?;
        let _ = self.free_exec(&half_tensor);
        let _ = self.free_exec(&variance_plus_abs);
        let _ = self.free_exec(&variance);

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

    pub(crate) fn covariance_to_correlation_exec(
        &self,
        matrix: &GpuTensorHandle,
    ) -> Result<ProviderCovarianceToCorrelationResult> {
        let entry = self.get_entry(matrix)?;
        ensure!(
            entry.storage != GpuTensorStorage::ComplexInterleaved,
            "covariance_to_correlation: complex covariance matrices are not supported"
        );
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "covariance_to_correlation: covariance matrix must be two-dimensional"
                ))
            }
        };
        ensure!(
            rows == cols,
            "covariance_to_correlation: covariance matrix must be square"
        );
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("covariance_to_correlation: matrix size overflow"))?;
        ensure!(
            total == entry.len,
            "covariance_to_correlation: shape does not match buffer length"
        );
        ensure!(
            total <= u32::MAX as usize && rows <= u32::MAX as usize,
            "covariance_to_correlation: matrix exceeds GPU dispatch limits"
        );

        let correlation_buffer =
            self.create_storage_buffer_checked(total, "runmat-cov2corr-correlation-out")?;
        let sigma_buffer = self.create_storage_buffer_checked(rows, "runmat-cov2corr-sigma-out")?;

        if total > 0 {
            let validation_buffer =
                self.create_storage_buffer_checked(total, "runmat-cov2corr-validation")?;
            let error_buffer =
                self.device_ref()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-cov2corr-error"),
                        contents: bytes_of(&0u32),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });
            let shader = covariance_to_correlation_shader(self.precision);
            let shader_module =
                self.device_ref()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("runmat-cov2corr-shader"),
                        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
                    });
            let bind_layout =
                self.device_ref()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-cov2corr-layout"),
                        entries: &covariance_to_correlation_bind_layout_entries(),
                    });
            let pipeline_layout =
                self.device_ref()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("runmat-cov2corr-pipeline-layout"),
                        bind_group_layouts: &[&bind_layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = self.device_ref().create_compute_pipeline(
                &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                    label: Some("runmat-cov2corr-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "main",
                },
            );
            let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
                * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
            let mut offset = 0usize;
            while offset < total {
                let chunk_len = (total - offset).min(chunk_capacity);
                let params = CovarianceToCorrelationParams {
                    n: rows as u32,
                    total: total as u32,
                    offset: offset as u32,
                    chunk: chunk_len as u32,
                };
                let params_buffer =
                    self.device_ref()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("runmat-cov2corr-params"),
                            contents: bytes_of(&params),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        });
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-cov2corr-bind"),
                        layout: &bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: correlation_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: sigma_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: error_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: validation_buffer.as_ref().as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                crate::backend::wgpu::dispatch::creation::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-cov2corr-encoder",
                    "runmat-cov2corr-pass",
                );
                offset += chunk_len;
            }

            let error_size = std::mem::size_of::<u32>() as u64;
            let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-cov2corr-error-staging"),
                size: error_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-cov2corr-error-copy"),
                    });
            encoder.copy_buffer_to_buffer(&error_buffer, 0, &staging, 0, error_size);
            self.submit(encoder);
            let bytes = self.map_readback_bytes_sync(staging, error_size, "cov2corr")?;
            let code = u32::from_le_bytes(
                bytes
                    .get(..4)
                    .ok_or_else(|| anyhow!("covariance_to_correlation: short error readback"))?
                    .try_into()
                    .map_err(|_| anyhow!("covariance_to_correlation: invalid error readback"))?,
            );
            match code {
                0 => {}
                1 => {
                    return Err(anyhow!(
                        "covariance_to_correlation: covariance matrix must contain finite values"
                    ))
                }
                2 => {
                    return Err(anyhow!(
                        "covariance_to_correlation: covariance matrix diagonal entries must be nonnegative"
                    ))
                }
                3 => {
                    return Err(anyhow!(
                        "covariance_to_correlation: covariance matrix must be symmetric"
                    ))
                }
                4 => {
                    return Err(anyhow!(
                        "covariance_to_correlation: covariance magnitude exceeds variance bounds"
                    ))
                }
                5 => {
                    return Err(anyhow!(
                        "covariance_to_correlation: covariance matrix must be positive semidefinite"
                    ))
                }
                other => {
                    return Err(anyhow!(
                        "covariance_to_correlation: validation failed with code {other}"
                    ))
                }
            }
        }

        Ok(ProviderCovarianceToCorrelationResult {
            correlation: self.register_existing_buffer(correlation_buffer, vec![rows, cols], total),
            sigma: self.register_existing_buffer(sigma_buffer, vec![rows, 1], rows),
        })
    }
}

fn covariance_to_correlation_bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 6] {
    std::array::from_fn(|binding| {
        let read_only = binding == 0;
        wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 4 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        }
    })
}

fn covariance_to_correlation_shader(precision: NumericPrecision) -> String {
    let ty = precision.as_str();
    let tol = match precision {
        NumericPrecision::F64 => "1.0e-10",
        NumericPrecision::F32 => "1.0e-5",
    };
    let max_finite = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let workgroup = crate::backend::wgpu::config::WORKGROUP_SIZE;
    format!(
        r#"
const MAX_FINITE_COV2CORR: {ty} = {ty}({max_finite});

struct Tensor {{
  data: array<{ty}>,
}};

struct ErrorState {{
  code: atomic<u32>,
}};

struct Params {{
  n: u32,
  total: u32,
  offset: u32,
  chunk: u32,
}};

@group(0) @binding(0) var<storage, read> covariance: Tensor;
@group(0) @binding(1) var<storage, read_write> correlation: Tensor;
@group(0) @binding(2) var<storage, read_write> sigma: Tensor;
@group(0) @binding(3) var<storage, read_write> errors: ErrorState;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> validation: Tensor;

fn is_nan_cov2corr(x: {ty}) -> bool {{
  return x != x;
}}

fn is_finite_cov2corr(x: {ty}) -> bool {{
  return (x == x) && (abs(x) < MAX_FINITE_COV2CORR);
}}

fn nan_cov2corr() -> {ty} {{
  let zero = {ty}(0.0);
  return zero / zero;
}}

fn flag_error(code: u32) {{
  atomicMax(&errors.code, code);
}}

@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= params.chunk) {{
    return;
  }}
  let idx = gid.x + params.offset;
  if (idx >= params.total) {{
    return;
  }}

  let n = params.n;
  let row = idx % n;
  let col = idx / n;
  let value = covariance.data[idx];
  let diag_row = covariance.data[row + row * n];
  let diag_col = covariance.data[col + col * n];

  if (idx == 0u) {{
    var finite_matrix = true;
    var scan = 0u;
    loop {{
      if (scan >= params.total) {{
        break;
      }}
      if (!is_finite_cov2corr(covariance.data[scan])) {{
        finite_matrix = false;
      }}
      validation.data[scan] = {ty}(0.0);
      scan = scan + 1u;
    }}
    if (finite_matrix) {{
      var scale = {ty}(1.0);
      var diagonal = 0u;
      loop {{
        if (diagonal >= n) {{
          break;
        }}
        scale = max(scale, abs(covariance.data[diagonal + diagonal * n]));
        diagonal = diagonal + 1u;
      }}
      let psd_tol = {ty}({tol}) * scale;
      let pivot_tol = sqrt(psd_tol);
      var factor_row = 0u;
      var psd = true;
      loop {{
        if (factor_row >= n || !psd) {{
          break;
        }}
        var factor_col = 0u;
        loop {{
          if (factor_col > factor_row || !psd) {{
            break;
          }}
          var residual = covariance.data[factor_row + factor_col * n];
          var k = 0u;
          loop {{
            if (k >= factor_col) {{
              break;
            }}
            residual = residual - validation.data[factor_row * n + k] * validation.data[factor_col * n + k];
            k = k + 1u;
          }}
          if (factor_row == factor_col) {{
            if (residual < -psd_tol) {{
              psd = false;
            }} else {{
              validation.data[factor_row * n + factor_col] = sqrt(max(residual, {ty}(0.0)));
            }}
          }} else {{
            let pivot = validation.data[factor_col * n + factor_col];
            if (pivot > pivot_tol) {{
              validation.data[factor_row * n + factor_col] = residual / pivot;
            }} else if (abs(residual) > psd_tol) {{
              psd = false;
            }}
          }}
          factor_col = factor_col + 1u;
        }}
        factor_row = factor_row + 1u;
      }}
      if (!psd) {{
        flag_error(5u);
      }}
    }}
  }}

  if (!is_finite_cov2corr(value)) {{
    flag_error(1u);
  }}
  if (row == col && value < {ty}(0.0)) {{
    flag_error(2u);
  }}
  if (row < col) {{
    let mirror = covariance.data[col + row * n];
    if ((is_nan_cov2corr(value) && !is_nan_cov2corr(mirror)) || (!is_nan_cov2corr(value) && is_nan_cov2corr(mirror))) {{
      flag_error(3u);
    }}
    if (!is_nan_cov2corr(value) && !is_nan_cov2corr(mirror)) {{
      let symmetry_tol = {ty}({tol}) * max(max(abs(value), abs(mirror)), {ty}(1.0));
      if (abs(value - mirror) > symmetry_tol) {{
        flag_error(3u);
      }}
      if (!is_nan_cov2corr(diag_row) && !is_nan_cov2corr(diag_col)) {{
        let max_covariance = sqrt(diag_row * diag_col);
        let bound_tol = {ty}({tol}) * max(max_covariance, max(abs(value), {ty}(1.0)));
        if (abs(value) > max_covariance + bound_tol) {{
          flag_error(4u);
        }}
      }}
    }}
  }}

  if (idx < n) {{
    sigma.data[idx] = sqrt(covariance.data[idx + idx * n]);
  }}

  let denom = sqrt(diag_row) * sqrt(diag_col);
  if (denom == {ty}(0.0)) {{
    correlation.data[idx] = nan_cov2corr();
  }} else {{
    correlation.data[idx] = value / denom;
  }}
}}
"#,
        ty = ty,
        max_finite = max_finite,
        tol = tol,
        workgroup = workgroup,
    )
}

#[cfg(test)]
mod covariance_conversion_tests {
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};

    #[test]
    fn covariance_to_correlation_wgpu_returns_resident_outputs() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let covariance = provider
            .upload(&HostTensorView {
                data: &[4.0, 2.0, 2.0, 9.0],
                shape: &[2, 2],
            })
            .expect("upload covariance");

        let result = provider
            .covariance_to_correlation(&covariance)
            .expect("covariance to correlation");
        let correlation =
            pollster::block_on(provider.download(&result.correlation)).expect("correlation");
        let sigma = pollster::block_on(provider.download(&result.sigma)).expect("sigma");

        assert_eq!(correlation.shape, vec![2, 2]);
        assert_eq!(sigma.shape, vec![2, 1]);
        let expected = [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0];
        for (actual, expected) in correlation.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        assert!((sigma.data[0] - 2.0).abs() < 1.0e-8);
        assert!((sigma.data[1] - 3.0).abs() < 1.0e-8);

        provider.free(&covariance).ok();
        provider.free(&result.correlation).ok();
        provider.free(&result.sigma).ok();
    }

    #[test]
    fn covariance_to_correlation_wgpu_rejects_invalid_covariance() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let covariance = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.1, 0.2, 1.0],
                shape: &[2, 2],
            })
            .expect("upload covariance");

        let err = provider
            .covariance_to_correlation(&covariance)
            .expect_err("invalid covariance");

        assert!(err.to_string().contains("symmetric"));
        provider.free(&covariance).ok();

        let indefinite = provider
            .upload(&HostTensorView {
                data: &[
                    1.0, 0.9, 0.9, //
                    0.9, 1.0, -0.9, //
                    0.9, -0.9, 1.0,
                ],
                shape: &[3, 3],
            })
            .expect("upload indefinite covariance");
        let err = provider
            .covariance_to_correlation(&indefinite)
            .expect_err("indefinite covariance");
        assert!(err.to_string().contains("positive semidefinite"));
        provider.free(&indefinite).ok();
    }
}
