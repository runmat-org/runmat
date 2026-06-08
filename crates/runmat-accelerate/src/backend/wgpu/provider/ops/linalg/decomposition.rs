use super::*;

impl WgpuProvider {
    pub(super) async fn cast_tensor_precision(
        &self,
        tensor: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(tensor)?;
        if entry.precision == target {
            return Ok(tensor.clone());
        }

        let mut host = self.download_exec(tensor).await?;
        if matches!(target, NumericPrecision::F32) {
            for value in host.data.iter_mut() {
                *value = (*value as f32) as f64;
            }
        }

        let view = HostTensorView {
            data: host.data.as_slice(),
            shape: host.shape.as_slice(),
        };
        self.upload_exec(&view)
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
                let _ = self.free_exec(&product);
                Ok(handle)
            }
            Err(err) => {
                let _ = self.free_exec(&product);
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
                    let _ = self.free_exec(&out);
                    Err(err)
                }
            }
        })();

        for h in &to_free {
            let _ = self.free_exec(h);
        }

        compute_result
    }

    pub(crate) fn qr_factor_device(
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

        let _ = self.free_exec(&gram_handle);

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
                let _ = self.free_exec(&q_temp);
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
            let _ = self.free_exec(&r_inv_handle);
            None
        };

        Ok((q_result, r_handle, r_inv_result))
    }

    pub(super) async fn qr_power_iter_host(
        &self,
        product: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> Result<Option<ProviderQrPowerIterResult>> {
        let host_product = self.download_exec(product).await?;
        let tensor =
            Tensor::new(host_product.data.clone(), host_product.shape.clone()).map_err(|e| {
                anyhow!("qr_power_iter: failed to construct host tensor for fallback: {e}")
            })?;
        let host_result = self.qr_host_result(tensor, options).await?;
        let _ = self.free_exec(product);
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
        if self.provider_precision_exec() != ProviderPrecision::F32 {
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
        let perm_matrix_handle = self.upload_exec(&HostTensorView {
            data: &perm_matrix,
            shape: &perm_matrix_shape,
        })?;
        let perm_vector_shape = vec![cols, 1];
        let perm_vector_handle = self.upload_exec(&HostTensorView {
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

        let q = self.upload_exec(&HostTensorView {
            data: &q_tensor.data,
            shape: &q_tensor.shape,
        })?;
        let r = self.upload_exec(&HostTensorView {
            data: &r_tensor.data,
            shape: &r_tensor.shape,
        })?;
        let perm_matrix = self.upload_exec(&HostTensorView {
            data: &perm_matrix_tensor.data,
            shape: &perm_matrix_tensor.shape,
        })?;
        let perm_vector = self.upload_exec(&HostTensorView {
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
        let host = self.download_exec(handle).await?;
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
        let combined = self.upload_exec(&HostTensorView {
            data: &combined,
            shape: &combined_shape,
        })?;
        let lower = self.upload_exec(&HostTensorView {
            data: &lower,
            shape: &lower_shape,
        })?;
        let upper = self.upload_exec(&HostTensorView {
            data: &upper,
            shape: &upper_shape,
        })?;
        let perm_matrix = self.upload_exec(&HostTensorView {
            data: &perm_matrix,
            shape: &perm_shape,
        })?;
        let perm_vector = self.upload_exec(&HostTensorView {
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
        let host = self.download_exec(handle).await?;
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
        let factor = self.upload_exec(&HostTensorView {
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
        let host = self.download_exec(handle).await?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("qr: {e}"))?;
        self.qr_host_result(tensor, &options).await
    }
}
