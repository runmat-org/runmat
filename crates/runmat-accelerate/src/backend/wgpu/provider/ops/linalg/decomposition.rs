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
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "dot: only 2D tensors are currently supported by the WGPU provider",
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
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                format!(
                    "dot: unsupported dimension {} for WGPU provider",
                    target_dim
                ),
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
                drop(product);
                Ok(handle)
            }
            Err(err) => {
                drop(product);
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

        let compute_result: Result<GpuTensorHandle> = (|| {
            let a1 = self.gather_linear_exec(lhs, &comp1, &reduced_shape)?;
            let a2 = self.gather_linear_exec(lhs, &comp2, &reduced_shape)?;
            let a3 = self.gather_linear_exec(lhs, &comp3, &reduced_shape)?;
            let b1 = self.gather_linear_exec(rhs, &comp1, &reduced_shape)?;
            let b2 = self.gather_linear_exec(rhs, &comp2, &reduced_shape)?;
            let b3 = self.gather_linear_exec(rhs, &comp3, &reduced_shape)?;

            let a2b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b3)?;
            let a3b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b2)?;
            let c1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a2b3, &a3b2)?;

            let a3b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b1)?;
            let a1b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b3)?;
            let c2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a3b1, &a1b3)?;

            let a1b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b2)?;
            let a2b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b1)?;
            let c3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a1b2, &a2b1)?;

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
                    drop(out);
                    Err(err)
                }
            }
        })();

        compute_result
    }

    pub(crate) fn qr_factor_device(
        &self,
        matrix: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        label: &str,
        retain_r_inv: bool,
    ) -> Result<(GpuTensorHandle, GpuTensorHandle, Option<GpuTensorHandle>)> {
        // This private Gram/Cholesky shader has a binary32 storage ABI. Public
        // Double factorization uses the native-class host path; it must never
        // reach this kernel with binary64 bytes or publish mislabeled results.
        if self.precision != NumericPrecision::F32
            || matrix.physical().element_type() != runmat_accelerate_api::NumericElementType::F32
            || matrix.physical().storage() != GpuTensorStorage::Real
            || matrix.view().is_logical()
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "device Gram QR requires an owning binary32 real route",
            ));
        }
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
        let len_out = cols * cols;
        let (r_buffer, r_inv_buffer) = self.with_emission(|emission, pending| -> Result<_> {
            let gram_scratch = self.kernel_resources.scratch_storage_buffer(
                emission,
                crate::backend::wgpu::resources::ScratchBufferKind::QrGram,
                gram_bytes,
                "runmat-qr-gram-scratch",
            )?;
            if gram_bytes > 0 {
                let gram_copy_label = format!("{label}-gram-copy");
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some(gram_copy_label.as_str()),
                    },
                );
                encoder.copy_buffer_to_buffer(
                    gram_entry.buffer.as_ref(),
                    0,
                    gram_scratch.as_ref(),
                    0,
                    gram_bytes,
                );
                self.submit(emission, pending, encoder)?;
            }

            // These buffers escape as owning results; reusable scratch cannot back
            // them while an earlier result or exported view remains live.
            let r_buffer = self.create_storage_buffer_checked(emission, len_out, "runmat-qr-r")?;
            let r_inv_buffer =
                self.create_storage_buffer_checked(emission, len_out, "runmat-qr-rinv")?;

            let params = QrPowerIterParams {
                cols: cols as u32,
                stride: cols as u32,
                _pad0: [0, 0],
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                emission,
                UniformBufferKey::QrPowerIterParams,
                std::mem::size_of::<QrPowerIterParams>() as u64,
                "runmat-qr-power-params",
            )?;
            emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));

            let layout = &self.pipelines.qr_power_iter.get(emission)?.layout;
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
            let bind_group = self.bind_group_cache.get_or_create(
                emission,
                layout,
                &bind_entries,
                |emission| {
                    Arc::new(
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-qr-power-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                },
            )?;
            crate::backend::wgpu::dispatch::qr_power_iter::run(
                emission,
                pending,
                &self.pipelines.qr_power_iter.get(emission)?.pipeline,
                bind_group.as_ref(),
            )?;
            Ok((r_buffer, r_inv_buffer))
        })??;

        drop(gram_handle);

        let r_shape = vec![cols, cols];
        let r_handle = self.register_existing_buffer_with_usage(
            r_buffer.clone(),
            r_shape.clone(),
            len_out,
            BufferUsageClass::FusionOut,
        )?;
        self.mark_buffer_usage(&r_handle, BufferUsageClass::FusionOut);

        let r_inv_handle = self.register_existing_buffer_with_usage(
            r_inv_buffer.clone(),
            r_shape,
            len_out,
            BufferUsageClass::FusionOut,
        )?;
        self.mark_buffer_usage(&r_inv_handle, BufferUsageClass::FusionOut);

        let q_result =
            self.matmul_exec_with_usage(matrix, &r_inv_handle, BufferUsageClass::FusionOut)?;

        let r_inv_result = if retain_r_inv {
            Some(r_inv_handle)
        } else {
            drop(r_inv_handle);
            None
        };

        Ok((q_result, r_handle, r_inv_result))
    }

    pub(super) async fn qr_power_iter_host(
        &self,
        product: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<ProviderQrPowerIterResult>> {
        let tensor = self.download_real_floating_tensor_exec(product).await?;
        Ok(Some(self.qr_host_result(tensor, options).await?))
    }

    pub(super) async fn qr_host_result(
        &self,
        tensor: Tensor,
        options: &ProviderQrOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderQrResult> {
        let eval = runmat_runtime::builtins::math::linalg::factor::qr::evaluate_provider(
            Value::Tensor(tensor),
            *options,
        )
        .await
        .map_err(runtime_flow_to_provider_error)?;
        let upload = |value| -> Result<GpuTensorHandle> {
            let tensor = host_tensor_from_value("qr", value)?;
            self.upload_real_floating_tensor_exec(&tensor)
        };
        match options.outputs {
            ProviderQrOutputs::R => Ok(ProviderQrResult::R {
                r: upload(eval.r())?,
            }),
            ProviderQrOutputs::Qr => Ok(ProviderQrResult::Qr {
                q: upload(eval.q())?,
                r: upload(eval.r())?,
            }),
            ProviderQrOutputs::Qrp(_) => Ok(ProviderQrResult::Qrp {
                q: upload(eval.q())?,
                r: upload(eval.r())?,
                permutation: upload(eval.permutation())?,
            }),
            ProviderQrOutputs::FullEnvelope => Ok(ProviderQrResult::FullEnvelope {
                q: upload(eval.q())?,
                r: upload(eval.r())?,
                perm_matrix: upload(eval.permutation_matrix())?,
                perm_vector: upload(eval.permutation_vector())?,
            }),
        }
    }

    pub(crate) async fn lu_exec(&self, handle: &GpuTensorHandle) -> Result<ProviderLuResult> {
        let source_precision = self.real_floating_input_precision_exec(handle)?;
        let host = self.download_real_floating_tensor_exec(handle).await?;
        let data: Cow<'_, [f64]> = if let Some(values) = host.as_f64_slice() {
            Cow::Borrowed(values)
        } else if let Some(values) = host.as_f32_slice() {
            Cow::Owned(values.iter().map(|&value| value as f64).collect())
        } else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "lu: real floating input must be F32 or F64",
            ));
        };
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
        } = lu_factor_host(&data, host.shape())?;
        let combined =
            self.upload_real_floating_values_exec(&combined, &combined_shape, source_precision)?;
        let lower =
            self.upload_real_floating_values_exec(&lower, &lower_shape, source_precision)?;
        let upper =
            self.upload_real_floating_values_exec(&upper, &upper_shape, source_precision)?;
        let perm_matrix =
            self.upload_real_floating_values_exec(&perm_matrix, &perm_shape, source_precision)?;
        let perm_vector =
            self.upload_real_floating_values_exec(&pivot_vector, &pivot_shape, source_precision)?;
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
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderCholResult> {
        let tensor = self.download_real_floating_tensor_exec(handle).await?;
        let mut args = Vec::new();
        if lower {
            args.push(Value::from("lower"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
            Value::Tensor(tensor),
            runmat_runtime::call::FunctionArgumentView::from_slice(&args),
        )
        .await
        .map_err(runtime_flow_to_provider_error)?;
        let factor_tensor = host_tensor_from_value("chol", eval.factor())?;
        let factor = self.upload_real_floating_tensor_exec(&factor_tensor)?;
        Ok(ProviderCholResult {
            factor,
            info: eval.flag_index() as u32,
        })
    }

    pub(crate) async fn qr_exec(
        &self,
        handle: &GpuTensorHandle,
        options: ProviderQrOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderQrResult> {
        let tensor = self.download_real_floating_tensor_exec(handle).await?;
        self.qr_host_result(tensor, &options).await
    }
}
