    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
    }
    fn image_normalize<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        desc: &'a runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
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
                    let cold_cap =
                        stream_hot_cap.min((Self::IMAGE_NORMALIZE_STREAM_COLD_CAP).max(1));
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
        })
    }
    fn matmul_power_step<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        epilogue: &'a runmat_accelerate_api::PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
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
            let norms =
                self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
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
                        let mut encoder = self.device_ref().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-power-step-copy"),
                            },
                        );
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
        })
    }
    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
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
        })
    }
    fn corrcoef<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: &'a CorrcoefOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.corrcoef_exec(matrix, options).await })
    }
    fn linsolve<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        options: &'a ProviderLinsolveOptions,
    ) -> AccelProviderFuture<'a, ProviderLinsolveResult> {
        Box::pin(async move {
            if let Some(result) = self.try_linsolve_device(lhs, rhs, options).await? {
                return Ok(result);
            }
            let start = Instant::now();
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;

            let (solution, rcond) =
                linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
                    .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_linsolve_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("linsolve:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &solution.data,
                shape: &solution.shape,
            })?;

            Ok(ProviderLinsolveResult {
                solution: handle,
                reciprocal_condition: rcond,
            })
        })
    }
    fn inv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("inv: {e}"))?;
            let result = inv_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
            self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })
        })
    }

    fn pinv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("pinv: {e}"))?;
            let result = pinv_host_real_for_provider(&tensor, options.tolerance)
                .map_err(|e| anyhow!("{e}"))?;
            self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })
        })
    }

    fn cond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        norm: ProviderCondNorm,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("cond: {e}"))?;
            let cond_value =
                cond_host_real_for_provider(&tensor, norm).map_err(|e| anyhow!("{e}"))?;
            let scalar = [cond_value];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn norm<'a>(
        &'a self,
        tensor: &'a GpuTensorHandle,
        order: ProviderNormOrder,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, tensor).await?;
            let host_tensor = Tensor::new(data, shape).map_err(|e| anyhow!("norm: {e}"))?;
            let value =
                norm_host_real_for_provider(&host_tensor, order).map_err(|e| anyhow!("{e}"))?;
            let scalar = [value];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn rank<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        tolerance: Option<f64>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rank: {e}"))?;
            let rank =
                rank_host_real_for_provider(&tensor, tolerance).map_err(|e| anyhow!("{e}"))? as f64;
            let scalar = [rank];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn rcond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rcond: {e}"))?;
            let estimate = rcond_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
            let scalar = [estimate];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn mldivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let start = Instant::now();
            if let Some(result) = self
                .try_linsolve_device(lhs, rhs, &ProviderLinsolveOptions::default())
                .await?
            {
                self.telemetry.record_mldivide_duration(start.elapsed());
                return Ok(result.solution);
            }
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;

            let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_mldivide_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("mldivide:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })?;
            Ok(handle)
        })
    }

    fn mrdivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let start = Instant::now();
            if let Some(result) = self.try_mrdivide_device(lhs, rhs).await? {
                self.telemetry.record_mrdivide_duration(start.elapsed());
                return Ok(result);
            }
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;

            let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_mrdivide_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("mrdivide:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })?;
            Ok(handle)
        })
    }

    fn dot<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        dim: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.dot_exec(lhs, rhs, dim) })
    }
    fn eig<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        compute_left: bool,
    ) -> AccelProviderFuture<'a, ProviderEigResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("eig: {e}"))?;
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
        })
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
        })
    }
    fn reduce_nnz_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(
                a,
                dim,
                crate::backend::wgpu::types::DimReduceOp::CountNonZero,
            )
        })
    }
    fn reduce_prod_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Prod)
        })
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Mean)
        })
    }
    fn reduce_any_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AnyOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AnyInclude
            };
            self.reduce_dim_sum_mean_exec(a, dim, op)
        })
    }
    fn reduce_any<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
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
        })
    }

    fn reduce_all_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AllOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AllInclude
            };
            self.reduce_dim_sum_mean_exec(a, dim, op)
        })
    }

    fn reduce_all<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
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
        })
    }

    fn reduce_median<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let median = median_from_slice(&host.data);
            let data = [median];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
        })
    }

    fn reduce_median_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
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
        })
    }

    fn reduce_sum<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)
        })
    }

    fn reduce_nnz<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::CountNonZero)
        })
    }

    fn reduce_prod<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Prod)
        })
    }

    fn reduce_mean<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            // Mean over all elements: compute via single-pass sum then divide by len
            let sum_handle =
                self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
            let total_elems: usize = self.get_entry(a)?.len.max(1);
            let scalar = 1.0 / (total_elems as f64);
            let out = self.scalar_op_exec(
                crate::backend::wgpu::types::ScalarOpCode::Mul,
                &sum_handle,
                scalar,
            )?;
            // Free temporary sum buffer
            let _ = self.free(&sum_handle);
            Ok(out)
        })
    }
    fn reduce_std<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_std_exec(a, normalization, nan_mode) })
    }

    fn reduce_std_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_std_dim_exec(a, dim, normalization, nan_mode) })
    }

    fn reduce_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Min)
        })
    }

    fn reduce_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Max)
        })
    }

    fn reduce_min_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Min)
        })
    }

    fn reduce_max_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Max)
        })
    }

