    fn qr_factor_device(
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

    async fn qr_power_iter_host(
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

    fn try_qr_device(
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

    async fn qr_host_result(
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

}
    fn qr_factor_device(
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

    async fn qr_power_iter_host(
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

    fn try_qr_device(
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

    async fn qr_host_result(
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

}
