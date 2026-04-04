use super::*;

impl WgpuProvider {
    fn matrix_dims_for_solve(shape: &[usize]) -> Result<(usize, usize)> {
        match shape.len() {
            0 => Ok((1, 1)),
            1 => Ok((shape[0], 1)),
            2 => Ok((shape[0], shape[1])),
            _ => Err(anyhow!("solve: input must be 2-D")),
        }
    }

    fn diagonal_rcond(min_diag: f64, max_diag: f64) -> f64 {
        if max_diag == 0.0 {
            0.0
        } else {
            min_diag / max_diag
        }
    }

    fn singular_value_rcond(singular_values: &[f64]) -> f64 {
        if singular_values.is_empty() {
            return 1.0;
        }
        let mut min_sv = f64::INFINITY;
        let mut max_sv = 0.0_f64;
        for &sv in singular_values {
            let abs = sv.abs();
            if !abs.is_finite() {
                return 0.0;
            }
            min_sv = min_sv.min(abs);
            max_sv = max_sv.max(abs);
        }
        if max_sv == 0.0 {
            0.0
        } else {
            min_sv / max_sv
        }
    }

    fn enforce_device_rcond(&self, options: &ProviderLinsolveOptions, rcond: f64) -> Result<()> {
        if let Some(threshold) = options.rcond {
            if rcond < threshold {
                return Err(anyhow!("linsolve: matrix is singular to working precision."));
            }
        }
        Ok(())
    }

    fn host_tensor_from_handle_sync(&self, label: &str, handle: &GpuTensorHandle) -> Result<Tensor> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let HostTensorOwned { data, shape } = block_on(<Self as AccelProvider>::download(
                self, handle,
            ))?;
            Tensor::new(data, shape).map_err(|e| anyhow!("{label}: {e}"))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = handle;
            Err(anyhow!("{label}: synchronous factor readback is unavailable on wasm"))
        }
    }

    fn triangular_rcond_sync(&self, lhs: &GpuTensorHandle) -> Result<f64> {
        let tensor = self.host_tensor_from_handle_sync("linsolve_triangular_rcond", lhs)?;
        let rows = tensor.rows();
        let cols = tensor.cols();
        ensure!(
            rows == cols,
            "linsolve: triangular rcond estimation requires square input"
        );
        let mut min_diag = f64::INFINITY;
        let mut max_diag = 0.0_f64;
        for i in 0..rows {
            let diag = tensor.data[i + i * rows].abs();
            if diag == 0.0 {
                return Err(anyhow!("linsolve: matrix is singular to working precision."));
            }
            min_diag = min_diag.min(diag);
            max_diag = max_diag.max(diag);
        }
        Ok(Self::diagonal_rcond(min_diag, max_diag))
    }

    fn svd_rcond_sync(&self, label: &str, factor: &GpuTensorHandle) -> Result<f64> {
        let tensor = self.host_tensor_from_handle_sync(label, factor)?;
        #[cfg(not(target_arch = "wasm32"))]
        let eval = block_on(runmat_runtime::builtins::math::linalg::factor::svd::evaluate(
            Value::Tensor(tensor),
            &[],
        ))
        .map_err(|err| runtime_flow_to_anyhow(label, err))?;
        #[cfg(target_arch = "wasm32")]
        let eval = {
            return Err(anyhow!(
                "{label}: synchronous singular-value estimation is unavailable on wasm"
            ));
        };
        let singular_values = host_tensor_from_value(label, eval.singular_values())?;
        Ok(Self::singular_value_rcond(&singular_values.data))
    }

    fn needs_rcond(options: &ProviderLinsolveOptions) -> bool {
        options.need_rcond || options.rcond.is_some()
    }

    fn triangular_solve_bind_group_layout(&self) -> Arc<wgpu::BindGroupLayout> {
        self.cached_bind_group_layout("runmat-triangular-solve-layout", |device| {
            let entries = [
                crate::backend::wgpu::bindings::storage_read_entry(0),
                crate::backend::wgpu::bindings::storage_read_entry(1),
                crate::backend::wgpu::bindings::storage_read_entry(2),
                crate::backend::wgpu::bindings::storage_read_write_entry(3),
                crate::backend::wgpu::bindings::uniform_entry(4),
            ];
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-triangular-solve-bgl"),
                entries: &entries,
            })
        })
    }

    fn triangular_solve_pipeline(
        &self,
        transpose: bool,
        lower: bool,
    ) -> Arc<wgpu::ComputePipeline> {
        let shader = crate::backend::wgpu::shaders::solve::triangular_linsolve_shader(
            self.precision,
            transpose,
            lower,
        );
        let bind_group_layout = self.triangular_solve_bind_group_layout();
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-triangular-solve-pipeline-layout",
            bind_group_layout.as_ref(),
        );
        let layout_tag = "runmat-triangular-solve-layout";
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            layout_tag,
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-triangular-solve-shader",
            &shader,
        );
        self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-triangular-solve-pipeline",
            Some(shader.as_bytes()),
            Some(layout_tag),
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        )
    }

    fn triangular_solve_bind_group(
        &self,
        bind_group_layout: &wgpu::BindGroupLayout,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        previous: &GpuTensorHandle,
        output: &GpuTensorHandle,
        params_buffer: &wgpu::Buffer,
    ) -> Result<Arc<wgpu::BindGroup>> {
        let lhs_entry = self.get_entry(lhs)?;
        let rhs_entry = self.get_entry(rhs)?;
        let prev_entry = self.get_entry(previous)?;
        let out_entry = self.get_entry(output)?;
        let bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: prev_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_entry.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        Ok(self
            .bind_group_cache
            .get_or_create(bind_group_layout, &bind_entries, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-triangular-solve-bind-group"),
                            layout: bind_group_layout,
                            entries: &bind_entries,
                        }),
                )
            }))
    }

    fn run_triangular_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        transpose: bool,
        effective_lower: bool,
    ) -> Result<ProviderLinsolveResult> {
        let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(&lhs.shape)?;
        let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(&rhs.shape)?;
        ensure!(
            lhs_rows == lhs_cols && rhs_rows == lhs_rows,
            "linsolve: triangular device path requires square lhs and matching rhs rows"
        );
        if lhs_rows == 0 || rhs_cols == 0 {
            return Ok(ProviderLinsolveResult {
                solution: self.zeros(&rhs.shape)?,
                reciprocal_condition: f64::NAN,
            });
        }

        let len = rhs_rows
            .checked_mul(rhs_cols)
            .ok_or_else(|| anyhow!("linsolve: rhs dimensions overflow"))?;
        let mut current = self.zeros(&rhs.shape)?;
        let mut scratch = self.zeros(&rhs.shape)?;
        let bind_group_layout = self.triangular_solve_bind_group_layout();
        let pipeline = self.triangular_solve_pipeline(transpose, effective_lower);
        crate::backend::wgpu::dispatch::solve::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        let mut params = crate::backend::wgpu::params::TriangularSolveParams {
            len: len as u32,
            offset: 0,
            total: len as u32,
            rows: lhs_rows as u32,
            rhs_cols: rhs_cols as u32,
            target_row: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-triangular-solve-params");
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        for step in 0..lhs_rows {
            params.target_row = if effective_lower {
                step
            } else {
                lhs_rows - 1 - step
            } as u32;
            self.queue_ref()
                .write_buffer(&params_buffer, 0, bytes_of(&params));
            let bind_group = self.triangular_solve_bind_group(
                bind_group_layout.as_ref(),
                lhs,
                rhs,
                &current,
                &scratch,
                &params_buffer,
            )?;
            crate::backend::wgpu::dispatch::solve::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                bind_group.as_ref(),
                workgroups,
            );
            std::mem::swap(&mut current, &mut scratch);
        }
        let _ = self.free(&scratch);
        Ok(ProviderLinsolveResult {
            solution: current,
            reciprocal_condition: f64::NAN,
        })
    }

    fn chol_factor_spd_device(
        &self,
        matrix: &GpuTensorHandle,
        rows: usize,
        label: &str,
    ) -> Result<(GpuTensorHandle, GpuTensorHandle)> {
        let matrix_entry = self.get_entry(matrix)?;
        let len_out = rows
            .checked_mul(rows)
            .ok_or_else(|| anyhow!("linsolve: SPD matrix dimensions overflow"))?;
        ensure!(
            matrix_entry.len == len_out,
            "linsolve: SPD factor expects square matrix buffer (expected {}, got {})",
            len_out,
            matrix_entry.len
        );

        let bytes = (len_out as u64) * (self.element_size as u64);
        let gram_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrGram,
            bytes,
            "runmat-chol-gram-scratch",
        );
        if bytes > 0 {
            let copy_label = format!("{label}-gram-copy");
            let mut encoder =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(copy_label.as_str()),
                    });
            encoder.copy_buffer_to_buffer(
                matrix_entry.buffer.as_ref(),
                0,
                gram_buffer.as_ref(),
                0,
                bytes,
            );
            self.submit(encoder);
        }

        let r_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrR,
            bytes,
            "runmat-chol-r-scratch",
        );
        let r_inv_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrRInv,
            bytes,
            "runmat-chol-rinv-scratch",
        );

        let params = QrPowerIterParams {
            cols: rows as u32,
            stride: rows as u32,
            _pad0: [0, 0],
        };
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::QrPowerIterParams,
            std::mem::size_of::<QrPowerIterParams>() as u64,
            "runmat-chol-params",
        );
        self.queue_ref()
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));

        let layout = &self.pipelines.qr_power_iter.layout;
        let bind_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gram_buffer.as_ref().as_entire_binding(),
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
        let bind_group = self.bind_group_cache.get_or_create(layout, &bind_entries, || {
            Arc::new(
                self.device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-chol-bind"),
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

        let shape = vec![rows, rows];
        let r_handle = self.register_existing_buffer_with_usage(
            r_buffer.clone(),
            shape.clone(),
            len_out,
            BufferUsageClass::FusionOut,
        );
        self.mark_buffer_usage(&r_handle, BufferUsageClass::FusionOut);
        let r_inv_handle = self.register_existing_buffer_with_usage(
            r_inv_buffer.clone(),
            shape,
            len_out,
            BufferUsageClass::FusionOut,
        );
        self.mark_buffer_usage(&r_inv_handle, BufferUsageClass::FusionOut);
        Ok((r_handle, r_inv_handle))
    }

    fn try_posdef_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<Option<ProviderLinsolveResult>> {
        if !options.posdef
            || options.lower
            || options.upper
            || options.rectangular
        {
            return Ok(None);
        }
        if self.precision() != ProviderPrecision::F32 {
            return Ok(None);
        }

        let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(&lhs.shape)?;
        let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(&rhs.shape)?;
        if lhs_rows != lhs_cols || rhs_rows != lhs_rows {
            return Ok(None);
        }
        if lhs_rows == 0 || rhs_cols == 0 {
            return Ok(Some(ProviderLinsolveResult {
                solution: self.zeros(&[lhs_cols, rhs_cols])?,
                reciprocal_condition: f64::NAN,
            }));
        }

        let start = Instant::now();
        let (r_handle, r_inv_handle) =
            self.chol_factor_spd_device(lhs, lhs_rows, "runmat-linsolve-posdef")?;
        let rcond = if Self::needs_rcond(options) {
            let factor_rcond = match self.svd_rcond_sync("linsolve_posdef_rcond", &r_handle) {
                Ok(value) => value,
                Err(err) => {
                    let _ = self.free(&r_handle);
                    let _ = self.free(&r_inv_handle);
                    return Err(err);
                }
            };
            let rcond = factor_rcond * factor_rcond;
            if let Err(err) = self.enforce_device_rcond(options, rcond) {
                let _ = self.free(&r_handle);
                let _ = self.free(&r_inv_handle);
                return Err(err);
            }
            rcond
        } else {
            f64::NAN
        };
        let projected_rhs = match self.run_triangular_linsolve_device(&r_handle, rhs, true, true) {
            Ok(value) => value,
            Err(err) => {
                let _ = self.free(&r_handle);
                let _ = self.free(&r_inv_handle);
                return Err(err);
            }
        };
        let mut solution = match self.run_triangular_linsolve_device(
            &r_handle,
            &projected_rhs.solution,
            false,
            false,
        ) {
            Ok(value) => value,
            Err(err) => {
                let _ = self.free(&projected_rhs.solution);
                let _ = self.free(&r_handle);
                let _ = self.free(&r_inv_handle);
                return Err(err);
            }
        };
        let _ = self.free(&projected_rhs.solution);
        let _ = self.free(&r_handle);
        let _ = self.free(&r_inv_handle);
        solution.reciprocal_condition = rcond;

        self.telemetry.record_linsolve_duration(start.elapsed());
        let shape = [
            ("rows", lhs_rows as u64),
            ("rhs_cols", rhs_cols as u64),
            ("transpose", if options.transposed { 1 } else { 0 }),
        ];
        let tuning = [("method", 4), ("symmetric", if options.symmetric { 1 } else { 0 })];
        self.record_kernel_launch_basic("linsolve_posdef_chol", &shape, &tuning);
        Ok(Some(solution))
    }

    fn try_qr_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<Option<ProviderLinsolveResult>> {
        if options.lower
            || options.upper
        {
            return Ok(None);
        }
        if self.precision() != ProviderPrecision::F32 {
            return Ok(None);
        }

        let transposed_lhs = if options.transposed {
            Some(self.permute_exec(lhs, &[1, 0])?)
        } else {
            None
        };
        let solve_lhs = transposed_lhs.as_ref().unwrap_or(lhs);
        let result = (|| -> Result<Option<ProviderLinsolveResult>> {
            let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(&solve_lhs.shape)?;
            let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(&rhs.shape)?;
            if rhs_rows != lhs_rows {
                return Ok(None);
            }
            if lhs_rows == 0 || lhs_cols == 0 || rhs_cols == 0 {
                return Ok(Some(ProviderLinsolveResult {
                    solution: self.zeros(&[lhs_cols, rhs_cols])?,
                    reciprocal_condition: f64::NAN,
                }));
            }
            if lhs_rows >= lhs_cols {
                if lhs_cols > QR_DEVICE_MAX_COLS {
                    return Ok(None);
                }
                if lhs_rows
                    .checked_mul(lhs_cols)
                    .map(|v| v > QR_DEVICE_MAX_ELEMS)
                    .unwrap_or(true)
                {
                    return Ok(None);
                }

                let start = Instant::now();
                let (q_handle, r_handle, _) = self.qr_factor_device(
                    solve_lhs,
                    lhs_rows,
                    lhs_cols,
                    None,
                    "runmat-linsolve-tall",
                    false,
                )?;
                let rcond = if Self::needs_rcond(options) {
                    let rcond = match self.svd_rcond_sync("linsolve_tall_qr_rcond", &r_handle) {
                        Ok(value) => value,
                        Err(err) => {
                            let _ = self.free(&q_handle);
                            let _ = self.free(&r_handle);
                            return Err(err);
                        }
                    };
                    if let Err(err) = self.enforce_device_rcond(options, rcond) {
                        let _ = self.free(&q_handle);
                        let _ = self.free(&r_handle);
                        return Err(err);
                    }
                    rcond
                } else {
                    f64::NAN
                };
                let q_t_handle = self.transpose_exec(&q_handle)?;
                let projected_rhs =
                    self.matmul_exec_with_usage(&q_t_handle, rhs, BufferUsageClass::FusionOut)?;
                let mut triangular = self.run_triangular_linsolve_device(
                    &r_handle,
                    &projected_rhs,
                    false,
                    false,
                )?;
                let _ = self.free(&projected_rhs);
                let _ = self.free(&q_t_handle);
                let _ = self.free(&q_handle);
                let _ = self.free(&r_handle);
                triangular.reciprocal_condition = rcond;
                self.telemetry.record_linsolve_duration(start.elapsed());
                let shape = [
                    ("rows", lhs_rows as u64),
                    ("cols", lhs_cols as u64),
                    ("rhs_cols", rhs_cols as u64),
                ];
                let tuning = [("method", 2), ("transpose", if options.transposed { 1 } else { 0 })];
                self.record_kernel_launch_basic("linsolve_tall_qr", &shape, &tuning);
                return Ok(Some(triangular));
            }

            if lhs_rows > QR_DEVICE_MAX_COLS {
                return Ok(None);
            }
            if lhs_rows
                .checked_mul(lhs_cols)
                .map(|v| v > QR_DEVICE_MAX_ELEMS)
                .unwrap_or(true)
            {
                return Ok(None);
            }

            let start = Instant::now();
            let lhs_t = self.permute_exec(solve_lhs, &[1, 0])?;
            let (q_handle, r_handle, _) = self.qr_factor_device(
                &lhs_t,
                lhs_cols,
                lhs_rows,
                None,
                "runmat-linsolve-wide",
                false,
            )?;
            let rcond = if Self::needs_rcond(options) {
                let rcond = match self.svd_rcond_sync("linsolve_wide_qr_rcond", &r_handle) {
                    Ok(value) => value,
                    Err(err) => {
                        let _ = self.free(&q_handle);
                        let _ = self.free(&r_handle);
                        let _ = self.free(&lhs_t);
                        return Err(err);
                    }
                };
                if let Err(err) = self.enforce_device_rcond(options, rcond) {
                    let _ = self.free(&q_handle);
                    let _ = self.free(&r_handle);
                    let _ = self.free(&lhs_t);
                    return Err(err);
                }
                rcond
            } else {
                f64::NAN
            };
            let intermediate = self.run_triangular_linsolve_device(&r_handle, rhs, true, true)?;
            let solution = self.matmul_exec_with_usage(
                &q_handle,
                &intermediate.solution,
                BufferUsageClass::FusionOut,
            )?;
            let _ = self.free(&intermediate.solution);
            let _ = self.free(&q_handle);
            let _ = self.free(&r_handle);
            let _ = self.free(&lhs_t);
            self.telemetry.record_linsolve_duration(start.elapsed());
            let shape = [
                ("rows", lhs_rows as u64),
                ("cols", lhs_cols as u64),
                ("rhs_cols", rhs_cols as u64),
            ];
            let tuning = [("method", 3), ("transpose", if options.transposed { 1 } else { 0 })];
            self.record_kernel_launch_basic("linsolve_wide_qr", &shape, &tuning);
            Ok(Some(ProviderLinsolveResult {
                solution,
                reciprocal_condition: rcond,
            }))
        })();
        if let Some(handle) = transposed_lhs.as_ref() {
            let _ = self.free(handle);
        }
        result
    }

    pub(super) fn try_triangular_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<Option<ProviderLinsolveResult>> {
        if options.rectangular
            || options.symmetric
            || options.posdef
        {
            return Ok(None);
        }
        if options.lower == options.upper {
            return Ok(None);
        }

        let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(&lhs.shape)?;
        let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(&rhs.shape)?;
        if lhs_rows != lhs_cols || rhs_rows != lhs_rows {
            return Ok(None);
        }
        if lhs_rows == 0 || rhs_cols == 0 {
            let solution = self.zeros(&rhs.shape)?;
            return Ok(Some(ProviderLinsolveResult {
                solution,
                reciprocal_condition: f64::NAN,
            }));
        }

        let effective_lower = if options.transposed {
            options.upper
        } else {
            options.lower
        };
        let effective_upper = if options.transposed {
            options.lower
        } else {
            options.upper
        };
        if effective_lower == effective_upper {
            return Ok(None);
        }

        let rcond = if Self::needs_rcond(options) {
            let rcond = self.triangular_rcond_sync(lhs)?;
            self.enforce_device_rcond(options, rcond)?;
            rcond
        } else {
            f64::NAN
        };
        let start = Instant::now();
        let mut result =
            self.run_triangular_linsolve_device(lhs, rhs, options.transposed, effective_lower)?;
        result.reciprocal_condition = rcond;
        self.telemetry.record_linsolve_duration(start.elapsed());
        let shape = [
            ("rows", lhs_rows as u64),
            ("rhs_cols", rhs_cols as u64),
            ("steps", lhs_rows as u64),
        ];
        let tuning = [
            ("transpose", if options.transposed { 1 } else { 0 }),
            ("lower", if effective_lower { 1 } else { 0 }),
        ];
        self.record_kernel_launch_basic("linsolve_triangular", &shape, &tuning);
        Ok(Some(result))
    }

    pub(super) fn try_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<Option<ProviderLinsolveResult>> {
        if let Some(result) = self.try_triangular_linsolve_device(lhs, rhs, options)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.try_posdef_linsolve_device(lhs, rhs, options)? {
            return Ok(Some(result));
        }
        self.try_qr_linsolve_device(lhs, rhs, options)
    }

    pub(super) fn try_mrdivide_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> Result<Option<GpuTensorHandle>> {
        let _ = Self::matrix_dims_for_solve(&lhs.shape)?;
        let _ = Self::matrix_dims_for_solve(&rhs.shape)?;
        let rhs_t = self.permute_exec(rhs, &[1, 0])?;
        let lhs_t = self.permute_exec(lhs, &[1, 0])?;
        let result = self.try_linsolve_device(&rhs_t, &lhs_t, &ProviderLinsolveOptions::default())?;
        let output = if let Some(result) = result {
            let transposed = self.permute_exec(&result.solution, &[1, 0])?;
            let _ = self.free(&result.solution);
            Some(transposed)
        } else {
            None
        };
        let _ = self.free(&rhs_t);
        let _ = self.free(&lhs_t);
        Ok(output)
    }
}
