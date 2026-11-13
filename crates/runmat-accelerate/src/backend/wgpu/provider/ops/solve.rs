use super::*;

#[path = "solve/validated_triangular.rs"]
mod validated_triangular;
#[path = "solve/condition.rs"]
mod condition;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "solve/ownership_tests.rs"]
mod ownership_tests;

struct TriangularSolveOperands<'a> {
    lhs: &'a GpuTensorHandle,
    rhs: &'a GpuTensorHandle,
    previous: &'a GpuTensorHandle,
    output: &'a GpuTensorHandle,
}

impl WgpuProvider {
    pub(crate) async fn linsolve_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderLinsolveResult> {
        if let Some(result) = self.try_linsolve_device(lhs, rhs, options).await? {
            return Ok(result);
        }
        let start = Instant::now();
        let lhs_tensor = self.download_real_floating_tensor_exec(lhs).await?;
        let rhs_tensor = self.download_real_floating_tensor_exec(rhs).await?;

        let (solution, rcond) = linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
            .map_err(|e| anyhow!("{e}"))?;
        self.telemetry.record_linsolve_duration(start.elapsed());
        self.telemetry
            .record_solve_fallback("linsolve:host_reupload");

        let handle = self.upload_real_floating_tensor_exec(&solution)?;

        Ok(ProviderLinsolveResult {
            solution: handle,
            reciprocal_condition: rcond,
        })
    }

    pub(crate) async fn inv_exec(
        &self,
        matrix: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (tensor, _) = self.real_floating_host_for_linalg_hook(matrix).await?;
        let result = inv_host_real_for_provider(&tensor).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from_source(source)
                .with_unproven_effects()
        })?;
        self.upload_real_floating_tensor_raw(&result)
            .map_err(runmat_accelerate_api::ProviderExecutionError::with_unproven_effects)
    }

    pub(crate) async fn pinv_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (tensor, _) = self.real_floating_host_for_linalg_hook(matrix).await?;
        let result = pinv_host_real_for_provider(&tensor, options.tolerance).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from_source(source)
                .with_unproven_effects()
        })?;
        self.upload_real_floating_tensor_raw(&result)
            .map_err(runmat_accelerate_api::ProviderExecutionError::with_unproven_effects)
    }

    pub(crate) async fn cond_exec(
        &self,
        matrix: &GpuTensorHandle,
        norm: ProviderCondNorm,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (tensor, precision) = self.real_floating_host_for_linalg_hook(matrix).await?;
        let cond_value = cond_host_real_for_provider(&tensor, norm).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from_source(source)
                .with_unproven_effects()
        })?;
        self.upload_real_floating_scalar_raw(cond_value, precision)
            .map_err(runmat_accelerate_api::ProviderExecutionError::with_unproven_effects)
    }

    pub(crate) async fn norm_exec(
        &self,
        tensor: &GpuTensorHandle,
        order: ProviderNormOrder,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (host_tensor, precision) = self.real_floating_host_for_linalg_hook(tensor).await?;
        let value = norm_host_real_for_provider(&host_tensor, order).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from_source(source)
                .with_unproven_effects()
        })?;
        self.upload_real_floating_scalar_raw(value, precision)
            .map_err(runmat_accelerate_api::ProviderExecutionError::with_unproven_effects)
    }

    pub(crate) async fn rank_exec(
        &self,
        matrix: &GpuTensorHandle,
        tolerance: Option<f64>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (tensor, _) = self.real_floating_host_for_linalg_hook(matrix).await?;
        let rank = rank_host_real_for_provider(&tensor, tolerance).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from_source(source)
                .with_unproven_effects()
        })? as f64;
        self.upload_real_floating_scalar_raw(rank, ProviderPrecision::F64)
            .map_err(runmat_accelerate_api::ProviderExecutionError::with_unproven_effects)
    }

    fn upload_real_floating_scalar_raw(
        &self,
        value: f64,
        precision: ProviderPrecision,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};

        let single = [value as f32];
        let double = [value];
        let data = match precision {
            ProviderPrecision::F32 => HostNumericDataView::F32(&single),
            ProviderPrecision::F64 => HostNumericDataView::F64(&double),
        };
        runmat_accelerate_api::AccelProvider::upload_numeric_raw(
            self,
            &HostNumericTensorView {
                data,
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            },
        )
    }

    pub(crate) async fn rcond_exec(
        &self,
        matrix: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (tensor, precision) = self.real_floating_host_for_linalg_hook(matrix).await?;
        let estimate = rcond_host_real_for_provider(&tensor).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from_source(source)
                .with_unproven_effects()
        })?;
        self.upload_real_floating_scalar_raw(estimate, precision)
            .map_err(runmat_accelerate_api::ProviderExecutionError::with_unproven_effects)
    }

    async fn real_floating_host_for_linalg_hook(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<(Tensor, ProviderPrecision)> {
        let precision = self
            .real_floating_input_precision_exec(handle)
            .map_err(|source| {
                let error = runmat_accelerate_api::ProviderExecutionError::from(source);
                if error.is_unsupported() {
                    error
                } else {
                    error.with_unproven_effects()
                }
            })?;
        // The shared readback has no phase receipt. A failure may precede or
        // follow staging, so it cannot authorize replay.
        let tensor = self
            .download_real_floating_tensor_exec(handle)
            .await
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })?;
        Ok((tensor, precision))
    }

    fn upload_real_floating_tensor_raw(
        &self,
        tensor: &Tensor,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};

        let data = if let Some(values) = tensor.as_f32_slice() {
            HostNumericDataView::F32(values)
        } else if let Some(values) = tensor.as_f64_slice() {
            HostNumericDataView::F64(values)
        } else {
            return Err(runmat_accelerate_api::ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation(
                    "real floating output must have F32 or F64 storage",
                ),
            ));
        };
        runmat_accelerate_api::AccelProvider::upload_numeric_raw(
            self,
            &HostNumericTensorView {
                data,
                shape: tensor.shape(),
                storage: GpuTensorStorage::Real,
            },
        )
    }

    pub(crate) async fn mldivide_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let start = Instant::now();
        if let Some(result) = self
            .try_linsolve_device(lhs, rhs, &ProviderLinsolveOptions::default())
            .await?
        {
            self.telemetry.record_mldivide_duration(start.elapsed());
            return Ok(result.solution);
        }
        let lhs_tensor = self.download_real_floating_tensor_exec(lhs).await?;
        let rhs_tensor = self.download_real_floating_tensor_exec(rhs).await?;

        let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;
        self.telemetry.record_mldivide_duration(start.elapsed());
        self.telemetry
            .record_solve_fallback("mldivide:host_reupload");

        let handle = self.upload_real_floating_tensor_exec(&result)?;
        Ok(handle)
    }

    pub(crate) async fn mrdivide_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let start = Instant::now();
        if let Some(result) = self.try_mrdivide_device(lhs, rhs).await? {
            self.telemetry.record_mrdivide_duration(start.elapsed());
            return Ok(result);
        }
        let lhs_tensor = self.download_real_floating_tensor_exec(lhs).await?;
        let rhs_tensor = self.download_real_floating_tensor_exec(rhs).await?;

        let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;
        self.telemetry.record_mrdivide_duration(start.elapsed());
        self.telemetry
            .record_solve_fallback("mrdivide:host_reupload");

        let handle = self.upload_real_floating_tensor_exec(&result)?;
        Ok(handle)
    }

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
                return Err(anyhow!(
                    "linsolve: matrix is singular to working precision."
                ));
            }
        }
        Ok(())
    }

    async fn host_tensor_from_handle(
        &self,
        label: &str,
        handle: &GpuTensorHandle,
    ) -> Result<Tensor> {
        let HostTensorOwned { data, shape, .. } = self.download_exec(handle).await?;
        Tensor::new(data, shape).map_err(|e| anyhow!("{label}: {e}"))
    }

    async fn triangular_rcond(&self, lhs: &GpuTensorHandle) -> Result<f64> {
        let tensor = self
            .host_tensor_from_handle("linsolve_triangular_rcond", lhs)
            .await?;
        let rows = tensor.rows();
        let cols = tensor.cols();
        ensure!(
            rows == cols,
            "linsolve: triangular rcond estimation requires square input"
        );
        let mut min_diag = f64::INFINITY;
        let mut max_diag = 0.0_f64;
        for i in 0..rows {
            let diag = tensor
                .numeric_value_at(i + i * rows)
                .expect("validated triangular tensor storage")
                .materialize_f64()
                .abs();
            if diag == 0.0 {
                return Err(anyhow!(
                    "linsolve: matrix is singular to working precision."
                ));
            }
            min_diag = min_diag.min(diag);
            max_diag = max_diag.max(diag);
        }
        Ok(Self::diagonal_rcond(min_diag, max_diag))
    }

    async fn svd_rcond(
        &self,
        label: &str,
        factor: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<f64> {
        let tensor = self.host_tensor_from_handle(label, factor).await?;
        let eval = runmat_runtime::builtins::math::linalg::factor::svd::evaluate(
            Value::Tensor(tensor),
            runmat_runtime::call::FunctionArgumentView::from_slice(&[]),
        )
        .await
        .map_err(runtime_flow_to_provider_error)?;
        let singular_values = host_tensor_from_value(label, eval.singular_values())?;
        Ok(Self::singular_value_rcond(
            &singular_values.materialize_f64(),
        ))
    }

    fn needs_rcond(options: &ProviderLinsolveOptions) -> bool {
        options.need_rcond || options.rcond.is_some()
    }

    fn triangular_solve_bind_group(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        bind_group_layout: &wgpu::BindGroupLayout,
        operands: TriangularSolveOperands<'_>,
        params_buffer: &wgpu::Buffer,
    ) -> Result<Arc<wgpu::BindGroup>> {
        let lhs_entry = self.get_entry(operands.lhs)?;
        let rhs_entry = self.get_entry(operands.rhs)?;
        let prev_entry = self.get_entry(operands.previous)?;
        let out_entry = self.get_entry(operands.output)?;
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
        self.bind_group_cache.get_or_create(
            emission,
            bind_group_layout,
            &bind_entries,
            |emission| {
                Arc::new(
                    emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-triangular-solve-bind-group"),
                            layout: bind_group_layout,
                            entries: &bind_entries,
                        }),
                )
            },
        )
    }

    async fn run_triangular_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        transpose: bool,
        effective_lower: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<ProviderLinsolveResult> {
        use runmat_accelerate_api::ProviderExecutionError;
        let frame = validated_triangular::TriangularSolveFrame::plan(
            self,
            lhs,
            rhs,
            transpose,
            effective_lower,
        )?;
        if frame.rows == 0 || frame.rhs_cols == 0 {
            return Ok(ProviderLinsolveResult {
                solution: self.zeros(rhs.shape())?,
                reciprocal_condition: f64::NAN,
            });
        }
        let validated = self
            .prepare_validated_triangular_solve_pipeline(&frame.recipe)
            .await?;
        validated
            .require_triangular_solve(&frame.recipe)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        frame
            .require_live_inputs(self, lhs, rhs)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let mut current = self
            .zeros(rhs.shape())
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let mut scratch = self
            .zeros(rhs.shape())
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        self.with_emission(|emission, pending| -> Result<()> {
            frame.require_live_inputs(self, lhs, rhs)?;
            let bind_group_layout = validated.layout.as_ref();
            let pipeline = validated.pipeline.as_ref();
            crate::backend::wgpu::dispatch::solve::warmup_noop(emission, pending, pipeline)?;
            let mut params = crate::backend::wgpu::params::TriangularSolveParams {
                len: frame.len as u32,
                offset: 0,
                total: frame.len as u32,
                rows: frame.rows as u32,
                rhs_cols: frame.rhs_cols as u32,
                target_row: 0,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer =
                self.uniform_buffer(emission, &params, "runmat-triangular-solve-params")?;
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                frame.len as u32,
                frame.recipe.workgroup,
            );
            for step in 0..frame.rows {
                params.target_row = if effective_lower {
                    step
                } else {
                    frame.rows - 1 - step
                } as u32;
                emission.write_buffer(&params_buffer, 0, bytes_of(&params));
                let bind_group = self.triangular_solve_bind_group(
                    emission,
                    bind_group_layout,
                    TriangularSolveOperands {
                        lhs,
                        rhs,
                        previous: &current,
                        output: &scratch,
                    },
                    &params_buffer,
                )?;
                crate::backend::wgpu::dispatch::solve::run(
                    emission,
                    pending,
                    pipeline,
                    bind_group.as_ref(),
                    workgroups,
                )?;
                std::mem::swap(&mut current, &mut scratch);
            }
            Ok(())
        })
        .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
        .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        drop(scratch);
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
        let (r_buffer, r_inv_buffer) = self.with_emission(|emission, pending| -> Result<_> {
            let gram_buffer = self.kernel_resources.scratch_storage_buffer(
                emission,
                crate::backend::wgpu::resources::ScratchBufferKind::QrGram,
                bytes,
                "runmat-chol-gram-scratch",
            )?;
            if bytes > 0 {
                let copy_label = format!("{label}-gram-copy");
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some(copy_label.as_str()),
                    },
                );
                encoder.copy_buffer_to_buffer(
                    matrix_entry.buffer.as_ref(),
                    0,
                    gram_buffer.as_ref(),
                    0,
                    bytes,
                );
                self.submit(emission, pending, encoder)?;
            }

            let r_buffer =
                self.create_storage_buffer_checked(emission, len_out, "runmat-chol-r")?;
            let r_inv_buffer =
                self.create_storage_buffer_checked(emission, len_out, "runmat-chol-rinv")?;

            let params = QrPowerIterParams {
                cols: rows as u32,
                stride: rows as u32,
                _pad0: [0, 0],
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                emission,
                crate::backend::wgpu::resources::UniformBufferKey::QrPowerIterParams,
                std::mem::size_of::<QrPowerIterParams>() as u64,
                "runmat-chol-params",
            )?;
            emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));

            let layout = &self.pipelines.qr_power_iter.get(emission)?.layout;
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
            let bind_group = self.bind_group_cache.get_or_create(
                emission,
                layout,
                &bind_entries,
                |emission| {
                    Arc::new(
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-chol-bind"),
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

        let shape = vec![rows, rows];
        let r_handle = self.register_existing_buffer_with_usage(
            r_buffer.clone(),
            shape.clone(),
            len_out,
            BufferUsageClass::FusionOut,
        )?;
        self.mark_buffer_usage(&r_handle, BufferUsageClass::FusionOut);
        let r_inv_handle = self.register_existing_buffer_with_usage(
            r_inv_buffer.clone(),
            shape,
            len_out,
            BufferUsageClass::FusionOut,
        )?;
        self.mark_buffer_usage(&r_inv_handle, BufferUsageClass::FusionOut);
        Ok((r_handle, r_inv_handle))
    }

    async fn try_posdef_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<ProviderLinsolveResult>> {
        if !options.posdef || options.lower || options.upper || options.rectangular {
            return Ok(None);
        }
        if self.provider_precision_exec() != ProviderPrecision::F32 {
            return Ok(None);
        }

        let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(lhs.shape())?;
        let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(rhs.shape())?;
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
            let rcond = match self
                .square_rcond_from_factors(
                    lhs,
                    condition::SquareDeviceFactor::Cholesky { r: &r_handle },
                )
                .await
            {
                Ok(value) => value,
                Err(err) => {
                    drop(r_handle);
                    drop(r_inv_handle);
                    return Err(err.with_unproven_effects());
                }
            };
            if let Err(err) = self.enforce_device_rcond(options, rcond) {
                drop(r_handle);
                drop(r_inv_handle);
                return Err(runmat_accelerate_api::ProviderExecutionError::from(err)
                    .with_unproven_effects());
            }
            rcond
        } else {
            f64::NAN
        };
        let projected_rhs = match self
            .run_triangular_linsolve_device(&r_handle, rhs, true, true)
            .await
        {
            Ok(value) => value,
            Err(err) => {
                drop(r_handle);
                drop(r_inv_handle);
                return Err(err.with_unproven_effects());
            }
        };
        let mut solution = match self
            .run_triangular_linsolve_device(&r_handle, &projected_rhs.solution, false, false)
            .await
        {
            Ok(value) => value,
            Err(err) => {
                drop(projected_rhs.solution);
                drop(r_handle);
                drop(r_inv_handle);
                return Err(err.with_unproven_effects());
            }
        };
        drop(projected_rhs.solution);
        drop(r_handle);
        drop(r_inv_handle);
        solution.reciprocal_condition = rcond;

        self.telemetry.record_linsolve_duration(start.elapsed());
        let shape = [
            ("rows", lhs_rows as u64),
            ("rhs_cols", rhs_cols as u64),
            ("transpose", if options.transposed { 1 } else { 0 }),
        ];
        let tuning = [
            ("method", 4),
            ("symmetric", if options.symmetric { 1 } else { 0 }),
        ];
        self.record_kernel_launch_basic("linsolve_posdef_chol", &shape, &tuning);
        Ok(Some(solution))
    }

    async fn try_qr_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<ProviderLinsolveResult>> {
        if options.lower || options.upper {
            return Ok(None);
        }
        if self.provider_precision_exec() != ProviderPrecision::F32 {
            return Ok(None);
        }

        let transposed_lhs = if options.transposed {
            Some(self.permute_exec(lhs, &[1, 0])?)
        } else {
            None
        };
        let solve_lhs = transposed_lhs.as_ref().unwrap_or(lhs);
        let result = async {
            let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(solve_lhs.shape())?;
            let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(rhs.shape())?;
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
                    "runmat-linsolve-tall",
                    false,
                )?;
                let rcond = if Self::needs_rcond(options) {
                    let rcond = match if lhs_rows == lhs_cols {
                        self.square_rcond_from_factors(
                            solve_lhs,
                            condition::SquareDeviceFactor::Qr {
                                q: &q_handle,
                                r: &r_handle,
                            },
                        )
                        .await
                    } else {
                        self.svd_rcond("linsolve_tall_qr_rcond", &r_handle).await
                    } {
                        Ok(value) => value,
                        Err(err) => {
                            drop(q_handle);
                            drop(r_handle);
                            return Err(err.with_unproven_effects());
                        }
                    };
                    if let Err(err) = self.enforce_device_rcond(options, rcond) {
                        drop(q_handle);
                        drop(r_handle);
                        return Err(runmat_accelerate_api::ProviderExecutionError::from(err)
                            .with_unproven_effects());
                    }
                    rcond
                } else {
                    f64::NAN
                };
                let q_t_handle = self.transpose_exec(&q_handle)?;
                let projected_rhs =
                    self.matmul_exec_with_usage(&q_t_handle, rhs, BufferUsageClass::FusionOut)?;
                let mut triangular = self
                    .run_triangular_linsolve_device(&r_handle, &projected_rhs, false, false)
                    .await
                    .map_err(|error| error.with_unproven_effects())?;
                drop(projected_rhs);
                drop(q_t_handle);
                drop(q_handle);
                drop(r_handle);
                triangular.reciprocal_condition = rcond;
                self.telemetry.record_linsolve_duration(start.elapsed());
                let shape = [
                    ("rows", lhs_rows as u64),
                    ("cols", lhs_cols as u64),
                    ("rhs_cols", rhs_cols as u64),
                ];
                let tuning = [
                    ("method", 2),
                    ("transpose", if options.transposed { 1 } else { 0 }),
                ];
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
            let (q_handle, r_handle, _) =
                self.qr_factor_device(&lhs_t, lhs_cols, lhs_rows, "runmat-linsolve-wide", false)?;
            let rcond = if Self::needs_rcond(options) {
                let rcond = match self.svd_rcond("linsolve_wide_qr_rcond", &r_handle).await {
                    Ok(value) => value,
                    Err(err) => {
                        drop(q_handle);
                        drop(r_handle);
                        drop(lhs_t);
                        return Err(err);
                    }
                };
                if let Err(err) = self.enforce_device_rcond(options, rcond) {
                    drop(q_handle);
                    drop(r_handle);
                    drop(lhs_t);
                    return Err(err.into());
                }
                rcond
            } else {
                f64::NAN
            };
            let intermediate = self
                .run_triangular_linsolve_device(&r_handle, rhs, true, true)
                .await
                .map_err(|error| error.with_unproven_effects())?;
            let solution = self.matmul_exec_with_usage(
                &q_handle,
                &intermediate.solution,
                BufferUsageClass::FusionOut,
            )?;
            drop(intermediate.solution);
            drop(q_handle);
            drop(r_handle);
            drop(lhs_t);
            self.telemetry.record_linsolve_duration(start.elapsed());
            let shape = [
                ("rows", lhs_rows as u64),
                ("cols", lhs_cols as u64),
                ("rhs_cols", rhs_cols as u64),
            ];
            let tuning = [
                ("method", 3),
                ("transpose", if options.transposed { 1 } else { 0 }),
            ];
            self.record_kernel_launch_basic("linsolve_wide_qr", &shape, &tuning);
            Ok(Some(ProviderLinsolveResult {
                solution,
                reciprocal_condition: rcond,
            }))
        }
        .await;
        drop(transposed_lhs);
        result
    }

    pub(super) async fn try_triangular_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<ProviderLinsolveResult>> {
        if options.rectangular || options.symmetric || options.posdef {
            return Ok(None);
        }
        if options.lower == options.upper {
            return Ok(None);
        }

        let (lhs_rows, lhs_cols) = Self::matrix_dims_for_solve(lhs.shape())?;
        let (rhs_rows, rhs_cols) = Self::matrix_dims_for_solve(rhs.shape())?;
        if lhs_rows != lhs_cols || rhs_rows != lhs_rows {
            return Ok(None);
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
        if lhs_rows == 0 || rhs_cols == 0 {
            return self
                .run_triangular_linsolve_device(lhs, rhs, options.transposed, effective_lower)
                .await
                .map(Some);
        }

        let rcond = if Self::needs_rcond(options) {
            let rcond = self.triangular_rcond(lhs).await?;
            self.enforce_device_rcond(options, rcond)?;
            rcond
        } else {
            f64::NAN
        };
        let start = Instant::now();
        let mut result = self
            .run_triangular_linsolve_device(lhs, rhs, options.transposed, effective_lower)
            .await
            .map_err(|error| {
                if Self::needs_rcond(options) {
                    error.with_unproven_effects()
                } else {
                    error
                }
            })?;
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

    pub(super) async fn try_linsolve_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<ProviderLinsolveResult>> {
        // These device solvers use the provider's F32 shader ABI. A provider
        // configured for F32 may still own F64 inputs; such operands must use
        // the exact-class host boundary instead of being reinterpreted.
        let lhs_precision = self.real_floating_input_precision_exec(lhs)?;
        let rhs_precision = self.real_floating_input_precision_exec(rhs)?;
        if self.provider_precision_exec() != ProviderPrecision::F32
            || lhs_precision != ProviderPrecision::F32
            || rhs_precision != ProviderPrecision::F32
        {
            return Ok(None);
        }
        if let Some(result) = self
            .try_triangular_linsolve_device(lhs, rhs, options)
            .await?
        {
            return Ok(Some(result));
        }
        if let Some(result) = self.try_posdef_linsolve_device(lhs, rhs, options).await? {
            return Ok(Some(result));
        }
        self.try_qr_linsolve_device(lhs, rhs, options).await
    }

    pub(super) async fn try_mrdivide_device(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let lhs_precision = self.real_floating_input_precision_exec(lhs)?;
        let rhs_precision = self.real_floating_input_precision_exec(rhs)?;
        if self.provider_precision_exec() != ProviderPrecision::F32
            || lhs_precision != ProviderPrecision::F32
            || rhs_precision != ProviderPrecision::F32
        {
            return Ok(None);
        }
        let _ = Self::matrix_dims_for_solve(lhs.shape())?;
        let _ = Self::matrix_dims_for_solve(rhs.shape())?;
        let rhs_t = self.permute_exec(rhs, &[1, 0])?;
        let lhs_t = self.permute_exec(lhs, &[1, 0])?;
        let result = self
            .try_linsolve_device(&rhs_t, &lhs_t, &ProviderLinsolveOptions::default())
            .await?;
        let output = if let Some(result) = result {
            let transposed = self.permute_exec(&result.solution, &[1, 0])?;
            drop(result.solution);
            Some(transposed)
        } else {
            None
        };
        drop(rhs_t);
        drop(lhs_t);
        Ok(output)
    }
}
