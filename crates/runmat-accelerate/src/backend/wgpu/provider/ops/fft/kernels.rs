use super::*;
use runmat_accelerate_api::unsupported_provider_operation;

impl WgpuProvider {
    pub(crate) async fn fft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry(handle)?.integer_type().is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(unsupported_provider_operation(
                "fft_dim: nonfloating handles require the runtime typed fallback",
            )
            .into());
        }
        if let Some(native) = self
            .try_fft_dim_exec_native(handle, len, dim, false)
            .await?
        {
            return Ok(native);
        }
        self.fft_dim_exec_host_fallback(handle, len, dim, false)
            .await
            .map_err(Into::into)
    }

    pub(crate) async fn ifft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry(handle)?.integer_type().is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(unsupported_provider_operation(
                "ifft_dim: nonfloating handles require the runtime typed fallback",
            )
            .into());
        }
        if let Some(native) = self.try_fft_dim_exec_native(handle, len, dim, true).await? {
            return Ok(native);
        }
        self.fft_dim_exec_host_fallback(handle, len, dim, true)
            .await
            .map_err(Into::into)
    }

    pub(crate) async fn fft_extract_real_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let mut out_shape = handle.shape().to_vec();
        if out_shape.last() != Some(&2) {
            return Ok(handle.clone());
        }
        out_shape.pop();
        out_shape = normalize_scalar_shape(&out_shape);

        let complex_len = entry.len / 2;
        if complex_len > u32::MAX as usize {
            return Err(anyhow!("fft_extract_real: tensor too large for GPU kernel").into());
        }

        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, complex_len, "runmat-fft-extract-real-out")
        })??;
        let output_handle =
            self.register_existing_buffer(out_buffer.clone(), out_shape, complex_len)?;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < complex_len {
            let chunk_len = (complex_len - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftExtractRealParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: complex_len as u32,
                _pad0: 0,
            };
            self.fft_dispatch_exec("runmat-fft-extract-real-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-extract-real-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-extract-real-bind"),
                        layout: &self.pipelines.fft_extract_real.get(emission)?.layout,
                        entries: &[
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
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_extract_real.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[handle, &output_handle],
                        &[&entry.buffer, &out_buffer, &params_buffer],
                    )?,
                ))
            })
            .await?;
            offset += chunk_len;
        }

        Ok(output_handle)
    }

    async fn try_fft_dim_exec_native(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
        inverse: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        if Self::fft_debug_enabled() {
            eprintln!(
                "[fft-debug] native enter len={len:?} dim={dim} inverse={inverse} shape={:?}",
                handle.shape()
            );
        }
        let entry = self.get_entry(handle)?;
        let mut shape = handle.shape().to_vec();
        let complex_axis =
            runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved;
        if shape.is_empty() {
            let scalar_len = if complex_axis {
                entry.len / 2
            } else {
                entry.len
            };
            shape = vec![scalar_len];
        }

        let origin_rank = shape.len();
        while shape.len() <= dim {
            shape.push(1);
        }

        let current_len = shape.get(dim).copied().unwrap_or(0);
        let target_len = len.unwrap_or(current_len);
        let inner_stride = shape[..dim]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let outer_stride = shape[dim + 1..]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let num_slices = inner_stride.saturating_mul(outer_stride);

        let mut out_shape = shape.clone();
        if dim < out_shape.len() {
            out_shape[dim] = target_len;
        }
        if target_len == 0 || num_slices == 0 {
            fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
            let buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-fft-empty-native")
            })??;
            return Ok(Some(self.register_existing_buffer_with_storage(
                buffer,
                out_shape,
                0,
                GpuTensorStorage::ComplexInterleaved,
            )?));
        }

        let total_out = target_len.saturating_mul(num_slices);
        let copy_len = current_len.min(target_len);

        if !(target_len <= u32::MAX as usize
            && inner_stride <= u32::MAX as usize
            && current_len <= u32::MAX as usize
            && copy_len <= u32::MAX as usize
            && total_out <= u32::MAX as usize)
        {
            return Err(anyhow!("fft_dim: dimensions exceed native GPU FFT kernel limits").into());
        }

        let out_scalar_len = total_out
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: output buffer length overflow"))?;

        if !fft_is_power_of_two(target_len) {
            if let Some(factors) = fft_factor_smooth_235(target_len) {
                let has_3 = factors.contains(&3);
                let has_5 = factors.contains(&5);
                if has_3 && !has_5 {
                    if let Some(digits) = fft_log3_pow3(target_len) {
                        return self
                            .try_fft_dim_exec_native_radix3(
                                handle,
                                entry.buffer,
                                shape,
                                dim,
                                origin_rank,
                                current_len,
                                copy_len,
                                target_len,
                                inner_stride,
                                total_out,
                                out_scalar_len,
                                complex_axis,
                                inverse,
                                digits,
                            )
                            .await;
                    }
                    return self
                        .try_fft_dim_exec_native_mixed(
                            handle,
                            entry.buffer,
                            shape,
                            dim,
                            origin_rank,
                            current_len,
                            copy_len,
                            target_len,
                            inner_stride,
                            total_out,
                            out_scalar_len,
                            complex_axis,
                            inverse,
                            &factors,
                        )
                        .await;
                } else if has_5 && !has_3 {
                    if let Some(digits) = fft_log5_pow5(target_len) {
                        return self
                            .try_fft_dim_exec_native_radix5(
                                handle,
                                entry.buffer,
                                shape,
                                dim,
                                origin_rank,
                                current_len,
                                copy_len,
                                target_len,
                                inner_stride,
                                total_out,
                                out_scalar_len,
                                complex_axis,
                                inverse,
                                digits,
                            )
                            .await;
                    }
                    return self
                        .try_fft_dim_exec_native_mixed(
                            handle,
                            entry.buffer,
                            shape,
                            dim,
                            origin_rank,
                            current_len,
                            copy_len,
                            target_len,
                            inner_stride,
                            total_out,
                            out_scalar_len,
                            complex_axis,
                            inverse,
                            &factors,
                        )
                        .await;
                } else {
                    return self
                        .try_fft_dim_exec_native_mixed(
                            handle,
                            entry.buffer,
                            shape,
                            dim,
                            origin_rank,
                            current_len,
                            copy_len,
                            target_len,
                            inner_stride,
                            total_out,
                            out_scalar_len,
                            complex_axis,
                            inverse,
                            &factors,
                        )
                        .await;
                }
            }
            if let Some(bluestein) = Box::pin(self.try_fft_dim_exec_native_bluestein(
                handle,
                entry.buffer.clone(),
                shape.clone(),
                dim,
                origin_rank,
                current_len,
                copy_len,
                target_len,
                inner_stride,
                num_slices,
                total_out,
                out_scalar_len,
                complex_axis,
                inverse,
            ))
            .await?
            {
                return Ok(Some(bluestein));
            }
            if let Some(digits) = fft_log3_pow3(target_len) {
                return self
                    .try_fft_dim_exec_native_radix3(
                        handle,
                        entry.buffer,
                        shape,
                        dim,
                        origin_rank,
                        current_len,
                        copy_len,
                        target_len,
                        inner_stride,
                        total_out,
                        out_scalar_len,
                        complex_axis,
                        inverse,
                        digits,
                    )
                    .await;
            }
            if let Some(digits) = fft_log5_pow5(target_len) {
                return self
                    .try_fft_dim_exec_native_radix5(
                        handle,
                        entry.buffer,
                        shape,
                        dim,
                        origin_rank,
                        current_len,
                        copy_len,
                        target_len,
                        inner_stride,
                        total_out,
                        out_scalar_len,
                        complex_axis,
                        inverse,
                        digits,
                    )
                    .await;
            }
            return Ok(None);
        }
        let Some(log2_len) = fft_log2_pow2(target_len) else {
            return Ok(None);
        };
        let (mut stage_a, mut stage_b) = self.with_emission(|emission, _| {
            self.fft_stage_buffer_pair(
                emission,
                out_scalar_len,
                "runmat-fft-stage-a",
                "runmat-fft-stage-b",
            )
        })??;
        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        // Each stage swaps A/B; reorder writes the opposite physical owner.
        let final_buffer = if log2_len % 2 == 0 {
            &stage_b
        } else {
            &stage_a
        };
        let output_handle = self.register_existing_buffer_with_storage(
            final_buffer.clone(),
            out_shape,
            out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            self.fft_dispatch_exec("runmat-fft-init-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-init-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-init-bind"),
                        layout: &self.pipelines.fft_init.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_init.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[handle, &output_handle],
                        &[&entry.buffer, &stage_a, &params_buffer],
                    )?,
                ))
            })
            .await?;
            offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("stage2:init", &stage_a, out_scalar_len);

        let mut stage_span = target_len;
        while stage_span >= 2 {
            let stage_half = stage_span / 2;
            let twiddle_step = target_len / stage_span;
            let mut stage_offset = 0usize;
            while stage_offset < total_out {
                let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                let params = crate::backend::wgpu::params::FftStageParams {
                    len: chunk_len as u32,
                    offset: stage_offset as u32,
                    total: total_out as u32,
                    target_len: target_len as u32,
                    inner_stride: inner_stride as u32,
                    stage_span: stage_span as u32,
                    stage_half: stage_half as u32,
                    twiddle_step: twiddle_step as u32,
                    inverse: if inverse { 1 } else { 0 },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                if Self::fft_debug_enabled() {
                    eprintln!(
                        "[fft-debug] stage2 params span={} half={} step={} chunk={} off={}",
                        params.stage_span,
                        params.stage_half,
                        params.twiddle_step,
                        params.len,
                        params.offset
                    );
                }
                self.fft_dispatch_exec("runmat-fft-stage-encoder", |emission| {
                    let params_buffer =
                        self.fft_uniform_buffer(emission, &params, "runmat-fft-stage-params")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fft-stage-bind"),
                                layout: &self.pipelines.fft_stage.get(emission)?.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: stage_a.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: stage_b.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                        chunk_len as u32,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                    );
                    if Self::fft_debug_enabled() {
                        eprintln!(
                            "[fft-debug] stage2 dispatch workgroups={workgroups} chunk={chunk_len}"
                        );
                    }
                    Ok(FftPreparedDispatch::new(
                        &self.pipelines.fft_stage.get(emission)?.pipeline,
                        bind_group,
                        workgroups,
                        FftDispatchResources::new(
                            &[handle, &output_handle],
                            &[&stage_a, &stage_b, &params_buffer],
                        )?,
                    ))
                })
                .await?;
                stage_offset += chunk_len;
            }
            std::mem::swap(&mut stage_a, &mut stage_b);
            self.fft_debug_dump_scalar_buffer("stage2:after_stage", &stage_a, out_scalar_len);
            stage_span /= 2;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorderParams {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                log2_len,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            self.fft_dispatch_exec("runmat-fft-reorder-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-reorder-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-reorder-bind"),
                        layout: &self.pipelines.fft_reorder.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_reorder.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[handle, &output_handle],
                        &[&stage_a, &stage_b, &params_buffer],
                    )?,
                ))
            })
            .await?;
            reorder_offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("stage2:after_reorder", &stage_b, out_scalar_len);

        Ok(Some(output_handle))
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_fft_dim_exec_native_bluestein(
        &self,
        input_handle: &GpuTensorHandle,
        in_buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        num_slices: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let two_n_minus_1 = target_len
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| anyhow!("fft_dim: bluestein length overflow"))?;
        let m_len = two_n_minus_1.next_power_of_two();
        if m_len <= target_len {
            return Ok(None);
        }
        if m_len > u32::MAX as usize || target_len > u32::MAX as usize {
            return Ok(None);
        }

        let mut ext_shape = shape.clone();
        ext_shape[dim] = m_len;

        let m_total = m_len
            .checked_mul(num_slices)
            .ok_or_else(|| anyhow!("fft_dim: bluestein total overflow"))?;
        let m_out_scalar_len = m_total
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: bluestein buffer overflow"))?;

        let chirp_scalar_len = target_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: chirp buffer overflow"))?;
        let chirp_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, chirp_scalar_len, "runmat-fft-bluestein-chirp")
        })??;
        let pi = std::f64::consts::PI;
        let sign = if inverse { 1.0 } else { -1.0 };
        match self.precision {
            NumericPrecision::F64 => {
                let mut chirp = Vec::with_capacity(chirp_scalar_len);
                for n in 0..target_len {
                    let angle = sign * pi * (n as f64) * (n as f64) / (target_len as f64);
                    chirp.push(angle.cos());
                    chirp.push(angle.sin());
                }
                self.with_emission(|emission, _| {
                    emission.write_buffer(chirp_buffer.as_ref(), 0, cast_slice(&chirp))
                })?;
            }
            NumericPrecision::F32 => {
                let mut chirp = Vec::with_capacity(chirp_scalar_len);
                for n in 0..target_len {
                    let angle = sign * pi * (n as f64) * (n as f64) / (target_len as f64);
                    chirp.push(angle.cos() as f32);
                    chirp.push(angle.sin() as f32);
                }
                self.with_emission(|emission, _| {
                    emission.write_buffer(chirp_buffer.as_ref(), 0, cast_slice(&chirp))
                })?;
            }
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let a_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, m_out_scalar_len, "runmat-fft-bluestein-a")
        })??;
        let a_handle = self.register_existing_buffer_with_storage(
            a_buffer.clone(),
            ext_shape.clone(),
            m_out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, out_scalar_len, "runmat-fft-bluestein-out")
        })??;
        let mut out_shape = shape;
        out_shape[dim] = target_len;
        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        let output_handle = self.register_existing_buffer_with_storage(
            out_buffer.clone(),
            out_shape,
            out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let mut offset = 0usize;
        while offset < m_total {
            let chunk_len = (m_total - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftBluesteinPrepParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: m_total as u32,
                target_len: m_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            self.fft_dispatch_exec("runmat-fft-blue-prep-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-blue-prep-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-blue-prep-bind"),
                        layout: &self.pipelines.fft_bluestein_prep.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: in_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: a_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: chirp_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_bluestein_prep.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &a_handle],
                        &[&in_buffer, &a_buffer, &chirp_buffer, &params_buffer],
                    )?,
                ))
            })
            .await?;
            offset += chunk_len;
        }

        let b_scalar_len = m_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: bluestein kernel buffer overflow"))?;
        let b_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, b_scalar_len, "runmat-fft-bluestein-b")
        })??;
        let b_handle = self.register_existing_buffer_with_storage(
            b_buffer.clone(),
            vec![m_len],
            b_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let mut boff = 0usize;
        while boff < m_len {
            let chunk_len = (m_len - boff).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftBluesteinKernelParams {
                len: chunk_len as u32,
                offset: boff as u32,
                total: m_len as u32,
                target_len: target_len as u32,
                m_len: m_len as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
            };
            self.fft_dispatch_exec("runmat-fft-blue-kernel-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-blue-kernel-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-blue-kernel-bind"),
                        layout: &self.pipelines.fft_bluestein_kernel.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: b_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: chirp_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_bluestein_kernel.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[&b_handle],
                        &[&b_buffer, &chirp_buffer, &params_buffer],
                    )?,
                ))
            })
            .await?;
            boff += chunk_len;
        }

        let Some(a_fft) = self
            .try_fft_dim_exec_native(&a_handle, Some(m_len), dim, false)
            .await?
        else {
            return Ok(None);
        };
        let Some(b_fft) = self
            .try_fft_dim_exec_native(&b_handle, Some(m_len), 0, false)
            .await?
        else {
            return Ok(None);
        };

        let c_fft_buf = self.with_emission(|emission, _| {
            self.create_storage_buffer(emission, m_out_scalar_len, "runmat-fft-bluestein-cfft")
        })??;
        let c_fft_handle = self.register_existing_buffer_with_storage(
            c_fft_buf.clone(),
            ext_shape.clone(),
            m_out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let mut poff = 0usize;
        while poff < m_total {
            let chunk_len = (m_total - poff).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftPointwiseBroadcastParams {
                len: chunk_len as u32,
                offset: poff as u32,
                total: m_total as u32,
                m_len: m_len as u32,
                inner_stride: inner_stride as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            self.fft_dispatch_exec("runmat-fft-blue-pointwise-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-blue-pointwise-params")?;
                let a_entry = self.get_entry(&a_fft)?;
                let b_entry = self.get_entry(&b_fft)?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-blue-pointwise-bind"),
                        layout: &self.pipelines.fft_pointwise_broadcast.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: c_fft_buf.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self
                        .pipelines
                        .fft_pointwise_broadcast
                        .get(emission)?
                        .pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[&a_fft, &b_fft, &c_fft_handle],
                        &[&a_entry.buffer, &b_entry.buffer, &c_fft_buf, &params_buffer],
                    )?,
                ))
            })
            .await?;
            poff += chunk_len;
        }

        let Some(c_time) = self
            .try_fft_dim_exec_native(&c_fft_handle, Some(m_len), dim, true)
            .await?
        else {
            return Ok(None);
        };

        let mut foff = 0usize;
        while foff < total_out {
            let chunk_len = (total_out - foff).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftBluesteinFinalizeParams {
                len: chunk_len as u32,
                offset: foff as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                m_len: m_len as u32,
                inner_stride: inner_stride as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            self.fft_dispatch_exec("runmat-fft-blue-finalize-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft-blue-finalize-params")?;
                let c_entry = self.get_entry(&c_time)?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-blue-finalize-bind"),
                        layout: &self.pipelines.fft_bluestein_finalize.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: c_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: chirp_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self
                        .pipelines
                        .fft_bluestein_finalize
                        .get(emission)?
                        .pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[&c_time, &output_handle],
                        &[&c_entry.buffer, &out_buffer, &chirp_buffer, &params_buffer],
                    )?,
                ))
            })
            .await?;
            foff += chunk_len;
        }

        drop(a_handle);
        drop(b_handle);
        drop(a_fft);
        drop(b_fft);
        drop(c_fft_handle);
        drop(c_time);

        Ok(Some(output_handle))
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_fft_dim_exec_native_mixed(
        &self,
        input_handle: &GpuTensorHandle,
        in_buffer: runmat_accelerate_api::WgpuBufferOwner,
        mut shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
        factors: &[u32],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        if factors.len() > 16 {
            return Ok(None);
        }

        let (mut stage_a, mut stage_b) = self.with_emission(|emission, _| {
            self.fft_stage_buffer_pair(
                emission,
                out_scalar_len,
                "runmat-fftm-stage-a",
                "runmat-fftm-stage-b",
            )
        })??;
        shape[dim] = target_len;
        fft_trim_trailing_ones(&mut shape, origin_rank.max(dim + 1));
        // Each stage swaps A/B; reorder writes the opposite physical owner.
        let final_buffer = if factors.len().is_multiple_of(2) {
            &stage_b
        } else {
            &stage_a
        };
        let output_handle = self.register_existing_buffer_with_storage(
            final_buffer.clone(),
            shape,
            out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let twiddle = self.with_emission(|emission, _| {
            self.fft_twiddle_buffer(emission, target_len, false, "runmat-fftm-twiddles")
        })??;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            self.fft_dispatch_exec("runmat-fftm-init-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fftm-init-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fftm-init-bind"),
                        layout: &self.pipelines.fft_init.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: in_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_init.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &output_handle],
                        &[&in_buffer, &stage_a, &params_buffer],
                    )?,
                ))
            })
            .await?;
            offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("mixed:init", &stage_a, out_scalar_len);

        let mut stage_span = target_len;
        for &radix in factors {
            if radix == 2 {
                let stage_half = stage_span / 2;
                let twiddle_step = target_len / stage_span;
                let mut stage_offset = 0usize;
                while stage_offset < total_out {
                    let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                    let params = crate::backend::wgpu::params::FftStageParams {
                        len: chunk_len as u32,
                        offset: stage_offset as u32,
                        total: total_out as u32,
                        target_len: target_len as u32,
                        inner_stride: inner_stride as u32,
                        stage_span: stage_span as u32,
                        stage_half: stage_half as u32,
                        twiddle_step: twiddle_step as u32,
                        inverse: if inverse { 1 } else { 0 },
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.fft_dispatch_exec("runmat-fftm-stage2-encoder", |emission| {
                        let params_buffer = self.fft_uniform_buffer(
                            emission,
                            &params,
                            "runmat-fftm-stage2-params",
                        )?;
                        let bind_group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-fftm-stage2-bind"),
                                    layout: &self.pipelines.fft_stage.get(emission)?.layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: stage_a.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: stage_b.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: params_buffer.as_entire_binding(),
                                        },
                                    ],
                                });
                        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                            chunk_len as u32,
                            crate::backend::wgpu::config::WORKGROUP_SIZE,
                        );
                        Ok(FftPreparedDispatch::new(
                            &self.pipelines.fft_stage.get(emission)?.pipeline,
                            bind_group,
                            workgroups,
                            FftDispatchResources::new(
                                &[input_handle, &output_handle],
                                &[&stage_a, &stage_b, &params_buffer],
                            )?,
                        ))
                    })
                    .await?;
                    stage_offset += chunk_len;
                }
                self.fft_debug_dump_scalar_buffer("mixed:after_stage2", &stage_b, out_scalar_len);
            } else if radix == 3 {
                let stage_third = stage_span / 3;
                let twiddle_step = target_len / stage_span;
                let mut stage_offset = 0usize;
                while stage_offset < total_out {
                    let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                    let params = crate::backend::wgpu::params::FftStage3Params {
                        len: chunk_len as u32,
                        offset: stage_offset as u32,
                        total: total_out as u32,
                        target_len: target_len as u32,
                        inner_stride: inner_stride as u32,
                        stage_span: stage_span as u32,
                        stage_third: stage_third as u32,
                        twiddle_step: twiddle_step as u32,
                        inverse: if inverse { 1 } else { 0 },
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.fft_dispatch_exec("runmat-fftm-stage3-encoder", |emission| {
                        let params_buffer = self.fft_uniform_buffer(
                            emission,
                            &params,
                            "runmat-fftm-stage3-params",
                        )?;
                        let bind_group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-fftm-stage3-bind"),
                                    layout: &self.pipelines.fft_stage3.get(emission)?.layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: stage_a.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: stage_b.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: twiddle.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 3,
                                            resource: params_buffer.as_entire_binding(),
                                        },
                                    ],
                                });
                        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                            chunk_len as u32,
                            crate::backend::wgpu::config::WORKGROUP_SIZE,
                        );
                        Ok(FftPreparedDispatch::new(
                            &self.pipelines.fft_stage3.get(emission)?.pipeline,
                            bind_group,
                            workgroups,
                            FftDispatchResources::new(
                                &[input_handle, &output_handle],
                                &[&stage_a, &stage_b, &twiddle, &params_buffer],
                            )?,
                        ))
                    })
                    .await?;
                    stage_offset += chunk_len;
                }
                self.fft_debug_dump_scalar_buffer("mixed:after_stage3", &stage_b, out_scalar_len);
            } else if radix == 5 {
                let stage_fifth = stage_span / 5;
                let twiddle_step = target_len / stage_span;
                let mut stage_offset = 0usize;
                while stage_offset < total_out {
                    let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                    let params = crate::backend::wgpu::params::FftStage5Params {
                        len: chunk_len as u32,
                        offset: stage_offset as u32,
                        total: total_out as u32,
                        target_len: target_len as u32,
                        inner_stride: inner_stride as u32,
                        stage_span: stage_span as u32,
                        stage_fifth: stage_fifth as u32,
                        twiddle_step: twiddle_step as u32,
                        inverse: if inverse { 1 } else { 0 },
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.fft_dispatch_exec("runmat-fftm-stage5-encoder", |emission| {
                        let params_buffer = self.fft_uniform_buffer(
                            emission,
                            &params,
                            "runmat-fftm-stage5-params",
                        )?;
                        let bind_group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-fftm-stage5-bind"),
                                    layout: &self.pipelines.fft_stage5.get(emission)?.layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: stage_a.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: stage_b.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: twiddle.as_ref().as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 3,
                                            resource: params_buffer.as_entire_binding(),
                                        },
                                    ],
                                });
                        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                            chunk_len as u32,
                            crate::backend::wgpu::config::WORKGROUP_SIZE,
                        );
                        Ok(FftPreparedDispatch::new(
                            &self.pipelines.fft_stage5.get(emission)?.pipeline,
                            bind_group,
                            workgroups,
                            FftDispatchResources::new(
                                &[input_handle, &output_handle],
                                &[&stage_a, &stage_b, &twiddle, &params_buffer],
                            )?,
                        ))
                    })
                    .await?;
                    stage_offset += chunk_len;
                }
                self.fft_debug_dump_scalar_buffer("mixed:after_stage5", &stage_b, out_scalar_len);
            } else {
                return Ok(None);
            }

            std::mem::swap(&mut stage_a, &mut stage_b);
            stage_span /= radix as usize;
        }

        let mut radices = [0u32; 16];
        for (i, &r) in factors.iter().enumerate() {
            radices[i] = r;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorderMixedParams {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                stage_count: factors.len() as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
                radices,
            };
            self.fft_dispatch_exec("runmat-fftm-reorder-encoder", |emission| {
                let params_buffer =
                    self.fft_storage_param_buffer(emission, &params, "runmat-fftm-reorder-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fftm-reorder-bind"),
                        layout: &self.pipelines.fft_reorder_mixed.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_reorder_mixed.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &output_handle],
                        &[&stage_a, &stage_b, &params_buffer],
                    )?,
                ))
            })
            .await?;
            reorder_offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("mixed:after_reorder", &stage_b, out_scalar_len);

        Ok(Some(output_handle))
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_fft_dim_exec_native_radix3(
        &self,
        input_handle: &GpuTensorHandle,
        in_buffer: runmat_accelerate_api::WgpuBufferOwner,
        mut shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
        digits: u32,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let (mut stage_a, mut stage_b) = self.with_emission(|emission, _| {
            self.fft_stage_buffer_pair(
                emission,
                out_scalar_len,
                "runmat-fft3-stage-a",
                "runmat-fft3-stage-b",
            )
        })??;
        shape[dim] = target_len;
        fft_trim_trailing_ones(&mut shape, origin_rank.max(dim + 1));
        // Each stage swaps A/B; reorder writes the opposite physical owner.
        let final_buffer = if digits.is_multiple_of(2) {
            &stage_b
        } else {
            &stage_a
        };
        let output_handle = self.register_existing_buffer_with_storage(
            final_buffer.clone(),
            shape,
            out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let twiddle = self.with_emission(|emission, _| {
            self.fft_twiddle_buffer(emission, target_len, false, "runmat-fft3-twiddles")
        })??;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            self.fft_dispatch_exec("runmat-fft3-init-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft3-init-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft3-init-bind"),
                        layout: &self.pipelines.fft_init.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: in_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_init.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &output_handle],
                        &[&in_buffer, &stage_a, &params_buffer],
                    )?,
                ))
            })
            .await?;
            offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("radix3:init", &stage_a, out_scalar_len);

        let mut stage_span = target_len;
        while stage_span >= 3 {
            let stage_third = stage_span / 3;
            let twiddle_step = target_len / stage_span;
            let mut stage_offset = 0usize;
            while stage_offset < total_out {
                let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                let params = crate::backend::wgpu::params::FftStage3Params {
                    len: chunk_len as u32,
                    offset: stage_offset as u32,
                    total: total_out as u32,
                    target_len: target_len as u32,
                    inner_stride: inner_stride as u32,
                    stage_span: stage_span as u32,
                    stage_third: stage_third as u32,
                    twiddle_step: twiddle_step as u32,
                    inverse: if inverse { 1 } else { 0 },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                if Self::fft_debug_enabled() {
                    eprintln!(
                        "[fft-debug] stage3 params span={} third={} step={} chunk={} off={}",
                        params.stage_span,
                        params.stage_third,
                        params.twiddle_step,
                        params.len,
                        params.offset
                    );
                }
                self.fft_dispatch_exec("runmat-fft3-stage-encoder", |emission| {
                    let params_buffer =
                        self.fft_uniform_buffer(emission, &params, "runmat-fft3-stage-params")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fft3-stage-bind"),
                                layout: &self.pipelines.fft_stage3.get(emission)?.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: stage_a.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: stage_b.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: twiddle.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                        chunk_len as u32,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                    );
                    if Self::fft_debug_enabled() {
                        eprintln!(
                            "[fft-debug] stage3 dispatch workgroups={workgroups} chunk={chunk_len}"
                        );
                    }
                    Ok(FftPreparedDispatch::new(
                        &self.pipelines.fft_stage3.get(emission)?.pipeline,
                        bind_group,
                        workgroups,
                        FftDispatchResources::new(
                            &[input_handle, &output_handle],
                            &[&stage_a, &stage_b, &twiddle, &params_buffer],
                        )?,
                    ))
                })
                .await?;
                stage_offset += chunk_len;
            }
            std::mem::swap(&mut stage_a, &mut stage_b);
            self.fft_debug_dump_scalar_buffer("stage3:after_stage", &stage_a, out_scalar_len);
            stage_span /= 3;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorder3Params {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                digits,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            self.fft_dispatch_exec("runmat-fft3-reorder-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft3-reorder-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft3-reorder-bind"),
                        layout: &self.pipelines.fft_reorder3.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_reorder3.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &output_handle],
                        &[&stage_a, &stage_b, &params_buffer],
                    )?,
                ))
            })
            .await?;
            reorder_offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("stage3:after_reorder", &stage_b, out_scalar_len);

        Ok(Some(output_handle))
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_fft_dim_exec_native_radix5(
        &self,
        input_handle: &GpuTensorHandle,
        in_buffer: runmat_accelerate_api::WgpuBufferOwner,
        mut shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
        digits: u32,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let (mut stage_a, mut stage_b) = self.with_emission(|emission, _| {
            self.fft_stage_buffer_pair(
                emission,
                out_scalar_len,
                "runmat-fft5-stage-a",
                "runmat-fft5-stage-b",
            )
        })??;
        shape[dim] = target_len;
        fft_trim_trailing_ones(&mut shape, origin_rank.max(dim + 1));
        // Each stage swaps A/B; reorder writes the opposite physical owner.
        let final_buffer = if digits.is_multiple_of(2) {
            &stage_b
        } else {
            &stage_a
        };
        let output_handle = self.register_existing_buffer_with_storage(
            final_buffer.clone(),
            shape,
            out_scalar_len,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        let twiddle = self.with_emission(|emission, _| {
            self.fft_twiddle_buffer(emission, target_len, false, "runmat-fft5-twiddles")
        })??;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            self.fft_dispatch_exec("runmat-fft5-init-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft5-init-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft5-init-bind"),
                        layout: &self.pipelines.fft_init.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: in_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_init.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &output_handle],
                        &[&in_buffer, &stage_a, &params_buffer],
                    )?,
                ))
            })
            .await?;
            offset += chunk_len;
        }

        let mut stage_span = target_len;
        while stage_span >= 5 {
            let stage_fifth = stage_span / 5;
            let twiddle_step = target_len / stage_span;
            let mut stage_offset = 0usize;
            while stage_offset < total_out {
                let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                let params = crate::backend::wgpu::params::FftStage5Params {
                    len: chunk_len as u32,
                    offset: stage_offset as u32,
                    total: total_out as u32,
                    target_len: target_len as u32,
                    inner_stride: inner_stride as u32,
                    stage_span: stage_span as u32,
                    stage_fifth: stage_fifth as u32,
                    twiddle_step: twiddle_step as u32,
                    inverse: if inverse { 1 } else { 0 },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                self.fft_dispatch_exec("runmat-fft5-stage-encoder", |emission| {
                    let params_buffer =
                        self.fft_uniform_buffer(emission, &params, "runmat-fft5-stage-params")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fft5-stage-bind"),
                                layout: &self.pipelines.fft_stage5.get(emission)?.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: stage_a.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: stage_b.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: twiddle.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                        chunk_len as u32,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                    );
                    Ok(FftPreparedDispatch::new(
                        &self.pipelines.fft_stage5.get(emission)?.pipeline,
                        bind_group,
                        workgroups,
                        FftDispatchResources::new(
                            &[input_handle, &output_handle],
                            &[&stage_a, &stage_b, &twiddle, &params_buffer],
                        )?,
                    ))
                })
                .await?;
                stage_offset += chunk_len;
            }
            std::mem::swap(&mut stage_a, &mut stage_b);
            stage_span /= 5;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorder5Params {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                digits,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            self.fft_dispatch_exec("runmat-fft5-reorder-encoder", |emission| {
                let params_buffer =
                    self.fft_uniform_buffer(emission, &params, "runmat-fft5-reorder-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft5-reorder-bind"),
                        layout: &self.pipelines.fft_reorder5.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                Ok(FftPreparedDispatch::new(
                    &self.pipelines.fft_reorder5.get(emission)?.pipeline,
                    bind_group,
                    workgroups,
                    FftDispatchResources::new(
                        &[input_handle, &output_handle],
                        &[&stage_a, &stage_b, &params_buffer],
                    )?,
                ))
            })
            .await?;
            reorder_offset += chunk_len;
        }

        Ok(Some(output_handle))
    }
}
