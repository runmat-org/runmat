use super::*;

impl WgpuProvider {
    pub(crate) fn issymmetric_exec(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderSymmetryKind,
        tolerance: f64,
    ) -> Result<bool> {
        let physical = self.get_entry_raw(matrix)?;
        if physical.storage != runmat_accelerate_api::GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "issymmetric_exec: real kernel cannot consume complex storage",
            ));
        }

        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_symmetry_shape(&entry.shape).map_err(|e| anyhow!("issymmetric: {e}"))?;
        if rows != cols {
            return Ok(false);
        }
        if rows == 0 || cols == 0 {
            return Ok(true);
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("issymmetric: matrix dimensions too large"))?;
        if total > entry.len {
            return Err(anyhow!(
                "issymmetric: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("issymmetric: matrix exceeds GPU limits"));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "issymmetric: tolerance must be finite and non-negative"
            ));
        }

        self.require_ieee_words(entry.precision)?;

        let mode = match kind {
            ProviderSymmetryKind::Symmetric => 0u32,
            ProviderSymmetryKind::Skew => 1u32,
        };

        let (staging, staging_size) = self.with_emission(|emission, pending| -> Result<_> {
            let output_init = [1u32];
            let output_buffer =
                emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-issymmetric-output"),
                    contents: cast_slice(&output_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

            let pipeline = self.pipelines.symmetry.get(emission)?;
            match entry.precision {
                NumericPrecision::F64 => {
                    let params = SymmetryParamsF64 {
                        rows: rows as u32,
                        cols: cols as u32,
                        len: total as u32,
                        mode,
                        tolerance,
                        _pad: 0.0,
                    };
                    let params_buffer =
                        self.uniform_buffer(emission, &params, "runmat-issymmetric-params-f64")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-issymmetric-bind-group-f64"),
                                layout: &pipeline.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: entry.buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: output_buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let groups =
                        crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        &pipeline.pipeline,
                        &bind_group,
                        groups,
                    )?;
                }
                NumericPrecision::F32 => {
                    let tol32 = tolerance.min(f32::MAX as f64).max(0.0) as f32;
                    let params = SymmetryParamsF32 {
                        rows: rows as u32,
                        cols: cols as u32,
                        len: total as u32,
                        mode,
                        tolerance: tol32,
                        _pad: [0.0; 3],
                    };
                    let params_buffer =
                        self.uniform_buffer(emission, &params, "runmat-issymmetric-params-f32")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-issymmetric-bind-group-f32"),
                                layout: &pipeline.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: entry.buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: output_buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let groups =
                        crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        &pipeline.pipeline,
                        &bind_group,
                        groups,
                    )?;
                }
            }

            let staging_size = std::mem::size_of::<u32>() as u64;
            let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-issymmetric-staging"),
                    size: staging_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-issymmetric-copy"),
                },
            );
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
            self.submit(emission, pending, encoder)?;

            Ok((staging, staging_size))
        })??;
        let bytes = self.map_readback_bytes_sync(staging, staging_size, "issymmetric")?;
        let words: &[u32] = cast_slice(&bytes);
        let flag = words.first().copied().unwrap_or(0);
        Ok(flag != 0)
    }

    pub(crate) async fn ishermitian_exec(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> Result<bool> {
        let physical = self.get_entry_raw(matrix)?;
        if physical.storage != runmat_accelerate_api::GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "ishermitian_exec: real kernel cannot consume complex storage",
            ));
        }

        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "ishermitian: tolerance must be finite and non-negative"
            ));
        }
        let host = self.download_exec(matrix).await?;
        let skew = matches!(kind, ProviderHermitianKind::Skew);
        ishermitian_host_real_data(&host.shape, &host.data, skew, tolerance).map_err(|e| anyhow!(e))
    }

    pub(crate) async fn sym_rcm_exec(&self, matrix: &GpuTensorHandle) -> Result<Vec<usize>> {
        let physical = self.get_entry_raw(matrix)?;
        if physical.storage != runmat_accelerate_api::GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "sym_rcm_exec: real kernel cannot consume complex storage",
            ));
        }

        let host = self.download_exec(matrix).await?;
        symrcm_host_real_data(&host.shape, &host.data).map_err(|e| anyhow!(e))
    }

    pub(crate) fn bandwidth_exec(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        let physical = self.get_entry_raw(matrix)?;
        if physical.storage != runmat_accelerate_api::GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "bandwidth_exec: real kernel cannot consume complex storage",
            ));
        }

        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_bandwidth_shape(&entry.shape).map_err(|e| anyhow!("bandwidth: {e}"))?;
        if rows == 0 || cols == 0 {
            return Ok(ProviderBandwidth { lower: 0, upper: 0 });
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("bandwidth: matrix dimensions too large"))?;
        if total == 0 {
            return Ok(ProviderBandwidth { lower: 0, upper: 0 });
        }
        if total > entry.len {
            return Err(anyhow!(
                "bandwidth: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("bandwidth: matrix exceeds GPU limits"));
        }

        let (staging, staging_size) = self.with_emission(|emission, pending| -> Result<_> {
            let pipeline = self.pipelines.bandwidth.get(emission)?;
            let output_init = [0u32, 0u32];
            let output_buffer =
                emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-bandwidth-output"),
                    contents: cast_slice(&output_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let params = BandwidthParams {
                rows: rows as u32,
                cols: cols as u32,
                len: total as u32,
                _pad: 0,
            };
            let params_buffer =
                self.uniform_buffer(emission, &params, "runmat-bandwidth-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-bandwidth-bind-group"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
            crate::backend::wgpu::dispatch::elementwise::run(
                emission,
                pending,
                &pipeline.pipeline,
                &bind_group,
                groups,
            )?;

            let staging_size = (std::mem::size_of::<u32>() * 2) as u64;
            let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-bandwidth-staging"),
                    size: staging_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-bandwidth-copy"),
                },
            );
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
            self.submit(emission, pending, encoder)?;

            Ok((staging, staging_size))
        })??;
        let bytes = self.map_readback_bytes_sync(staging, staging_size, "bandwidth")?;
        let words: &[u32] = cast_slice(&bytes);
        let lower = words.first().copied().unwrap_or(0);
        let upper = words.get(1).copied().unwrap_or(0);

        Ok(ProviderBandwidth { lower, upper })
    }
}
