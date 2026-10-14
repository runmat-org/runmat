use anyhow::{anyhow, ensure, Result};

#[cfg(test)]
#[path = "polynomial/admission_tests.rs"]
mod admission_tests;
use bytemuck::cast_slice;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderConv1dOptions, ProviderConvMode,
    ProviderPolyderQuotient, ProviderPolyfitOutputs, ProviderPolyfitResult, ProviderPolyvalOptions,
};
use runmat_runtime::builtins::math::poly::polyfit::polyfit_host_real_for_provider;

use super::{
    conv_orientation_for, polynomial_orientation, shape_for_orientation, trim_leading_zeros_real,
    NumericPrecision, PolyderParams, PolyintParamsF32, PolyintParamsF64, PolynomialOrientation,
    WgpuProvider,
};

impl WgpuProvider {
    pub(crate) async fn polyfit_exec(
        &self,
        x: &GpuTensorHandle,
        y: &GpuTensorHandle,
        degree: usize,
        weights: Option<&GpuTensorHandle>,
        outputs: ProviderPolyfitOutputs,
    ) -> Result<ProviderPolyfitResult> {
        // Authenticate every borrowed operand before storage eligibility or work.
        let entries = [Some(x), Some(y), weights]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .iter()
            .map(|input| self.get_entry_raw(input))
            .collect::<Result<Vec<_>>>()?;
        if entries
            .iter()
            .any(|entry| entry.integer_type().is_some() || entry.storage != GpuTensorStorage::Real)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "polyfit_exec: kernel cannot consume this physical storage",
            ));
        }

        let x_host = self.download_exec(x).await?;
        let y_host = self.download_exec(y).await?;
        ensure!(
            x_host.data.len() == y_host.data.len(),
            "polyfit: X and Y vectors must match in length"
        );
        let weights_host = match weights {
            Some(handle) => Some(self.download_exec(handle).await?),
            None => None,
        };
        let weights_slice = weights_host.as_ref().map(|w| w.data.as_slice());
        let host_result = polyfit_host_real_for_provider(
            &x_host.data,
            &y_host.data,
            degree,
            weights_slice,
            outputs,
        )
        .map_err(|err| anyhow!(err))?;
        Ok(ProviderPolyfitResult {
            coefficients: host_result.coefficients,
            r_matrix: host_result.r_matrix,
            normr: host_result.normr,
            df: host_result.df,
            mu: host_result.mu,
        })
    }

    pub(crate) fn polyval_exec(
        &self,
        coeffs: &GpuTensorHandle,
        points: &GpuTensorHandle,
        options: &ProviderPolyvalOptions,
    ) -> Result<GpuTensorHandle> {
        // Authenticate every borrowed operand before storage eligibility or work.
        let entries = [coeffs, points]
            .iter()
            .map(|input| self.get_entry_raw(input))
            .collect::<Result<Vec<_>>>()?;
        if entries
            .iter()
            .any(|entry| entry.integer_type().is_some() || entry.storage != GpuTensorStorage::Real)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "polyval_exec: kernel cannot consume this physical storage",
            ));
        }

        let coeff_entry = self.get_entry(coeffs)?;
        let points_entry = self.get_entry(points)?;

        ensure!(
            coeff_entry.precision == self.precision && points_entry.precision == self.precision,
            "polyval: precision mismatch between tensors and provider"
        );
        ensure!(
            coeff_entry.len > 0,
            "polyval: coefficient vector must contain at least one element"
        );

        let len = points_entry.len;
        let shape = points_entry.shape.clone();
        if len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-polyval-out")
            })??;
            return self.register_existing_buffer(out_buffer, shape, 0);
        }

        ensure!(
            len <= u32::MAX as usize,
            "polyval: evaluation tensor exceeds GPU limits"
        );
        ensure!(
            coeff_entry.len <= u32::MAX as usize,
            "polyval: coefficient vector exceeds GPU limits"
        );

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-polyval-warmup"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-polyval-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.polyval.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;

        self.access.poll(wgpu::Maintain::Poll);
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-polyval-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

            let out_buffer =
                self.create_storage_buffer_checked(emission, len, "runmat-polyval-out")?;
            let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
                * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
            let (mu_mean, mu_scale) = options.mu.map(|m| (m.mean, m.scale)).unwrap_or((0.0, 1.0));
            let has_mu = options.mu.is_some() as u32;

            let mut offset = 0usize;
            while offset < len {
                let remaining = len - offset;
                let chunk_len = remaining.min(chunk_capacity).max(1);
                let params_buffer = match self.precision {
                    NumericPrecision::F64 => {
                        let params = crate::backend::wgpu::params::PolyvalParamsF64 {
                            len: chunk_len as u32,
                            coeff_len: coeff_entry.len as u32,
                            offset: offset as u32,
                            has_mu,
                            mu_mean,
                            mu_scale,
                        };
                        self.uniform_buffer(emission, &params, "runmat-polyval-params-f64")?
                    }
                    NumericPrecision::F32 => {
                        let params = crate::backend::wgpu::params::PolyvalParamsF32 {
                            len: chunk_len as u32,
                            coeff_len: coeff_entry.len as u32,
                            offset: offset as u32,
                            has_mu,
                            mu_mean: mu_mean as f32,
                            mu_scale: mu_scale as f32,
                            _pad0: 0,
                            _pad1: 0,
                        };
                        self.uniform_buffer(emission, &params, "runmat-polyval-params-f32")?
                    }
                };

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-polyval-bind"),
                        layout: &self.pipelines.polyval.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: coeff_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: points_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buffer.as_ref().as_entire_binding(),
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

                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &self.pipelines.polyval.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;

                offset += chunk_len;
            }

            Ok(out_buffer)
        })??;
        self.register_existing_buffer(out_buffer, shape, len)
    }

    pub(crate) fn polyint_exec(
        &self,
        polynomial: &GpuTensorHandle,
        constant: f64,
    ) -> Result<GpuTensorHandle> {
        // Authenticate every borrowed operand before storage eligibility or work.
        let entries = [polynomial]
            .iter()
            .map(|input| self.get_entry_raw(input))
            .collect::<Result<Vec<_>>>()?;
        if entries.iter().any(|entry| entry.integer_type().is_some()) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "polyint_exec: kernel cannot consume this physical storage",
            ));
        }

        let entry = self.get_entry(polynomial)?;
        ensure!(
            entry.precision == self.precision,
            "polyint: precision mismatch between tensor and provider"
        );
        let orientation = polynomial_orientation(&entry.shape)?;
        let storage = if runmat_accelerate_api::handle_storage(polynomial)
            == GpuTensorStorage::ComplexInterleaved
        {
            GpuTensorStorage::ComplexInterleaved
        } else {
            entry.storage
        };
        let storage_factor = match storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        ensure!(
            entry.len.is_multiple_of(storage_factor),
            "polyint: storage length does not match tensor storage"
        );
        let input_len = entry.len / storage_factor;

        if entry.len == 0 {
            let shape = shape_for_orientation(orientation, 1);
            let output_len = storage_factor;
            let buffer = match self.precision {
                NumericPrecision::F64 => {
                    let values = if storage_factor == 1 {
                        vec![constant]
                    } else {
                        vec![constant, 0.0]
                    };
                    self.with_emission(|emission, _| {
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyint-const-f64"),
                                contents: cast_slice(&values),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    })?
                }
                NumericPrecision::F32 => {
                    let values = if storage_factor == 1 {
                        vec![constant as f32]
                    } else {
                        vec![constant as f32, 0.0]
                    };
                    self.with_emission(|emission, _| {
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyint-const-f32"),
                                contents: cast_slice(&values),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    })?
                }
            };
            return self.register_existing_buffer_with_storage(buffer, shape, output_len, storage);
        }

        ensure!(
            input_len <= u32::MAX as usize,
            "polyint: polynomial length exceeds GPU limits"
        );

        let output_logical_len = input_len + 1;
        let output_len = output_logical_len
            .checked_mul(storage_factor)
            .ok_or_else(|| anyhow!("polyint: output length exceeds GPU limits"))?;
        ensure!(
            output_len <= u32::MAX as usize,
            "polyint: output length exceeds GPU limits"
        );
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, output_len, "runmat-polyint-out")?;
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = PolyintParamsF64 {
                        input_len: input_len as u32,
                        output_len: output_len as u32,
                        storage_factor: storage_factor as u32,
                        _pad0: 0,
                        constant,
                    };
                    self.uniform_buffer(emission, &params, "runmat-polyint-params-f64")?
                }
                NumericPrecision::F32 => {
                    let params = PolyintParamsF32 {
                        input_len: input_len as u32,
                        output_len: output_len as u32,
                        storage_factor: storage_factor as u32,
                        _pad0: 0,
                        constant: constant as f32,
                        _pad1: 0.0,
                        _pad2: 0.0,
                        _pad3: 0.0,
                    };
                    self.uniform_buffer(emission, &params, "runmat-polyint-params-f32")?
                }
            };

            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-polyint-bind"),
                    layout: &self.pipelines.polyint.get(emission)?.layout,
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
                output_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-polyint-encoder"),
                },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-polyint-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.polyint.get(emission)?.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.submit(emission, pending, encoder)?;
            Ok(out_buffer)
        })??;

        let out_shape = shape_for_orientation(orientation, output_logical_len);
        self.register_existing_buffer_with_storage(out_buffer, out_shape, output_len, storage)
    }
    pub(crate) async fn polyder_exec(
        &self,
        polynomial: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        // Authenticate every borrowed operand before storage eligibility or work.
        let entries = [polynomial]
            .iter()
            .map(|input| self.get_entry_raw(input))
            .collect::<Result<Vec<_>>>()?;
        if entries
            .iter()
            .any(|entry| entry.integer_type().is_some() || entry.storage != GpuTensorStorage::Real)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "polyder_exec: kernel cannot consume this physical storage",
            ));
        }

        let entry = self.get_entry(polynomial)?;
        ensure!(
            entry.precision == self.precision,
            "polyder: precision mismatch between tensor and provider"
        );
        let orientation = polynomial_orientation(&entry.shape)?;
        if entry.len <= 1 {
            let shape = shape_for_orientation(orientation, 1);
            let buffer = match self.precision {
                NumericPrecision::F64 => self.with_emission(|emission, _| {
                    runmat_accelerate_api::WgpuBufferOwner::new(emission.create_initialized_buffer(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("runmat-polyder-const-f64"),
                            contents: cast_slice(&[0.0f64]),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ))
                })?,
                NumericPrecision::F32 => {
                    let zeros: [f32; 1] = [0.0];
                    self.with_emission(|emission, _| {
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyder-const-f32"),
                                contents: cast_slice(&zeros),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    })?
                }
            };
            return self.register_existing_buffer(buffer, shape, 1);
        }

        ensure!(
            entry.len <= u32::MAX as usize,
            "polyder: polynomial length exceeds GPU limits"
        );
        let output_len = entry.len - 1;
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, output_len, "runmat-polyder-out")?;
            let params = PolyderParams {
                input_len: entry.len as u32,
                output_len: output_len as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-polyder-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-polyder-bind"),
                    layout: &self.pipelines.polyder.get(emission)?.layout,
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                params.output_len,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-polyder-encoder"),
                },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-polyder-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.polyder.get(emission)?.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.submit(emission, pending, encoder)?;
            Ok(out_buffer)
        })??;

        let out_shape = shape_for_orientation(orientation, output_len);
        let handle = self.register_existing_buffer(out_buffer, out_shape, output_len)?;
        self.trim_polynomial_handle(&handle, orientation).await
    }

    pub(crate) async fn polyder_product_exec(
        &self,
        p: &GpuTensorHandle,
        q: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        // Authenticate every borrowed operand before storage eligibility or work.
        let entries = [p, q]
            .iter()
            .map(|input| self.get_entry_raw(input))
            .collect::<Result<Vec<_>>>()?;
        if entries
            .iter()
            .any(|entry| entry.integer_type().is_some() || entry.storage != GpuTensorStorage::Real)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "polyder_product_exec: kernel cannot consume this physical storage",
            ));
        }

        let p_entry = self.get_entry(p)?;
        let q_entry = self.get_entry(q)?;
        ensure!(
            p_entry.precision == self.precision && q_entry.precision == self.precision,
            "polyder: precision mismatch between tensors and provider"
        );
        let orientation = polynomial_orientation(&p_entry.shape)?;
        let conv_orientation = conv_orientation_for(orientation);

        let dp = self.polyder_exec(p).await?;
        let dq = self.polyder_exec(q).await?;
        let options = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation,
        };
        let term1 = self.conv1d_exec(&dp, q, options)?;
        let term2 = self.conv1d_exec(p, &dq, options)?;
        let result = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &term1,
            &term2,
        )?;
        self.trim_polynomial_handle(&result, orientation).await
    }

    pub(crate) async fn polyder_quotient_exec(
        &self,
        u: &GpuTensorHandle,
        v: &GpuTensorHandle,
    ) -> Result<ProviderPolyderQuotient> {
        // Authenticate every borrowed operand before storage eligibility or work.
        let entries = [u, v]
            .iter()
            .map(|input| self.get_entry_raw(input))
            .collect::<Result<Vec<_>>>()?;
        if entries
            .iter()
            .any(|entry| entry.integer_type().is_some() || entry.storage != GpuTensorStorage::Real)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "polyder_quotient_exec: kernel cannot consume this physical storage",
            ));
        }

        let u_entry = self.get_entry(u)?;
        let v_entry = self.get_entry(v)?;
        ensure!(
            u_entry.precision == self.precision && v_entry.precision == self.precision,
            "polyder: precision mismatch between tensors and provider"
        );
        let orientation_u = polynomial_orientation(&u_entry.shape)?;
        let orientation_v = polynomial_orientation(&v_entry.shape)?;
        let options_num = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation_for(orientation_u),
        };
        let options_den = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation_for(orientation_v),
        };

        let du = self.polyder_exec(u).await?;
        let dv = self.polyder_exec(v).await?;
        let term1 = self.conv1d_exec(&du, v, options_num)?;
        let term2 = self.conv1d_exec(u, &dv, options_num)?;
        let numerator_handle = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &term1,
            &term2,
        )?;
        let denominator_handle = self.conv1d_exec(v, v, options_den)?;

        let numerator = self
            .trim_polynomial_handle(&numerator_handle, orientation_u)
            .await?;
        let denominator = self
            .trim_polynomial_handle(&denominator_handle, orientation_v)
            .await?;
        Ok(ProviderPolyderQuotient {
            numerator,
            denominator,
        })
    }

    async fn trim_polynomial_handle(
        &self,
        handle: &GpuTensorHandle,
        orientation: PolynomialOrientation,
    ) -> Result<GpuTensorHandle> {
        let host = self.download_exec(handle).await?;
        let trimmed = trim_leading_zeros_real(&host.data);
        if trimmed.len() == host.data.len() {
            return Ok(handle.clone());
        }
        let shape_vec = shape_for_orientation(orientation, trimmed.len());
        let new_handle = if trimmed.is_empty() {
            self.register_existing_buffer(
                self.with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-polyder-trim-empty")
                })??,
                shape_vec,
                0,
            )?
        } else {
            match self.precision {
                NumericPrecision::F64 => {
                    let buffer = self.with_emission(|emission, _| {
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyder-trim-f64"),
                                contents: cast_slice(trimmed.as_slice()),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    })?;
                    self.register_existing_buffer(buffer, shape_vec, trimmed.len())?
                }
                NumericPrecision::F32 => {
                    let data_f32: Vec<f32> = trimmed.iter().map(|v| *v as f32).collect();
                    let buffer = self.with_emission(|emission, _| {
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyder-trim-f32"),
                                contents: cast_slice(&data_f32),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    })?;
                    self.register_existing_buffer(buffer, shape_vec, trimmed.len())?
                }
            }
        };
        Ok(new_handle)
    }
}
