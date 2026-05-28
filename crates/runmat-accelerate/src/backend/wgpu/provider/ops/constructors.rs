use super::*;

impl WgpuProvider {
    pub(crate) fn eye_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let normalized = normalize_eye_shape(shape);
        if normalized.len() < 2 {
            return Err(anyhow!("eye: expected at least 2 dimensions"));
        }
        let total_len = product_checked(&normalized)
            .ok_or_else(|| anyhow!("eye: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-eye-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized, total_len));
        }

        let rows = normalized[0];
        let cols = normalized[1];
        let diag_len = rows.min(cols);
        if diag_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized, total_len));
        }
        let slice_stride = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("eye: matrix slice exceeds GPU limits"))?;
        let slices = if normalized.len() <= 2 {
            1usize
        } else {
            product_checked(&normalized[2..])
                .ok_or_else(|| anyhow!("eye: slice count exceeds GPU limits"))?
        };
        let diag_total = diag_len
            .checked_mul(slices)
            .ok_or_else(|| anyhow!("eye: diagonal count exceeds GPU limits"))?;
        if diag_total == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized, total_len));
        }
        if rows > (u32::MAX as usize)
            || cols > (u32::MAX as usize)
            || slice_stride > (u32::MAX as usize)
            || diag_total > (u32::MAX as usize)
        {
            return Err(anyhow!("eye: dimensions exceed GPU dispatch limits"));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-eye-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-eye-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.eye.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let params = crate::backend::wgpu::params::EyeParams {
            rows: rows as u32,
            cols: cols as u32,
            diag_len: diag_len as u32,
            slices: slices as u32,
            stride_slice: slice_stride as u32,
            diag_total: diag_total as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-eye-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-eye-bind"),
                layout: &self.pipelines.eye.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            diag_total as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.eye.pipeline,
            &bind_group,
            workgroups,
            "runmat-eye-encoder",
            "runmat-eye-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, normalized, total_len))
    }
    pub(crate) fn fill_exec(&self, shape: &[usize], value: f64) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("fill: tensor size exceeds GPU limits"))?;
        let shape_vec = shape.to_vec();
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-fill-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "fill: tensor length exceeds GPU dispatch limits"
        );

        // chunked dispatch below will build per-chunk params buffers

        // Dispatch in chunks to satisfy per-dimension group limits (<= 65535)
        let wg_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let max_groups: u32 = 65535;
        let max_elems_per_dispatch = (max_groups as usize) * (wg_size as usize);
        let mut processed: usize = 0;
        while processed < total_len {
            let remain = total_len - processed;
            let chunk_len = remain.min(max_elems_per_dispatch);

            // Per-chunk params (length)
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::FillParamsF64 {
                        value,
                        len: chunk_len as u32,
                        _pad: [0, 0, 0],
                    };
                    self.uniform_buffer(&params, "runmat-fill-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::FillParamsF32 {
                        value: value as f32,
                        len: chunk_len as u32,
                        _pad: [0, 0],
                    };
                    self.uniform_buffer(&params, "runmat-fill-params-f32")
                }
            };

            // Bind only the range for this chunk
            let byte_offset = (processed * self.element_size) as u64;
            let byte_size = std::num::NonZeroU64::new((chunk_len * self.element_size) as u64);
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fill-bind"),
                    layout: &self.pipelines.fill.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: out_buffer.as_ref(),
                                offset: byte_offset,
                                size: byte_size,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups =
                crate::backend::wgpu::dispatch::common::dispatch_size(chunk_len as u32, wg_size);
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fill.pipeline,
                &bind_group,
                workgroups,
                "runmat-fill-encoder",
                "runmat-fill-pass",
            );

            processed += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape_vec, total_len))
    }

    pub(crate) fn zeros_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let buffer = self.create_storage_buffer_checked(len, "zeros")?;
        // Explicitly zero-initialize the storage buffer; pooled buffers may contain old data
        let size_bytes = (len.max(1) as u64) * (self.element_size as u64);
        if size_bytes > 0 {
            // Write zeros across the entire buffer
            let zero_bytes = vec![0u8; size_bytes as usize];
            self.queue.write_buffer(buffer.as_ref(), 0, &zero_bytes);
        }
        Ok(self.register_existing_buffer(buffer, shape.to_vec(), len))
    }

    pub(crate) fn meshgrid_exec(
        &self,
        axes: &[MeshgridAxisView<'_>],
    ) -> Result<ProviderMeshgridResult> {
        ensure!(
            axes.len() == 2 || axes.len() == 3,
            "meshgrid: provider expects two or three axes"
        );

        let x_axis = axes
            .first()
            .ok_or_else(|| anyhow!("meshgrid: missing X axis"))?
            .data;
        let y_axis = axes
            .get(1)
            .ok_or_else(|| anyhow!("meshgrid: missing Y axis"))?
            .data;
        let z_axis = axes.get(2).map(|axis| axis.data);

        let nx = x_axis.len();
        let ny = y_axis.len();
        let nz = z_axis.map(|axis| axis.len()).unwrap_or(1);

        let shape = if nz == 1 {
            vec![ny, nx]
        } else {
            vec![ny, nx, nz]
        };

        let total = product_checked(&shape)
            .ok_or_else(|| anyhow!("meshgrid: tensor size exceeds GPU limits"))?;

        let mut x_data = Vec::with_capacity(total);
        let mut y_data = Vec::with_capacity(total);
        let mut z_data = z_axis.map(|_| Vec::with_capacity(total));

        if let Some(axis) = z_axis {
            for &z_value in axis.iter().take(nz) {
                for &x_value in x_axis.iter().take(nx) {
                    for &y_value in y_axis.iter().take(ny) {
                        x_data.push(x_value);
                        y_data.push(y_value);
                        if let Some(ref mut z_vec) = z_data {
                            z_vec.push(z_value);
                        }
                    }
                }
            }
        } else {
            for &x_value in x_axis.iter().take(nx) {
                for &y_value in y_axis.iter().take(ny) {
                    x_data.push(x_value);
                    y_data.push(y_value);
                }
            }
        }

        let shape_slice = &shape;
        let x_view = HostTensorView {
            data: &x_data,
            shape: shape_slice,
        };
        let y_view = HostTensorView {
            data: &y_data,
            shape: shape_slice,
        };
        let x_handle = self.upload_exec(&x_view)?;
        let y_handle = self.upload_exec(&y_view)?;

        let mut outputs = vec![x_handle, y_handle];

        if let Some(z_vec) = z_data {
            let z_view = HostTensorView {
                data: &z_vec,
                shape: shape_slice,
            };
            let z_handle = self.upload_exec(&z_view)?;
            outputs.push(z_handle);
        }

        Ok(ProviderMeshgridResult { outputs })
    }
    pub(crate) fn fspecial_exec(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        let spec =
            runtime_fspecial_spec_from_request(&request.filter).map_err(|err| anyhow!(err))?;

        let (rows, cols, kind, sigma, alpha, norm, center_x, center_y) = match &spec {
            FspecialFilterSpec::Average { rows, cols } => (
                *rows,
                *cols,
                0u32,
                0.0,
                0.0,
                1.0 / ((*rows as f64) * (*cols as f64)),
                0.0,
                0.0,
            ),
            FspecialFilterSpec::Gaussian { rows, cols, sigma } => {
                let norm = gaussian_normalizer(*rows, *cols, *sigma);
                ensure!(
                    norm.is_finite() && norm > 0.0,
                    "fspecial: gaussian normaliser invalid"
                );
                (
                    *rows,
                    *cols,
                    1u32,
                    *sigma,
                    0.0,
                    norm,
                    ((*cols as f64) - 1.0) / 2.0,
                    ((*rows as f64) - 1.0) / 2.0,
                )
            }
            FspecialFilterSpec::Laplacian { alpha } => {
                let norm = 4.0 / (alpha + 1.0);
                (3, 3, 2u32, 0.0, *alpha, norm, 0.0, 0.0)
            }
            FspecialFilterSpec::Prewitt => (3, 3, 3u32, 0.0, 0.0, 1.0, 0.0, 0.0),
            FspecialFilterSpec::Sobel => (3, 3, 4u32, 0.0, 0.0, 1.0, 0.0, 0.0),
            FspecialFilterSpec::Unsharp { alpha } => {
                let norm = 1.0 / (alpha + 1.0);
                (3, 3, 5u32, 0.0, *alpha, norm, 0.0, 0.0)
            }
            _ => {
                return Err(anyhow!(
                    "fspecial: filter not yet accelerated on the WGPU backend"
                ))
            }
        };

        ensure!(
            rows <= u32::MAX as usize && cols <= u32::MAX as usize,
            "fspecial: kernel dimensions exceed GPU limits"
        );
        let shape_vec = vec![rows, cols];
        let total_len = product_checked(&shape_vec)
            .ok_or_else(|| anyhow!("fspecial: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-fspecial-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "fspecial: tensor length exceeds GPU dispatch limits"
        );

        let params_buffer = match self.precision {
            NumericPrecision::F64 => {
                let params = crate::backend::wgpu::params::FspecialParamsF64 {
                    rows: rows as u32,
                    cols: cols as u32,
                    kind,
                    len: total_len as u32,
                    sigma,
                    alpha,
                    norm,
                    center_x,
                    center_y,
                    extra0: 0.0,
                };
                self.uniform_buffer(&params, "runmat-fspecial-params-f64")
            }
            NumericPrecision::F32 => {
                let params = crate::backend::wgpu::params::FspecialParamsF32 {
                    rows: rows as u32,
                    cols: cols as u32,
                    kind,
                    len: total_len as u32,
                    sigma: sigma as f32,
                    alpha: alpha as f32,
                    norm: norm as f32,
                    _pad0: 0.0,
                    center_x: center_x as f32,
                    center_y: center_y as f32,
                    _pad1: [0.0, 0.0],
                };
                self.uniform_buffer(&params, "runmat-fspecial-params-f32")
            }
        };

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-fspecial-bind"),
                layout: &self.pipelines.fspecial.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            total_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.fspecial.pipeline,
            &bind_group,
            workgroups,
            "runmat-fspecial-encoder",
            "runmat-fspecial-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, shape_vec, total_len))
    }

    pub(crate) fn peaks_exec(&self, n: usize) -> Result<GpuTensorHandle> {
        if n > u32::MAX as usize {
            return Err(anyhow!("peaks: dimension exceeds GPU limits"));
        }
        let total_len = n
            .checked_mul(n)
            .ok_or_else(|| anyhow!("peaks: tensor size overflows"))?;
        ensure!(
            total_len <= u32::MAX as usize,
            "peaks: tensor length exceeds GPU dispatch limits"
        );
        let shape_vec = vec![n, n];
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-peaks-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }

        let n_u32 = n as u32;
        let total_u32 = total_len as u32;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;

        while offset < total_len {
            let chunk_len = (total_len - offset).min(chunk_capacity).max(1);
            let offset_u32 = offset as u32;
            let chunk_u32 = chunk_len as u32;

            let params = crate::backend::wgpu::params::PeaksParams {
                n: n_u32,
                total: total_u32,
                chunk: chunk_u32,
                offset: offset_u32,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-peaks-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-peaks-bind"),
                    layout: &self.pipelines.peaks.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.peaks.pipeline,
                &bind_group,
                workgroups,
                "runmat-peaks-encoder",
                "runmat-peaks-pass",
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape_vec, total_len))
    }

    pub(crate) fn peaks_xy_exec(
        &self,
        x: &GpuTensorHandle,
        y: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            x.shape == y.shape,
            "peaks: X and Y must have the same shape"
        );
        let total_len =
            product_checked(&x.shape).ok_or_else(|| anyhow!("peaks: tensor size overflows"))?;
        ensure!(
            total_len <= u32::MAX as usize,
            "peaks: tensor length exceeds GPU dispatch limits"
        );
        let shape_vec = x.shape.clone();
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-peaks-xy-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }

        let x_entry = self.get_entry(x)?;
        let y_entry = self.get_entry(y)?;
        let total_u32 = total_len as u32;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;

        while offset < total_len {
            let chunk_len = (total_len - offset).min(chunk_capacity).max(1);
            let offset_u32 = offset as u32;
            let chunk_u32 = chunk_len as u32;

            let params_buffer = {
                let params = crate::backend::wgpu::params::PeaksXYParams {
                    total: total_u32,
                    chunk: chunk_u32,
                    offset: offset_u32,
                    _pad: 0,
                };
                self.uniform_buffer(&params, "runmat-peaks-xy-params")
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-peaks-xy-bind"),
                    layout: &self.pipelines.peaks_xy.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: y_entry.buffer.as_ref().as_entire_binding(),
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
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.peaks_xy.pipeline,
                &bind_group,
                workgroups,
                "runmat-peaks-xy-encoder",
                "runmat-peaks-xy-pass",
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape_vec, total_len))
    }
    pub(crate) fn linspace_exec(
        &self,
        start: f64,
        stop: f64,
        count: usize,
    ) -> Result<GpuTensorHandle> {
        if count > u32::MAX as usize {
            return Err(anyhow!("linspace: sequence length exceeds GPU limits"));
        }

        let shape = vec![1, count];
        let out_buffer = self.create_storage_buffer(count, "runmat-linspace-out");
        if count == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-linspace-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-linspace-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.linspace.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let step = if count <= 1 {
            0.0
        } else {
            (stop - start) / ((count - 1) as f64)
        };
        let total_u32 = count as u32;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;

        while offset < count {
            let chunk_len = (count - offset).min(chunk_capacity).max(1);
            let offset_u32 = offset as u32;
            let chunk_u32 = chunk_len as u32;

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::LinspaceParamsF64 {
                        start,
                        step,
                        stop,
                        total: total_u32,
                        chunk: chunk_u32,
                        offset: offset_u32,
                        _pad: 0,
                    };
                    self.uniform_buffer(&params, "runmat-linspace-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::LinspaceParamsF32 {
                        start: start as f32,
                        step: step as f32,
                        stop: stop as f32,
                        _pad0: 0.0,
                        total: total_u32,
                        chunk: chunk_u32,
                        offset: offset_u32,
                        _pad1: 0,
                    };
                    self.uniform_buffer(&params, "runmat-linspace-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-linspace-bind"),
                    layout: &self.pipelines.linspace.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.linspace.pipeline,
                &bind_group,
                workgroups,
                "runmat-linspace-encoder",
                "runmat-linspace-pass",
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape, count))
    }
    pub(crate) fn diag_from_vector_exec(
        &self,
        vector: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(vector)?;
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: input must be a vector"
        );

        let len = entry.len;
        if len == 0 {
            return Err(anyhow!("diag: empty vector fallback"));
        }
        let (size, total) = diag_matrix_size_checked(len, offset)?;
        ensure!(
            len <= u32::MAX as usize,
            "diag: vector is too large for GPU dispatch"
        );
        ensure!(
            size <= u32::MAX as usize,
            "diag: result dimension exceeds GPU dispatch limits"
        );
        ensure!(
            total <= u32::MAX as usize,
            "diag: result size exceeds GPU dispatch limits"
        );
        let offset_i32 = i32::try_from(offset)
            .map_err(|_| anyhow!("diag: offset magnitude exceeds GPU limits"))?;

        let out_buffer = self.create_storage_buffer(total, "runmat-diag-vec-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-diag-vec-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-diag-vec-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.diag_from_vector.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let params = crate::backend::wgpu::params::DiagFromVectorParams {
            len: len as u32,
            size: size as u32,
            offset: offset_i32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diag-vec-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diag-vec-bind"),
                layout: &self.pipelines.diag_from_vector.layout,
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
            len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diag::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diag_from_vector.pipeline,
            &bind_group,
            workgroups,
            "runmat-diag-vec-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, vec![size, size], total))
    }
    pub(crate) fn diag_extract_exec(
        &self,
        matrix: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(matrix)?;
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            !diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: matrix input required"
        );
        let diag_len = diag_length(rows, cols, offset);
        if diag_len == 0 {
            return Err(anyhow!("diag: empty diagonal fallback"));
        }
        ensure!(
            diag_len <= u32::MAX as usize,
            "diag: diagonal length exceeds GPU dispatch limits"
        );
        ensure!(
            rows <= u32::MAX as usize && cols <= u32::MAX as usize,
            "diag: matrix dimensions exceed GPU dispatch limits"
        );
        let offset_i32 = i32::try_from(offset)
            .map_err(|_| anyhow!("diag: offset magnitude exceeds GPU limits"))?;

        let out_buffer = self.create_storage_buffer(diag_len, "runmat-diag-extract-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-diag-extract-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-diag-extract-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.diag_extract.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let params = crate::backend::wgpu::params::DiagExtractParams {
            rows: rows as u32,
            cols: cols as u32,
            offset: offset_i32,
            diag_len: diag_len as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diag-extract-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diag-extract-bind"),
                layout: &self.pipelines.diag_extract.layout,
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
            diag_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diag::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diag_extract.pipeline,
            &bind_group,
            workgroups,
            "runmat-diag-extract-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, vec![diag_len, 1], diag_len))
    }
}
