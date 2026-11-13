use super::*;
#[path = "constructors/diag_launch.rs"]
mod diag_launch;
#[path = "constructors/optional.rs"]
mod optional;

#[cfg(test)]
#[path = "constructors/typed_tests.rs"]
mod typed_tests;

#[cfg(test)]
#[path = "constructors/validation_tests.rs"]
mod validation_tests;

fn constructor_elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        Some(0)
    } else {
        product_checked(shape)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NdgridParams {
    total: u32,
    axis_len: u32,
    stride: u32,
    offset: u32,
    chunk: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl WgpuProvider {
    pub(crate) fn eye_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let normalized = normalize_eye_shape(shape);
        if normalized.len() < 2 {
            return Err(anyhow!("eye: expected at least 2 dimensions"));
        }
        let total_len = product_checked(&normalized)
            .ok_or_else(|| anyhow!("eye: tensor size exceeds GPU limits"))?;
        let dispatch = if total_len == 0 {
            None
        } else {
            let rows = normalized[0];
            let cols = normalized[1];
            let diag_len = rows.min(cols);
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
            if rows > (u32::MAX as usize)
                || cols > (u32::MAX as usize)
                || slice_stride > (u32::MAX as usize)
                || diag_total > (u32::MAX as usize)
            {
                return Err(anyhow!("eye: dimensions exceed GPU dispatch limits"));
            }
            let workgroups = self.pipelines.eye.linear_dispatch_groups(diag_total)?;
            Some((
                rows,
                cols,
                diag_len,
                slice_stride,
                slices,
                diag_total,
                workgroups,
            ))
        };
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, total_len, "runmat-eye-out")?;
            let Some((rows, cols, diag_len, slice_stride, slices, diag_total, workgroups)) =
                dispatch
            else {
                return Ok(out_buffer);
            };

            let pipeline = self.pipelines.eye.get(emission)?;
            {
                let mut enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-eye-noop"),
                    },
                );
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-eye-noop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                drop(pass);
                self.submit(emission, pending, enc)?;
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
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-eye-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-eye-bind"),
                    layout: &pipeline.layout,
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
            crate::backend::wgpu::dispatch::creation::run(
                emission,
                pending,
                &pipeline.pipeline,
                &bind_group,
                workgroups,
                "runmat-eye-encoder",
                "runmat-eye-pass",
            )?;

            Ok(out_buffer)
        })??;
        self.register_existing_buffer(out_buffer, normalized, total_len)
    }
    pub(crate) fn fill_exec(&self, shape: &[usize], value: f64) -> Result<GpuTensorHandle> {
        self.fill_produced_exec(shape, value, false, None, None, None)
    }

    pub(super) fn fill_std_selected_exec(
        &self,
        shape: &[usize],
        value: f64,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
        effect: &std::cell::Cell<bool>,
        after_allocation: Option<&super::reduction::phase_context::AllocationFault<'_>>,
    ) -> Result<GpuTensorHandle> {
        self.fill_produced_exec(
            shape,
            value,
            false,
            Some(validated),
            Some(effect),
            after_allocation,
        )
    }

    pub(crate) fn fill_buffer_demand(&self, elements: u64) -> Option<u64> {
        let precision = match self.precision {
            NumericPrecision::F32 => ProviderPrecision::F32,
            NumericPrecision::F64 => ProviderPrecision::F64,
        };
        crate::backend::wgpu_geometry::fill_physical_work(
            elements,
            self.linear_launch.workgroup(),
            precision,
        )?
        .total_bytes()
    }

    pub(crate) fn logical_fill_exec(
        &self,
        shape: &[usize],
        value: bool,
    ) -> Result<GpuTensorHandle> {
        self.fill_produced_exec(shape, if value { 1.0 } else { 0.0 }, true, None, None, None)
    }

    fn fill_produced_exec(
        &self,
        shape: &[usize],
        value: f64,
        logical: bool,
        validated: Option<&super::validated_fusion::ValidatedFusionPipeline>,
        effect: Option<&std::cell::Cell<bool>>,
        after_allocation: Option<&super::reduction::phase_context::AllocationFault<'_>>,
    ) -> Result<GpuTensorHandle> {
        let total_len = constructor_elements(shape)
            .ok_or_else(|| anyhow!("fill: tensor size exceeds GPU limits"))?;
        let chunk_capacity = self
            .pipelines
            .fill
            .linear_chunk_capacity()
            .map_err(|cause| anyhow!("fill: {cause}"))?;
        let shape_vec = shape.to_vec();
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, total_len, "runmat-fill-out")?;
            if let Some(effect) = effect {
                effect.set(true);
            }
            if let Some(after_allocation) = after_allocation {
                after_allocation(&out_buffer)?;
            }
            if total_len == 0 {
                return Ok(out_buffer);
            }
            ensure!(
                total_len <= u32::MAX as usize,
                "fill: tensor length exceeds GPU dispatch limits"
            );

            let legacy = if validated.is_none() {
                Some(self.pipelines.fill.get(emission)?)
            } else {
                None
            };
            let layout = validated
                .map(|selected| selected.layout.as_ref())
                .or_else(|| legacy.as_ref().map(|bundle| &bundle.layout))
                .expect("selected or legacy fill layout");
            let pipeline = validated
                .map(|selected| selected.pipeline.as_ref())
                .or_else(|| legacy.as_ref().map(|bundle| &bundle.pipeline))
                .expect("selected or legacy fill pipeline");
            // chunked dispatch below will build per-chunk params buffers

            let mut processed: usize = 0;
            while processed < total_len {
                let remain = total_len - processed;
                let chunk_len = remain.min(chunk_capacity);

                // Per-chunk params (length)
                let params_buffer = match self.precision {
                    NumericPrecision::F64 => {
                        let params = crate::backend::wgpu::params::FillParamsF64 {
                            value,
                            len: chunk_len as u32,
                            _pad: [0, 0, 0],
                        };
                        self.uniform_buffer(emission, &params, "runmat-fill-params-f64")?
                    }
                    NumericPrecision::F32 => {
                        let params = crate::backend::wgpu::params::FillParamsF32 {
                            value: value as f32,
                            len: chunk_len as u32,
                            _pad: [0, 0],
                        };
                        self.uniform_buffer(emission, &params, "runmat-fill-params-f32")?
                    }
                };

                // Bind only the range for this chunk
                let byte_offset = (processed * self.element_size) as u64;
                let byte_size = std::num::NonZeroU64::new((chunk_len * self.element_size) as u64);
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fill-bind"),
                        layout,
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

                let workgroups = self.pipelines.fill.linear_dispatch_groups(chunk_len)?;
                crate::backend::wgpu::dispatch::creation::run(
                    emission,
                    pending,
                    pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-fill-encoder",
                    "runmat-fill-pass",
                )?;

                processed += chunk_len;
            }

            Ok(out_buffer)
        })??;
        if logical {
            self.register_logical_buffer(
                out_buffer,
                shape_vec,
                total_len,
                BufferUsageClass::Generic,
            )
        } else {
            self.register_existing_buffer(out_buffer, shape_vec, total_len)
        }
    }

    pub(crate) fn zeros_like_exec(
        &self,
        prototype: &GpuTensorHandle,
        shape: &[usize],
        effect: &std::cell::Cell<bool>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(prototype)?;
        let logical =
            self.allocations.requires_logical_encoding(prototype)? && prototype.view().is_logical();
        let elements =
            constructor_elements(shape).ok_or_else(|| anyhow!("zeros_like: shape overflow"))?;
        let lanes = elements
            .checked_mul(if entry.storage == GpuTensorStorage::ComplexInterleaved {
                2
            } else {
                1
            })
            .ok_or_else(|| anyhow!("zeros_like: lane count overflow"))?;
        let words_per_lane = entry.element_type.integer_type().map_or(
            entry.element_type.element_size() / 4,
            crate::backend::wgpu_storage::integer_words_per_element,
        );
        let words = lanes
            .checked_mul(words_per_lane)
            .ok_or_else(|| anyhow!("zeros_like: word count overflow"))?;
        let buffer = self.with_emission(|emission, pending| -> Result<_> {
            let buffer = self.create_integer_word_buffer(emission, words, "runmat-zeros-like")?;
            effect.set(true);
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-zeros-like-clear"),
                },
            );
            encoder.clear_buffer(&buffer, 0, None);
            self.submit(emission, pending, encoder)?;
            Ok(buffer)
        })??;
        let allocated_bytes = buffer.size();
        self.register_numeric_buffer_produced(
            buffer,
            NumericBufferRegistration {
                shape: shape.to_vec(),
                len: lanes,
                physical_element_type: entry.element_type,
                storage: entry.storage,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
            logical,
        )
    }

    pub(crate) fn zeros_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.zeros_with_storage_exec(shape, GpuTensorStorage::Real)
    }

    pub(crate) fn zeros_with_storage_exec(
        &self,
        shape: &[usize],
        storage: GpuTensorStorage,
    ) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let lane_factor = match storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let raw_len = len
            .checked_mul(lane_factor)
            .ok_or_else(|| anyhow!("zeros_with_storage: tensor size exceeds GPU limits"))?;
        let buffer = self.with_emission(|emission, _| -> Result<_> {
            let buffer = self.create_storage_buffer_checked(emission, raw_len, "zeros")?;
            // Explicitly zero-initialize the storage buffer; pooled buffers may contain old data
            let size_bytes = (raw_len.max(1) as u64) * (self.element_size as u64);
            if size_bytes > 0 {
                // Write zeros across the entire buffer
                let zero_bytes = vec![0u8; size_bytes as usize];
                emission.write_buffer(buffer.as_ref(), 0, &zero_bytes);
            }
            Ok(buffer)
        })??;
        self.register_existing_buffer_with_storage(buffer, shape.to_vec(), raw_len, storage)
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

    pub(crate) fn ndgrid_exec(
        &self,
        request: &ProviderNdgridRequest<'_>,
    ) -> Result<ProviderNdgridResult> {
        ensure!(request.output_count > 0, "ndgrid: missing outputs");
        ensure!(
            request.output_count <= request.axes.len(),
            "ndgrid: too many outputs for axes"
        );
        ensure!(!request.output_shape.is_empty(), "ndgrid: missing shape");
        ensure!(
            request.output_count <= request.output_shape.len(),
            "ndgrid: output rank does not cover requested axes"
        );

        let total_len = product_checked(request.output_shape)
            .ok_or_else(|| anyhow!("ndgrid: tensor size exceeds GPU limits"))?;
        ensure!(
            total_len <= u32::MAX as usize,
            "ndgrid: tensor length exceeds GPU dispatch limits"
        );
        let total_u32 = total_len as u32;
        let chunk_capacity = self
            .linear_launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("ndgrid: enabled launch extent overflow"))?;
        let strides = column_major_strides_checked(request.output_shape)?;
        let expected_element_type = match self.precision {
            NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
            NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
        };
        let entries = {
            let mut entries = Vec::with_capacity(request.output_count);
            for (dim, axis) in request.axes.iter().take(request.output_count).enumerate() {
                ensure!(
                    axis.handle.device_id() == self.runtime_device_id,
                    "ndgrid: axis belongs to a different device"
                );
                let entry = self.get_entry_raw(axis.handle)?;
                if entry.storage == GpuTensorStorage::ComplexInterleaved {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "ndgrid: complex axes are not supported",
                    ));
                }
                if entry.element_type != expected_element_type {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        format!(
                            "ndgrid: axis physical type {} does not match provider type {}",
                            entry.element_type.class_name(),
                            expected_element_type.class_name()
                        ),
                    ));
                }
                let expected = request.output_shape.get(dim).copied().unwrap_or(1);
                ensure!(
                    entry.len == expected,
                    "ndgrid: axis {} length {} does not match output extent {}",
                    dim + 1,
                    entry.len,
                    expected
                );
                ensure!(
                    entry.len <= u32::MAX as usize && strides[dim] <= u32::MAX as usize,
                    "ndgrid: axis dimensions exceed GPU dispatch limits"
                );
                entries.push(entry);
            }
            entries
        };

        let buffers = self.with_emission(|emission, pending| -> Result<_> {
            let shader = ndgrid_shader(self.precision, self.linear_launch.workgroup());
            let shader_module = crate::backend::wgpu::shader_compilation::create_shader_module(
                emission,
                "runmat-ndgrid-shader",
                &shader,
                None,
            )?;
            let bind_layout =
                emission
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-ndgrid-layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });
            let pipeline_layout =
                emission
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("runmat-ndgrid-pipeline-layout"),
                        bind_group_layouts: &[&bind_layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = emission.device().create_compute_pipeline(
                &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                    label: Some("runmat-ndgrid-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "main",
                },
            );

            let mut outputs = Vec::with_capacity(request.output_count);
            for dim in 0..request.output_count {
                let out_buffer =
                    self.create_storage_buffer_checked(emission, total_len, "runmat-ndgrid-out")?;
                if total_len > 0 {
                    let mut offset = 0usize;
                    while offset < total_len {
                        let chunk_len = (total_len - offset).min(chunk_capacity);
                        let params = NdgridParams {
                            total: total_u32,
                            axis_len: entries[dim].len as u32,
                            stride: strides[dim] as u32,
                            offset: offset as u32,
                            chunk: chunk_len as u32,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        };
                        let params_buffer =
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-ndgrid-params"),
                                contents: bytes_of(&params),
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            });
                        let bind_group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-ndgrid-bind"),
                                    layout: &bind_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: entries[dim].buffer.as_entire_binding(),
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
                        let workgroups =
                            self.linear_launch
                                .dispatch_groups(chunk_len)
                                .ok_or_else(|| {
                                    anyhow!("ndgrid: chunk exceeds enabled launch limits")
                                })?;
                        crate::backend::wgpu::dispatch::creation::run(
                            emission,
                            pending,
                            &pipeline,
                            &bind_group,
                            workgroups,
                            "runmat-ndgrid-encoder",
                            "runmat-ndgrid-pass",
                        )?;
                        offset += chunk_len;
                    }
                }
                outputs.push(out_buffer);
            }
            Ok(outputs)
        })??;
        let outputs = buffers
            .into_iter()
            .map(|buffer| {
                self.register_existing_buffer(buffer, request.output_shape.to_vec(), total_len)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ProviderNdgridResult { outputs })
    }

    pub(crate) fn fspecial_exec(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        let operation = runmat_accelerate_api::ProviderOperationIdentity::new("fspecial");
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        let checkpoint = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );
        let spec = runtime_fspecial_spec_from_request(&request.filter);

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
        ensure!(
            total_len <= u32::MAX as usize,
            "fspecial: tensor length exceeds GPU dispatch limits"
        );
        let workgroups = self.pipelines.fspecial.linear_dispatch_groups(total_len)?;
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, total_len, "runmat-fspecial-out")?;
            if total_len == 0 {
                return Ok(out_buffer);
            }

            let pipeline = self.pipelines.fspecial.get(emission)?;

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
                    self.uniform_buffer(emission, &params, "runmat-fspecial-params-f64")?
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
                    self.uniform_buffer(emission, &params, "runmat-fspecial-params-f32")?
                }
            };

            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fspecial-bind"),
                    layout: &pipeline.layout,
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

            crate::backend::wgpu::dispatch::creation::run(
                emission,
                pending,
                &pipeline.pipeline,
                &bind_group,
                workgroups,
                "runmat-fspecial-encoder",
                "runmat-fspecial-pass",
            )?;

            Ok(out_buffer)
        })??;

        let output = self.register_existing_buffer(out_buffer, shape_vec, total_len)?;
        if let (Some(recorder), Some(checkpoint)) = (observed, checkpoint) {
            recorder.confirm_submitted_provider_execution_after(
                self.device_id(),
                &operation,
                checkpoint,
            );
        }
        Ok(output)
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
        let chunk_capacity = self
            .linear_launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("peaks: launch extent overflow"))?;
        let shape_vec = vec![n, n];
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, total_len, "runmat-peaks-out")?;
            if total_len == 0 {
                return Ok(out_buffer);
            }

            let n_u32 = n as u32;
            let total_u32 = total_len as u32;
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
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-peaks-params")?;

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-peaks-bind"),
                        layout: &self.pipelines.peaks.get(emission)?.layout,
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

                let workgroups = self.pipelines.peaks.linear_dispatch_groups(chunk_len)?;
                crate::backend::wgpu::dispatch::creation::run(
                    emission,
                    pending,
                    &self.pipelines.peaks.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-peaks-encoder",
                    "runmat-peaks-pass",
                )?;

                offset += chunk_len;
            }

            Ok(out_buffer)
        })??;

        self.register_existing_buffer(out_buffer, shape_vec, total_len)
    }

    pub(crate) fn peaks_xy_exec(
        &self,
        x: &GpuTensorHandle,
        y: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            x.shape() == y.shape(),
            "peaks: X and Y must have the same shape"
        );
        let total_len =
            product_checked(x.shape()).ok_or_else(|| anyhow!("peaks: tensor size overflows"))?;
        ensure!(
            total_len <= u32::MAX as usize,
            "peaks: tensor length exceeds GPU dispatch limits"
        );
        let chunk_capacity = self
            .linear_launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("peaks: launch extent overflow"))?;
        let shape_vec = x.shape().to_vec();
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer_checked(emission, total_len, "runmat-peaks-xy-out")?;
            if total_len == 0 {
                return Ok(out_buffer);
            }

            let x_entry = self.get_entry(x)?;
            let y_entry = self.get_entry(y)?;
            let total_u32 = total_len as u32;
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
                    self.uniform_buffer(emission, &params, "runmat-peaks-xy-params")?
                };

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-peaks-xy-bind"),
                        layout: &self.pipelines.peaks_xy.get(emission)?.layout,
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

                let workgroups = self.pipelines.peaks_xy.linear_dispatch_groups(chunk_len)?;
                crate::backend::wgpu::dispatch::creation::run(
                    emission,
                    pending,
                    &self.pipelines.peaks_xy.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-peaks-xy-encoder",
                    "runmat-peaks-xy-pass",
                )?;

                offset += chunk_len;
            }

            Ok(out_buffer)
        })??;

        self.register_existing_buffer(out_buffer, shape_vec, total_len)
    }
    pub(crate) fn linspace_exec(
        &self,
        start: f64,
        stop: f64,
        count: usize,
    ) -> Result<GpuTensorHandle> {
        let request = runmat_accelerate_api::ProviderSequenceRequest::linear(
            start,
            stop,
            count,
            match self.precision {
                NumericPrecision::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                NumericPrecision::F64 => runmat_accelerate_api::ProviderPrecision::F64,
            },
        );
        self.sequence_exec_observed(&request, &std::cell::Cell::new(false))
    }

    pub(crate) fn sequence_exec_observed(
        &self,
        request: &runmat_accelerate_api::ProviderSequenceRequest,
        effect: &std::cell::Cell<bool>,
    ) -> Result<GpuTensorHandle> {
        let start = request.start;
        let stop = request.stop;
        let count = request.count;
        if count > u32::MAX as usize {
            return Err(anyhow!("linspace: sequence length exceeds GPU limits"));
        }
        let chunk_capacity = if count == 0 {
            0
        } else {
            self.pipelines.linspace.linear_chunk_capacity()?
        };

        let shape = vec![1, count];
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer = self.create_storage_buffer(emission, count, "runmat-linspace-out")?;
            effect.set(true);
            if count == 0 {
                return Ok(out_buffer);
            }

            {
                let mut enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-linspace-noop"),
                    },
                );
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-linspace-noop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.linspace.get(emission)?.pipeline);
                drop(pass);
                self.submit(emission, pending, enc)?;
            }

            let step = if count <= 1 {
                0.0
            } else {
                (stop - start) / ((count - 1) as f64)
            };
            let total_u32 = count as u32;
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
                            first: request.first,
                            last: request.last,
                            total: total_u32,
                            chunk: chunk_u32,
                            offset: offset_u32,
                            mode: u32::from(matches!(
                                request.kind,
                                runmat_accelerate_api::ProviderSequenceKind::Log10
                            )),
                            _pad: 0.0,
                        };
                        self.uniform_buffer(emission, &params, "runmat-linspace-params-f64")?
                    }
                    NumericPrecision::F32 => {
                        let params = crate::backend::wgpu::params::LinspaceParamsF32 {
                            start: start as f32,
                            step: step as f32,
                            stop: stop as f32,
                            first: request.first as f32,
                            last: request.last as f32,
                            total: total_u32,
                            chunk: chunk_u32,
                            offset: offset_u32,
                            mode: u32::from(matches!(
                                request.kind,
                                runmat_accelerate_api::ProviderSequenceKind::Log10
                            )),
                            _pad: [0; 3],
                        };
                        self.uniform_buffer(emission, &params, "runmat-linspace-params-f32")?
                    }
                };

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-linspace-bind"),
                        layout: &self.pipelines.linspace.get(emission)?.layout,
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

                let workgroups = self.pipelines.linspace.linear_dispatch_groups(chunk_len)?;
                crate::backend::wgpu::dispatch::creation::run(
                    emission,
                    pending,
                    &self.pipelines.linspace.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-linspace-encoder",
                    "runmat-linspace-pass",
                )?;

                offset += chunk_len;
            }

            Ok(out_buffer)
        })??;

        self.register_existing_buffer(out_buffer, shape, count)
    }
    pub(crate) fn diag_from_vector_exec(
        &self,
        vector: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(vector)?;
        if entry.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "diag: complex input requires a complex provider kernel",
            ));
        }
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: input must be a vector"
        );

        let len = entry.len;
        let (size, _) = diag_matrix_size_checked(len, offset)?;
        self.diag_from_vector_sized_exec(vector, offset, size, size)
    }

    pub(crate) fn diag_from_vector_sized_exec(
        &self,
        vector: &GpuTensorHandle,
        offset: isize,
        out_rows: usize,
        out_cols: usize,
    ) -> Result<GpuTensorHandle> {
        let declared = self.get_entry(vector)?;
        if declared.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "diag: complex input requires a complex provider kernel",
            ));
        }
        diag_ensure_shape(&declared.shape)?;
        let (declared_rows, declared_cols) = diag_rows_cols(&declared.shape);
        ensure!(
            diag_is_vector_like(declared_rows, declared_cols, declared.shape.len()),
            "diag: input must be a vector"
        );
        let declared_len = declared.len;
        let total = out_rows
            .checked_mul(out_cols)
            .ok_or_else(|| anyhow!("diag: result size exceeds GPU dispatch limits"))?;
        let dispatched_len =
            diag_launch::vector_dispatch_len(declared_len, out_rows, out_cols, offset);
        diag_launch::ensure_storage_domain(
            declared.allocated_bytes,
            total,
            dispatched_len,
            self.element_size,
            &self.adapter_limits,
        )?;
        // A bound diagonal of length d requires at least d*d floating output
        // elements. The WGPU storage-binding byte limit is u32-sized and a
        // floating element is at least four bytes, so every admitted diagonal
        // is smaller than even the WG=1, 65_535-group linear-launch capacity.
        let workgroups = self
            .pipelines
            .diag_from_vector
            .linear_dispatch_groups(dispatched_len)
            .map_err(|cause| anyhow!("diag: {cause}"))?;
        let contiguous = self.reshape_exec(vector, vector.shape())?;
        let vector = &contiguous;

        let entry = self.get_entry(vector)?;
        if entry.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "diag: complex input requires a complex provider kernel",
            ));
        }
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: input must be a vector"
        );

        let len = entry.len;
        ensure!(
            len == declared_len,
            "diag: authenticated vector size disagrees with its view"
        );
        ensure!(
            dispatched_len <= u32::MAX as usize,
            "diag: written diagonal is too large for GPU dispatch"
        );
        ensure!(
            out_rows <= u32::MAX as usize && out_cols <= u32::MAX as usize,
            "diag: result dimension exceeds GPU dispatch limits"
        );
        ensure!(
            total <= u32::MAX as usize,
            "diag: result size exceeds GPU dispatch limits"
        );
        let offset_i32 = i32::try_from(offset)
            .map_err(|_| anyhow!("diag: offset magnitude exceeds GPU limits"))?;
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer = self.create_storage_buffer(emission, total, "runmat-diag-vec-out")?;
            if dispatched_len == 0 || total == 0 {
                return Ok(out_buffer);
            }
            {
                let mut enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-diag-vec-noop"),
                    },
                );
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-diag-vec-noop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.diag_from_vector.get(emission)?.pipeline);
                drop(pass);
                self.submit(emission, pending, enc)?;
            }

            let params = crate::backend::wgpu::params::DiagFromVectorParams {
                len: dispatched_len as u32,
                rows: out_rows as u32,
                cols: out_cols as u32,
                _pad0: 0,
                offset: offset_i32,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            let params_buffer = self.uniform_buffer(emission, &params, "runmat-diag-vec-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-diag-vec-bind"),
                    layout: &self.pipelines.diag_from_vector.get(emission)?.layout,
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
            crate::backend::wgpu::dispatch::diag::run(
                emission,
                pending,
                &self.pipelines.diag_from_vector.get(emission)?.pipeline,
                &bind_group,
                workgroups,
                "runmat-diag-vec-pass",
            )?;

            Ok(out_buffer)
        })??;
        if vector.view().is_logical() {
            self.register_logical_buffer(
                out_buffer,
                vec![out_rows, out_cols],
                total,
                BufferUsageClass::Generic,
            )
        } else {
            self.register_existing_buffer(out_buffer, vec![out_rows, out_cols], total)
        }
    }
    pub(crate) fn diag_extract_exec(
        &self,
        matrix: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let declared = self.get_entry(matrix)?;
        if declared.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "diag: complex input requires a complex provider kernel",
            ));
        }
        diag_ensure_shape(&declared.shape)?;
        let (declared_rows, declared_cols) = diag_rows_cols(&declared.shape);
        ensure!(
            !diag_is_vector_like(declared_rows, declared_cols, declared.shape.len()),
            "diag: matrix input required"
        );
        let declared_diag_len = diag_length(declared_rows, declared_cols, offset);
        diag_launch::ensure_storage_domain(
            declared.allocated_bytes,
            declared_diag_len,
            declared_diag_len,
            self.element_size,
            &self.adapter_limits,
        )?;
        // The same d*d storage bound applies to the source matrix here.
        let workgroups = self
            .pipelines
            .diag_extract
            .linear_dispatch_groups(declared_diag_len)
            .map_err(|cause| anyhow!("diag: {cause}"))?;
        let contiguous = self.reshape_exec(matrix, matrix.shape())?;
        let matrix = &contiguous;

        let entry = self.get_entry(matrix)?;
        if entry.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "diag: complex input requires a complex provider kernel",
            ));
        }
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            !diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: matrix input required"
        );
        let diag_len = diag_length(rows, cols, offset);
        ensure!(
            rows == declared_rows && cols == declared_cols && diag_len == declared_diag_len,
            "diag: authenticated matrix shape disagrees with its view"
        );
        if diag_len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-diag-extract-empty")
            })??;
            return if matrix.view().is_logical() {
                self.register_logical_buffer(out_buffer, vec![0, 1], 0, BufferUsageClass::Generic)
            } else {
                self.register_existing_buffer(out_buffer, vec![0, 1], 0)
            };
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
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer =
                self.create_storage_buffer(emission, diag_len, "runmat-diag-extract-out")?;
            {
                let mut enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-diag-extract-noop"),
                    },
                );
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-diag-extract-noop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.diag_extract.get(emission)?.pipeline);
                drop(pass);
                self.submit(emission, pending, enc)?;
            }

            let params = crate::backend::wgpu::params::DiagExtractParams {
                rows: rows as u32,
                cols: cols as u32,
                offset: offset_i32,
                diag_len: diag_len as u32,
            };
            let params_buffer =
                self.uniform_buffer(emission, &params, "runmat-diag-extract-params")?;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-diag-extract-bind"),
                    layout: &self.pipelines.diag_extract.get(emission)?.layout,
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
            crate::backend::wgpu::dispatch::diag::run(
                emission,
                pending,
                &self.pipelines.diag_extract.get(emission)?.pipeline,
                &bind_group,
                workgroups,
                "runmat-diag-extract-pass",
            )?;

            Ok(out_buffer)
        })??;
        if matrix.view().is_logical() {
            self.register_logical_buffer(
                out_buffer,
                vec![diag_len, 1],
                diag_len,
                BufferUsageClass::Generic,
            )
        } else {
            self.register_existing_buffer(out_buffer, vec![diag_len, 1], diag_len)
        }
    }
}

fn ndgrid_shader(precision: NumericPrecision, workgroup: u32) -> String {
    let scalar = precision.as_str();
    format!(
        r#"
struct Tensor {{
  data: array<{scalar}>,
}};

struct Params {{
  total: u32,
  axis_len: u32,
  stride: u32,
  offset: u32,
  chunk: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}};

@group(0) @binding(0) var<storage, read> axis: Tensor;
@group(0) @binding(1) var<storage, read_write> out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= params.chunk) {{
    return;
  }}
  let linear = gid.x + params.offset;
  if (linear >= params.total) {{
    return;
  }}
  let coord = (linear / params.stride) % params.axis_len;
  out.data[linear] = axis.data[coord];
}}
"#
    )
}

fn column_major_strides_checked(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| anyhow!("ndgrid: stride exceeds GPU limits"))?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use runmat_accelerate_api::{
        AccelProvider, HostTensorView, ProviderNdgridAxis, ProviderNdgridRequest,
    };

    fn register_wgpu_provider_for_test() -> Option<crate::backend::wgpu::provider::WgpuTestSession>
    {
        match crate::backend::wgpu::provider::register_test_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => Some(provider),
            Err(err) if err.to_string() == "wgpu: no compatible adapter found" => None,
            Err(err) => panic!("register wgpu provider failed: {err:?}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires an actual WGPU device"]
    async fn wgpu_diagonal_outputs_preserve_logical_producer() {
        let provider = crate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("actual WGPU qualification requires a device");
        let input = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1, 0, 1],
                shape: &[3, 1],
            })
            .unwrap();
        let matrix = provider.diag_from_vector(&input, 0).unwrap();
        let output = provider.diag_extract(&matrix, 0).unwrap();
        let empty = provider.diag_extract(&matrix, 9).unwrap();
        for result in [&matrix, &output, &empty] {
            assert!(result.view().is_logical());
            assert!(!result.shares_allocation_with(&input));
        }
        drop(input);
        drop(matrix);
        assert_eq!(
            provider.download(&output).await.unwrap().data,
            vec![1.0, 0.0, 1.0]
        );
        assert_eq!(empty.shape(), &[0, 1]);
        let zero = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[],
                shape: &[0, 1],
            })
            .unwrap();
        assert!(provider
            .diag_from_vector_sized(&zero, 0, 0, 0)
            .unwrap()
            .view()
            .is_logical());
    }

    #[tokio::test]
    async fn wgpu_diag_vector_sized_and_extract_match_column_major_cpu() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let vector = [3.0, 5.0, 7.0];
        let handle = provider
            .upload(&HostTensorView {
                data: &vector,
                shape: &[3, 1],
            })
            .expect("upload vector");
        let placed = provider
            .diag_from_vector_sized(&handle, -1, 5, 3)
            .expect("diag from vector");
        let placed_host = provider.download(&placed).await.expect("download placed");
        assert_eq!(placed_host.shape, vec![5, 3]);
        assert_eq!(
            placed_host.data,
            vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0]
        );

        let extracted = provider.diag_extract(&placed, -1).expect("diag extract");
        let extracted_host = provider
            .download(&extracted)
            .await
            .expect("download extracted");
        assert_eq!(extracted_host.shape, vec![3, 1]);
        assert_eq!(extracted_host.data, vector.to_vec());

        let empty = provider.diag_extract(&placed, 10).expect("empty extract");
        let empty_host = provider.download(&empty).await.expect("download empty");
        assert_eq!(empty_host.shape, vec![0, 1]);
        assert!(empty_host.data.is_empty());
    }

    #[tokio::test]
    async fn wgpu_ndgrid_resident_axes_match_column_major_cpu() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let x = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("upload x");
        let y = provider
            .upload(&HostTensorView {
                data: &[10.0, 20.0, 30.0],
                shape: &[3, 1],
            })
            .expect("upload y");
        let axes = [
            ProviderNdgridAxis { handle: &x },
            ProviderNdgridAxis { handle: &y },
        ];
        let result = provider
            .ndgrid(&ProviderNdgridRequest {
                axes: &axes,
                output_shape: &[2, 3],
                output_count: 2,
            })
            .expect("ndgrid");
        assert_eq!(result.outputs.len(), 2);
        let gx = provider
            .download(&result.outputs[0])
            .await
            .expect("download gx");
        let gy = provider
            .download(&result.outputs[1])
            .await
            .expect("download gy");
        assert_eq!(gx.shape, vec![2, 3]);
        assert_eq!(gy.shape, vec![2, 3]);
        assert_eq!(gx.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(gy.data, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    }
}
