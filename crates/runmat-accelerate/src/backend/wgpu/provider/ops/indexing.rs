use super::*;

fn linear_storage_lane_factor(
    integer_type: Option<runmat_accelerate_api::IntegerElementType>,
    storage: GpuTensorStorage,
) -> usize {
    match integer_type {
        Some(
            runmat_accelerate_api::IntegerElementType::I64
            | runmat_accelerate_api::IntegerElementType::U64,
        ) => 2,
        Some(_) => 1,
        None => match storage {
            GpuTensorStorage::Real => 1,
            GpuTensorStorage::ComplexInterleaved => 2,
        },
    }
}

fn linear_values_entry_len(
    integer_type: Option<runmat_accelerate_api::IntegerElementType>,
    storage: GpuTensorStorage,
    index_count: usize,
) -> Option<usize> {
    // Integer BufferEntry lengths are logical element counts; only float/complex
    // buffers expose their storage-lane count as len.
    if integer_type.is_some() {
        Some(index_count)
    } else {
        index_count.checked_mul(linear_storage_lane_factor(integer_type, storage))
    }
}

impl WgpuProvider {
    pub(crate) fn scatter_column_exec(
        &self,
        matrix: &GpuTensorHandle,
        col_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let raw_matrix = self.get_entry_raw(matrix)?;
        if raw_matrix.integer_type().is_some() {
            ensure!(
                raw_matrix.shape.len() == 2,
                "scatter_column: only 2D tensors supported"
            );
            let rows = raw_matrix.shape[0];
            let cols = raw_matrix.shape[1];
            ensure!(
                col_index < cols,
                "scatter_column: column index out of bounds"
            );
            let raw_values = self.get_entry_raw(values)?;
            ensure!(
                raw_values.integer_type() == raw_matrix.integer_type(),
                "scatter_column: integer storage mismatch"
            );
            ensure!(
                raw_values.shape.len() <= 2 && raw_values.shape[0] == rows,
                "scatter_column: length mismatch"
            );
            let indices: Vec<u32> = (0..rows)
                .map(|row| {
                    u32::try_from(col_index * rows + row)
                        .map_err(|_| anyhow!("scatter_column: index exceeds u32"))
                })
                .collect::<Result<_>>()?;
            let all_indices: Vec<u32> = (0..rows * cols)
                .map(|index| {
                    u32::try_from(index).map_err(|_| anyhow!("scatter_column: index exceeds u32"))
                })
                .collect::<Result<_>>()?;
            let out = self.gather_linear_exec(matrix, &all_indices, &raw_matrix.shape)?;
            self.scatter_linear_exec(&out, &indices, values)?;
            return Ok(out);
        }
        let m_entry = self.get_entry(matrix)?;
        if m_entry.shape.len() != 2 {
            return Err(anyhow!("scatter_column: only 2D tensors supported"));
        }
        let rows = m_entry.shape[0];
        let cols = m_entry.shape[1];
        if col_index >= cols {
            return Err(anyhow!("scatter_column: column index out of bounds"));
        }
        let v_entry = self.get_entry(values)?;
        let v_rows = match v_entry.shape.len() {
            1 | 2 => v_entry.shape[0],
            _ => {
                return Err(anyhow!("scatter_column: values must be vector or [rows,1]"));
            }
        };
        if v_rows != rows {
            return Err(anyhow!("scatter_column: length mismatch"));
        }
        let shader = crate::backend::wgpu::shaders::scatter::SCATTER_COL_SHADER_F32;
        let out_buffer = self.create_storage_buffer(rows * cols, "runmat-scatter-col-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-scatter-col-copy"),
                    });
            enc.copy_buffer_to_buffer(
                m_entry.buffer.as_ref(),
                0,
                out_buffer.as_ref(),
                0,
                (rows * cols * self.element_size) as u64,
            );
            self.submit(enc);
        }
        let bgl = crate::backend::wgpu::bindings::build_scatter_col_bgl(self.device_ref());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-scatter-col-pl",
            &bgl,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-scatter-col-module",
            shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-scatter-col-bgl",
            Some(256),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-scatter-col",
            Some(shader.as_bytes()),
            Some("runmat-scatter-col-bgl"),
            Some(256),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pm {
            rows: u32,
            cols: u32,
            j: u32,
        }
        let params = Pm {
            rows: rows as u32,
            cols: cols as u32,
            j: col_index as u32,
        };
        let pbuf = self.uniform_buffer(&params, "runmat-scatter-col-params");
        let bg = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-col-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: m_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pbuf.as_entire_binding(),
                    },
                ],
            });
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(rows as u32, 256);
        crate::backend::wgpu::dispatch::scatter::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bg,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols))
    }
    pub(crate) fn sub2ind_exec(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        if inputs.len() != dims.len() || inputs.len() != scalar_mask.len() {
            return Err(anyhow!(
                "sub2ind: expected {} subscripts for {} dimensions",
                dims.len(),
                dims.len()
            ));
        }
        let expected_len: usize = output_shape.iter().copied().product();
        if expected_len != len {
            return Err(anyhow!(
                "sub2ind: output shape does not match subscript sizes"
            ));
        }
        if len == 0 {
            let buffer = self.create_storage_buffer(0, "runmat-sub2ind-empty");
            return Ok(self.register_existing_buffer(buffer, output_shape.to_vec(), 0));
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || len > u32::MAX as usize
        {
            return Err(anyhow!("sub2ind: dimensions exceed GPU kernel limits"));
        }

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&s| s as u32).collect();
        let mask_u32: Vec<u32> = scalar_mask
            .iter()
            .map(|&m| if m { 1u32 } else { 0u32 })
            .collect();

        let mut input_buffers = Vec::with_capacity(inputs.len());
        for handle in inputs {
            input_buffers.push(self.get_entry(handle)?.buffer.clone());
        }

        let output_buffer = self.create_storage_buffer_checked(len, "runmat-sub2ind-out")?;
        let error_bytes = vec![0u8; std::mem::size_of::<u32>() * 4];
        let error_buffer = Arc::new(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-sub2ind-error"),
                contents: &error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let (scalar_ty, epsilon) = match self.provider_precision_exec() {
            runmat_accelerate_api::ProviderPrecision::F64 => ("f64", "1.0e-12"),
            runmat_accelerate_api::ProviderPrecision::F32 => ("f32", "1.0e-5"),
        };
        let workgroup_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let shader = crate::backend::wgpu::shaders::sub2ind::build_sub2ind_shader(
            scalar_ty,
            &dims_u32,
            &strides_u32,
            &mask_u32,
            workgroup_size,
            epsilon,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-sub2ind-module",
            &shader,
        );
        let bgl =
            crate::backend::wgpu::bindings::build_sub2ind_bgl(self.device_ref(), inputs.len());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-sub2ind-pl",
            &bgl,
        );
        let layout_tag = format!("runmat-sub2ind-layout-{}", inputs.len());
        let key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), &layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-sub2ind",
            Some(shader.as_bytes()),
            Some(layout_tag.as_str()),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let params = Params {
            len: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-sub2ind-params");

        let mut bind_entries = Vec::with_capacity(inputs.len() + 3);
        for (idx, buffer) in input_buffers.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 1) as u32,
            resource: error_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 2) as u32,
            resource: params_buffer.as_entire_binding(),
        });
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-sub2ind-bg"),
                layout: &bgl,
                entries: &bind_entries,
            });

        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(len as u32, workgroup_size);
        crate::backend::wgpu::dispatch::sub2ind::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bind_group,
            groups,
        );

        let error_size = (std::mem::size_of::<u32>() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-sub2ind-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-sub2ind-error-copy"),
            });
        encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, error_size, "sub2ind")?;
        let words: &[u32] = cast_slice(&bytes);
        let code = words.first().copied().unwrap_or(0);
        let dim_word = words.get(1).copied().unwrap_or(0);
        let extra = words.get(2).copied().unwrap_or(0);

        if code != 0 {
            let dim_index = dim_word.max(1) as usize;
            let dim_size = dims.get(dim_index.saturating_sub(1)).copied().unwrap_or(0);
            let err = match code {
                1 => anyhow!(
                    "sub2ind: subscript in dimension {} must be finite",
                    dim_index
                ),
                2 => anyhow!(
                    "sub2ind: subscript in dimension {} must be an integer",
                    dim_index
                ),
                3 => anyhow!(
                    "sub2ind: subscript {} exceeds dimension {} (size {})",
                    extra as isize,
                    dim_index,
                    dim_size
                ),
                _ => anyhow!("sub2ind: kernel reported error code {}", code),
            };
            return Err(err);
        }

        Ok(self.register_existing_buffer_with_usage(
            output_buffer,
            output_shape.to_vec(),
            len,
            BufferUsageClass::FusionOut,
        ))
    }

    pub(crate) fn ind2sub_exec(
        &self,
        dims: &[usize],
        strides: &[usize],
        indices: &GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &[usize],
    ) -> Result<Vec<GpuTensorHandle>> {
        if dims.len() != strides.len() {
            return Err(anyhow!("ind2sub: size vector mismatch"));
        }
        let expected_len: usize = output_shape.iter().copied().product();
        if expected_len != len {
            return Err(anyhow!("ind2sub: output shape does not match index tensor"));
        }
        if len == 0 {
            let mut handles = Vec::with_capacity(dims.len());
            for _ in 0..dims.len() {
                let buffer = self.create_storage_buffer(0, "runmat-ind2sub-empty");
                handles.push(self.register_existing_buffer(buffer, output_shape.to_vec(), 0));
            }
            return Ok(handles);
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || total > u32::MAX as usize
            || len > u32::MAX as usize
        {
            return Err(anyhow!("ind2sub: dimensions exceed GPU kernel limits"));
        }
        let storage_bindings =
            super::backend_shared::checked_binding_count("ind2sub", dims.len(), 2)?;
        let total_bindings =
            super::backend_shared::checked_binding_count("ind2sub", dims.len(), 3)?;
        super::backend_shared::validate_compute_binding_counts(
            "ind2sub",
            storage_bindings,
            total_bindings,
            &self.device_ref().limits(),
        )?;

        let entry = self.get_entry(indices)?;
        if entry.len != len {
            return Err(anyhow!(
                "ind2sub: index tensor length does not match provided shape"
            ));
        }

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&s| s as u32).collect();

        let mut output_buffers = Vec::with_capacity(dims.len());
        for _ in 0..dims.len() {
            output_buffers.push(self.create_storage_buffer_checked(len, "runmat-ind2sub-out")?);
        }

        let error_bytes = vec![0u8; std::mem::size_of::<u32>() * 4];
        let error_buffer = Arc::new(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-ind2sub-error"),
                contents: &error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let (scalar_ty, epsilon) = match self.provider_precision_exec() {
            runmat_accelerate_api::ProviderPrecision::F64 => ("f64", "1.0e-12"),
            runmat_accelerate_api::ProviderPrecision::F32 => ("f32", "1.0e-5"),
        };
        let workgroup_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let shader = crate::backend::wgpu::shaders::ind2sub::build_ind2sub_shader(
            scalar_ty,
            &dims_u32,
            &strides_u32,
            total as u32,
            workgroup_size,
            epsilon,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-ind2sub-module",
            &shader,
        );
        let bgl = crate::backend::wgpu::bindings::build_ind2sub_bgl(self.device_ref(), dims.len());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-ind2sub-pl",
            &bgl,
        );
        let layout_tag = format!("runmat-ind2sub-layout-{}", dims.len());
        let key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), &layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-ind2sub",
            Some(shader.as_bytes()),
            Some(layout_tag.as_str()),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let params = Params {
            len: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-ind2sub-params");

        let mut bind_entries = Vec::with_capacity(dims.len() + 3);
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: entry.buffer.as_ref().as_entire_binding(),
        });
        for (idx, buffer) in output_buffers.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: (idx + 1) as u32,
                resource: buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (dims.len() + 1) as u32,
            resource: error_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (dims.len() + 2) as u32,
            resource: params_buffer.as_entire_binding(),
        });
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-ind2sub-bg"),
                layout: &bgl,
                entries: &bind_entries,
            });

        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(len as u32, workgroup_size);
        crate::backend::wgpu::dispatch::ind2sub::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bind_group,
            groups,
        );

        let error_size = (std::mem::size_of::<u32>() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-ind2sub-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-ind2sub-error-copy"),
            });
        encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, error_size, "ind2sub")?;
        let words: &[u32] = cast_slice(&bytes);
        let code = words.first().copied().unwrap_or(0);

        if code != 0 {
            let err = match code {
                1..=3 => anyhow!("Linear indices must be positive integers."),
                4 => anyhow!("Linear index exceeds GPU kernel limits."),
                _ => anyhow!("ind2sub: kernel reported error code {}", code),
            };
            return Err(err);
        }

        let mut handles = Vec::with_capacity(output_buffers.len());
        for buffer in output_buffers {
            handles.push(self.register_existing_buffer(buffer, output_shape.to_vec(), len));
        }
        Ok(handles)
    }

    pub(crate) fn scatter_row_exec(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let raw_matrix = self.get_entry_raw(matrix)?;
        if raw_matrix.integer_type().is_some() {
            ensure!(
                raw_matrix.shape.len() == 2,
                "scatter_row: only 2D tensors supported"
            );
            let rows = raw_matrix.shape[0];
            let cols = raw_matrix.shape[1];
            ensure!(row_index < rows, "scatter_row: row index out of bounds");
            let raw_values = self.get_entry_raw(values)?;
            ensure!(
                raw_values.integer_type() == raw_matrix.integer_type(),
                "scatter_row: integer storage mismatch"
            );
            ensure!(
                raw_values.shape.len() <= 2
                    && (raw_values.shape.len() == 1 || raw_values.shape[1] == cols),
                "scatter_row: length mismatch"
            );
            let indices: Vec<u32> = (0..cols)
                .map(|col| {
                    u32::try_from(col * rows + row_index)
                        .map_err(|_| anyhow!("scatter_row: index exceeds u32"))
                })
                .collect::<Result<_>>()?;
            let all_indices: Vec<u32> = (0..rows * cols)
                .map(|index| {
                    u32::try_from(index).map_err(|_| anyhow!("scatter_row: index exceeds u32"))
                })
                .collect::<Result<_>>()?;
            let out = self.gather_linear_exec(matrix, &all_indices, &raw_matrix.shape)?;
            self.scatter_linear_exec(&out, &indices, values)?;
            return Ok(out);
        }
        let m_entry = self.get_entry(matrix)?;
        if m_entry.shape.len() != 2 {
            return Err(anyhow!("scatter_row: only 2D tensors supported"));
        }
        let rows = m_entry.shape[0];
        let cols = m_entry.shape[1];
        if row_index >= rows {
            return Err(anyhow!("scatter_row: row index out of bounds"));
        }
        let v_entry = self.get_entry(values)?;
        let v_cols = if v_entry.shape.len() == 1 {
            v_entry.shape[0]
        } else if v_entry.shape.len() == 2 {
            v_entry.shape[1]
        } else {
            return Err(anyhow!("scatter_row: values must be vector or [1,cols]"));
        };
        if v_cols != cols {
            return Err(anyhow!("scatter_row: length mismatch"));
        }
        let shader = crate::backend::wgpu::shaders::scatter::SCATTER_ROW_SHADER_F32;
        let out_buffer = self.create_storage_buffer(rows * cols, "runmat-scatter-row-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-scatter-row-copy"),
                    });
            enc.copy_buffer_to_buffer(
                m_entry.buffer.as_ref(),
                0,
                out_buffer.as_ref(),
                0,
                (rows * cols * self.element_size) as u64,
            );
            self.submit(enc);
        }
        let bgl = crate::backend::wgpu::bindings::build_scatter_row_bgl(self.device_ref());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-scatter-row-pl",
            &bgl,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-scatter-row-module",
            shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-scatter-row-bgl",
            Some(256),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-scatter-row",
            Some(shader.as_bytes()),
            Some("runmat-scatter-row-bgl"),
            Some(256),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pm {
            rows: u32,
            cols: u32,
            i: u32,
        }
        let params = Pm {
            rows: rows as u32,
            cols: cols as u32,
            i: row_index as u32,
        };
        let pbuf = self.uniform_buffer(&params, "runmat-scatter-row-params");
        let bg = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-row-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: m_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pbuf.as_entire_binding(),
                    },
                ],
            });
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            cols as u32,
            crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scatter::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bg,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols))
    }

    pub(crate) fn gather_linear_exec(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(source)?;
        let integer_type = entry.integer_type();
        let expected = product_checked(output_shape)
            .ok_or_else(|| anyhow!("gather_linear: output shape product overflow"))?;
        let lane_factor = linear_storage_lane_factor(integer_type, entry.storage);
        if integer_type.is_none() {
            ensure!(
                entry.len % lane_factor == 0,
                "gather_linear: source raw length {} is not aligned to storage lane factor {}",
                entry.len,
                lane_factor
            );
        }
        let source_logical_len = if integer_type.is_some() {
            entry.len
        } else {
            entry.len / lane_factor
        };
        let _span = info_span!(
            "gpu.gather_linear",
            source_len = entry.len,
            source_logical_len,
            index_count = indices.len(),
            output_size = expected
        )
        .entered();
        ensure!(
            expected == indices.len(),
            "gather_linear: index count {} does not match output size {}",
            indices.len(),
            expected
        );
        if expected == 0 {
            let out = self.create_storage_buffer(0, "runmat-gather-linear-empty");
            let handle = if let Some(element_type) = integer_type {
                self.register_integer_buffer(out, output_shape.to_vec(), 0, element_type, 0)
            } else {
                self.register_existing_buffer_with_storage(
                    out,
                    output_shape.to_vec(),
                    0,
                    entry.storage,
                )
            };
            if runmat_accelerate_api::handle_is_logical(source) {
                runmat_accelerate_api::set_handle_logical(&handle, true);
            }
            return Ok(handle);
        }
        ensure!(
            indices.len() <= u32::MAX as usize,
            "gather_linear: index count exceeds GPU limits"
        );
        ensure!(
            expected.checked_mul(lane_factor).is_some(),
            "gather_linear: output raw length overflow"
        );
        ensure!(
            indices
                .iter()
                .all(|&idx| (idx as usize) < source_logical_len),
            "gather_linear: index out of bounds for source tensor"
        );
        let indices_len_bytes = std::mem::size_of_val(indices) as u64;
        let indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-gather-linear-indices"),
            size: indices_len_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        if !indices.is_empty() {
            self.queue
                .write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
        }
        log::trace!(
            "gather_linear begin source_buffer={} ptr=0x{:x} out_shape={:?} count={}",
            source.buffer_id,
            entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            output_shape,
            indices.len()
        );

        let raw_output_len = expected * lane_factor;
        let out_buffer =
            self.create_storage_buffer_checked(raw_output_len, "runmat-gather-linear-out")?;
        let params = LinearGatherParams {
            count: indices.len() as u32,
            lane_factor: lane_factor as u32,
            _pad: [0; 2],
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-gather-linear-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-gather-linear-bind"),
                layout: &self.pipelines.gather_linear.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: indices_buffer.as_ref().as_entire_binding(),
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
            indices.len() as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.gather_linear.pipeline,
            &bind_group,
            workgroups,
            "runmat-gather-linear-encoder",
            "runmat-gather-linear-pass",
        );
        log::trace!(
            "gather_linear complete source_buffer={} out_ptr=0x{:x} count={}",
            source.buffer_id,
            out_buffer.as_ref() as *const wgpu::Buffer as usize,
            indices.len()
        );

        let handle = if let Some(element_type) = integer_type {
            self.register_integer_buffer(
                out_buffer,
                output_shape.to_vec(),
                expected,
                element_type,
                (raw_output_len * std::mem::size_of::<u32>()) as u64,
            )
        } else {
            self.register_existing_buffer_with_storage(
                out_buffer,
                output_shape.to_vec(),
                raw_output_len,
                entry.storage,
            )
        };
        if runmat_accelerate_api::handle_is_logical(source) {
            runmat_accelerate_api::set_handle_logical(&handle, true);
        }
        Ok(handle)
    }

    pub(crate) fn scatter_linear_exec(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        ensure!(
            indices.len() <= u32::MAX as usize,
            "scatter_linear: index count exceeds GPU limits"
        );
        let target_entry = self.get_entry_raw(target)?;
        let values_entry = self.get_entry_raw(values)?;
        let integer_type = target_entry.integer_type();
        ensure!(
            integer_type == values_entry.integer_type(),
            "scatter_linear: integer storage mismatch target={:?} values={:?}",
            integer_type,
            values_entry.integer_type()
        );
        ensure!(
            target_entry.storage == values_entry.storage,
            "scatter_linear: storage mismatch target={:?} values={:?}",
            target_entry.storage,
            values_entry.storage
        );
        let lane_factor = linear_storage_lane_factor(integer_type, target_entry.storage);
        if integer_type.is_none() {
            ensure!(
                target_entry.len % lane_factor == 0,
                "scatter_linear: target raw length {} is not aligned to storage lane factor {}",
                target_entry.len,
                lane_factor
            );
        }
        let target_logical_len = if integer_type.is_some() {
            target_entry.len
        } else {
            target_entry.len / lane_factor
        };
        let _span = info_span!(
            "gpu.scatter_linear",
            target_len = target_entry.len,
            target_logical_len,
            index_count = indices.len(),
            values_len = values_entry.len
        )
        .entered();
        let expected_values_len =
            linear_values_entry_len(integer_type, target_entry.storage, indices.len())
                .ok_or_else(|| anyhow!("scatter_linear: values length overflow"))?;
        ensure!(
            values_entry.len == expected_values_len,
            "scatter_linear: values length {} does not match index count {} for lane factor {}",
            values_entry.len,
            indices.len(),
            lane_factor
        );
        ensure!(
            indices
                .iter()
                .all(|&idx| (idx as usize) < target_logical_len),
            "scatter_linear: index out of bounds for target tensor"
        );
        let indices_len_bytes = std::mem::size_of_val(indices) as u64;
        let indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-scatter-linear-indices"),
            size: indices_len_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.queue
            .write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
        log::trace!(
            "scatter_linear begin target_buffer={} target_ptr=0x{:x} values_buffer={} values_ptr=0x{:x} count={}",
            target.buffer_id,
            target_entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            values.buffer_id,
            values_entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            indices.len()
        );
        let params = LinearScatterParams {
            count: indices.len() as u32,
            lane_factor: lane_factor as u32,
            _pad: [0; 2],
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-scatter-linear-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-linear-bind"),
                layout: &self.pipelines.scatter_linear.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: target_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            indices.len() as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.scatter_linear.pipeline,
            &bind_group,
            workgroups,
            "runmat-scatter-linear-encoder",
            "runmat-scatter-linear-pass",
        );
        log::trace!(
            "scatter_linear complete target_buffer={} values_buffer={} count={}",
            target.buffer_id,
            values.buffer_id,
            indices.len()
        );
        Ok(())
    }

    pub(crate) fn find_exec(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        let entry = self.get_entry(a)?;
        ensure!(
            entry.precision == NumericPrecision::F64,
            "find: exact double index outputs require f64 provider precision"
        );
        let total = entry.len;
        if total == 0 {
            let shape = vec![0, 1];
            let indices = self.create_storage_buffer(0, "runmat-find-empty-indices");
            let rows = self.create_storage_buffer(0, "runmat-find-empty-rows");
            let cols = self.create_storage_buffer(0, "runmat-find-empty-cols");
            let values = self.create_storage_buffer(0, "runmat-find-empty-values");
            let linear = self.register_existing_buffer(indices, shape.clone(), 0);
            let rows_handle = self.register_existing_buffer(rows, shape.clone(), 0);
            let cols_handle = self.register_existing_buffer(cols, shape.clone(), 0);
            let values_handle = self.register_existing_buffer(values, shape, 0);
            return Ok(ProviderFindResult {
                linear,
                rows: rows_handle,
                cols: cols_handle,
                values: Some(values_handle),
            });
        }

        ensure!(
            total <= u32::MAX as usize,
            "find: tensor length exceeds GPU support"
        );

        let rows_extent = entry.shape.first().copied().unwrap_or(1).max(1);
        ensure!(
            rows_extent <= u32::MAX as usize,
            "find: row extent exceeds GPU support"
        );

        let raw_cap = match direction {
            FindDirection::First => limit.unwrap_or(total),
            FindDirection::Last => limit.unwrap_or(1),
        };
        let cap = raw_cap.min(total);

        if cap == 0 {
            let shape = vec![0, 1];
            let indices = self.create_storage_buffer(0, "runmat-find-zero-limit-indices");
            let rows = self.create_storage_buffer(0, "runmat-find-zero-limit-rows");
            let cols = self.create_storage_buffer(0, "runmat-find-zero-limit-cols");
            let values = self.create_storage_buffer(0, "runmat-find-zero-limit-values");
            let linear = self.register_existing_buffer(indices, shape.clone(), 0);
            let rows_handle = self.register_existing_buffer(rows, shape.clone(), 0);
            let cols_handle = self.register_existing_buffer(cols, shape.clone(), 0);
            let values_handle = self.register_existing_buffer(values, shape, 0);
            return Ok(ProviderFindResult {
                linear,
                rows: rows_handle,
                cols: cols_handle,
                values: Some(values_handle),
            });
        }

        ensure!(cap <= u32::MAX as usize, "find: limit exceeds GPU support");

        let indices_buffer = self.create_storage_buffer(cap, "runmat-find-indices");
        let rows_buffer = self.create_storage_buffer(cap, "runmat-find-rows");
        let cols_buffer = self.create_storage_buffer(cap, "runmat-find-cols");
        let values_buffer = self.create_storage_buffer(cap, "runmat-find-values");

        let count_storage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-find-count-storage"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&count_storage, 0, bytemuck::cast_slice(&[0u32, 0u32]));
        let count_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-find-count-staging"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = crate::backend::wgpu::params::FindParams {
            len: total as u32,
            limit: cap as u32,
            rows: rows_extent as u32,
            direction: match direction {
                FindDirection::First => 0,
                FindDirection::Last => 1,
            },
            include_values: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-find-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-find-bind"),
                layout: &self.pipelines.find.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rows_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cols_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: values_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: count_storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        crate::backend::wgpu::dispatch::find::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.find.pipeline,
            &bind_group,
        );

        let mut copy_encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-find-copy-count"),
                });
        copy_encoder.copy_buffer_to_buffer(&count_storage, 0, &count_staging, 0, 8);
        self.submit(copy_encoder);
        let bytes = self.map_readback_bytes_sync(count_staging, 8, "find")?;
        let counts: &[u32] = cast_slice(&bytes);
        let count = counts.first().copied().unwrap_or(0) as usize;

        let shape = vec![count, 1];
        let linear = self.register_existing_buffer(indices_buffer, shape.clone(), count);
        let rows_handle = self.register_existing_buffer(rows_buffer, shape.clone(), count);
        let cols_handle = self.register_existing_buffer(cols_buffer, shape.clone(), count);
        let values_handle = self.register_existing_buffer(values_buffer, shape, count);

        Ok(ProviderFindResult {
            linear,
            rows: rows_handle,
            cols: cols_handle,
            values: Some(values_handle),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, HostNumericDataView,
        HostNumericTensorView,
    };

    #[test]
    fn linear_scatter_length_uses_logical_counts_for_all_integer_types() {
        use runmat_accelerate_api::IntegerElementType;

        for integer_type in [
            IntegerElementType::I8,
            IntegerElementType::I16,
            IntegerElementType::I32,
            IntegerElementType::I64,
            IntegerElementType::U8,
            IntegerElementType::U16,
            IntegerElementType::U32,
            IntegerElementType::U64,
        ] {
            assert_eq!(
                linear_values_entry_len(Some(integer_type), GpuTensorStorage::Real, 3),
                Some(3),
                "{integer_type:?} must use logical element count"
            );
        }
        assert_eq!(
            linear_values_entry_len(None, GpuTensorStorage::Real, 3),
            Some(3)
        );
        assert_eq!(
            linear_values_entry_len(None, GpuTensorStorage::ComplexInterleaved, 3),
            Some(6)
        );
    }

    #[test]
    fn wgpu_linear_gather_and_scatter_copy_complex_logical_elements() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let source = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F64(&data),
                shape: &[1, 4],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("upload");

        let gathered = provider
            .gather_linear_exec(&source, &[1, 3], &[1, 2])
            .expect("gather");
        assert_eq!(
            runmat_accelerate_api::handle_storage(&gathered),
            GpuTensorStorage::ComplexInterleaved
        );
        let host = block_on(provider.download_exec(&gathered)).expect("download gathered");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 2]);
        assert_eq!(host.data, vec![2.0, 20.0, 4.0, 40.0]);

        let target = provider
            .zeros_with_storage_exec(&[1, 4], GpuTensorStorage::ComplexInterleaved)
            .expect("zeros");
        provider
            .scatter_linear_exec(&target, &[0, 2], &gathered)
            .expect("scatter");
        let host = block_on(provider.download_exec(&target)).expect("download target");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 4]);
        assert_eq!(host.data, vec![2.0, 20.0, 0.0, 0.0, 4.0, 40.0, 0.0, 0.0]);
    }

    #[test]
    fn wgpu_linear_gather_and_scatter_preserve_u64_words() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [0_u64, 1_u64 << 63, u64::MAX];
        let source = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[1, 3],
            })
            .expect("upload exact u64");
        let gathered = provider
            .gather_linear_exec(&source, &[2, 1], &[1, 2])
            .expect("gather exact u64");
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&gathered),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let host = block_on(provider.download_integer_exec(&gathered)).expect("download gathered");
        assert_eq!(
            host.data,
            HostIntegerDataOwned::U64(vec![u64::MAX, 1_u64 << 63])
        );

        let target = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0, 0, 0]),
                shape: &[1, 3],
            })
            .expect("upload target");
        provider
            .scatter_linear_exec(&target, &[0, 2], &gathered)
            .expect("scatter exact u64");
        let host = block_on(provider.download_integer_exec(&target)).expect("download target");
        assert_eq!(
            host.data,
            HostIntegerDataOwned::U64(vec![u64::MAX, 0, 1_u64 << 63])
        );
    }

    #[test]
    fn wgpu_linear_gather_and_scatter_preserve_all_native_integer_classes() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        macro_rules! check {
            ($view:ident, $owned:ident, $values:expr, $expected:expr) => {{
                let source = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&$values),
                        shape: &[1, 3],
                    })
                    .expect("upload integer source");
                let gathered = provider
                    .gather_linear_exec(&source, &[2, 0], &[1, 2])
                    .expect("gather integer values");
                assert_eq!(
                    block_on(provider.download_integer_exec(&gathered))
                        .expect("download gathered")
                        .data,
                    HostIntegerDataOwned::$owned(vec![($expected)[0], ($expected)[1]])
                );
                let target = provider
                    .upload_integer_exec(&HostIntegerTensorView {
                        data: HostIntegerDataView::$view(&[0 as _, 0 as _, 0 as _]),
                        shape: &[1, 3],
                    })
                    .expect("upload integer target");
                provider
                    .scatter_linear_exec(&target, &[0, 2], &gathered)
                    .expect("scatter integer values");
                assert_eq!(
                    block_on(provider.download_integer_exec(&target))
                        .expect("download target")
                        .data,
                    HostIntegerDataOwned::$owned(vec![($expected)[0], 0, ($expected)[1]])
                );
            }};
        }
        check!(I8, I8, [i8::MIN, -1, i8::MAX], [i8::MAX, i8::MIN]);
        check!(I16, I16, [i16::MIN, -1, i16::MAX], [i16::MAX, i16::MIN]);
        check!(I32, I32, [i32::MIN, -1, i32::MAX], [i32::MAX, i32::MIN]);
        check!(I64, I64, [i64::MIN, -1, i64::MAX], [i64::MAX, i64::MIN]);
        check!(U8, U8, [0, 1, u8::MAX], [u8::MAX, 0]);
        check!(U16, U16, [0, 1, u16::MAX], [u16::MAX, 0]);
        check!(U32, U32, [0, 1, u32::MAX], [u32::MAX, 0]);
        check!(U64, U64, [0, 1, u64::MAX], [u64::MAX, 0]);
    }

    #[test]
    fn wgpu_row_and_column_scatter_preserve_native_u64_words() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let matrix = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0, 0, 0, 0, 0, 0]),
                shape: &[2, 3],
            })
            .expect("upload matrix");
        let column = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[u64::MAX, 1 << 63]),
                shape: &[2, 1],
            })
            .expect("upload column");
        let after_column = provider
            .scatter_column_exec(&matrix, 1, &column)
            .expect("scatter column");
        let row = provider
            .upload_integer_exec(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[7, 8, 9]),
                shape: &[1, 3],
            })
            .expect("upload row");
        let after_row = provider
            .scatter_row_exec(&after_column, 0, &row)
            .expect("scatter row");
        assert_eq!(
            block_on(provider.download_integer_exec(&after_row))
                .expect("download")
                .data,
            HostIntegerDataOwned::U64(vec![7, 0, 8, 1 << 63, 9, 0])
        );
        for handle in [&matrix, &column, &after_column, &row, &after_row] {
            provider.free_exec(handle).expect("free handle");
        }
    }
}
