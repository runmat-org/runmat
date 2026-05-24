use super::*;

impl WgpuProvider {
    pub(crate) fn bandwidth_exec(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
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

        let pipeline = &self.pipelines.bandwidth;
        let output_init = [0u32, 0u32];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        let params_buffer = self.uniform_buffer(&params, "runmat-bandwidth-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            self.device_ref(),
            self.queue_ref(),
            &pipeline.pipeline,
            &bind_group,
            groups,
        );

        let staging_size = (std::mem::size_of::<u32>() * 2) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-bandwidth-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-bandwidth-copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
        self.submit(encoder);

        let bytes = self.map_readback_bytes_sync(staging, staging_size, "bandwidth")?;
        let words: &[u32] = cast_slice(&bytes);
        let lower = words.first().copied().unwrap_or(0);
        let upper = words.get(1).copied().unwrap_or(0);

        Ok(ProviderBandwidth { lower, upper })
    }
    pub(crate) fn syrk_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("syrk: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let out_shape = vec![cols, cols];
        let len = cols
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("syrk: output size overflow"))?;

        let out_bytes = (len as u64) * (self.element_size as u64);
        let out_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::SyrkOut,
            out_bytes,
            "runmat-syrk-out-scratch",
        );
        if len == 0 {
            return Ok(self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                0,
                BufferUsageClass::SyrkOut,
            ));
        }

        let rows_u32 =
            u32::try_from(rows).map_err(|_| anyhow!("syrk: row count exceeds GPU limits"))?;
        let cols_u32 =
            u32::try_from(cols).map_err(|_| anyhow!("syrk: column count exceeds GPU limits"))?;
        let lda_u32 = rows_u32;
        let ldc_u32 = cols_u32;

        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(cols_u32, tile);
        let groups_y = groups_x;

        const SYRK_ROW_CHUNK: usize = 32768;
        let mut offset = 0usize;
        let mut first_chunk = true;
        while offset < rows {
            let remaining = rows - offset;
            let chunk_rows = remaining.min(SYRK_ROW_CHUNK.max(1));
            let chunk_rows_u32 = u32::try_from(chunk_rows)
                .map_err(|_| anyhow!("syrk: chunk rows exceed GPU limits"))?;
            let row_offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("syrk: row offset exceeds GPU limits"))?;

            let mut flags = SYRK_FLAG_FILL_BOTH;
            if !first_chunk {
                flags |= SYRK_FLAG_ACCUMULATE;
            }

            let params = SyrkParams {
                rows_total: rows_u32,
                cols: cols_u32,
                lda: lda_u32,
                ldc: ldc_u32,
                row_offset: row_offset_u32,
                chunk_rows: chunk_rows_u32,
                flags,
                offset_out: 0,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::SyrkParams,
                std::mem::size_of::<crate::backend::wgpu::params::SyrkParams>() as u64,
                "runmat-syrk-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
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
                    resource: params_buffer.as_entire_binding(),
                },
            ];
            let layout = &self.pipelines.syrk.layout;
            let bind_group = self
                .bind_group_cache
                .get_or_create(layout, &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-syrk-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                });

            crate::backend::wgpu::dispatch::syrk::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.syrk.pipeline,
                bind_group.as_ref(),
                groups_x,
                groups_y,
            );

            offset += chunk_rows;
            first_chunk = false;
        }

        Ok(self.register_existing_buffer_with_usage(
            out_buffer,
            out_shape,
            len,
            BufferUsageClass::SyrkOut,
        ))
    }

    pub(crate) fn matmul_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.matmul_exec_with_usage(a, b, BufferUsageClass::MatmulOut)
    }
    fn matmul_exec_with_usage(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        out_usage: BufferUsageClass,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
            return Err(anyhow!("matmul: only 2D tensors supported"));
        }

        let view_a = build_matrix_operand_view(a, &entry_a).map_err(|e| anyhow!("matmul: {e}"))?;
        let view_b = build_matrix_operand_view(b, &entry_b).map_err(|e| anyhow!("matmul: {e}"))?;

        if view_a.cols != view_b.rows {
            return Err(anyhow!("matmul: inner dimensions must match"));
        }

        let m = view_a.rows;
        let n = view_b.cols;
        let k = view_a.cols;

        let debug_matmul = std::env::var("RUNMAT_DEBUG_MATMUL").is_ok();
        if debug_matmul {
            log::debug!(
                "[matmul_debug] ptr_a={:p} ptr_b={:p}",
                entry_a.buffer.as_ref(),
                entry_b.buffer.as_ref()
            );
            log::debug!(
                "[matmul_debug] m={} n={} k={} lda={} ldb={} transpose_a={} transpose_b={}",
                m,
                n,
                k,
                view_a.lda,
                view_b.lda,
                view_a.transpose,
                view_b.transpose
            );
        }

        let out_shape = vec![m, n];
        let len = m * n;
        if len == 0 {
            let (out_buffer, _) =
                self.create_storage_buffer_for_usage(out_usage, 0, "runmat-matmul-out");
            return Ok(
                self.register_existing_buffer_with_usage(out_buffer, out_shape, 0, out_usage)
            );
        }

        let m_u32 = u32::try_from(m).map_err(|_| anyhow!("matmul: m exceeds GPU limits"))?;
        let n_u32 = u32::try_from(n).map_err(|_| anyhow!("matmul: n exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k).map_err(|_| anyhow!("matmul: k exceeds GPU limits"))?;

        const K_CHUNK: usize = 8192;
        const K_CHUNK_SWITCH: usize = 65536; // only chunk for very large k to avoid regressions

        let can_vec4 = self.precision == NumericPrecision::F32
            && !view_a.transpose
            && !view_b.transpose
            && m % 4 == 0
            && m >= 4
            && n > 0;
        let disable_vec4 = std::env::var("RUNMAT_DISABLE_VEC4")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True"))
            .unwrap_or(false);
        let use_vec4 = can_vec4 && k < K_CHUNK_SWITCH && !disable_vec4;
        let enable_chunk = !view_a.transpose && !view_b.transpose && k >= K_CHUNK_SWITCH;
        if debug_matmul {
            log::debug!(
                "[matmul_debug] can_vec4={} use_vec4={} enable_chunk={} usage={:?}",
                can_vec4,
                use_vec4,
                enable_chunk,
                out_usage
            );
        }

        let start = Instant::now();

        if enable_chunk {
            self.prepare_matmul_pipeline();
            self.device_ref().poll(wgpu::Maintain::Poll);
            let lda_u32 = view_a.lda;
            let ldb_u32 = view_b.lda;
            // Accumulator handle across chunks
            let mut acc: Option<GpuTensorHandle> = None;
            let mut k_off: usize = 0;
            let partial_storage = self.create_storage_buffer_checked_with_usage(
                len,
                "runmat-matmul-partial",
                BufferUsageClass::MatmulPartial,
            )?;
            while k_off < k {
                let k_sub = std::cmp::min(K_CHUNK, k - k_off);
                // Create partial output buffer and bind group
                let partial_buffer = partial_storage.clone();
                let offset_a_elems = k_off
                    .checked_mul(view_a.rows)
                    .ok_or_else(|| anyhow!("matmul: offset overflow"))?;
                let offset_a_u32 = u32::try_from(offset_a_elems)
                    .map_err(|_| anyhow!("matmul: A offset exceeds GPU limits"))?;
                let offset_b_u32 = u32::try_from(k_off)
                    .map_err(|_| anyhow!("matmul: B offset exceeds GPU limits"))?;
                let params = crate::backend::wgpu::params::MatmulParams {
                    m: m_u32,
                    n: n_u32,
                    k: k_sub as u32,
                    lda: lda_u32,
                    ldb: ldb_u32,
                    ldc: m_u32,
                    offset_a: offset_a_u32,
                    offset_b: offset_b_u32,
                    offset_out: 0,
                    flags: 0,
                };
                let params_buffer = self.kernel_resources.uniform_buffer(
                    self.device_ref(),
                    UniformBufferKey::MatmulParams,
                    std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
                    "runmat-matmul-params",
                );
                self.queue
                    .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
                let bind_entries = [
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry_a.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: entry_b.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: partial_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ];
                let layout = &self.pipelines.matmul.layout;
                let bg =
                    self.bind_group_cache
                        .get_or_create(layout, &bind_entries, || {
                            Arc::new(self.device_ref().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-matmul-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        });
                let tile = crate::backend::wgpu::config::effective_matmul_tile();
                let groups_x =
                    crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
                let groups_y =
                    crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);
                crate::backend::wgpu::dispatch::matmul::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.matmul.pipeline,
                    bg.as_ref(),
                    groups_x,
                    groups_y,
                );
                // Wrap partial buffer into handle
                let partial = self.register_existing_buffer_with_usage(
                    partial_buffer,
                    out_shape.clone(),
                    len,
                    BufferUsageClass::MatmulPartial,
                );
                acc = match acc {
                    None => Some(partial),
                    Some(prev) => {
                        let sum = self.binary_op_exec(
                            crate::backend::wgpu::types::BinaryOpCode::Add,
                            &prev,
                            &partial,
                        )?;
                        self.free(&prev).ok();
                        self.free(&partial).ok();
                        Some(sum)
                    }
                };
                k_off += k_sub;
            }
            let handle = acc.expect("matmul chunking produced no output");
            self.remember_matmul_sources(&handle, a, b);
            self.mark_buffer_usage(&handle, out_usage);
            self.telemetry.record_matmul_duration(start.elapsed());
            self.record_matmul_kernel_launch(m, n, k, use_vec4, true);
            return Ok(handle);
        }

        // Default single-dispatch path
        let out_buffer =
            self.create_storage_buffer_checked_with_usage(len, "runmat-matmul-out", out_usage)?;
        if use_vec4 {
            self.prepare_matmul_vec4_pipeline();
        } else {
            self.prepare_matmul_pipeline();
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        let mut flags = 0u32;
        if view_a.transpose {
            flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
        }
        if view_b.transpose {
            flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
        }
        let params = crate::backend::wgpu::params::MatmulParams {
            m: m_u32,
            n: n_u32,
            k: k_u32,
            lda: view_a.lda,
            ldb: view_b.lda,
            ldc: m_u32,
            offset_a: 0,
            offset_b: 0,
            offset_out: 0,
            flags,
        };
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::MatmulParams,
            std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
            "runmat-matmul-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
        let layout = if use_vec4 {
            &self.pipelines.matmul_vec4.layout
        } else {
            &self.pipelines.matmul.layout
        };
        let pipeline = if use_vec4 {
            &self.pipelines.matmul_vec4.pipeline
        } else {
            &self.pipelines.matmul.pipeline
        };
        let bind_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: entry_a.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: entry_b.buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bg = if out_usage == BufferUsageClass::MatmulOut {
            Arc::new(
                self.device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-matmul-bind"),
                        layout,
                        entries: &bind_entries,
                    }),
            )
        } else {
            self.bind_group_cache
                .get_or_create(layout, &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-matmul-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                })
        };
        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
        let groups_y = if use_vec4 {
            let rows_vec = (m as u32) / 4;
            crate::backend::wgpu::dispatch::common::dispatch_size_dim(rows_vec, tile)
        } else {
            crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile)
        };
        crate::backend::wgpu::dispatch::matmul::run(
            self.device_ref(),
            self.queue_ref(),
            pipeline,
            bg.as_ref(),
            groups_x,
            groups_y,
        );
        let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer;
        let handle =
            self.register_existing_buffer_with_usage(out_buffer, out_shape, len, out_usage);
        if debug_matmul {
            log::debug!("[matmul_debug] out_ptr={:p} len={}", out_ptr, len);
        }
        self.remember_matmul_sources(&handle, a, b);
        self.telemetry.record_matmul_duration(start.elapsed());
        self.record_matmul_kernel_launch(m, n, k, use_vec4, false);
        Ok(handle)
    }
    pub(crate) fn pagefun_exec(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        match request.op {
            PagefunOp::Mtimes => self.pagefun_mtimes_exec(request),
        }
    }
    fn pagefun_mtimes_exec(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        ensure!(
            request.inputs.len() == 2,
            "pagefun: @mtimes expects exactly two inputs"
        );
        ensure!(
            request.input_page_dims.len() == request.inputs.len(),
            "pagefun: input metadata mismatch"
        );

        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let entry_a = self.get_entry(lhs)?;
        let entry_b = self.get_entry(rhs)?;

        let view_a = build_matrix_operand_view(lhs, &entry_a)
            .map_err(|e| anyhow!("pagefun @mtimes: {e}"))?;
        let view_b = build_matrix_operand_view(rhs, &entry_b)
            .map_err(|e| anyhow!("pagefun @mtimes: {e}"))?;

        let canonical_a = canonical_matrix_shape(&entry_a.shape);
        let canonical_b = canonical_matrix_shape(&entry_b.shape);
        ensure!(
            canonical_a.len() >= 2 && canonical_b.len() >= 2,
            "pagefun: @mtimes operands must be at least 2-D"
        );

        let rows = view_a.rows;
        let k_a = view_a.cols;
        let k_b = view_b.rows;
        let cols = view_b.cols;
        ensure!(
            k_a == k_b,
            "pagefun: inner matrix dimensions must agree ({} vs {})",
            k_a,
            k_b
        );

        let rank = request.page_dims.len();
        let lhs_dims = pad_dims(request.input_page_dims[0].clone(), rank);
        let rhs_dims = pad_dims(request.input_page_dims[1].clone(), rank);
        let lhs_strides = compute_page_strides(&lhs_dims);
        let rhs_strides = compute_page_strides(&rhs_dims);

        let lhs_page_size = rows
            .checked_mul(k_a)
            .ok_or_else(|| anyhow!("pagefun: lhs page size overflow"))?;
        let rhs_page_size = k_b
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("pagefun: rhs page size overflow"))?;
        let out_page_size = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("pagefun: output page size overflow"))?;

        let page_volume = if rank == 0 {
            1
        } else {
            product_checked(&request.page_dims)
                .ok_or_else(|| anyhow!("pagefun: page dimensions overflow"))?
        };

        let total_len = out_page_size
            .checked_mul(page_volume)
            .ok_or_else(|| anyhow!("pagefun: output size overflow"))?;
        let out_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-pagefun-mtimes-out")?;

        if total_len == 0 {
            return Ok(self.register_existing_buffer(
                out_buffer,
                request.output_shape.clone(),
                total_len,
            ));
        }

        let m_u32 = u32::try_from(rows)
            .map_err(|_| anyhow!("pagefun: matrix row count exceeds GPU limits"))?;
        let n_u32 = u32::try_from(cols)
            .map_err(|_| anyhow!("pagefun: matrix column count exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k_a)
            .map_err(|_| anyhow!("pagefun: shared dimension exceeds GPU limits"))?;

        let lda = view_a.lda;
        let ldb = view_b.lda;
        let ldc = m_u32;

        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);

        self.prepare_matmul_pipeline();
        self.device_ref().poll(wgpu::Maintain::Poll);

        let start = Instant::now();

        let mut multi_index = vec![0usize; rank];
        for page_idx in 0..page_volume {
            if rank > 0 {
                decode_multi_index(page_idx, &request.page_dims, &mut multi_index);
            }

            let lhs_linear = broadcast_linear_index(&lhs_dims, &lhs_strides, &multi_index);
            let rhs_linear = broadcast_linear_index(&rhs_dims, &rhs_strides, &multi_index);

            let lhs_offset_elements = lhs_linear
                .checked_mul(lhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: lhs offset overflow"))?;
            let rhs_offset_elements = rhs_linear
                .checked_mul(rhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: rhs offset overflow"))?;
            let out_offset_elements = page_idx
                .checked_mul(out_page_size)
                .ok_or_else(|| anyhow!("pagefun: output offset overflow"))?;

            let lhs_end = lhs_offset_elements
                .checked_add(lhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: lhs offset overflow"))?;
            let rhs_end = rhs_offset_elements
                .checked_add(rhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: rhs offset overflow"))?;
            let out_end = out_offset_elements
                .checked_add(out_page_size)
                .ok_or_else(|| anyhow!("pagefun: output offset overflow"))?;

            ensure!(
                lhs_end <= entry_a.len,
                "pagefun: lhs page out of bounds (page {})",
                page_idx
            );
            ensure!(
                rhs_end <= entry_b.len,
                "pagefun: rhs page out of bounds (page {})",
                page_idx
            );
            ensure!(
                out_end <= total_len,
                "pagefun: output page out of bounds (page {})",
                page_idx
            );

            let offset_a_u32 = u32::try_from(lhs_offset_elements)
                .map_err(|_| anyhow!("pagefun: lhs offset exceeds GPU limits"))?;
            let offset_b_u32 = u32::try_from(rhs_offset_elements)
                .map_err(|_| anyhow!("pagefun: rhs offset exceeds GPU limits"))?;
            let offset_out_u32 = u32::try_from(out_offset_elements)
                .map_err(|_| anyhow!("pagefun: output offset exceeds GPU limits"))?;

            let mut flags = 0u32;
            if view_a.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
            }
            if view_b.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
            }

            let params = crate::backend::wgpu::params::MatmulParams {
                m: m_u32,
                n: n_u32,
                k: k_u32,
                lda,
                ldb,
                ldc,
                offset_a: offset_a_u32,
                offset_b: offset_b_u32,
                offset_out: offset_out_u32,
                flags,
            };

            let params_buffer = self.uniform_buffer(&params, "runmat-pagefun-mtimes-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-pagefun-mtimes-bind"),
                    layout: &self.pipelines.matmul.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_b.buffer.as_ref().as_entire_binding(),
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

            crate::backend::wgpu::dispatch::matmul::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.matmul.pipeline,
                &bind_group,
                groups_x,
                groups_y,
            );
        }

        self.telemetry.record_matmul_duration(start.elapsed());

        let handle =
            self.register_existing_buffer(out_buffer, request.output_shape.clone(), total_len);

        Ok(handle)
    }
    async fn centered_gram_exec_kernel(
        &self,
        matrix: &GpuTensorHandle,
        matrix_entry: &BufferEntry,
        means: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        denom: f64,
    ) -> Result<GpuTensorHandle> {
        let rows_f64 = rows as f64;
        let means_entry = self.get_entry(means)?;
        let mut means_used = means.clone();
        let mut casted_means = false;
        if means_entry.precision != matrix_entry.precision {
            means_used = self
                .cast_tensor_precision(means, matrix_entry.precision)
                .await?;
            casted_means = true;
        }

        // Compute X^T * X using the SYRK pipeline (no explicit transpose required).
        let xtx = self.syrk_exec(matrix)?;

        // Form n * μ μᵀ without materialising a centered copy of X.
        let means_scaled = self.scalar_mul(&means_used, rows_f64)?;
        let means_col = self
            .reshape(&means_scaled, &[cols, 1])
            .map_err(|e| anyhow!("centered_gram: reshape means col failed: {e}"))?;
        let means_row_scaled = self
            .reshape(&means_scaled, &[1, cols])
            .map_err(|e| anyhow!("centered_gram: reshape means row failed: {e}"))?;

        let outer_scaled = self.matmul_exec_with_usage(
            &means_col,
            &means_row_scaled,
            BufferUsageClass::FusionOut,
        )?;
        let outer = self.scalar_mul(&outer_scaled, 1.0 / rows_f64)?;

        let _ = self.free(&means_col);
        let _ = self.free(&means_row_scaled);
        let _ = self.free(&outer_scaled);

        let centered =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &xtx, &outer)?;

        let _ = self.free(&xtx);
        let _ = self.free(&outer);
        let _ = self.free(&means_scaled);

        let handle = self.scalar_mul(&centered, 1.0 / denom)?;
        let _ = self.free(&centered);

        self.mark_buffer_usage(&handle, BufferUsageClass::FusionOut);

        if std::env::var("RUNMAT_DEBUG_CENTERED_GRAM").is_ok() {
            if let Err(err) = self
                .debug_centered_gram(
                    matrix,
                    matrix_entry.precision,
                    &means_used,
                    &handle,
                    rows,
                    cols,
                    denom,
                )
                .await
            {
                log::warn!("centered_gram debug instrumentation failed: {err}");
            }
        }

        if casted_means {
            let _ = self.free(&means_used);
        }

        Ok(handle)
    }
    #[allow(clippy::too_many_arguments)]
    async fn debug_centered_gram(
        &self,
        matrix: &GpuTensorHandle,
        precision: NumericPrecision,
        means: &GpuTensorHandle,
        output: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        denom: f64,
    ) -> Result<()> {
        let matrix_host = <Self as AccelProvider>::download(self, matrix).await?;
        let means_gpu = <Self as AccelProvider>::download(self, means).await?;
        let output_gpu = <Self as AccelProvider>::download(self, output).await?;
        if matrix_host.data.len() != rows * cols {
            return Err(anyhow!(
                "centered_gram debug: matrix download length mismatch ({} vs {})",
                matrix_host.data.len(),
                rows * cols
            ));
        }

        let mut mean_ref = vec![0.0f64; cols];
        for (col, mean_slot) in mean_ref.iter_mut().enumerate().take(cols) {
            let mut sum = 0.0f64;
            let base = col * rows;
            for row in 0..rows {
                sum += matrix_host.data[base + row];
            }
            *mean_slot = sum / (rows as f64);
        }

        let mut max_mean_diff = 0.0f64;
        for (mean, gpu_mean) in mean_ref.iter().zip(means_gpu.data.iter()) {
            let diff = (*mean - *gpu_mean).abs();
            if diff > max_mean_diff {
                max_mean_diff = diff;
            }
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..cols).collect();
        indices.shuffle(&mut rng);
        indices.truncate(cols.min(32));
        indices.sort_unstable();

        let mut max_abs_err = 0.0f64;
        let mut max_abs_idx = (0usize, 0usize);
        let mut max_rel_err = 0.0f64;
        let mut max_rel_idx = (0usize, 0usize);
        let mut max_diag_neg = 0.0f64;
        let mut max_diag_idx = 0usize;

        for &j in &indices {
            for &i in &indices {
                let mut sum = 0.0f64;
                let base_i = i * rows;
                let base_j = j * rows;
                for row in 0..rows {
                    let centered_i = matrix_host.data[base_i + row] - mean_ref[i];
                    let centered_j = matrix_host.data[base_j + row] - mean_ref[j];
                    sum += centered_i * centered_j;
                }
                sum /= denom;

                let gpu_val = output_gpu.data[i + j * cols];
                let abs_err = (gpu_val - sum).abs();
                if i == j && std::env::var("RUNMAT_DEBUG_CENTERED_GRAM_TRACE").is_ok() {
                    log::info!(
                        "centered_gram diag sample col={} gpu={:.6e} ref={:.6e}",
                        i,
                        gpu_val,
                        sum
                    );
                }
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                    max_abs_idx = (i, j);
                }
                if sum.abs() > 0.0 {
                    let rel_err = abs_err / sum.abs();
                    if rel_err > max_rel_err {
                        max_rel_err = rel_err;
                        max_rel_idx = (i, j);
                    }
                }
                if i == j && gpu_val < 0.0 {
                    let neg = gpu_val.abs();
                    if neg > max_diag_neg {
                        max_diag_neg = neg;
                        max_diag_idx = i;
                    }
                }
            }
        }

        let sample_preview: Vec<usize> = indices.iter().copied().take(16).collect();
        let rows_out = output_gpu.shape.first().copied().unwrap_or(cols);
        let diag_len = cols.min(rows_out);
        let mut trace = 0.0f64;
        for d in 0..diag_len {
            let idx = d + d * rows_out;
            if let Some(val) = output_gpu.data.get(idx) {
                trace += *val;
            }
        }
        log::info!(
            "centered_gram debug [{}]: rows={} cols={} sample_cols={} trace={:.6e} max_mean_diff={:.3e} max_abs_err={:.3e} at ({}, {}) max_rel_err={:.3e} at ({}, {}) max_diag_neg={:.3e} at ({}) samples={:?}",
            match precision {
                NumericPrecision::F32 => "f32",
                NumericPrecision::F64 => "f64",
            },
            rows,
            cols,
            indices.len(),
            trace,
            max_mean_diff,
            max_abs_err,
            max_abs_idx.0,
            max_abs_idx.1,
            max_rel_err,
            max_rel_idx.0,
            max_rel_idx.1,
            max_diag_neg,
            max_diag_idx,
            sample_preview
        );

        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    async fn debug_qr_power_iter(
        &self,
        product: &GpuTensorHandle,
        product_entry: &BufferEntry,
        pre_product_max: Option<f64>,
        pre_q_max: Option<f64>,
        q_result: &GpuTensorHandle,
        r_handle: &GpuTensorHandle,
        r_inv_handle: &GpuTensorHandle,
        gram_host: Option<&HostTensorOwned>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let product_host = <Self as AccelProvider>::download(self, product).await?;
        let q_gpu_host = <Self as AccelProvider>::download(self, q_result).await?;
        let r_gpu_host = <Self as AccelProvider>::download(self, r_handle).await?;
        let r_inv_gpu_host = <Self as AccelProvider>::download(self, r_inv_handle).await?;
        let max_r_inv_abs = r_inv_gpu_host
            .data
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));

        if product_host.data.len() != rows * cols
            || q_gpu_host.data.len() != rows * cols
            || r_gpu_host.data.len() != cols * cols
            || r_inv_gpu_host.data.len() != cols * cols
        {
            return Err(anyhow!(
                "qr_power_iter debug: length mismatch (rows={}, cols={})",
                rows,
                cols
            ));
        }

        let gram_cow: Cow<'_, HostTensorOwned> = if let Some(host) = gram_host {
            Cow::Borrowed(host)
        } else {
            let product_t_tmp = self.transpose_exec(product)?;
            let gram_tmp =
                self.matmul_exec_with_usage(&product_t_tmp, product, BufferUsageClass::FusionOut)?;
            let _ = self.free(&product_t_tmp);
            let owned = <Self as AccelProvider>::download(self, &gram_tmp).await?;
            let _ = self.free(&gram_tmp);
            Cow::Owned(owned)
        };
        let gram_view: &HostTensorOwned = gram_cow.as_ref();

        if gram_view.data.len() != cols * cols {
            return Err(anyhow!(
                "qr_power_iter debug: Gram data mismatch (cols={})",
                cols
            ));
        }

        let mut min_r_diag = f64::MAX;
        let mut max_r_diag = f64::MIN;
        for i in 0..cols {
            let diag = r_gpu_host.data[i + i * cols];
            min_r_diag = min_r_diag.min(diag);
            max_r_diag = max_r_diag.max(diag);
        }

        let mut min_gram_diag = f64::MAX;
        let mut max_gram_diag = f64::MIN;
        for i in 0..cols {
            let diag = gram_view.data[i + i * cols];
            min_gram_diag = min_gram_diag.min(diag);
            max_gram_diag = max_gram_diag.max(diag);
        }

        let mut q_ref = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    sum += product_host.data[row + k * rows] * r_inv_gpu_host.data[k + col * cols];
                }
                q_ref[row + col * rows] = sum;
            }
        }

        let mut max_q_diff = 0.0f64;
        let mut max_q_diff_idx = 0usize;
        let mut max_q_abs = 0.0f64;
        let mut min_q_abs = f64::MAX;
        let mut non_zero_q = false;
        for (idx, (val, ref_val)) in q_gpu_host
            .data
            .iter()
            .zip(q_ref.iter())
            .enumerate()
            .take(rows * cols)
        {
            let diff = (*val - *ref_val).abs();
            if diff > max_q_diff {
                max_q_diff = diff;
                max_q_diff_idx = idx;
            }
            let abs_val = val.abs();
            if abs_val > max_q_abs {
                max_q_abs = abs_val;
            }
            if abs_val < min_q_abs {
                min_q_abs = abs_val;
            }
            if abs_val > 1.0e-12 {
                non_zero_q = true;
            }
        }
        if min_q_abs == f64::MAX {
            min_q_abs = 0.0;
        }

        let mut max_qtq_diag = 0.0f64;
        let mut max_qtq_diag_idx = 0usize;
        let mut max_qtq_off = 0.0f64;
        let mut max_qtq_off_idx = (0usize, 0usize);
        let mut min_diag_val = f64::MAX;
        let mut max_diag_val = f64::MIN;
        for j in 0..cols {
            for i in 0..cols {
                let mut sum = 0.0f64;
                for row in 0..rows {
                    sum += q_gpu_host.data[row + i * rows] * q_gpu_host.data[row + j * rows];
                }
                if i == j {
                    let err = (sum - 1.0).abs();
                    if err > max_qtq_diag {
                        max_qtq_diag = err;
                        max_qtq_diag_idx = i;
                    }
                    if sum < min_diag_val {
                        min_diag_val = sum;
                    }
                    if sum > max_diag_val {
                        max_diag_val = sum;
                    }
                } else {
                    let err = sum.abs();
                    if err > max_qtq_off {
                        max_qtq_off = err;
                        max_qtq_off_idx = (i, j);
                    }
                }
            }
        }

        let mut max_residual = 0.0f64;
        let mut max_residual_idx = (0usize, 0usize);
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    sum += q_gpu_host.data[row + k * rows] * r_gpu_host.data[k + col * cols];
                }
                let diff = (sum - product_host.data[row + col * rows]).abs();
                if diff > max_residual {
                    max_residual = diff;
                    max_residual_idx = (row, col);
                }
            }
        }

        let mut gq_gpu = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for l in 0..cols {
                    sum += gram_view.data[l + col * cols] * q_gpu_host.data[row + l * rows];
                }
                gq_gpu[row + col * rows] = sum;
            }
        }
        let mut gq_ref = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for l in 0..cols {
                    sum += gram_view.data[l + col * cols] * q_ref[row + l * rows];
                }
                gq_ref[row + col * rows] = sum;
            }
        }

        let mut gpu_topk = 0.0f64;
        let mut ref_topk = 0.0f64;
        for col in 0..cols {
            let mut diag_gpu = 0.0f64;
            let mut diag_ref = 0.0f64;
            for row in 0..rows {
                diag_gpu += q_gpu_host.data[row + col * rows] * gq_gpu[row + col * rows];
                diag_ref += q_ref[row + col * rows] * gq_ref[row + col * rows];
            }
            gpu_topk += diag_gpu;
            ref_topk += diag_ref;
        }
        let topk_diff = gpu_topk - ref_topk;
        let max_product_abs = product_host
            .data
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));

        log::info!(
            "qr_power_iter debug: rows={} cols={} max_q_diff={:.3e} at idx={} max_q_abs={:.3e} min_q_abs={:.3e} non_zero_q={} max_qtq_diag_err={:.3e} at col={} max_qtq_off={:.3e} at ({}, {}) min_diag={:.3e} max_diag={:.3e} max_residual={:.3e} at ({}, {}) max_product_abs_pre={:?} max_product_abs={:.3e} max_q_abs_pre={:?} max_r_inv_abs={:.3e} min_r_diag={:.3e} max_r_diag={:.3e} min_gram_diag={:.3e} max_gram_diag={:.3e} gpu_topk={:.6e} ref_topk={:.6e} diff={:.3e}",
            rows,
            cols,
            max_q_diff,
            max_q_diff_idx,
            max_q_abs,
            min_q_abs,
            non_zero_q,
            max_qtq_diag,
            max_qtq_diag_idx,
            max_qtq_off,
            max_qtq_off_idx.0,
            max_qtq_off_idx.1,
            min_diag_val,
            max_diag_val,
            max_residual,
            max_residual_idx.0,
            max_residual_idx.1,
            pre_product_max,
            max_product_abs,
            pre_q_max,
            max_r_inv_abs,
            min_r_diag,
            max_r_diag,
            min_gram_diag,
            max_gram_diag,
            gpu_topk,
            ref_topk,
            topk_diff
        );

        if !non_zero_q || max_product_abs <= 1.0e-12 {
            let active = active_fusion();
            let plan = active_group_plan_clone();
            log::warn!(
                "qr_power_iter zero-data alert: product={} len={} non_zero_q={} max_product_abs_pre={:?} max_product_abs={:.3e} max_q_abs_pre={:?} active={:?} plan_inputs={:?} stack_pattern={:?}",
                product.buffer_id,
                product_entry.len,
                non_zero_q,
                pre_product_max,
                max_product_abs,
                pre_q_max,
                active,
                plan.as_ref().map(|p| p.inputs.clone()),
                plan.as_ref().map(|p| p.stack_pattern.clone())
            );
        }

        Ok(())
    }
    pub(crate) async fn covariance_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CovRows::All {
            return Err(anyhow!(
                "covariance: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }
        if options.has_weight_vector {
            return Err(anyhow!(
                "covariance: weight vectors are not supported by WGPU provider"
            ));
        }

        let entry = self.get_entry(matrix)?;
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                    shape
                ))
            }
        };

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cov-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let denom = match options.normalization {
            CovNormalization::Unbiased => (rows as f64) - 1.0,
            CovNormalization::Biased => rows as f64,
        };

        if denom <= 0.0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let means = self.reduce_dim_sum_mean_exec(
            matrix,
            0,
            crate::backend::wgpu::types::DimReduceOp::Mean,
        )?;
        let result = self
            .centered_gram_exec_kernel(matrix, &entry, &means, rows, cols, denom)
            .await;
        let _ = self.free(&means);
        result
    }
    pub(crate) async fn corrcoef_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: &CorrcoefOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CorrcoefRows::All {
            return Err(anyhow!(
                "corrcoef: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }

        let entry = self.get_entry(matrix)?;
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "corrcoef: inputs must be 2-D matrices or vectors (got shape {:?})",
                    shape
                ))
            }
        };

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-corrcoef-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let denom = match options.normalization {
            CorrcoefNormalization::Unbiased => (rows as f64) - 1.0,
            CorrcoefNormalization::Biased => rows as f64,
        };

        if denom <= 0.0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let means = self.reduce_dim_sum_mean_exec(
            matrix,
            0,
            crate::backend::wgpu::types::DimReduceOp::Mean,
        )?;
        let ones = self.fill_exec(&[rows, 1], 1.0)?;
        let means_full = self.matmul_exec(&ones, &means)?;
        let centered = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            matrix,
            &means_full,
        )?;
        let squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &centered,
            &centered,
        )?;
        let centered_t = self.transpose_exec(&centered)?;
        let covariance = self.matmul_exec(&centered_t, &centered)?;
        let inv_denom = 1.0 / denom;
        let inv_cov = self.fill_exec(&covariance.shape, inv_denom)?;
        let covariance_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &covariance,
            &inv_cov,
        )?;

        let variance_sum = self.reduce_dim_sum_mean_exec(
            &squared,
            0,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let inv_var = self.fill_exec(&variance_sum.shape, inv_denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_sum,
            &inv_var,
        )?;

        // Clamp tiny negative variances to zero to stabilise sqrt
        let mut host_variance = <Self as AccelProvider>::download(self, &variance).await?;
        for value in host_variance.data.iter_mut() {
            if *value < 0.0 && *value > -1.0e-12 {
                *value = 0.0;
            }
        }
        let view = HostTensorView {
            data: &host_variance.data,
            shape: &host_variance.shape,
        };
        let variance_adjusted = self.upload(&view)?;
        self.free(&variance)?;

        let std = self.unary_op_exec(
            crate::backend::wgpu::types::UnaryOpCode::Sqrt,
            &variance_adjusted,
        )?;
        let std_t = self.transpose_exec(&std)?;
        let std_outer = self.matmul_exec(&std_t, &std)?;
        let correlation = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &covariance_scaled,
            &std_outer,
        )?;

        // Free temporaries
        let _ = self.free(&means);
        let _ = self.free(&ones);
        let _ = self.free(&means_full);
        let _ = self.free(&centered);
        let _ = self.free(&centered_t);
        let _ = self.free(&covariance);
        let _ = self.free(&inv_cov);
        let _ = self.free(&covariance_scaled);
        let _ = self.free(&squared);
        let _ = self.free(&variance_sum);
        let _ = self.free(&inv_var);
        let _ = self.free(&variance_adjusted);
        let _ = self.free(&std);
        let _ = self.free(&std_t);
        let _ = self.free(&std_outer);

        Ok(correlation)
    }
    async fn cast_tensor_precision(
        &self,
        tensor: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(tensor)?;
        if entry.precision == target {
            return Ok(tensor.clone());
        }

        let mut host = <Self as AccelProvider>::download(self, tensor).await?;
        if matches!(target, NumericPrecision::F32) {
            for value in host.data.iter_mut() {
                *value = (*value as f32) as f64;
            }
        }

        let view = HostTensorView {
            data: host.data.as_slice(),
            shape: host.shape.as_slice(),
        };
        self.upload(&view)
    }

    pub(crate) fn dot_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry_lhs = self.get_entry(lhs)?;
        let entry_rhs = self.get_entry(rhs)?;
        ensure!(
            entry_lhs.shape == entry_rhs.shape,
            "dot: shape mismatch between inputs"
        );
        if entry_lhs.shape.is_empty() {
            return self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs);
        }
        if entry_lhs.shape.len() != 2 {
            return Err(anyhow!(
                "dot: only 2D tensors are currently supported by the WGPU provider"
            ));
        }

        let shape = entry_lhs.shape.clone();
        let default_dim = shape
            .iter()
            .position(|&extent| extent != 1)
            .map(|idx| idx + 1)
            .unwrap_or(1);
        let target_dim = dim.unwrap_or(default_dim);
        let dim_index = target_dim.saturating_sub(1);

        if dim_index >= shape.len() {
            return self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs);
        }
        if dim_index > 1 {
            return Err(anyhow!(
                "dot: unsupported dimension {} for WGPU provider",
                target_dim
            ));
        }

        let product =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs)?;

        let reduce = self.reduce_dim_sum_mean_exec(
            &product,
            dim_index,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        );
        match reduce {
            Ok(handle) => {
                let _ = self.free(&product);
                Ok(handle)
            }
            Err(err) => {
                let _ = self.free(&product);
                Err(err)
            }
        }
    }

    pub(crate) fn cross_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry_lhs = self.get_entry(lhs)?;
        let entry_rhs = self.get_entry(rhs)?;
        ensure!(
            entry_lhs.shape == entry_rhs.shape,
            "cross: shape mismatch between inputs"
        );

        let shape = if entry_lhs.shape.is_empty() {
            vec![1, 1]
        } else {
            entry_lhs.shape.clone()
        };
        let rank = shape.len();
        let target_dim = match dim {
            Some(target_dim) => {
                ensure!(
                    target_dim >= 1 && target_dim <= rank,
                    "cross: dimension {} exceeds the number of array dimensions ({})",
                    target_dim,
                    rank
                );
                ensure!(
                    shape[target_dim - 1] == 3,
                    "cross: dimension {} must have length 3",
                    target_dim
                );
                target_dim
            }
            None => shape
                .iter()
                .position(|&extent| extent == 3)
                .map(|idx| idx + 1)
                .ok_or_else(|| anyhow!("cross: inputs must have a dimension of length 3"))?,
        };
        let dim_index = target_dim - 1;
        let total_len = entry_lhs.len;
        if total_len == 0 {
            return self.zeros_exec(&shape);
        }

        let stride_before = product_checked(&shape[..dim_index])
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let stride_after = product_checked(&shape[dim_index + 1..])
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let slice_stride = stride_before
            .checked_mul(3)
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let slice_count = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;

        let mut comp1 = Vec::with_capacity(slice_count);
        let mut comp2 = Vec::with_capacity(slice_count);
        let mut comp3 = Vec::with_capacity(slice_count);
        for after in 0..stride_after {
            let slice_base = after
                .checked_mul(slice_stride)
                .ok_or_else(|| anyhow!("cross: internal index overflow"))?;
            for before in 0..stride_before {
                let idx1 = slice_base + before;
                let idx2 = idx1 + stride_before;
                let idx3 = idx2 + stride_before;
                comp1.push(
                    u32::try_from(idx1).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
                comp2.push(
                    u32::try_from(idx2).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
                comp3.push(
                    u32::try_from(idx3).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
            }
        }

        let mut reduced_shape = shape.clone();
        reduced_shape[dim_index] = 1;

        // Track every intermediate handle outside the computation closure so that
        // handles allocated before a failing `?` are still freed on error.
        let mut to_free: Vec<GpuTensorHandle> = Vec::with_capacity(15);

        let compute_result: Result<GpuTensorHandle> = (|| {
            let a1 = self.gather_linear_exec(lhs, &comp1, &reduced_shape)?;
            to_free.push(a1.clone());
            let a2 = self.gather_linear_exec(lhs, &comp2, &reduced_shape)?;
            to_free.push(a2.clone());
            let a3 = self.gather_linear_exec(lhs, &comp3, &reduced_shape)?;
            to_free.push(a3.clone());
            let b1 = self.gather_linear_exec(rhs, &comp1, &reduced_shape)?;
            to_free.push(b1.clone());
            let b2 = self.gather_linear_exec(rhs, &comp2, &reduced_shape)?;
            to_free.push(b2.clone());
            let b3 = self.gather_linear_exec(rhs, &comp3, &reduced_shape)?;
            to_free.push(b3.clone());

            let a2b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b3)?;
            to_free.push(a2b3.clone());
            let a3b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b2)?;
            to_free.push(a3b2.clone());
            let c1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a2b3, &a3b2)?;
            to_free.push(c1.clone());

            let a3b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b1)?;
            to_free.push(a3b1.clone());
            let a1b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b3)?;
            to_free.push(a1b3.clone());
            let c2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a3b1, &a1b3)?;
            to_free.push(c2.clone());

            let a1b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b2)?;
            to_free.push(a1b2.clone());
            let a2b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b1)?;
            to_free.push(a2b1.clone());
            let c3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a1b2, &a2b1)?;
            to_free.push(c3.clone());

            let out = self.zeros_exec(&shape)?;
            let scatter_result = (|| -> Result<()> {
                self.scatter_linear_exec(&out, &comp1, &c1)?;
                self.scatter_linear_exec(&out, &comp2, &c2)?;
                self.scatter_linear_exec(&out, &comp3, &c3)?;
                Ok(())
            })();

            match scatter_result {
                Ok(()) => Ok(out),
                Err(err) => {
                    let _ = self.free(&out);
                    Err(err)
                }
            }
        })();

        for h in &to_free {
            let _ = self.free(h);
        }

        compute_result
    }
}

}
