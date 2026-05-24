use runmat_accelerate_api::AccelProvider;

use super::*;
use std::collections::HashSet;
use std::time::Duration;

impl WgpuProvider {
    pub(crate) fn reduce_any_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AnyOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AnyInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    pub(crate) fn reduce_all_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AllOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AllInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fused_reduction_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        if inputs.is_empty() {
            return Err(anyhow!("fused_reduction: no inputs"));
        }
        if reduce_len == 0 {
            return Err(anyhow!("fused_reduction: zero reduce_len"));
        }
        let out_elems: usize = output_shape.iter().product();
        if out_elems != num_slices.max(1) {
            return Err(anyhow!(
                "fused_reduction: output_shape {:?} inconsistent with num_slices {}",
                output_shape,
                num_slices
            ));
        }

        let workgroup_size = if workgroup_size == 0 {
            self.default_reduction_workgroup_size()
        } else {
            workgroup_size
        };
        let tuning_key = ReductionAutotuneKey::new(self.precision, num_slices, reduce_len);
        if self.reduction_autotune.is_enabled() {
            if let Some(tuning) = self.reduction_autotune.get(&tuning_key) {
                return self.execute_reduction_with_strategy(
                    &tuning,
                    inputs,
                    output_shape,
                    shader,
                    reduce_len,
                    num_slices,
                    workgroup_size,
                    flavor,
                );
            }
            if let Some(handle) = self.maybe_autotune_reduction(
                &tuning_key,
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
                flavor,
            )? {
                return Ok(handle);
            }
            if let Some(tuning) = self.reduction_autotune.get(&tuning_key) {
                return self.execute_reduction_with_strategy(
                    &tuning,
                    inputs,
                    output_shape,
                    shader,
                    reduce_len,
                    num_slices,
                    workgroup_size,
                    flavor,
                );
            }
        }

        let fallback_tuning =
            self.heuristic_reduction_tuning(reduce_len, num_slices, workgroup_size);
        self.execute_reduction_with_strategy(
            &fallback_tuning,
            inputs,
            output_shape,
            shader,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_reduction_with_strategy(
        &self,
        tuning: &ReductionTuning,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let mut prepared =
            self.prepare_reduction_tuning(tuning, reduce_len, num_slices, workgroup_size);
        if prepared.is_none()
            && !matches!(tuning.mode, ReductionMode::SinglePass)
            && self.can_use_single_pass(num_slices)
        {
            prepared = Some(ReductionTuning {
                mode: ReductionMode::SinglePass,
            });
        }
        let prepared = prepared.ok_or_else(|| {
            anyhow!(
                "fused_reduction: unable to schedule tuning {:?} for slices={} reduce_len={}",
                tuning.mode,
                num_slices,
                reduce_len
            )
        })?;

        match prepared.mode {
            ReductionMode::SinglePass => self.run_reduction_single_pass(
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
            ),
            ReductionMode::TwoPass { chunk_rows } => self.run_reduction_two_pass(
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
                chunk_rows,
                flavor,
            ),
        }
    }

    fn run_reduction_single_pass(
        &self,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Result<GpuTensorHandle> {
        let layout_tag = &format!("runmat-reduction-layout-{}", inputs.len());
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-fused-reduction-module",
            shader,
        );
        let bgl = self
            .cached_bind_group_layout_for_tag(layout_tag)
            .expect("reduction bgl");
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-pl",
            bgl.as_ref(),
        );
        if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
            let out_len = num_slices.max(1);
            let out_buffer = self.create_storage_buffer_checked(out_len, "runmat-reduction-out")?;
            return Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len));
        }
        let key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-reduction-pipeline",
            Some(shader.as_bytes()),
            Some(layout_tag),
            Some(workgroup_size),
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        self.device_ref().poll(wgpu::Maintain::Poll);
        let flush_enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-flush-single-pass-gap"),
            });
        self.submit(flush_enc);
        let out_len = num_slices.max(1);
        let mut out_buffer = self.create_storage_buffer_checked_with_usage(
            out_len,
            "runmat-reduction-out",
            BufferUsageClass::FusionOut,
        )?;
        {
            let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer as usize;
            let mut alias = false;
            for h in inputs.iter() {
                let ptr = self.get_entry(h)?.buffer.as_ref() as *const wgpu::Buffer as usize;
                if ptr == out_ptr {
                    alias = true;
                    break;
                }
            }
            if alias {
                out_buffer = self.create_storage_buffer_checked_with_usage(
                    out_len,
                    "runmat-reduction-out-unique",
                    BufferUsageClass::FusionOut,
                )?;
            }
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            nrows: u32,
            ncols: u32,
            ld: u32,
            flags: u32,
        }
        let flags = if shader.contains("const OMITNAN: bool = true") {
            1u32
        } else {
            0u32
        };
        let params = Params {
            nrows: reduce_len as u32,
            ncols: num_slices as u32,
            ld: reduce_len as u32,
            flags,
        };
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::ReductionParams,
            std::mem::size_of::<Params>() as u64,
            "runmat-reduction-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
        let mut entries_vec: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(inputs.len() + 2);
        let mut input_bufs: Vec<(Arc<wgpu::Buffer>, u64)> = Vec::with_capacity(inputs.len());
        for h in inputs.iter() {
            let e = self.get_entry(h)?;
            let bytes = (e.len * self.element_size) as u64;
            input_bufs.push((e.buffer.clone(), bytes));
        }
        let snapshot_inputs = std::env::var("RUNMAT_FUSED_SNAPSHOT_INPUTS").is_ok();
        let mut bind_input_bufs: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(inputs.len());
        if snapshot_inputs {
            for (buf, bytes) in input_bufs.iter() {
                let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-fused-input-snapshot"),
                    size: *bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                let mut enc =
                    self.device_ref()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-fused-input-snapshot-copy"),
                        });
                enc.copy_buffer_to_buffer(buf.as_ref(), 0, snap.as_ref(), 0, *bytes);
                self.submit(enc);
                bind_input_bufs.push(snap);
            }
        } else {
            for (buf, _bytes) in input_bufs.iter() {
                bind_input_bufs.push(buf.clone());
            }
        }
        for (i, buf_arc) in bind_input_bufs.iter().enumerate() {
            entries_vec.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf_arc.as_ref().as_entire_binding(),
            });
        }
        entries_vec.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: out_buffer.as_ref().as_entire_binding(),
        });
        entries_vec.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 1) as u32,
            resource: params_buffer.as_ref().as_entire_binding(),
        });
        {
            let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer as usize;
            let mut alias_found = false;
            for b in bind_input_bufs.iter() {
                let in_ptr = b.as_ref() as *const wgpu::Buffer as usize;
                if in_ptr == out_ptr {
                    alias_found = true;
                    break;
                }
            }
            if alias_found {
                return Err(anyhow!("fused_reduction(single-pass): input/output alias"));
            }
        }
        let groups = (num_slices as u32).max(1);
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            for (i, buf) in input_bufs.iter().enumerate() {
                log::debug!(
                    "[fused-reduction] binding={} role=read ptr={:p}",
                    i,
                    buf.0.as_ref()
                );
            }
            log::debug!(
                "[fused-reduction] binding={} role=read_write ptr={:p}",
                inputs.len(),
                out_buffer.as_ref()
            );
            log::debug!(
                "[fused-reduction] binding={} role=uniform ptr={:p}",
                inputs.len() + 1,
                params_buffer.as_ref()
            );
            log::debug!(
                "[fused-reduction] reduce_len={} slices={} wg={} groups={}",
                reduce_len,
                num_slices,
                workgroup_size,
                groups
            );
        }
        let disable_bg_cache = std::env::var("RUNMAT_DISABLE_FUSED_BG_CACHE").is_ok();
        let bg = if disable_bg_cache {
            Arc::new(
                self.device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduction-bg-direct"),
                        layout: bgl.as_ref(),
                        entries: &entries_vec,
                    }),
            )
        } else {
            self.bind_group_cache
                .get_or_create(bgl.as_ref(), &entries_vec, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-reduction-bg"),
                                layout: bgl.as_ref(),
                                entries: &entries_vec,
                            }),
                    )
                })
        };
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            bg.as_ref(),
            groups,
        );
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            log::debug!("[fused-reduction] single-pass dispatch complete");
        }
        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_reduction_two_pass(
        &self,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        chunk_rows: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let scalar_ty = match self.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => "f64",
            _ => "f32",
        };
        let flags = if shader.contains("const OMITNAN: bool = true") {
            1u32
        } else {
            0u32
        };
        let chunk_rows = chunk_rows.max(workgroup_size.max(1));
        let chunk_rows_u32 = chunk_rows;
        let total_chunks = (reduce_len as u64).div_ceil(chunk_rows as u64);
        let total_chunks_u32 =
            u32::try_from(total_chunks).map_err(|_| anyhow!("reduction: too many chunks"))?;
        let partials_len = num_slices.max(1) * (total_chunks as usize);
        let (pass1, pass2) = crate::backend::wgpu::shaders::reduction::build_two_pass_shaders(
            scalar_ty,
            workgroup_size,
        );
        let m1 = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-reduction-pass1",
            &pass1,
        );
        let m2 = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-reduction-pass2",
            &pass2,
        );
        let bgl1 = self
            .cached_bind_group_layout_for_tag("runmat-reduction-p1-bgl")
            .expect("p1 bgl");
        let bgl2 = self
            .cached_bind_group_layout_for_tag("runmat-reduction-p2-bgl")
            .expect("p2 bgl");
        let pl1 = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-p1-pl",
            bgl1.as_ref(),
        );
        let pl2 = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-p2-pl",
            bgl2.as_ref(),
        );
        if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
            let out_len = num_slices.max(1);
            let out_buffer = self.create_storage_buffer_checked(out_len, "runmat-reduction-out")?;
            return Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len));
        }
        let p2_key = self.compute_pipeline_hash_bytes(
            pass2.as_bytes(),
            "runmat-reduction-p2-bgl",
            Some(workgroup_size),
        );
        let pipeline_p2 = self.get_or_create_pipeline(
            p2_key,
            &pl2,
            &m2,
            "runmat-reduction-pass2",
            Some(pass2.as_bytes()),
            Some("runmat-reduction-p2-bgl"),
            Some(workgroup_size),
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        let flush_enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-flush-before-pass1"),
            });
        self.submit(flush_enc);
        crate::backend::wgpu::dispatch::reduction::warmup_noop_after_pass2(
            self.device_ref(),
            self.queue_ref(),
            &pipeline_p2,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        let p1_key = self.compute_pipeline_hash_bytes(
            pass1.as_bytes(),
            "runmat-reduction-p1-bgl",
            Some(workgroup_size),
        );
        let pipeline_p1 = self.get_or_create_pipeline(
            p1_key,
            &pl1,
            &m1,
            "runmat-reduction-pass1",
            Some(pass1.as_bytes()),
            Some("runmat-reduction-p1-bgl"),
            Some(workgroup_size),
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        let input_buf = self.get_entry(&inputs[0])?.buffer.clone();
        let input_buf = if std::env::var("RUNMAT_FUSED_SNAPSHOT_INPUTS").is_ok() {
            let bytes = (reduce_len * num_slices.max(1) * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fused-p1-input-snapshot"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fused-p1-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(input_buf.as_ref(), 0, snap.as_ref(), 0, bytes);
            self.submit(enc);
            snap
        } else {
            input_buf
        };
        let partials_bytes = (partials_len * self.element_size) as u64;
        let mut partials_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::ReductionPartials,
            partials_bytes,
            "runmat-reduction-partials-scratch",
        );
        let out_len = num_slices.max(1);
        let mut out_buffer = self
            .create_storage_buffer_for_usage(
                BufferUsageClass::FusionOut,
                out_len,
                "runmat-reduction-out",
            )
            .0;
        {
            let in_ptr = input_buf.as_ref() as *const wgpu::Buffer as usize;
            let mut part_ptr = partials_buffer.as_ref() as *const wgpu::Buffer as usize;
            let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer as usize;
            if part_ptr == in_ptr {
                let unique =
                    self.create_storage_buffer(partials_len, "runmat-reduction-partials-unique");
                partials_buffer = unique;
                part_ptr = partials_buffer.as_ref() as *const wgpu::Buffer as usize;
            }
            if out_ptr == in_ptr || out_ptr == part_ptr {
                out_buffer = self
                    .create_storage_buffer_for_usage(
                        BufferUsageClass::FusionOut,
                        out_len,
                        "runmat-reduction-out-unique",
                    )
                    .0;
            }
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P1 {
            nrows: u32,
            ncols: u32,
            ld: u32,
            flags: u32,
            chunks: u32,
            chunk_stride: u32,
            chunk_rows: u32,
            _pad: u32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P2F32 {
            ncols: u32,
            chunks: u32,
            flags: u32,
            scale: f32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P2F64 {
            ncols: u32,
            chunks: u32,
            flags: u32,
            _pad: u32,
            scale: f64,
        }
        let max_dispatch = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS;
        let mut chunk_stride = total_chunks_u32.min(max_dispatch).max(1);
        let mut chunk_tiles = (total_chunks_u32 as u64).div_ceil(chunk_stride as u64);
        if chunk_tiles > max_dispatch as u64 {
            let required_stride = (total_chunks_u32 as u64).div_ceil(max_dispatch as u64);
            chunk_stride = required_stride.max(1).min(max_dispatch as u64) as u32;
            chunk_tiles = (total_chunks_u32 as u64).div_ceil(chunk_stride as u64);
            if chunk_tiles > max_dispatch as u64 {
                return Err(anyhow!(
                    "fused_reduction: chunk grid {} exceeds dispatch limits (stride {}, tiles {})",
                    total_chunks_u32,
                    chunk_stride,
                    chunk_tiles
                ));
            }
        }
        let p1u = P1 {
            nrows: reduce_len as u32,
            ncols: num_slices as u32,
            ld: reduce_len as u32,
            flags,
            chunks: total_chunks_u32,
            chunk_stride,
            chunk_rows: chunk_rows_u32,
            _pad: 0,
        };
        let p1_buf = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::ReductionPass1Params,
            std::mem::size_of::<P1>() as u64,
            "runmat-reduction-p1-params",
        );
        let p2_size = match self.precision {
            NumericPrecision::F64 => std::mem::size_of::<P2F64>() as u64,
            NumericPrecision::F32 => std::mem::size_of::<P2F32>() as u64,
        };
        let p2_buf = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::ReductionPass2Params,
            p2_size,
            "runmat-reduction-p2-params",
        );

        self.queue.write_buffer(p1_buf.as_ref(), 0, bytes_of(&p1u));
        let scale_value = flavor.scale(reduce_len);
        match self.precision {
            NumericPrecision::F64 => {
                let p2u = P2F64 {
                    ncols: num_slices as u32,
                    chunks: total_chunks_u32,
                    flags,
                    _pad: 0,
                    scale: scale_value,
                };
                self.queue.write_buffer(p2_buf.as_ref(), 0, bytes_of(&p2u));
            }
            NumericPrecision::F32 => {
                let p2u = P2F32 {
                    ncols: num_slices as u32,
                    chunks: total_chunks_u32,
                    flags,
                    scale: scale_value as f32,
                };
                self.queue.write_buffer(p2_buf.as_ref(), 0, bytes_of(&p2u));
            }
        }
        let entries_bg1 = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: partials_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: p1_buf.as_ref().as_entire_binding(),
            },
        ];
        let bg1 = self
            .bind_group_cache
            .get_or_create(bgl1.as_ref(), &entries_bg1, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-reduction-p1-bg"),
                            layout: bgl1.as_ref(),
                            entries: &entries_bg1,
                        }),
                )
            });
        let entries_bg2 = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: partials_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: p2_buf.as_ref().as_entire_binding(),
            },
        ];
        let bg2 = self
            .bind_group_cache
            .get_or_create(bgl2.as_ref(), &entries_bg2, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-reduction-p2-bg"),
                            layout: bgl2.as_ref(),
                            entries: &entries_bg2,
                        }),
                )
            });
        let g0 = (num_slices as u32).max(1);
        let g1 = chunk_stride.max(1);
        let g2 = (chunk_tiles as u32).max(1);
        crate::backend::wgpu::dispatch::reduction::run_two_pass(
            self.device_ref(),
            self.queue_ref(),
            &pipeline_p1,
            &pipeline_p2,
            bg1.as_ref(),
            bg2.as_ref(),
            g0,
            g1,
            g2,
        );
        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), num_slices.max(1)))
    }

    fn reduction_partials_budget_bytes(&self) -> u64 {
        if let Ok(raw) = std::env::var("RUNMAT_REDUCTION_PARTIALS_BUDGET_BYTES") {
            if let Ok(parsed) = raw.parse::<u64>() {
                if parsed > 0 {
                    return parsed;
                }
            }
        }
        let fallback = 4u64 << 30;
        let adapter_limit = if self.adapter_limits.max_buffer_size > 0 {
            self.adapter_limits.max_buffer_size
        } else {
            fallback
        };
        let mut budget = adapter_limit.saturating_mul(40) / 100;
        if budget == 0 {
            budget = adapter_limit;
        }
        let floor = 256u64 << 20;
        if adapter_limit >= floor {
            budget = budget.max(floor);
        }
        budget.min(adapter_limit)
    }

    fn sanitize_chunk_rows_for_limits(
        &self,
        chunk_rows: u32,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Option<u32> {
        if reduce_len == 0 {
            return None;
        }
        let wg = workgroup_size.max(1);
        let reduce_cap = reduce_len.min(u32::MAX as usize) as u32;
        let mut rows = chunk_rows.clamp(wg, reduce_cap.max(wg));
        if !rows.is_multiple_of(wg) {
            rows = rows.div_ceil(wg) * wg;
        }
        let slices = num_slices.max(1) as u64;
        let elem_bytes = self.element_size as u64;
        if elem_bytes == 0 {
            return None;
        }
        let per_chunk_bytes = slices.checked_mul(elem_bytes)?;
        let budget = self.reduction_partials_budget_bytes();
        if budget < per_chunk_bytes {
            return None;
        }
        let max_chunks = (budget / per_chunk_bytes).max(1);
        let mut required_chunk_rows = (reduce_len as u64).div_ceil(max_chunks);
        required_chunk_rows = required_chunk_rows.max(wg as u64);
        if required_chunk_rows > u32::MAX as u64 {
            return None;
        }
        let mut rows_u64 = rows as u64;
        if rows_u64 < required_chunk_rows {
            rows_u64 = required_chunk_rows;
        }
        rows_u64 = rows_u64.min(reduce_len.min(u32::MAX as usize) as u64);
        let mut rows_u32 = rows_u64 as u32;
        if !rows_u32.is_multiple_of(wg) {
            rows_u32 = rows_u32.div_ceil(wg) * wg;
        }
        rows_u32 = rows_u32.min(reduce_cap.max(wg));
        Some(rows_u32.max(wg))
    }

    fn prepare_reduction_tuning(
        &self,
        tuning: &ReductionTuning,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Option<ReductionTuning> {
        match tuning.mode {
            ReductionMode::SinglePass => Some(*tuning),
            ReductionMode::TwoPass { chunk_rows } => {
                let rows = self.sanitize_chunk_rows_for_limits(
                    chunk_rows,
                    reduce_len,
                    num_slices,
                    workgroup_size,
                )?;
                Some(ReductionTuning {
                    mode: ReductionMode::TwoPass { chunk_rows: rows },
                })
            }
        }
    }

    fn heuristic_reduction_tuning(
        &self,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> ReductionTuning {
        let default_two_pass = ReductionTuning {
            mode: ReductionMode::TwoPass {
                chunk_rows: self.default_chunk_rows(reduce_len, workgroup_size),
            },
        };
        let single = ReductionTuning {
            mode: ReductionMode::SinglePass,
        };
        match self.reduction_two_pass_mode {
            ReductionTwoPassMode::ForceOn => default_two_pass,
            ReductionTwoPassMode::ForceOff => single,
            ReductionTwoPassMode::Auto => {
                if self.can_use_single_pass(num_slices) && reduce_len <= self.two_pass_threshold() {
                    single
                } else {
                    default_two_pass
                }
            }
        }
    }

    fn default_chunk_rows(&self, reduce_len: usize, workgroup_size: u32) -> u32 {
        let wg = workgroup_size.max(1) as usize;
        if reduce_len <= wg {
            return reduce_len.min(u32::MAX as usize) as u32;
        }
        let max_dispatch = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
        let target_chunks = max_dispatch.max(1);
        let mut rows = reduce_len.div_ceil(target_chunks).max(wg);
        rows = rows.div_ceil(wg) * wg;
        rows = rows.clamp(wg, reduce_len.max(wg));
        rows.min(u32::MAX as usize) as u32
    }

    fn can_use_single_pass(&self, num_slices: usize) -> bool {
        num_slices as u64 <= crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as u64
    }

    fn reduction_chunk_row_candidates(&self, reduce_len: usize, workgroup_size: u32) -> Vec<u32> {
        let mut values = Vec::new();
        let mut current = workgroup_size.max(1) as usize;
        while current < reduce_len {
            values.push(current.min(u32::MAX as usize) as u32);
            current = current.saturating_mul(2);
        }
        values.push(reduce_len.min(u32::MAX as usize) as u32);
        values.sort_unstable();
        values.dedup();
        values
    }

    fn reduction_strategy_candidates(
        &self,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Vec<ReductionTuning> {
        let mut strategies = Vec::new();
        let chunk_candidates = self.reduction_chunk_row_candidates(reduce_len, workgroup_size);
        let can_single = self.can_use_single_pass(num_slices);
        match self.reduction_two_pass_mode {
            ReductionTwoPassMode::ForceOff => {
                if can_single {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::SinglePass,
                    });
                }
            }
            ReductionTwoPassMode::ForceOn => {
                for chunk in &chunk_candidates {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::TwoPass { chunk_rows: *chunk },
                    });
                }
            }
            ReductionTwoPassMode::Auto => {
                if can_single {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::SinglePass,
                    });
                }
                for chunk in &chunk_candidates {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::TwoPass { chunk_rows: *chunk },
                    });
                }
            }
        }
        if strategies.is_empty() {
            strategies.push(ReductionTuning {
                mode: ReductionMode::TwoPass {
                    chunk_rows: self.default_chunk_rows(reduce_len, workgroup_size),
                },
            });
        }
        strategies
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_autotune_reduction(
        &self,
        key: &ReductionAutotuneKey,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<Option<GpuTensorHandle>> {
        if !self.reduction_autotune.is_enabled() {
            return Ok(None);
        }
        let candidates = self.reduction_strategy_candidates(reduce_len, num_slices, workgroup_size);
        if candidates.len() <= 1 {
            if let Some(tuning) = candidates.first() {
                self.reduction_autotune.insert(key.clone(), *tuning);
            }
            return Ok(None);
        }
        let mut best_tuning: Option<ReductionTuning> = None;
        let mut best_time: Option<Duration> = None;
        let mut best_handle: Option<GpuTensorHandle> = None;
        let mut tested = HashSet::new();
        let mut last_err: Option<anyhow::Error> = None;
        for tuning in candidates {
            let sanitized = self
                .prepare_reduction_tuning(&tuning, reduce_len, num_slices, workgroup_size)
                .or_else(|| {
                    if !matches!(tuning.mode, ReductionMode::SinglePass)
                        && self.can_use_single_pass(num_slices)
                    {
                        Some(ReductionTuning {
                            mode: ReductionMode::SinglePass,
                        })
                    } else {
                        None
                    }
                });
            let Some(sanitized) = sanitized else {
                continue;
            };
            if !tested.insert(sanitized) {
                continue;
            }
            let start = Instant::now();
            match self.execute_reduction_with_strategy(
                &sanitized,
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
                flavor,
            ) {
                Ok(handle) => {
                    let elapsed = start.elapsed();
                    if best_time.is_none_or(|t| elapsed < t) {
                        if let Some(existing) = best_handle.replace(handle) {
                            let _ = self.free(&existing);
                        }
                        best_time = Some(elapsed);
                        best_tuning = Some(sanitized);
                    } else {
                        let _ = self.free(&handle);
                    }
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }
        if let (Some(tuning), Some(handle)) = (best_tuning, best_handle) {
            self.reduction_autotune.insert(key.clone(), tuning);
            return Ok(Some(handle));
        }
        if let Some(err) = last_err {
            return Err(err);
        }
        Ok(None)
    }
    pub(crate) fn reduce_global_exec(
        &self,
        a: &GpuTensorHandle,
        op: crate::backend::wgpu::types::GlobalReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            log::debug!(
                "[reduce-global] in ptr={:p} len={} op={}",
                entry.buffer.as_ref(),
                entry.len,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_GLOBAL").is_ok() {
            return Err(anyhow!(
                "reduce_global disabled via RUNMAT_DISABLE_REDUCE_GLOBAL"
            ));
        }
        if entry.len == 0 {
            let default = match op {
                crate::backend::wgpu::types::GlobalReduceOp::Sum => 0.0,
                crate::backend::wgpu::types::GlobalReduceOp::Prod => 1.0,
                crate::backend::wgpu::types::GlobalReduceOp::Min => f64::INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::Max => f64::NEG_INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::CountNonZero => 0.0,
            };
            let data = [default];
            let shape = [1usize, 1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            return self.upload(&view);
        }
        let mut current = if std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-global-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-global-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(enc);
            snap
        } else {
            entry.buffer.clone()
        };
        let mut current_len = entry.len;
        while current_len > 1 {
            let wg = crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE as usize;
            let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
            let elems_per_group = 2 * wg;
            let max_input_per_pass = max_groups * elems_per_group;

            let output_len_total = current_len.div_ceil(elems_per_group).max(1);
            // Metal-safe (opt-in): snapshot input buffer for this pass to avoid any SR/SRW conflicts
            // Enabled only if RUNMAT_FORCE_REDUCE_SNAPSHOT is set to avoid perf impact by default.
            let mut input_for_pass = current.clone();
            if self.adapter_info.backend == wgpu::Backend::Metal
                && std::env::var("RUNMAT_FORCE_REDUCE_SNAPSHOT").is_ok()
            {
                let size_bytes = (current_len * self.element_size) as u64;
                let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-input-snapshot"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                let mut enc =
                    self.device_ref()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-reduce-pass-input-snapshot-copy"),
                        });
                enc.copy_buffer_to_buffer(current.as_ref(), 0, snap.as_ref(), 0, size_bytes);
                self.submit(enc);
                input_for_pass = snap;
            }
            let mut out_buffer =
                self.create_storage_buffer_checked(output_len_total, "runmat-reduce-pass")?;
            // Prevent aliasing: output buffer must not be the same as input buffer
            if std::ptr::eq(out_buffer.as_ref(), input_for_pass.as_ref()) {
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: alias detected; current_ptr={:p} out_ptr={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                    eprintln!(
                        "[reduction] alias current={:p} out={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                }
                let size_bytes = (output_len_total * self.element_size) as u64;
                if size_bytes > self.adapter_limits.max_buffer_size {
                    return Err(gpu_per_buffer_limit_error(
                        "runmat-reduce-pass-unique",
                        size_bytes,
                        self.adapter_limits.max_buffer_size,
                    ));
                }
                out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-unique"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            let mut in_offset_elems = 0usize;
            let mut _out_offset_elems = 0usize;
            while in_offset_elems < current_len {
                let remain = current_len - in_offset_elems;
                let chunk_in = remain.min(max_input_per_pass);
                let chunk_out = chunk_in.div_ceil(elems_per_group).max(1);

                let params = crate::backend::wgpu::params::ReduceGlobalParams {
                    len: chunk_in as u32,
                    op: op as u32,
                    offset: in_offset_elems as u32,
                    total: current_len as u32,
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-reduce-global-params");

                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduce-global-bind"),
                        layout: &self.pipelines.reduce_global.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_for_pass.as_ref().as_entire_binding(),
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
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size_reduce(
                    chunk_in as u32,
                    crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
                );
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: dispatch groups={} current_ptr={:p} out_ptr={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                    eprintln!(
                        "[reduction] dispatch groups={} in={:p} out={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                }
                crate::backend::wgpu::dispatch::reduction::run_single_pass(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_global.pipeline,
                    &bind_group,
                    groups,
                );
                in_offset_elems += chunk_in;
                _out_offset_elems += chunk_out;
            }

            current = out_buffer;
            current_len = output_len_total;
        }
        Ok(self.register_existing_buffer(current, vec![1, 1], 1))
    }
    pub(crate) fn reduce_dim_sum_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_DIM").is_ok() {
            return Err(anyhow!("reduce_dim disabled via RUNMAT_DISABLE_REDUCE_DIM"));
        }
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        // Optional snapshot of input
        let in_buf = if std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-dim-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(enc);
            snap
        } else {
            entry.buffer.clone()
        };
        let mut out_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-out")?;
        // Prevent aliasing: output must not be identical to input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                log::debug!(
                    "reduce_dim_sum_mean_exec: alias detected; in_ptr={:p} out_ptr={:p} rows={} cols={} out_len={}",
                    entry.buffer.as_ref(),
                    out_buffer.as_ref(),
                    rows,
                    cols,
                    out_len
                );
            }
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-out-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if out_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-dim-bind"),
                layout: &self.pipelines.reduce_dim_sum_mean.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_ref().as_entire_binding(),
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
            out_len as u32,
            crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
        );
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean] in ptr={:p} out ptr={:p} rows={} cols={} dim={} groups={}",
                entry.buffer.as_ref(),
                out_buffer.as_ref(),
                rows,
                cols,
                reduce_dim,
                groups
            );
        }
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.reduce_dim_sum_mean.pipeline,
            &bind_group,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }
    pub(crate) fn reduce_dim_minmax_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceExtrema,
    ) -> Result<ReduceDimResult> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        let mut values_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-ext-values")?;
        let mut indices_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-ext-indices")?;
        // Prevent aliasing either output with input buffer
        if std::ptr::eq(values_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-ext-values-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            values_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-ext-values-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if std::ptr::eq(indices_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (out_len * self.element_size) as u64;
            if size_bytes > self.adapter_limits.max_buffer_size {
                return Err(gpu_per_buffer_limit_error(
                    "runmat-reduce-dim-ext-indices-unique",
                    size_bytes,
                    self.adapter_limits.max_buffer_size,
                ));
            }
            indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-ext-indices-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if out_len == 0 {
            let values_handle =
                self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
            let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
            return Ok(ReduceDimResult {
                values: values_handle,
                indices: indices_handle,
            });
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-ext-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-dim-ext-bind"),
                layout: &self.pipelines.reduce_dim_minmax.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_buffer.as_ref().as_entire_binding(),
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            out_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.reduce_dim_minmax.pipeline,
            &bind_group,
            groups,
        );
        let values_handle =
            self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
        let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
        Ok(ReduceDimResult {
            values: values_handle,
            indices: indices_handle,
        })
    }
    pub(crate) fn reduce_std_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(anyhow!(
                "reduce_std_dim: omitnan is not supported by the wgpu provider"
            ));
        }
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce_std_dim: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_len = match dim {
            0 => rows,
            1 => cols,
            _ => return Err(anyhow!("reduce_std_dim: only dims 0 or 1 supported")),
        };
        let out_shape = if dim == 0 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        if reduce_len == 0 {
            return self.fill_exec(&out_shape, f64::NAN);
        }

        let inv_len = 1.0 / (reduce_len as f64);

        let sum_handle =
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)?;
        let squared = self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, a)?;
        let sum_sq_handle = self.reduce_dim_sum_mean_exec(
            &squared,
            dim,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let _ = self.free(&squared);

        let sum_squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_handle,
            &sum_handle,
        )?;
        let _ = self.free(&sum_handle);

        let sum_shape = self.get_entry(&sum_squared)?.shape.clone();
        let scale_tensor = self.fill_exec(&sum_shape, inv_len)?;
        let sum_squared_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_squared,
            &scale_tensor,
        )?;
        let _ = self.free(&scale_tensor);
        let variance_numer = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &sum_sq_handle,
            &sum_squared_scaled,
        )?;
        let _ = self.free(&sum_sq_handle);
        let _ = self.free(&sum_squared);
        let _ = self.free(&sum_squared_scaled);

        let denom = match normalization {
            ProviderStdNormalization::Sample => {
                if reduce_len > 1 {
                    (reduce_len - 1) as f64
                } else {
                    1.0
                }
            }
            ProviderStdNormalization::Population => reduce_len as f64,
        };
        if denom == 0.0 {
            let _ = self.free(&variance_numer);
            return self.fill_exec(&out_shape, f64::NAN);
        }

        let variance_shape = self.get_entry(&variance_numer)?.shape.clone();
        let denom_tensor = self.fill_exec(&variance_shape, denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &variance_numer,
            &denom_tensor,
        )?;
        let _ = self.free(&denom_tensor);
        let _ = self.free(&variance_numer);

        let abs_variance =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, &variance)?;
        let variance_plus_abs = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
        )?;
        let _ = self.free(&abs_variance);
        let _ = self.free(&variance);

        let half_shape = self.get_entry(&variance_plus_abs)?.shape.clone();
        let half_tensor = self.fill_exec(&half_shape, 0.5)?;
        let clamped = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
        )?;
        let _ = self.free(&half_tensor);
        let _ = self.free(&variance_plus_abs);

        let std_handle =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &clamped)?;
        let _ = self.free(&clamped);

        Ok(std_handle)
    }

    pub(crate) fn reduce_std_exec(
        &self,
        a: &GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(anyhow!(
                "reduce_std: omitnan is not supported by the wgpu provider"
            ));
        }
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let scalar_shape = [1usize, 1usize];
        if len == 0 {
            return self.fill_exec(&scalar_shape, f64::NAN);
        }

        let inv_len = 1.0 / (len as f64);

        let sum_handle =
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let squared = self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, a)?;
        let sum_sq_handle =
            self.reduce_global_exec(&squared, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let _ = self.free(&squared);

        let sum_squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_handle,
            &sum_handle,
        )?;
        let _ = self.free(&sum_handle);

        let sum_shape = self.get_entry(&sum_squared)?.shape.clone();
        let scale_tensor = self.fill_exec(&sum_shape, inv_len)?;
        let sum_squared_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_squared,
            &scale_tensor,
        )?;
        let _ = self.free(&scale_tensor);
        let variance_numer = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &sum_sq_handle,
            &sum_squared_scaled,
        )?;
        let _ = self.free(&sum_sq_handle);
        let _ = self.free(&sum_squared);
        let _ = self.free(&sum_squared_scaled);

        let denom = match normalization {
            ProviderStdNormalization::Sample => {
                if len > 1 {
                    (len - 1) as f64
                } else {
                    1.0
                }
            }
            ProviderStdNormalization::Population => len as f64,
        };
        if denom == 0.0 {
            let _ = self.free(&variance_numer);
            return self.fill_exec(&scalar_shape, f64::NAN);
        }

        let variance_shape = self.get_entry(&variance_numer)?.shape.clone();
        let denom_tensor = self.fill_exec(&variance_shape, denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &variance_numer,
            &denom_tensor,
        )?;
        let _ = self.free(&denom_tensor);
        let _ = self.free(&variance_numer);

        let abs_variance =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, &variance)?;
        let variance_plus_abs = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
        )?;
        let _ = self.free(&abs_variance);
        let _ = self.free(&variance);

        let half_shape = self.get_entry(&variance_plus_abs)?.shape.clone();
        let half_tensor = self.fill_exec(&half_shape, 0.5)?;
        let clamped = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
        )?;
        let _ = self.free(&half_tensor);
        let _ = self.free(&variance_plus_abs);

        let std_handle =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &clamped)?;
        let _ = self.free(&clamped);

        Ok(std_handle)
    }
}
impl WgpuProvider {
    pub(crate) async fn reduce_nd_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<GpuTensorHandle> {
        // If input is a known square of a base tensor, reuse or compute ex2 via moments
        if let Ok(map) = self.pow2_of.lock() {
            if let Some(&base_id) = map.get(&a.buffer_id) {
                drop(map);
                // Try cache
                if let Ok(cache) = self.moments_cache.lock() {
                    if let Some((_mean_h, ex2_h)) = cache.get(&(base_id, dims_zero_based.to_vec()))
                    {
                        return Ok(ex2_h.clone());
                    }
                }
                // Build a handle for the base buffer id
                let base_entry = {
                    let guard = self.buffers.lock().expect("buffer mutex poisoned");
                    guard.get(&base_id).cloned()
                };
                if let Some(entry) = base_entry {
                    let base_handle = GpuTensorHandle {
                        shape: entry.shape.clone(),
                        device_id: self.runtime_device_id,
                        buffer_id: base_id,
                    };
                    let moments = self.reduce_moments_nd_exec(&base_handle, dims_zero_based)?;
                    if let Ok(mut cache2) = self.moments_cache.lock() {
                        cache2.insert(
                            (base_id, dims_zero_based.to_vec()),
                            (moments.mean.clone(), moments.ex2.clone()),
                        );
                    }
                    return Ok(moments.ex2);
                }
            }
        }
        // Prefer computing moments once and caching ex2 for future reuse
        if let Ok(cache) = self.moments_cache.lock() {
            let key = (a.buffer_id, dims_zero_based.to_vec());
            if let Some((mean_h, _ex2_h)) = cache.get(&key) {
                return Ok(mean_h.clone());
            }
            // Compute moments and store
            drop(cache);
            let moments = self.reduce_moments_nd_exec(a, dims_zero_based)?;
            if let Ok(mut cache2) = self.moments_cache.lock() {
                cache2.insert(
                    (a.buffer_id, dims_zero_based.to_vec()),
                    (moments.mean.clone(), moments.ex2.clone()),
                );
            }
            return Ok(moments.mean);
        }
        let entry = self.get_entry(a)?;
        let rank = entry.shape.len();
        ensure!(rank > 0, "reduce_mean_nd: rank must be > 0");
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&d| d < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(
            !reduce.is_empty(),
            "reduce_mean_nd: no valid dims to reduce"
        );
        let kept: Vec<usize> = (0..rank).filter(|i| !reduce.contains(i)).collect();

        // Compute strides (MATLAB/column-major)
        let mut strides: Vec<usize> = vec![0; rank];
        let mut s = 1usize;
        for (i, stride_slot) in strides.iter_mut().enumerate().take(rank) {
            *stride_slot = s;
            s = s
                .checked_mul(entry.shape[i])
                .ok_or_else(|| anyhow!("reduce_mean_nd: shape too large"))?;
        }

        let kept_sizes: Vec<u32> = kept.iter().map(|&i| entry.shape[i] as u32).collect();
        let reduce_sizes: Vec<u32> = reduce.iter().map(|&i| entry.shape[i] as u32).collect();
        let kept_strides: Vec<u32> = kept.iter().map(|&i| strides[i] as u32).collect();
        let reduce_strides: Vec<u32> = reduce.iter().map(|&i| strides[i] as u32).collect();

        let rows: usize = reduce
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        let cols: usize = kept
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        ensure!(rows > 0 && cols > 0, "reduce_mean_nd: empty tensor");
        if rows as u64 > u32::MAX as u64 || cols as u64 > u32::MAX as u64 {
            return Err(anyhow!("reduce_mean_nd: tensor exceeds GPU limits"));
        }

        // Heuristic fallback: for very large row extents, prefer sequenced dim-reductions.
        if rows >= self.two_pass_threshold() {
            let mut current = a.clone();
            let mut owned = false;
            for &d in &reduce {
                let next = self.reduce_mean_dim(&current, d).await?;
                if owned {
                    let _ = self.free(&current);
                }
                current = next;
                owned = true;
            }
            return Ok(current);
        }

        let mut out_buffer = self.create_storage_buffer(cols, "runmat-reduce-nd-mean-out");
        // Prevent aliasing: output must not be the same as input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (cols * self.element_size) as u64;
            out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-nd-mean-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let mut out_shape = entry.shape.clone();
        for &d in &reduce {
            out_shape[d] = 1;
        }

        match entry.precision {
            NumericPrecision::F64 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-mean-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-mean-bind-f64"),
                    layout: &self.pipelines.reduce_nd_mean.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    eprintln!(
                        "[reduce-nd-mean f64] in ptr={:p} out ptr={:p} rows={} cols={}",
                        entry.buffer.as_ref(),
                        out_buffer.as_ref(),
                        rows,
                        cols
                    );
                }
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_mean.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
            NumericPrecision::F32 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-mean-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-mean-bind-f32"),
                    layout: &self.pipelines.reduce_nd_mean.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    eprintln!(
                        "[reduce-nd-mean f32] in ptr={:p} out ptr={:p} rows={} cols={}",
                        entry.buffer.as_ref(),
                        out_buffer.as_ref(),
                        rows,
                        cols
                    );
                }
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_mean.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, cols))
    }

    pub(crate) fn reduce_moments_nd_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        let entry = self.get_entry(a)?;
        let rank = entry.shape.len();
        ensure!(rank > 0, "reduce_moments_nd: rank must be > 0");
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&d| d < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(
            !reduce.is_empty(),
            "reduce_moments_nd: no valid dims to reduce"
        );
        let kept: Vec<usize> = (0..rank).filter(|i| !reduce.contains(i)).collect();

        // Strides in column-major
        let mut strides: Vec<usize> = vec![0; rank];
        let mut s = 1usize;
        for (i, stride_slot) in strides.iter_mut().enumerate().take(rank) {
            *stride_slot = s;
            s = s
                .checked_mul(entry.shape[i])
                .ok_or_else(|| anyhow!("reduce_moments_nd: shape too large"))?;
        }

        let kept_sizes: Vec<u32> = kept.iter().map(|&i| entry.shape[i] as u32).collect();
        let reduce_sizes: Vec<u32> = reduce.iter().map(|&i| entry.shape[i] as u32).collect();
        let kept_strides: Vec<u32> = kept.iter().map(|&i| strides[i] as u32).collect();
        let reduce_strides: Vec<u32> = reduce.iter().map(|&i| strides[i] as u32).collect();

        let rows: usize = reduce
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        let cols: usize = kept
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        ensure!(rows > 0 && cols > 0, "reduce_moments_nd: empty tensor");
        if rows as u64 > u32::MAX as u64 || cols as u64 > u32::MAX as u64 {
            return Err(anyhow!("reduce_moments_nd: tensor exceeds GPU limits"));
        }

        // Allocate outputs
        let mean_out = self.create_storage_buffer(cols, "runmat-reduce-nd-moments-mean");
        let ex2_out = self.create_storage_buffer(cols, "runmat-reduce-nd-moments-ex2");
        let mut out_shape = entry.shape.clone();
        for &d in &reduce {
            out_shape[d] = 1;
        }

        match entry.precision {
            NumericPrecision::F64 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-moments-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-moments-bind-f64"),
                    layout: &self.pipelines.reduce_nd_moments.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mean_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ex2_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_moments.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
            NumericPrecision::F32 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-moments-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-moments-bind-f32"),
                    layout: &self.pipelines.reduce_nd_moments.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mean_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ex2_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_moments.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
        }

        let mean_handle = self.register_existing_buffer(mean_out, out_shape.clone(), cols);
        let ex2_handle = self.register_existing_buffer(ex2_out, out_shape, cols);
        Ok(runmat_accelerate_api::ProviderMoments2 {
            mean: mean_handle,
            ex2: ex2_handle,
        })
    }
}
