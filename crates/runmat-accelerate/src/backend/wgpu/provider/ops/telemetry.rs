use super::*;

pub(crate) struct FusedReductionTelemetryRequest<'a> {
    pub(crate) shader: &'a str,
    pub(crate) inputs: &'a [GpuTensorHandle],
    pub(crate) output_shape: &'a [usize],
    pub(crate) reduce_len: usize,
    pub(crate) num_slices: usize,
    pub(crate) workgroup_size: u32,
    pub(crate) flavor: ReductionFlavor,
}

impl WgpuProvider {
    pub(crate) fn fused_elementwise_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_elementwise_exec(shader, inputs, output_shape, len);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    pub(crate) fn fused_elementwise_multi_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        num_outputs: usize,
    ) -> Result<Vec<GpuTensorHandle>> {
        let start = Instant::now();
        let result =
            self.fused_elementwise_multi_exec(shader, inputs, output_shape, len, num_outputs);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
                ("num_outputs", num_outputs as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise_multi", &shape, &tuning);
        }
        result
    }

    pub(crate) fn map_nan_to_zero_exec(&self, handle: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-nan-to-zero-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F32,
        };
        self.fused_elementwise_with_telemetry_exec(
            shader,
            std::slice::from_ref(handle),
            &entry.shape,
            len,
        )
    }

    pub(crate) fn not_nan_mask_exec(&self, handle: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-not-nan-mask-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F32,
        };
        self.fused_elementwise_with_telemetry_exec(
            shader,
            std::slice::from_ref(handle),
            &entry.shape,
            len,
        )
    }

    pub(crate) fn fused_reduction_with_telemetry_exec(
        &self,
        request: FusedReductionTelemetryRequest<'_>,
    ) -> Result<GpuTensorHandle> {
        let FusedReductionTelemetryRequest {
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        } = request;
        let start = Instant::now();
        let result = self.fused_reduction_exec(
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        );
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_reduction_duration(elapsed);
            let actual_wg = if workgroup_size == 0 {
                self.default_reduction_workgroup_size_exec()
            } else {
                workgroup_size
            } as u64;
            let flavor_tag = match flavor {
                ReductionFlavor::Sum => 0,
                ReductionFlavor::Mean => 1,
                ReductionFlavor::CustomScale(_) => 2,
            };
            let shape = [
                ("reduce_len", reduce_len as u64),
                ("slices", num_slices as u64),
                ("rank", output_shape.len() as u64),
            ];
            let tuning = [("wg", actual_wg), ("flavor", flavor_tag)];
            self.record_kernel_launch_basic("fused_reduction", &shape, &tuning);
        }
        result
    }

    pub(crate) fn warmup_exec(&self) {
        if std::env::var("RUNMAT_WGPU_SKIP_WARMUP")
            .ok()
            .and_then(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    None
                } else if trimmed.eq_ignore_ascii_case("1")
                    || trimmed.eq_ignore_ascii_case("true")
                    || trimmed.eq_ignore_ascii_case("yes")
                {
                    Some(true)
                } else if trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("no")
                {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(false)
        {
            log::info!("RunMat Accelerate: skipping wgpu warmup (RUNMAT_WGPU_SKIP_WARMUP=1)");
            return;
        }

        let start = Instant::now();
        self.warmup_from_disk();
        let pl = &self.pipelines;
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary_broadcast.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.unary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.scalar.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_dim_sum_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_nd_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_global.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.fill.pipeline,
        );

        let ms = start.elapsed().as_millis() as u64;
        self.metrics.set_last_warmup_millis(ms);
    }

    pub(crate) fn fused_cache_counters_exec(&self) -> (u64, u64) {
        self.metrics.counters()
    }

    pub(crate) fn last_warmup_millis_exec(&self) -> Option<u64> {
        Some(self.metrics.last_warmup_millis())
    }

    pub(crate) fn telemetry_snapshot_exec(&self) -> runmat_accelerate_api::ProviderTelemetry {
        let (fusion_hits, fusion_misses) = self.metrics.counters();
        let (bind_hits, bind_misses) = self.bind_group_cache.counters();
        let mut by_layout: Vec<runmat_accelerate_api::BindGroupLayoutTelemetry> = Vec::new();
        let per = self.bind_group_cache.per_layout_counters();
        if let Ok(tags) = self.bind_group_layout_tags.lock() {
            for (ptr, (h, m)) in per {
                let tag = tags
                    .get(&ptr)
                    .cloned()
                    .unwrap_or_else(|| format!("layout_ptr_{:#x}", ptr));
                by_layout.push(runmat_accelerate_api::BindGroupLayoutTelemetry {
                    tag,
                    hits: h,
                    misses: m,
                });
            }
        }
        self.telemetry.snapshot(
            fusion_hits,
            fusion_misses,
            bind_hits,
            bind_misses,
            Some(by_layout),
        )
    }

    pub(crate) fn reset_telemetry_exec(&self) {
        self.telemetry.reset();
        self.metrics.reset();
        self.bind_group_cache.reset_counters();
    }

    pub(crate) fn default_reduction_workgroup_size_exec(&self) -> u32 {
        self.reduction_workgroup_size_default
    }

    pub(crate) fn two_pass_threshold_exec(&self) -> usize {
        self.reduction_two_pass_threshold
    }

    pub(crate) fn reduction_two_pass_mode_exec(&self) -> ReductionTwoPassMode {
        self.reduction_two_pass_mode
    }
}
