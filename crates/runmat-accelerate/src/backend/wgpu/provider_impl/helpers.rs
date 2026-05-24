use super::*;

impl WgpuProvider {
    fn precision_tag(&self) -> &'static str {
        match self.precision {
            NumericPrecision::F64 => "f64",
            NumericPrecision::F32 => "f32",
        }
    }

    fn record_kernel_launch_basic(
        &self,
        kernel: &'static str,
        shape: &[(&'static str, u64)],
        tuning: &[(&'static str, u64)],
    ) {
        self.telemetry
            .record_kernel_launch(kernel, Some(self.precision_tag()), shape, tuning);
    }

    fn record_matmul_kernel_launch(
        &self,
        m: usize,
        n: usize,
        k: usize,
        use_vec4: bool,
        chunked: bool,
    ) {
        let shape = [("m", m as u64), ("n", n as u64), ("k", k as u64)];
        let tuning = [
            ("vec4", if use_vec4 { 1 } else { 0 }),
            ("chunked", if chunked { 1 } else { 0 }),
        ];
        self.record_kernel_launch_basic("matmul", &shape, &tuning);
    }

    fn create_storage_buffer_checked_with_usage(
        &self,
        len: usize,
        label: &str,
        usage: BufferUsageClass,
    ) -> Result<Arc<wgpu::Buffer>> {
        // Centralised guard + warning for oversized allocations
        let size_bytes = (len as u64) * self.element_size as u64;
        if size_bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                label,
                size_bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        let (buffer, reused) = self.create_storage_buffer_for_usage(usage, len, label);
        if reused && std::env::var("RUNMAT_DEBUG_RESIDENCY").is_ok() {
            log::debug!(
                "[residency_debug] reused buffer label={} usage={:?} len={} ptr={:p}",
                label,
                usage,
                len,
                buffer.as_ref()
            );
        }
        if !reused && size_bytes >= (256u64 << 20) {
            log::warn!(
                "{}: large GPU allocation ({} bytes) len={} elems",
                label,
                size_bytes,
                len
            );
        }
        Ok(buffer)
    }

    fn create_storage_buffer_checked(&self, len: usize, label: &str) -> Result<Arc<wgpu::Buffer>> {
        self.create_storage_buffer_checked_with_usage(len, label, BufferUsageClass::Generic)
    }

    fn image_normalize_vector_width(&self) -> u32 {
        match self.precision {
            NumericPrecision::F64 => 2,
            NumericPrecision::F32 => 4,
        }
    }

    fn round_up_to_multiple(value: u32, mult: u32) -> u32 {
        if mult <= 1 {
            return value;
        }
        let remainder = value % mult;
        if remainder == 0 {
            value
        } else {
            value.saturating_add(mult - remainder).max(mult)
        }
    }

    fn select_image_normalize_tuning(&self, batches: u32, plane: u32) -> ImageNormalizeTuning {
        let batches = batches.max(1);
        let plane = plane.max(1);
        let mut lane =
            ((plane as f64) / Self::IMAGE_NORMALIZE_TARGET_SAMPLES_PER_LANE).ceil() as u32;
        lane = lane.max(32);
        let max_lane_dim = self.workgroup_config.max_x.max(32);
        lane = lane.min(max_lane_dim);
        lane = Self::round_up_to_multiple(lane, 32).max(32);
        let plane_per_lane = (plane as f64 / lane as f64).max(1.0);
        let mut values_per_thread =
            ((plane_per_lane / Self::IMAGE_NORMALIZE_TARGET_LOOP_ITERS_PER_LANE).ceil() as u32)
                .clamp(1, 8);
        if plane <= 512 {
            values_per_thread = values_per_thread.min(4);
        }
        let spatial_tile = if plane <= 1024 {
            1
        } else if plane <= 4096 {
            2
        } else {
            4
        };
        let mut batch_tile = if plane >= 8192 {
            batches.min(16)
        } else {
            batches.min(32)
        };
        if batches <= 4 {
            batch_tile = batches;
        }
        let tuning = ImageNormalizeTuning {
            batch_tile: batch_tile.max(1),
            values_per_thread,
            lane_count: lane,
            spatial_tile,
        };
        let sanitized = self
            .workgroup_config
            .sanitize_image_normalize_tuning(tuning, batches);
        debug!(
            "select_image_normalize_tuning batches={} plane={} raw={:?} sanitized={:?}",
            batches, plane, tuning, sanitized
        );
        sanitized
    }

    fn resolve_image_normalize_tuning(
        &self,
        batches: u32,
        plane: u32,
    ) -> (ImageNormalizeTuning, bool) {
        let key = ImageNormalizeKey::new(self.precision, batches, plane);
        if self.image_norm_autotune.is_enabled() {
            if let Some(tuning) = self.image_norm_autotune.get(&key) {
                let sanitized = self
                    .workgroup_config
                    .sanitize_image_normalize_tuning(tuning, batches);
                if sanitized != tuning {
                    debug!(
                        "image_normalize autotune sanitized cached key {:?}: {:?} -> {:?}",
                        key, tuning, sanitized
                    );
                    debug!(
                        "resolve_image_normalize_tuning returning cached {:?} for key {:?}",
                        sanitized, key
                    );
                    self.image_norm_autotune.insert(key, sanitized);
                } else {
                    debug!(
                        "image_normalize autotune reusing cached key {:?}: {:?}",
                        key, tuning
                    );
                }
                return (sanitized, true);
            }
            let tuning = self.select_image_normalize_tuning(batches, plane);
            debug!(
                "image_normalize autotune inserted key {:?}: {:?}",
                key, tuning
            );
            self.image_norm_autotune.insert(key, tuning);
            (tuning, false)
        } else {
            let tuning = self.select_image_normalize_tuning(batches, plane);
            debug!(
                "resolve_image_normalize_tuning returning fresh {:?} for key {:?}",
                tuning, key
            );
            (tuning, false)
        }
    }

    fn image_normalize_hot_stream_cap(&self, plane: u32, batches: u32) -> u32 {
        if batches == 0 {
            return 0;
        }
        let plane = plane.max(1);
        let bytes_per_batch = plane as u64 * self.element_size as u64;
        if bytes_per_batch == 0 {
            return batches;
        }
        let target_bytes = self
            .image_normalize_stream_target_bytes()
            .max(bytes_per_batch);
        let max_batches = target_bytes / bytes_per_batch;
        max_batches
            .clamp(1, batches as u64)
            .try_into()
            .unwrap_or(batches)
    }

    fn image_normalize_stream_target_bytes(&self) -> u64 {
        if let Ok(raw) = std::env::var("RUNMAT_IMAGE_NORMALIZE_STREAM_TARGET_BYTES") {
            if let Ok(parsed) = raw.parse::<u64>() {
                return parsed.max(1);
            }
        }
        let limit = self.adapter_limits.max_buffer_size;
        let default = 6u64 * 1024 * 1024 * 1024;
        default.min(limit).max((self.element_size as u64) * 4)
    }

    fn image_normalize_pipeline(
        &self,
        tuning: &ImageNormalizeTuning,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        if let Ok(cache) = self.image_norm_pipeline_cache.lock() {
            if let Some(existing) = cache.get(tuning) {
                return Ok(existing.clone());
            }
        }
        info!(
            "Compiling image_normalize pipeline tuning: batch_tile={} values/thread={} lane={} spatial={}",
            tuning.batch_tile, tuning.values_per_thread, tuning.lane_count, tuning.spatial_tile
        );
        let template = match self.precision {
            NumericPrecision::F64 => IMAGE_NORMALIZE_SHADER_F64,
            NumericPrecision::F32 => IMAGE_NORMALIZE_SHADER_F32,
        };
        let shader_src = template
            .replace("@BT@", &tuning.batch_tile.to_string())
            .replace("@VP@", &tuning.values_per_thread.to_string())
            .replace("@WG@", &tuning.lane_count.to_string())
            .replace("@ST@", &tuning.spatial_tile.to_string())
            .replace("@BV@", &self.image_normalize_vector_width().to_string());
        let module = self
            .device_ref()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-image-normalize-shader-dyn"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
            });
        let pipeline_layout = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
            self.device_ref(),
            "runmat-image-normalize-pipeline-dyn",
            &self.pipelines.image_normalize.layout,
        );
        let pipeline = crate::backend::wgpu::cache::factory::create_compute_pipeline(
            self.device_ref(),
            "runmat-image-normalize-pipeline-dyn",
            &pipeline_layout,
            &module,
        );
        let arc = Arc::new(pipeline);
        if let Ok(mut cache) = self.image_norm_pipeline_cache.lock() {
            cache.insert(*tuning, arc.clone());
        }
        Ok(arc)
    }
    pub fn device_id(&self) -> u32 {
        self.cache_device_id
    }

    pub(crate) fn device_ref(&self) -> &wgpu::Device {
        self.device.as_ref()
    }
    pub(crate) fn queue_ref(&self) -> &wgpu::Queue {
        self.queue.as_ref()
    }

    fn warmup_from_disk(&self) {
        if std::env::var("RUNMAT_DISABLE_PIPELINE_WARMUP").is_ok() {
            return;
        }
        crate::backend::wgpu::warmup::warmup_from_disk(
            &self.device,
            self.pipeline_cache_dir.as_deref(),
            self.precision,
            |bytes, tag, wg| self.compute_pipeline_hash_bytes(bytes, tag, wg),
            |key, pl, module, label, src, tag, wg| {
                self.get_or_create_pipeline(key, pl, module, label, src, tag, wg)
            },
            |pipeline| {
                crate::backend::wgpu::warmup::noop_after_create(&self.device, &self.queue, pipeline)
            },
        );
    }

    fn cached_bind_group_layout<F>(&self, key: &str, build: F) -> Arc<wgpu::BindGroupLayout>
    where
        F: FnOnce(&wgpu::Device) -> wgpu::BindGroupLayout,
    {
        if let Ok(cache) = self.bind_group_layout_cache.lock() {
            if let Some(layout) = cache.get(key).cloned() {
                return layout;
            }
        }
        let layout = Arc::new(build(self.device_ref()));
        let ptr = layout.as_ref() as *const wgpu::BindGroupLayout as usize;
        if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
            tags.entry(ptr).or_insert_with(|| key.to_string());
        }
        if let Ok(mut cache) = self.bind_group_layout_cache.lock() {
            cache.insert(key.to_string(), layout.clone());
        }
        layout
    }

    fn cached_bind_group_layout_for_tag(&self, tag: &str) -> Option<Arc<wgpu::BindGroupLayout>> {
        if let Ok(cache) = self.bind_group_layout_cache.lock() {
            if let Some(layout) = cache.get(tag).cloned() {
                return Some(layout);
            }
        }
        let layout =
            crate::backend::wgpu::bindings::build_bgl_for_layout_tag(self.device_ref(), tag)?;
        let layout = Arc::new(layout);
        let ptr = layout.as_ref() as *const wgpu::BindGroupLayout as usize;
        if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
            tags.entry(ptr).or_insert_with(|| tag.to_string());
        }
        if let Ok(mut cache) = self.bind_group_layout_cache.lock() {
            cache.insert(tag.to_string(), layout.clone());
        }
        Some(layout)
    }

    fn cached_fusion_bind_group_layout(&self, inputs_len: usize) -> Arc<wgpu::BindGroupLayout> {
        let key = format!("runmat-fusion-layout-{}", inputs_len);
        self.cached_bind_group_layout(&key, |device| {
            crate::backend::wgpu::bindings::build_fusion_bgl(device, inputs_len)
        })
    }

    pub fn try_compile_kernel(&self, label: &str, wgsl_src: &str) -> Result<()> {
        crate::backend::wgpu::debug::try_compile_kernel(&self.device, label, wgsl_src);
        Ok(())
    }

    pub fn probe_kernel_with_buffers(&self, label: &str, wgsl_src: &str, wg: u32) -> Result<()> {
        crate::backend::wgpu::debug::probe_kernel_with_buffers(
            &self.device,
            &self.queue,
            label,
            wgsl_src,
            wg,
        );
        Ok(())
    }

    async fn image_normalize_cpu_fallback(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        let mut host = <Self as AccelProvider>::download(self, input).await?;
        ensure!(
            host.shape.len() == 3,
            "image_normalize: expected 3-D tensor, got {:?}",
            host.shape
        );
        ensure!(
            host.shape[0] == desc.batch
                && host.shape[1] == desc.height
                && host.shape[2] == desc.width,
            "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
            (desc.batch, desc.height, desc.width),
            host.shape
        );

        let batch = desc.batch;
        let height = desc.height;
        let width = desc.width;
        let plane = height * width;

        if plane == 0 {
            let view = HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            return self.upload(&view);
        }

        let stride_h = batch;
        let stride_w = batch * height;

        let gain = desc.gain.unwrap_or(1.0);
        let bias = desc.bias.unwrap_or(0.0);
        let gamma = desc.gamma;

        for b in 0..batch {
            let mut sum = 0.0;
            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    sum += host.data[idx];
                }
            }
            let mean = sum / plane as f64;

            let mut sq_sum = 0.0;
            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    let diff = host.data[idx] - mean;
                    sq_sum += diff * diff;
                }
            }
            let variance = sq_sum / plane as f64;
            let sigma = (variance + desc.epsilon).sqrt();
            let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };

            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    let mut value = (host.data[idx] - mean) * inv_sigma;
                    if desc.gain.is_some() {
                        value *= gain;
                    }
                    if desc.bias.is_some() {
                        value += bias;
                    }
                    value = value.max(0.0);
                    if let Some(gamma) = gamma {
                        value = value.powf(gamma);
                    }
                    host.data[idx] = value;
                }
            }
        }

        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        self.upload(&view)
    }

    /// Get or create a compute pipeline from cache using a caller-provided hash key.
    #[allow(clippy::too_many_arguments)]
    fn get_or_create_pipeline(
        &self,
        hash_key: u64,
        pipeline_layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        label: &str,
        persist_wgsl_src: Option<&[u8]>,
        persist_layout_tag: Option<&str>,
        persist_workgroup_size: Option<u32>,
    ) -> Arc<wgpu::ComputePipeline> {
        if let Some(p) = self
            .fused_pipeline_cache
            .try_lock()
            .ok()
            .and_then(|guard| guard.get(&hash_key).cloned())
        {
            self.metrics.inc_hit();
            return p;
        }
        self.metrics.inc_miss();
        // Persist WGSL + meta for warmup on next run
        self.persist_pipeline_meta(
            hash_key,
            label,
            persist_layout_tag,
            persist_workgroup_size,
            persist_wgsl_src,
        );
        let p = Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(pipeline_layout),
                    module,
                    entry_point: "main",
                }),
        );
        if let Ok(mut guard) = self.fused_pipeline_cache.try_lock() {
            guard.insert(hash_key, p.clone());
        }
        p
    }

    pub fn compute_pipeline_hash_bytes(
        &self,
        shader_bytes: &[u8],
        layout_tag: &str,
        workgroup_size: Option<u32>,
    ) -> u64 {
        cache_key::compute_pipeline_hash_bytes(shader_bytes, layout_tag, workgroup_size)
    }

    fn persist_pipeline_meta(
        &self,
        hash_key: u64,
        label: &str,
        layout_tag: Option<&str>,
        workgroup_size: Option<u32>,
        wgsl_src: Option<&[u8]>,
    ) {
        if let Some(dir) = &self.pipeline_cache_dir {
            cache_persist::persist_pipeline_meta(
                dir,
                hash_key,
                label,
                layout_tag,
                workgroup_size,
                self.precision,
                wgsl_src,
            );
        }
    }

}
