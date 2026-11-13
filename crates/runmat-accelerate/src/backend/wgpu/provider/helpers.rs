use runmat_accelerate_api::AccelProvider;

use anyhow::{ensure, Result};
use log::{debug, info};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, WgpuEmission};
use std::sync::Arc;

use super::backend_shared::gpu_per_buffer_limit_error;
use super::backend_types::{ImageNormalizeKey, ImageNormalizeTuning, WgpuProvider};
use super::pipeline_cache::{CacheLookup, PipelineAllocation, PipelineCache, PipelineOwner};
use crate::backend::wgpu::cache::{key as cache_key, persist as cache_persist};
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::shaders::image_normalize::{self, ImageNormalizeShaderGeometry};
use crate::backend::wgpu::types::NumericPrecision;
use runmat_execution::resource::ProviderHostMetadataAdmission;

impl WgpuProvider {
    pub(super) fn precision_tag(&self) -> &'static str {
        match self.precision {
            NumericPrecision::F64 => "f64",
            NumericPrecision::F32 => "f32",
        }
    }

    pub(super) fn record_kernel_launch_basic(
        &self,
        kernel: &'static str,
        shape: &[(&'static str, u64)],
        tuning: &[(&'static str, u64)],
    ) {
        self.telemetry
            .record_kernel_launch(kernel, Some(self.precision_tag()), shape, tuning);
    }

    pub(super) fn record_matmul_kernel_launch(
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

    pub(super) fn create_storage_buffer_checked_with_usage(
        &self,
        emission: &WgpuEmission<'_>,
        len: usize,
        label: &str,
        usage: BufferUsageClass,
    ) -> Result<runmat_accelerate_api::WgpuBufferOwner> {
        self.create_storage_buffer_checked_with_element_size(
            emission,
            len,
            label,
            usage,
            self.element_size,
        )
    }

    /// Same checked pool owner, with the operation's admitted native element
    /// size rather than the provider's preferred floating representation.
    pub(super) fn create_storage_buffer_checked_with_element_size(
        &self,
        emission: &WgpuEmission<'_>,
        len: usize,
        label: &str,
        usage: BufferUsageClass,
        element_size: usize,
    ) -> Result<runmat_accelerate_api::WgpuBufferOwner> {
        // Centralised guard + warning for oversized allocations
        let size_bytes = BufferResidency::requested_bytes(len, element_size)
            .ok_or_else(|| anyhow::anyhow!("{label}: allocation byte extent overflow"))?;
        if size_bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                label,
                size_bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        emission.ensure_device(&self.access)?;
        let (buffer, reused) =
            self.allocations
                .acquire(emission, usage, len, element_size, label)?;
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

    pub(super) fn create_storage_buffer_checked(
        &self,
        emission: &WgpuEmission<'_>,
        len: usize,
        label: &str,
    ) -> Result<runmat_accelerate_api::WgpuBufferOwner> {
        self.create_storage_buffer_checked_with_usage(
            emission,
            len,
            label,
            BufferUsageClass::Generic,
        )
    }

    pub(super) fn round_up_to_multiple(value: u32, mult: u32) -> u32 {
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

    pub(super) fn select_image_normalize_tuning(
        &self,
        batches: u32,
        plane: u32,
    ) -> ImageNormalizeTuning {
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

    pub(super) fn resolve_image_normalize_tuning(
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

    pub(super) fn image_normalize_hot_stream_cap(&self, plane: u32, batches: u32) -> u32 {
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

    pub(super) fn image_normalize_stream_target_bytes(&self) -> u64 {
        if let Ok(raw) = std::env::var("RUNMAT_IMAGE_NORMALIZE_STREAM_TARGET_BYTES") {
            if let Ok(parsed) = raw.parse::<u64>() {
                return parsed.max(1);
            }
        }
        let limit = self.adapter_limits.max_buffer_size;
        let default = 6u64 * 1024 * 1024 * 1024;
        default.min(limit).max((self.element_size as u64) * 4)
    }

    pub(super) fn image_normalize_pipeline(
        &self,
        emission: &WgpuEmission<'_>,
        tuning: &ImageNormalizeTuning,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        emission.ensure_device(&self.access)?;
        let observation = super::pipeline_observation::PipelineObservation::begin();
        let cached = self
            .image_norm_pipeline_cache
            .lock()
            .ok()
            .and_then(|cache| cache.get(tuning).cloned());
        if let Some(existing) = cached {
            observation.finish(runmat_execution::CandidatePreparationState::Warm);
            return Ok(existing);
        }
        crate::backend::wgpu::cache::scope::allow_creation(emission.mode())?;
        info!(
            "Compiling image_normalize pipeline tuning: batch_tile={} values/thread={} lane={} spatial={}",
            tuning.batch_tile, tuning.values_per_thread, tuning.lane_count, tuning.spatial_tile
        );
        let shader_src = image_normalize::materialize(
            self.precision,
            ImageNormalizeShaderGeometry {
                batch_tile: tuning.batch_tile,
                values_per_thread: tuning.values_per_thread,
                lane_count: tuning.lane_count,
                spatial_tile: tuning.spatial_tile,
            },
        );
        let module = crate::backend::wgpu::shader_compilation::create_shader_module(
            emission,
            "runmat-image-normalize-shader-dyn",
            &shader_src,
            None,
        )?;
        let pipeline_layout = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
            emission,
            "runmat-image-normalize-pipeline-dyn",
            &self.pipelines.image_normalize.get(emission)?.layout,
        );
        let pipeline = crate::backend::wgpu::cache::factory::create_compute_pipeline(
            emission,
            "runmat-image-normalize-pipeline-dyn",
            &pipeline_layout,
            &module,
        );
        let arc = Arc::new(pipeline);
        if let Ok(mut cache) = self.image_norm_pipeline_cache.lock() {
            cache.insert(*tuning, arc.clone());
        }
        observation.finish(runmat_execution::CandidatePreparationState::Cold);
        Ok(arc)
    }
    pub fn device_id(&self) -> u32 {
        self.runtime_device_id
    }

    /// Hardware identifier used to partition persistent pipeline caches.
    /// It is not the process-local owner identity carried by tensor handles.
    pub fn pipeline_cache_device_id(&self) -> u32 {
        self.cache_device_id
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn warmup_from_disk(&self) {
        if std::env::var("RUNMAT_DISABLE_PIPELINE_WARMUP").is_ok() {
            return;
        }
        // The legacy disk records contain WGSL but no typed shader-feature
        // requirement. Do not compile an unidentified F64 record when IEEE
        // word shaders are unavailable. Selected reduction caches carry their
        // own typed requirement and remain independently available.
        if !self.ieee_words_available(self.precision) {
            return;
        }
        crate::backend::wgpu::warmup::warmup_from_disk(
            self.pipeline_cache_dir.as_deref(),
            self.precision,
            self.access.enabled_features(),
            |bytes, tag, wg| self.compute_pipeline_hash_bytes(bytes, tag, wg),
            |key, tag, src, _bytes, workgroup| {
                // Build under the device's exclusive validation scopes, but
                // resolve their detached receipt only after releasing access.
                // Never publish an invalid driver object into the shared cache.
                let receipt = self.access.with_profiled_scoped(
                    &self.emission_authority,
                    |emission| -> Result<Option<_>> {
                        let Some(bgl) =
                            crate::backend::wgpu::bindings::build_bgl_for_layout_tag(emission, tag)
                        else {
                            return Ok(None);
                        };
                        let bgl = Arc::new(bgl);
                        let pl =
                            crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                                emission,
                                "warmup-pipeline-layout",
                                bgl.as_ref(),
                            );
                        let module = crate::backend::wgpu::pipelines::create_shader_module(
                            emission,
                            "warmup-shader-module",
                            src,
                        )?;
                        Ok(Some((
                            bgl,
                            crate::backend::wgpu::cache::factory::create_compute_pipeline(
                                emission,
                                "warmup-precompiled-pipeline",
                                &pl,
                                &module,
                            ),
                        )))
                    },
                )?;
                let Some((layout, pipeline)) = pollster::block_on(receipt.resolve())?? else {
                    return Ok(false);
                };
                let pipeline = PipelineOwner::unbounded(pipeline);
                let warmed = if let Some(workgroup) = workgroup {
                    let recipe = super::pipeline_recipe::PipelineRecipe {
                        shader: src.to_owned(),
                        layout: tag.to_owned(),
                        workgroup,
                    };
                    anyhow::ensure!(recipe.key() == key, "warmup recipe key changed");
                    Some(Arc::new(super::validated_fusion::WarmedPipelineReceipt {
                        recipe,
                        layout,
                        pipeline: pipeline.clone(),
                    }))
                } else {
                    None
                };
                // Both locks are acquired only after the scoped receipt has
                // resolved; a poisoned cache publishes neither half.
                let mut raw_cache = self
                    .fused_pipeline_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("warmup pipeline cache is poisoned"))?;
                let mut warmed_cache = self
                    .warmed_pipeline_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("warmed pipeline cache is poisoned"))?;
                let (_, retired) = raw_cache.insert_unbounded(key, &pipeline)?;
                if let Some(warmed) = warmed {
                    warmed_cache.entry(key).or_insert(warmed);
                }
                drop(warmed_cache);
                drop(raw_cache);
                drop(retired);
                Ok(true)
            },
        );
    }

    pub(super) fn cached_bind_group_layout<F>(
        &self,
        emission: &WgpuEmission<'_>,
        key: &str,
        build: F,
    ) -> Result<Arc<wgpu::BindGroupLayout>>
    where
        F: FnOnce(&WgpuEmission<'_>) -> wgpu::BindGroupLayout,
    {
        emission.ensure_device(&self.access)?;
        if let Ok(cache) = self.bind_group_layout_cache.lock() {
            if let Some(layout) = cache.get(key).cloned() {
                return Ok(layout);
            }
        }
        crate::backend::wgpu::cache::scope::allow_creation(emission.mode())?;
        let layout = Arc::new(build(emission));
        let ptr = layout.as_ref() as *const wgpu::BindGroupLayout as usize;
        if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
            tags.entry(ptr).or_insert_with(|| key.to_string());
        }
        if let Ok(mut cache) = self.bind_group_layout_cache.lock() {
            cache.insert(key.to_string(), layout.clone());
        }
        Ok(layout)
    }

    pub(super) fn cached_bind_group_layout_for_tag(
        &self,
        emission: &WgpuEmission<'_>,
        tag: &str,
    ) -> Result<Option<Arc<wgpu::BindGroupLayout>>> {
        emission.ensure_device(&self.access)?;
        if let Ok(cache) = self.bind_group_layout_cache.lock() {
            if let Some(layout) = cache.get(tag).cloned() {
                return Ok(Some(layout));
            }
        }
        crate::backend::wgpu::cache::scope::allow_creation(emission.mode())?;
        let Some(layout) = crate::backend::wgpu::bindings::build_bgl_for_layout_tag(emission, tag)
        else {
            return Ok(None);
        };
        let layout = Arc::new(layout);
        let ptr = layout.as_ref() as *const wgpu::BindGroupLayout as usize;
        if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
            tags.entry(ptr).or_insert_with(|| tag.to_string());
        }
        if let Ok(mut cache) = self.bind_group_layout_cache.lock() {
            cache.insert(tag.to_string(), layout.clone());
        }
        Ok(Some(layout))
    }

    pub(super) fn cached_fusion_bind_group_layout(
        &self,
        emission: &WgpuEmission<'_>,
        inputs_len: usize,
    ) -> Result<Arc<wgpu::BindGroupLayout>> {
        let key = format!("runmat-fusion-layout-{}", inputs_len);
        self.cached_bind_group_layout(emission, &key, |emission| {
            crate::backend::wgpu::bindings::build_fusion_bgl(emission, inputs_len)
        })
    }

    pub fn try_compile_kernel(&self, label: &str, wgsl_src: &str) -> Result<()> {
        self.with_emission(|emission, _| {
            crate::backend::wgpu::debug::try_compile_kernel(emission, label, wgsl_src)
        })??;
        Ok(())
    }

    pub fn probe_kernel_with_buffers(&self, label: &str, wgsl_src: &str, wg: u32) -> Result<()> {
        self.with_emission(|emission, pending| {
            crate::backend::wgpu::debug::probe_kernel_with_buffers(
                emission, pending, label, wgsl_src, wg,
            )
        })??;
        Ok(())
    }

    pub(super) async fn image_normalize_cpu_fallback(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            desc.epsilon.is_finite(),
            "image_normalize: epsilon must be finite"
        );
        ensure!(
            desc.epsilon >= 0.0,
            "image_normalize: epsilon must be non-negative"
        );

        if input.physical().element_type() == runmat_accelerate_api::NumericElementType::F32 {
            use runmat_accelerate_api::{
                GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
            };
            let host = <Self as AccelProvider>::download_numeric(self, input).await?;
            ensure!(
                host.shape == [desc.batch, desc.height, desc.width],
                "image_normalize: descriptor dimensions do not match native single input"
            );
            let HostNumericDataOwned::F32(data) = host.data else {
                return Err(anyhow::anyhow!(
                    "image_normalize: provider lost native single input precision"
                ));
            };
            let batch = desc.batch;
            let plane = desc.height * desc.width;
            let mut output = vec![0.0f32; data.len()];
            if plane != 0 {
                for b in 0..batch {
                    let mut sum = 0.0f32;
                    for index in 0..plane {
                        sum += data[b + batch * index];
                    }
                    let mean = sum / plane as f32;
                    let mut squared_sum = 0.0f32;
                    for index in 0..plane {
                        let diff = data[b + batch * index] - mean;
                        squared_sum += diff * diff;
                    }
                    let sigma = (squared_sum / plane as f32 + desc.epsilon as f32).sqrt();
                    let inverse_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };
                    for index in 0..plane {
                        let offset = b + batch * index;
                        let mut value = (data[offset] - mean) * inverse_sigma;
                        if let Some(gain) = desc.gain {
                            value *= gain as f32;
                        }
                        if let Some(bias) = desc.bias {
                            value += bias as f32;
                        }
                        if desc.clamp_zero {
                            value = value.max(0.0);
                        }
                        if let Some(gamma) = desc.gamma {
                            value = value.powf(gamma as f32);
                        }
                        output[offset] = value;
                    }
                }
            }
            return self.upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F32(&output),
                shape: &host.shape,
                storage: GpuTensorStorage::Real,
            });
        }

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
                    if desc.clamp_zero {
                        value = value.max(0.0);
                    }
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
    pub(super) fn get_or_create_pipeline(
        &self,
        emission: &WgpuEmission<'_>,
        hash_key: u64,
        pipeline_layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        label: &str,
        persist_wgsl_src: Option<&[u8]>,
        persist_layout_tag: Option<&str>,
        persist_workgroup_size: Option<u32>,
    ) -> Result<PipelineOwner> {
        emission.ensure_device(&self.access)?;
        self.get_or_create_pipeline_with(
            emission,
            None,
            hash_key, label, persist_wgsl_src, persist_layout_tag, persist_workgroup_size,
            || Ok(emission.device().create_compute_pipeline(
                &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                    label: Some(label), layout: Some(pipeline_layout), module: module, entry_point: "main",
                },
            )),
        )
    }

    pub(super) fn recipe_pipeline(
        &self,
        emission: &WgpuEmission<'_>,
        recipe: &super::pipeline_recipe::PipelineRecipe,
        bindings: &wgpu::BindGroupLayout,
        label: &str,
    ) -> Result<PipelineOwner> {
        emission.ensure_device(&self.access)?;
        self.get_or_create_pipeline_with(emission, None, recipe.key(), label, Some(recipe.shader.as_bytes()),
            Some(&recipe.layout), Some(recipe.workgroup), || {
                let prepared = crate::backend::wgpu::shader_compilation::prepare(
                    &recipe.shader, self.access.enabled_features(), None)?;
                let layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
                    emission, label, bindings);
                let module = crate::backend::wgpu::shader_compilation::create_prepared(
                    emission, label, &prepared)?;
                Ok(emission.device().create_compute_pipeline(
                    &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                        label: Some(label), layout: Some(&layout), module: &module, entry_point: "main",
                    },
                ))
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_or_create_pipeline_with(
        &self,
        emission: &WgpuEmission<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
        hash_key: u64,
        label: &str,
        persist_wgsl_src: Option<&[u8]>,
        persist_layout_tag: Option<&str>,
        persist_workgroup_size: Option<u32>,
        create: impl FnOnce() -> Result<wgpu::ComputePipeline>,
    ) -> Result<PipelineOwner> {
        emission.ensure_device(&self.access)?;
        if let (Some(source), Some(layout_tag)) = (persist_wgsl_src, persist_layout_tag) {
            ensure!(
                !cache_key::has_unresolved_launch_tokens(source),
                "{label}: persisted shader source has unresolved launch tokens"
            );
            ensure!(
                hash_key
                    == self.compute_pipeline_hash_bytes(source, layout_tag, persist_workgroup_size),
                "{label}: persisted shader source does not match its pipeline cache key"
            );
        }
        let observation = super::pipeline_observation::PipelineObservation::begin();
        let reservation =
            match PipelineCache::reserve(&self.fused_pipeline_cache, hash_key, metadata) {
                Ok(CacheLookup::Hit(p)) => {
                    self.metrics.inc_hit();
                    observation.finish(runmat_execution::CandidatePreparationState::Warm);
                    return Ok(p);
                }
                Ok(CacheLookup::Miss(reservation)) => Some(reservation),
                // Historical unmanaged poison behavior is an uncached pipeline.
                // Managed calls never bypass their actual table admission.
                Err(error) if metadata.is_none() && self.fused_pipeline_cache.is_poisoned() => {
                    drop(error);
                    super::pipeline_observation::poisoned_cache();
                    None
                }
                Err(error) => return Err(error),
            };
        crate::backend::wgpu::cache::scope::allow_creation(emission.mode())?;
        let allocation = PipelineAllocation::prepare(metadata)?;
        // Admission failure must precede publication and all driver creation.
        let persistence = self
            .pipeline_cache_dir
            .as_ref()
            .map(|dir| cache_persist::PreparedPipelinePersistence::prepare(dir, hash_key, metadata))
            .transpose()?;
        self.metrics.inc_miss();
        // Cache lock is released before any driver/compiler callback.
        let p = allocation.allocate(create()?)?;
        let (p, loser, inserted) = if let Some(reservation) = reservation {
            reservation.publish(p)?
        } else {
            (p, None, true)
        };
        // A concurrent winner retains its identity. Neither loser destruction
        // nor metadata callbacks/persistence run under the table mutex.
        drop(loser);
        if inserted {
            if let (Some(prepared), Some(dir)) = (persistence, self.pipeline_cache_dir.as_ref()) {
                prepared.write(
                    dir,
                    label,
                    persist_layout_tag,
                    persist_workgroup_size,
                    self.precision,
                    persist_wgsl_src,
                );
            }
        }
        observation.finish(runmat_execution::CandidatePreparationState::Cold);
        Ok(p)
    }

    pub fn compute_pipeline_hash_bytes(
        &self,
        shader_bytes: &[u8],
        layout_tag: &str,
        workgroup_size: Option<u32>,
    ) -> u64 {
        cache_key::compute_pipeline_hash_bytes(shader_bytes, layout_tag, workgroup_size)
    }

    pub(super) fn persist_pipeline_meta(
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
