use super::*;
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderObservationFailure,
};

#[cfg(test)]
#[path = "telemetry/map_nan_tests.rs"]
mod map_nan_tests;

pub(crate) struct FusedReductionTelemetryRequest<'a> {
    pub(crate) shader: &'a str,
    pub(crate) requirements: runmat_accelerate_api::ProviderShaderRequirements,
    pub(crate) inputs: &'a [GpuTensorHandle],
    pub(crate) output_shape: &'a [usize],
    pub(crate) reduce_len: usize,
    pub(crate) num_slices: usize,
    pub(crate) workgroup_size: u32,
    pub(crate) flavor: ReductionFlavor,
}

impl WgpuProvider {
    pub(super) fn fused_logical_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_with_representation_telemetry_exec(
            shader,
            inputs,
            output_shape,
            len,
            super::elementwise::FusedElementwiseRepresentation {
                input_precision: self.precision,
                output_precision: self.precision,
                output_storage: GpuTensorStorage::Real,
                logical_output: true,
            },
        )
    }
    pub(crate) fn fused_elementwise_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_with_telemetry_and_storage_exec(
            shader,
            inputs,
            output_shape,
            len,
            GpuTensorStorage::Real,
        )
    }

    pub(crate) fn fused_elementwise_with_telemetry_and_storage_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        output_storage: GpuTensorStorage,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_with_representation_telemetry_exec(
            shader,
            inputs,
            output_shape,
            len,
            super::elementwise::FusedElementwiseRepresentation {
                input_precision: self.precision,
                output_precision: self.precision,
                output_storage,
                logical_output: false,
            },
        )
    }

    pub(super) fn fused_elementwise_with_representation_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        representation: super::elementwise::FusedElementwiseRepresentation,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_with_representation_telemetry_observed_exec(
            shader,
            inputs,
            output_shape,
            len,
            representation,
            |_| Ok(()),
        )
    }

    pub(super) fn fused_elementwise_validated_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        frame: &super::elementwise::SingleFrame,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_elementwise_validated_exec(
            shader,
            inputs,
            output_shape,
            len,
            frame,
            validated,
        );
        if result.is_ok() {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
            ];
            let tuning = [("wg", u64::from(self.linear_launch.workgroup()))];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    pub(super) fn fused_elementwise_with_representation_telemetry_observed_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        representation: super::elementwise::FusedElementwiseRepresentation,
        after_allocation: impl FnMut(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_elementwise_typed_exec_observed(
            shader,
            inputs,
            output_shape,
            len,
            representation,
            after_allocation,
        );
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

    #[cfg(test)]
    pub(super) fn fused_elementwise_prepared_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: super::elementwise::FusedElementwiseWork<'_>,
        prepared: &super::elementwise::FrozenFusionRecipe,
    ) -> Result<(
        GpuTensorHandle,
        runmat_accelerate_api::ProviderRealElementwiseObserved,
    )> {
        let start = Instant::now();
        let len = work.len;
        let rank = work.output_shape.len();
        let result = self.fused_elementwise_prepared_exec(shader, inputs, work, prepared);
        if result.is_ok() {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", rank as u64),
            ];
            let tuning = [("wg", u64::from(prepared.launch().workgroup()))];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    pub(super) fn fused_elementwise_prepared_probe_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: super::elementwise::FusedElementwiseWork<'_>,
        prepared: &super::elementwise::FrozenFusionRecipe,
    ) -> Result<(
        GpuTensorHandle,
        runmat_accelerate_api::ProviderRealElementwiseObserved,
        Vec<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    )> {
        self.fused_elementwise_prepared_probe_with_telemetry_admitted_exec(
            shader, inputs, work, prepared, None,
        )
    }

    pub(super) fn fused_elementwise_prepared_probe_with_telemetry_admitted_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: super::elementwise::FusedElementwiseWork<'_>,
        prepared: &super::elementwise::FrozenFusionRecipe,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<(
        GpuTensorHandle,
        runmat_accelerate_api::ProviderRealElementwiseObserved,
        Vec<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    )> {
        let start = Instant::now();
        let len = work.len;
        let rank = work.output_shape.len();
        let result = self.fused_elementwise_prepared_probe_admitted_exec(
            shader, inputs, work, prepared, admission,
        );
        if result.is_ok() {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", rank as u64),
            ];
            let tuning = [("wg", u64::from(prepared.launch().workgroup()))];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    pub(super) fn fused_elementwise_multi_validated_with_telemetry_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        frame: &super::elementwise::MultiFrame,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
    ) -> Result<Vec<GpuTensorHandle>> {
        let start = Instant::now();
        let result = self.fused_elementwise_multi_validated_exec(shader, inputs, frame, validated);
        if result.is_ok() {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
            let shape = [
                ("len", frame.len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", frame.shape.len() as u64),
                ("num_outputs", frame.output_classes.len() as u64),
            ];
            let tuning = [("wg", u64::from(self.workgroup_config.scalar))];
            self.record_kernel_launch_basic("fused_elementwise_multi", &shape, &tuning);
        }
        result
    }

    pub(crate) fn fused_indexed_multi_with_telemetry_exec(
        &self,
        request: super::elementwise::IndexedFusionMultiRequest<'_>,
    ) -> Result<Vec<GpuTensorHandle>> {
        let start = Instant::now();
        let shape = [
            ("len", request.len as u64),
            ("inputs", request.inputs.len() as u64),
            ("rank", request.output_shape.len() as u64),
            ("num_outputs", request.output_classes.len() as u64),
        ];
        let result = self.fused_indexed_multi_exec(request);
        if result.is_ok() {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            self.record_kernel_launch_basic("fused_indexed_multi", &shape, &[("wg", wg)]);
        }
        result
    }

    pub(crate) fn map_nan_to_zero_exec_typed(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.map_nan_to_zero_exec_typed_after_allocation(handle, |_| Ok(()))
    }

    fn map_nan_to_zero_exec_typed_after_allocation(
        &self,
        handle: &GpuTensorHandle,
        mut checkpoint: impl FnMut(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let produced = std::cell::Cell::new(false);
        self.map_nan_to_zero_exec_observed(handle, |output| {
            produced.set(true);
            checkpoint(output)
        })
        .map_err(|source| {
            let error = runmat_accelerate_api::ProviderExecutionError::from(source);
            if produced.get() {
                error.after_effect()
            } else {
                error
            }
        })
    }

    fn map_nan_to_zero_exec_observed(
        &self,
        handle: &GpuTensorHandle,
        mut after_allocation: impl FnMut(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let len = entry.len;
        if len == 0 {
            let out = self.with_emission(
                |emission, _| -> Result<runmat_accelerate_api::WgpuBufferOwner> {
                    let output =
                        self.create_storage_buffer(emission, 0, "runmat-nan-to-zero-empty")?;
                    after_allocation(&output)?;
                    Ok(output)
                },
            )??;
            return self.register_existing_buffer(out, entry.shape, 0);
        }
        let shader = crate::backend::wgpu::shaders::nan::map_source(self.precision);
        self.fused_elementwise_with_representation_telemetry_observed_exec(
            &shader,
            std::slice::from_ref(handle),
            &entry.shape,
            len,
            super::elementwise::FusedElementwiseRepresentation {
                input_precision: self.precision,
                output_precision: self.precision,
                output_storage: GpuTensorStorage::Real,
                logical_output: false,
            },
            after_allocation,
        )
    }

    pub(crate) fn not_nan_mask_exec(&self, handle: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let len = entry.len;
        if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-not-nan-mask-empty")
            })??;
            return self.register_existing_buffer(out, entry.shape, 0);
        }
        let shader = crate::backend::wgpu::shaders::nan::mask_source(self.precision);
        self.fused_elementwise_with_telemetry_exec(
            &shader,
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
            requirements,
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
            requirements,
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
        #[cfg(not(target_arch = "wasm32"))]
        self.warmup_from_disk();
        let warmup = self
            .with_emission(|emission, pending| -> Result<()> {
                let pl = &self.pipelines;
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission,
                    pending,
                    &pl.binary.get(emission)?.pipeline,
                )?;
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission,
                    pending,
                    &pl.binary_broadcast.get(emission)?.pipeline,
                )?;
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission,
                    pending,
                    &pl.unary.get(emission)?.pipeline,
                )?;
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission,
                    pending,
                    &pl.scalar.get(emission)?.pipeline,
                )?;
                if self.ieee_words_available(self.precision) {
                    crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
                        emission,
                        pending,
                        &pl.reduce_dim_sum_mean.get(emission)?.pipeline,
                    )?;
                }
                crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
                    emission,
                    pending,
                    &pl.reduce_nd_mean.get(emission)?.pipeline,
                )?;
                if self.ieee_words_available(self.precision) {
                    crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
                        emission,
                        pending,
                        &pl.reduce_global.get(emission)?.pipeline,
                    )?;
                }
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission,
                    pending,
                    &pl.fill.get(emission)?.pipeline,
                )?;

                Ok(())
            })
            .and_then(|result| result);
        if let Err(error) = warmup {
            if let Some(recorder) = current_provider_observation() {
                recorder.fail(ProviderObservationFailure::ProviderFailed);
            }
            log::warn!("RunMat Accelerate: WGPU warmup failed: {error:#}");
            return;
        }
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
