use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_layer_norm_exec(
        &self,
        input: &GpuTensorHandle,
        scale: &GpuTensorHandle,
        offset: &GpuTensorHandle,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<GpuTensorHandle> {
        Ok(self
            .deep_learning_layer_norm_dispatch(input, scale, offset, request, false)?
            .0)
    }

    pub(crate) fn deep_learning_layer_norm_recorded_exec(
        &self,
        input: &GpuTensorHandle,
        scale: &GpuTensorHandle,
        offset: &GpuTensorHandle,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<runmat_accelerate_api::DeepLearningLayerNormRecorded> {
        let (output, statistics) =
            self.deep_learning_layer_norm_dispatch(input, scale, offset, request, true)?;
        Ok(runmat_accelerate_api::DeepLearningLayerNormRecorded {
            output,
            statistics: statistics
                .ok_or_else(|| anyhow!("normalization statistics were not retained"))?,
        })
    }

    fn deep_learning_layer_norm_dispatch(
        &self,
        input: &GpuTensorHandle,
        scale: &GpuTensorHandle,
        offset: &GpuTensorHandle,
        request: &DeepLearningLayerNormRequest,
        capture_statistics: bool,
    ) -> Result<(GpuTensorHandle, Option<GpuTensorHandle>)> {
        ensure!(
            self.deep_learning_layer_norm_capability_exec(
                input,
                Some(scale),
                Some(offset),
                request
            )? == DeepLearningLayerNormCapability::Ready,
            "normalization route was not admitted by this provider"
        );
        let (groups, _) = geometry(request)?;
        let total = shape_count(&request.shape)?;
        let base = super::geometry::params(request, self.precision)?;
        if total == 0 {
            let (output, statistics) = self.with_emission(|emission, _| {
                let output =
                    self.create_storage_buffer_checked(emission, 0, "runmat-layer-norm-empty")?;
                let statistics = capture_statistics
                    .then(|| {
                        self.create_storage_buffer_checked(
                            emission,
                            0,
                            "runmat-layer-norm-empty-statistics",
                        )
                    })
                    .transpose()?;
                Ok::<_, anyhow::Error>((output, statistics))
            })??;
            let output = self.register_existing_buffer_with_storage(
                output,
                request.shape.clone(),
                0,
                GpuTensorStorage::Real,
            )?;
            let statistics = statistics
                .map(|buffer| {
                    self.register_existing_buffer_with_storage(
                        buffer,
                        vec![3, groups],
                        0,
                        GpuTensorStorage::Real,
                    )
                })
                .transpose()?;
            return Ok((output, statistics));
        }
        let input_materialized = self.materialize_nd_reduction_input(input)?;
        let scale_materialized = self.materialize_nd_reduction_input(scale)?;
        let offset_materialized = self.materialize_nd_reduction_input(offset)?;
        let source = self.get_entry(input_materialized.as_ref().unwrap_or(input))?;
        let gamma = self.get_entry(scale_materialized.as_ref().unwrap_or(scale))?;
        let beta = self.get_entry(offset_materialized.as_ref().unwrap_or(offset))?;
        let (output_buffer, statistics_buffer) =
            self.with_emission(|emission, pending| -> Result<_> {
                let statistics = self.create_storage_buffer_checked(
                    emission,
                    groups * 3,
                    "runmat-layer-norm-statistics",
                )?;
                let output = self.create_storage_buffer_checked(
                    emission,
                    total,
                    "runmat-layer-norm-output",
                )?;
                let stats_pipeline = self
                    .pipelines
                    .deep_learning_layer_norm_statistics
                    .get(emission)?;
                let affine_pipeline = self
                    .pipelines
                    .deep_learning_layer_norm_affine
                    .get(emission)?;
                let mut offset = 0u32;
                while let Some(chunk) = next_chunk(groups as u32, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &NormParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..base
                        },
                        "runmat-layer-norm-stat-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-layer-norm-stat-bind"),
                            layout: &stats_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: source.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: statistics.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params.as_entire_binding(),
                                },
                            ],
                        });
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        &stats_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }
                offset = 0;
                while let Some(chunk) = next_chunk(total as u32, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &NormParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..base
                        },
                        "runmat-layer-norm-affine-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-layer-norm-affine-bind"),
                            layout: &affine_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: source.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: gamma.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: beta.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: statistics.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: output.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: params.as_entire_binding(),
                                },
                            ],
                        });
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        &affine_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }
                Ok((output, statistics))
            })??;
        let output = self.register_existing_buffer_with_storage(
            output_buffer,
            request.shape.clone(),
            total,
            GpuTensorStorage::Real,
        )?;
        let statistics = if capture_statistics {
            Some(self.register_existing_buffer_with_storage(
                statistics_buffer,
                vec![3, groups],
                groups * 3,
                GpuTensorStorage::Real,
            )?)
        } else {
            None
        };
        Ok((output, statistics))
    }
}
