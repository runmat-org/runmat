use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_layer_norm_adjoint_exec(
        &self,
        input: &GpuTensorHandle,
        scale: &GpuTensorHandle,
        offset: &GpuTensorHandle,
        statistics: &GpuTensorHandle,
        upstream: &GpuTensorHandle,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<runmat_accelerate_api::DeepLearningLayerNormGradients> {
        ensure!(
            input.shape() == request.shape
                && self.deep_learning_layer_norm_adjoint_capability_exec(
                    input,
                    Some(scale),
                    Some(offset),
                    Some(statistics),
                    Some(upstream),
                    request,
                )? == DeepLearningLayerNormCapability::Ready,
            "normalization adjoint route was not admitted by this provider"
        );
        let (groups, _) = geometry(request)?;
        let total = shape_count(&request.shape)?;
        let channels = request.shape[request.channel_axis];
        let base = super::geometry::params(request, self.precision)?;

        let input_view = self.materialize_nd_reduction_input(input)?;
        let scale_view = self.materialize_nd_reduction_input(scale)?;
        let statistics_view = self.materialize_nd_reduction_input(statistics)?;
        let upstream_view = self.materialize_nd_reduction_input(upstream)?;
        let x = self.get_entry(input_view.as_ref().unwrap_or(input))?;
        let gamma = self.get_entry(scale_view.as_ref().unwrap_or(scale))?;
        let stats = self.get_entry(statistics_view.as_ref().unwrap_or(statistics))?;
        let dy = self.get_entry(upstream_view.as_ref().unwrap_or(upstream))?;

        let (dx_buffer, dgamma_buffer, dbeta_buffer) =
            self.with_emission(|emission, pending| -> Result<_> {
                let sums = self.create_storage_buffer_checked(
                    emission,
                    groups * 2,
                    "runmat-layer-norm-adjoint-sums",
                )?;
                let dx = self.create_storage_buffer_checked(
                    emission,
                    total,
                    "runmat-layer-norm-adjoint-input",
                )?;
                let dgamma = self.create_storage_buffer_checked(
                    emission,
                    channels,
                    "runmat-layer-norm-adjoint-scale",
                )?;
                let dbeta = self.create_storage_buffer_checked(
                    emission,
                    channels,
                    "runmat-layer-norm-adjoint-offset",
                )?;
                // Channels still dispatch when the batch is empty: pooled
                // buffers can contain older bytes and the zero gradients
                // must be written, not inferred from allocation.
                let groups_pipeline = self
                    .pipelines
                    .deep_learning_layer_norm_adjoint_groups
                    .get(emission)?;
                let channels_pipeline = self
                    .pipelines
                    .deep_learning_layer_norm_adjoint_channels
                    .get(emission)?;
                let input_pipeline = self
                    .pipelines
                    .deep_learning_layer_norm_adjoint_input
                    .get(emission)?;

                let mut offset = 0u32;
                while let Some(chunk) = next_chunk(u32::try_from(groups)?, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &NormParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..base
                        },
                        "runmat-layer-norm-adjoint-group-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-layer-norm-adjoint-groups-bind"),
                            layout: &groups_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: x.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: dy.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: gamma.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: stats.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: sums.as_ref().as_entire_binding(),
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
                        &groups_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }

                offset = 0;
                while let Some(chunk) = next_chunk(u32::try_from(channels)?, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &NormParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..base
                        },
                        "runmat-layer-norm-adjoint-channel-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-layer-norm-adjoint-channels-bind"),
                            layout: &channels_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: x.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: dy.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: stats.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: dgamma.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: dbeta.as_ref().as_entire_binding(),
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
                        &channels_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }

                offset = 0;
                while let Some(chunk) = next_chunk(u32::try_from(total)?, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &NormParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..base
                        },
                        "runmat-layer-norm-adjoint-input-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-layer-norm-adjoint-input-bind"),
                            layout: &input_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: x.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: dy.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: gamma.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: stats.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: sums.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: dx.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: params.as_entire_binding(),
                                },
                            ],
                        });
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        &input_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }
                Ok((dx, dgamma, dbeta))
            })??;

        let input = self.register_existing_buffer_with_storage(
            dx_buffer,
            request.shape.clone(),
            total,
            GpuTensorStorage::Real,
        )?;
        let scale = self.register_existing_buffer_with_storage(
            dgamma_buffer,
            request.scale_shape.clone(),
            channels,
            GpuTensorStorage::Real,
        )?;
        let offset = self.register_existing_buffer_with_storage(
            dbeta_buffer,
            request.offset_shape.clone(),
            channels,
            GpuTensorStorage::Real,
        )?;
        Ok(runmat_accelerate_api::DeepLearningLayerNormGradients {
            input,
            scale,
            offset,
        })
    }
}
