use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_layer_norm_exec(
        &self,
        input: &GpuTensorHandle,
        scale: &GpuTensorHandle,
        offset: &GpuTensorHandle,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            self.deep_learning_layer_norm_capability_exec(
                input,
                Some(scale),
                Some(offset),
                request
            )? == DeepLearningLayerNormCapability::Ready,
            "normalization route was not admitted by this provider"
        );
        let (groups, group_size) = geometry(request)?;
        let total = shape_count(&request.shape)?;
        let mut shape = [1u32; 4];
        for (target, &extent) in shape.iter_mut().zip(&request.shape) {
            *target = extent as u32;
        }
        let epsilon_bits = match self.precision {
            crate::backend::wgpu::types::NumericPrecision::F32 => {
                (request.epsilon as f32).to_bits() as u64
            }
            crate::backend::wgpu::types::NumericPrecision::F64 => request.epsilon.to_bits(),
        };
        let base = NormParams {
            shape,
            rank: request.shape.len() as u32,
            reduce_mask: request.reduce_mask as u32,
            groups: groups as u32,
            group_size: group_size as u32,
            channel_axis: request.channel_axis as u32,
            total: total as u32,
            chunk_offset: 0,
            chunk_count: 0,
            epsilon_low: epsilon_bits as u32,
            epsilon_high: (epsilon_bits >> 32) as u32,
        };
        if total == 0 {
            let output = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked(emission, 0, "runmat-layer-norm-empty")
            })??;
            return self.register_existing_buffer_with_storage(
                output,
                request.shape.clone(),
                0,
                GpuTensorStorage::Real,
            );
        }
        let input_materialized = self.materialize_nd_reduction_input(input)?;
        let scale_materialized = self.materialize_nd_reduction_input(scale)?;
        let offset_materialized = self.materialize_nd_reduction_input(offset)?;
        let source = self.get_entry(input_materialized.as_ref().unwrap_or(input))?;
        let gamma = self.get_entry(scale_materialized.as_ref().unwrap_or(scale))?;
        let beta = self.get_entry(offset_materialized.as_ref().unwrap_or(offset))?;
        let output = self.with_emission(|emission, pending| -> Result<_> {
            let statistics = self.create_storage_buffer_checked(
                emission,
                groups * 3,
                "runmat-layer-norm-statistics",
            )?;
            let output =
                self.create_storage_buffer_checked(emission, total, "runmat-layer-norm-output")?;
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
            Ok(output)
        })??;
        self.register_existing_buffer_with_storage(
            output,
            request.shape.clone(),
            total,
            GpuTensorStorage::Real,
        )
    }
}
