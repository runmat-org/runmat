use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_conv1d_adjoint_exec(
        &self,
        input: &GpuTensorHandle,
        weights: &GpuTensorHandle,
        bias: &GpuTensorHandle,
        upstream: &GpuTensorHandle,
        request: &DeepLearningConv1DRequest,
    ) -> Result<runmat_accelerate_api::DeepLearningConv1DGradients> {
        ensure!(
            input.shape() == request.input_shape,
            "convolution adjoint input differs from admitted shape"
        );
        ensure!(
            self.deep_learning_conv1d_capability_exec(
                DeepLearningConv1DOperands {
                    anchor: input,
                    input: Some(input),
                    weights: Some(weights),
                    bias: Some(bias),
                    upstream: Some(upstream),
                },
                request,
            )? == DeepLearningConv1DCapability::Ready,
            "convolution adjoint route was not admitted by this provider"
        );
        let geometry = Geometry::new(request, self.precision)?
            .ok_or_else(|| anyhow!("convolution geometry exceeds shader address range"))?;
        let source = self.get_entry(input)?;
        let kernel = self.get_entry(weights)?;
        let gradient = self.get_entry(upstream)?;
        let (dx_buffer, dw_buffer, db_buffer) =
            self.with_emission(|emission, pending| -> Result<_> {
                let dx = self.create_storage_buffer_checked(
                    emission,
                    geometry.input_elements,
                    "runmat-learned-conv1d-dinput",
                )?;
                let dw = self.create_storage_buffer_checked(
                    emission,
                    geometry.weight_elements,
                    "runmat-learned-conv1d-dweights",
                )?;
                let db = self.create_storage_buffer_checked(
                    emission,
                    geometry.bias_elements,
                    "runmat-learned-conv1d-dbias",
                )?;

                let weight_pipeline = self
                    .pipelines
                    .deep_learning_conv1d_adjoint_weight
                    .get(emission)?;
                let total_weights = u32::try_from(geometry.weight_elements)?;
                let mut offset = 0u32;
                while let Some(chunk) = next_chunk(total_weights, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &ConvParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..geometry.params
                        },
                        "runmat-learned-conv1d-dweights-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-learned-conv1d-dweights-bind"),
                            layout: &weight_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: source.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: gradient.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: dw.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: params.as_entire_binding(),
                                },
                            ],
                        });
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        &weight_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }

                let bias_pipeline = self
                    .pipelines
                    .deep_learning_conv1d_adjoint_bias
                    .get(emission)?;
                let total_bias = u32::try_from(geometry.bias_elements)?;
                offset = 0;
                while let Some(chunk) = next_chunk(total_bias, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &ConvParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..geometry.params
                        },
                        "runmat-learned-conv1d-dbias-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-learned-conv1d-dbias-bind"),
                            layout: &bias_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: gradient.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: db.as_ref().as_entire_binding(),
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
                        &bias_pipeline.pipeline,
                        &bind,
                        chunk.workgroups,
                    )?;
                    offset += chunk.count;
                }

                let input_pipeline = self
                    .pipelines
                    .deep_learning_conv1d_adjoint_input
                    .get(emission)?;
                let total_input = u32::try_from(geometry.input_elements)?;
                offset = 0;
                while let Some(chunk) = next_chunk(total_input, offset) {
                    let params = self.uniform_buffer(
                        emission,
                        &ConvParams {
                            chunk_offset: chunk.offset,
                            chunk_count: chunk.count,
                            ..geometry.params
                        },
                        "runmat-learned-conv1d-dinput-params",
                    )?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-learned-conv1d-dinput-bind"),
                            layout: &input_pipeline.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: kernel.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: gradient.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: dx.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
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
                Ok((dx, dw, db))
            })??;

        let input = self.register_existing_buffer_with_storage(
            dx_buffer,
            request.input_shape.clone(),
            geometry.input_elements,
            GpuTensorStorage::Real,
        )?;
        let weights = self.register_existing_buffer_with_storage(
            dw_buffer,
            request.weights_shape.to_vec(),
            geometry.weight_elements,
            GpuTensorStorage::Real,
        )?;
        let bias = self.register_existing_buffer_with_storage(
            db_buffer,
            request.bias_shape.to_vec(),
            geometry.bias_elements,
            GpuTensorStorage::Real,
        )?;
        Ok(runmat_accelerate_api::DeepLearningConv1DGradients {
            input,
            weights,
            bias,
        })
    }
}
