use super::*;

impl WgpuProvider {
    pub(crate) fn deep_learning_conv1d_exec(
        &self,
        input: &GpuTensorHandle,
        weights: &GpuTensorHandle,
        bias: &GpuTensorHandle,
        request: &DeepLearningConv1DRequest,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            input.shape() == request.input_shape,
            "convolution input differs from admitted shape"
        );
        ensure!(
            self.deep_learning_conv1d_capability_exec(
                DeepLearningConv1DOperands {
                    anchor: input,
                    input: Some(input),
                    weights: Some(weights),
                    bias: Some(bias),
                    upstream: None,
                },
                request,
            )? == DeepLearningConv1DCapability::Ready,
            "convolution forward route was not admitted by this provider"
        );
        let geometry = Geometry::new(request, self.precision)?
            .ok_or_else(|| anyhow!("convolution geometry exceeds shader address range"))?;
        let source = self.get_entry(input)?;
        let kernel = self.get_entry(weights)?;
        let bias_entry = self.get_entry(bias)?;
        let output_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let output = self.create_storage_buffer_checked(
                emission,
                geometry.output_elements,
                "runmat-learned-conv1d-output",
            )?;
            if geometry.output_elements == 0 {
                return Ok(output);
            }
            let pipeline = self.pipelines.deep_learning_conv1d_forward.get(emission)?;
            let total = u32::try_from(geometry.output_elements)?;
            let mut offset = 0u32;
            while let Some(chunk) = next_chunk(total, offset) {
                let params = self.uniform_buffer(
                    emission,
                    &ConvParams {
                        chunk_offset: chunk.offset,
                        chunk_count: chunk.count,
                        ..geometry.params
                    },
                    "runmat-learned-conv1d-forward-params",
                )?;
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-learned-conv1d-forward-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: source.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: kernel.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: bias_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: output.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &pipeline.pipeline,
                    &bind,
                    chunk.workgroups,
                )?;
                offset += chunk.count;
            }
            Ok(output)
        })??;
        self.register_existing_buffer_with_storage(
            output_buffer,
            request.output_shape.clone(),
            geometry.output_elements,
            GpuTensorStorage::Real,
        )
    }
}
