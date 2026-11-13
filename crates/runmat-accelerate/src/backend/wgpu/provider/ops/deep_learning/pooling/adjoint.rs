use super::*;

#[cfg(test)]
#[path = "adjoint/tests.rs"]
mod tests;

impl WgpuProvider {
    pub(crate) fn deep_learning_pool_1d_adjoint_capability_exec(
        &self,
        anchor: &GpuTensorHandle,
        input_shape: &[usize],
        upstream: Option<&GpuTensorHandle>,
        request: DeepLearningPool1DRequest,
    ) -> Result<DeepLearningPool1DCapability> {
        let owner = self.get_entry_raw(anchor)?;
        ensure!(
            owner.element_type == anchor.physical().element_type()
                && owner.storage == anchor.physical().storage()
                && owner.len == checked_shape_count(anchor.shape())?
                && owner.logical == runmat_accelerate_api::handle_is_logical(anchor)
                && owner.integer_type() == runmat_accelerate_api::handle_integer_type(anchor),
            "pool adjoint anchor differs from provider-owned descriptor"
        );
        let axis = request.axis_zero_based;
        let Some(&pool_size) = input_shape.get(axis) else {
            return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
        };
        if pool_size == 0 {
            return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
        }
        let total = checked_shape_count(input_shape)?;
        let inner = checked_shape_count(&input_shape[..axis])?;
        let mut reduced_shape = input_shape.to_vec();
        reduced_shape[axis] = 1;
        let reduced = total / pool_size;
        if let Some(upstream) = upstream {
            let entry = self.get_entry_raw(upstream)?;
            ensure!(
                upstream.shape() == reduced_shape
                    && entry.shape == reduced_shape
                    && entry.len == reduced
                    && entry.element_type == upstream.physical().element_type()
                    && entry.storage == upstream.physical().storage()
                    && entry.logical == runmat_accelerate_api::handle_is_logical(upstream)
                    && entry.integer_type() == runmat_accelerate_api::handle_integer_type(upstream),
                "pool adjoint upstream differs from admitted provider-owned geometry"
            );
            if entry.element_type != owner.element_type || entry.storage != owner.storage {
                return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
            }
        }
        if owner.storage != GpuTensorStorage::Real
            || owner.logical
            || owner.integer_type().is_some()
            || !self.supports_native_real_element_type(owner.element_type)
        {
            return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
        }
        if total > u32::MAX as usize
            || pool_size > u32::MAX as usize
            || inner > u32::MAX as usize
            || input_shape.iter().any(|&extent| extent > u32::MAX as usize)
        {
            return Ok(DeepLearningPool1DCapability::ResourceLimit);
        }
        let bytes = u64::try_from(total)?
            .checked_mul(u64::try_from(self.element_size)?)
            .ok_or_else(|| anyhow!("pool adjoint output byte count overflows"))?;
        let limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        if bytes > limit {
            return Ok(DeepLearningPool1DCapability::ResourceLimit);
        }
        Ok(DeepLearningPool1DCapability::Ready)
    }

    pub(crate) fn deep_learning_pool_1d_adjoint_exec(
        &self,
        input: &GpuTensorHandle,
        upstream: &GpuTensorHandle,
        request: DeepLearningPool1DRequest,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            self.deep_learning_pool_1d_adjoint_capability_exec(
                input,
                input.shape(),
                Some(upstream),
                request,
            )? == DeepLearningPool1DCapability::Ready,
            "pool adjoint route was not admitted by this provider"
        );
        let axis = request.axis_zero_based;
        let total = checked_shape_count(input.shape())?;
        let pool_size = u32::try_from(input.shape()[axis])?;
        let inner_stride = u32::try_from(checked_shape_count(&input.shape()[..axis])?)?;
        if total == 0 {
            let output = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked(
                    emission,
                    0,
                    "runmat-deep-learning-empty-pool-adjoint",
                )
            })??;
            return self.register_existing_buffer_with_storage(
                output,
                input.shape().to_vec(),
                0,
                GpuTensorStorage::Real,
            );
        }
        let materialized = self.materialize_nd_reduction_input(upstream)?;
        let source = self.get_entry(materialized.as_ref().unwrap_or(upstream))?;
        let output = self.with_emission(|emission, pending| -> Result<_> {
            let output = self.create_storage_buffer_checked(
                emission,
                total,
                "runmat-deep-learning-pool-1d-adjoint",
            )?;
            let pipeline = self.pipelines.deep_learning_pool_1d_adjoint.get(emission)?;
            let mut offset = 0u32;
            while let Some(chunk) = next_pool_chunk(
                u32::try_from(total)?,
                offset,
                crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS,
            )? {
                let params = self.uniform_buffer(
                    emission,
                    &Pool1DParams {
                        pool_size,
                        chunk_output_count: chunk.output_count,
                        inner_stride,
                        output_offset: chunk.offset,
                    },
                    "runmat-deep-learning-pool-1d-adjoint-params",
                )?;
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-deep-learning-pool-1d-adjoint-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: source.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: output.as_ref().as_entire_binding(),
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
                    &pipeline.pipeline,
                    &bind,
                    chunk.workgroups,
                )?;
                offset += chunk.output_count;
            }
            Ok(output)
        })??;
        self.register_existing_buffer_with_storage(
            output,
            input.shape().to_vec(),
            total,
            GpuTensorStorage::Real,
        )
    }
}
