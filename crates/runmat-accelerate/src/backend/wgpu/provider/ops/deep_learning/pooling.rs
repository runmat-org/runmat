use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Pool1DParams {
    pool_size: u32,
    output_count: u32,
    inner_stride: u32,
    output_offset: u32,
}

impl WgpuProvider {
    pub(crate) fn deep_learning_global_average_pool_1d_exec(
        &self,
        input: &GpuTensorHandle,
        request: DeepLearningPool1DRequest,
    ) -> Result<GpuTensorHandle> {
        let logical = self.get_entry(input)?;
        ensure!(
            logical.len <= u32::MAX as usize,
            "pool input exceeds GPU indexing limits"
        );
        ensure!(
            logical.storage == GpuTensorStorage::Real
                && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(input)
                && runmat_accelerate_api::handle_integer_type(input).is_none()
                && logical.precision == self.precision,
            "deep-learning global-average-pool-1d requires real floating input"
        );
        let shape = input.shape();
        let axis = request.axis_zero_based;
        ensure!(axis < shape.len(), "pool axis exceeds input rank");
        let pool_size = shape[axis];
        ensure!(pool_size > 0, "cannot pool an empty axis");
        let inner_stride = shape[..axis].iter().try_fold(1usize, |n, &extent| {
            n.checked_mul(extent)
                .ok_or_else(|| anyhow!("pool stride overflows"))
        })?;
        let output_count = logical.len / pool_size;
        let pool_size =
            u32::try_from(pool_size).map_err(|_| anyhow!("pool axis exceeds GPU limits"))?;
        let inner_stride =
            u32::try_from(inner_stride).map_err(|_| anyhow!("pool stride exceeds GPU limits"))?;
        let output_count =
            u32::try_from(output_count).map_err(|_| anyhow!("pool output exceeds GPU limits"))?;
        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        if output_count == 0 {
            let output = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked(emission, 0, "runmat-deep-learning-empty-pool")
            })??;
            return self.register_existing_buffer_with_storage(
                output,
                output_shape,
                0,
                GpuTensorStorage::Real,
            );
        }

        // The kernel uses physical column-major offsets. A lazy transpose
        // must be materialized by its owner before reading the backing buffer.
        let materialized = self.materialize_nd_reduction_input(input)?;
        let source = materialized.as_ref().unwrap_or(input);
        let entry = self.get_entry(source)?;
        let output = self.with_emission(|emission, pending| -> Result<_> {
            let output = self.create_storage_buffer_checked(
                emission,
                output_count as usize,
                "runmat-deep-learning-pool-1d-out",
            )?;
            let pipeline = self.pipelines.deep_learning_stable_pool_1d.get(emission)?;
            let chunk = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS;
            for offset in (0..output_count).step_by(chunk as usize) {
                let params = self.uniform_buffer(
                    emission,
                    &Pool1DParams {
                        pool_size,
                        output_count,
                        inner_stride,
                        output_offset: offset,
                    },
                    "runmat-deep-learning-pool-1d-params",
                )?;
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-deep-learning-pool-1d-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
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
                    (output_count - offset).min(chunk),
                )?;
            }
            Ok(output)
        })??;
        self.register_existing_buffer(output, output_shape, output_count as usize)
    }
}
