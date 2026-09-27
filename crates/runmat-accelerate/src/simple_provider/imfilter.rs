//! Native floating snapshots and publication around the shared filter kernel.
use super::*;

impl InProcessProvider {
    pub(super) fn native_imfilter(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        let (image_tensor, kernel_tensor) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("imfilter: allocation owner is poisoned"))?;
            let snapshot = |handle: &GpuTensorHandle| -> Result<Tensor> {
                let data = guard
                    .get(handle)
                    .ok_or_else(|| anyhow!("imfilter: input is not a live owned allocation"))?;
                if handle.physical().storage() != GpuTensorStorage::Real
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_is_transposed(handle)
                    || matches!(data, InProcessNumericData::Integer(_))
                {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "imfilter requires contiguous real floating operands",
                    ));
                }
                ensure!(
                    data.len() == logical_len_for_shape("imfilter", handle.shape())?,
                    "imfilter payload extent mismatch"
                );
                match data {
                    InProcessNumericData::F32(values) => {
                        Tensor::from_f32(values.clone(), handle.shape().to_vec())
                    }
                    InProcessNumericData::F64(values) => {
                        Tensor::new(values.clone(), handle.shape().to_vec())
                    }
                    InProcessNumericData::Integer(_) => unreachable!("integer rejected above"),
                }
                .map_err(|error| anyhow!(error))
            };
            (snapshot(image)?, snapshot(kernel)?)
        };
        let result = runmat_runtime::builtins::image::filters::imfilter::apply_imfilter_native(
            &image_tensor,
            &kernel_tensor,
            options,
            runmat_accelerate_api::handle_is_explicit(image),
            "imfilter",
        )?;
        let shape = result.shape.clone();
        let data = match result
            .into_numeric_storage()
            .map_err(|error| anyhow!(error))?
        {
            runmat_value::NumericStorage::F32(values) => InProcessNumericData::F32(values),
            runmat_value::NumericStorage::F64(values) => InProcessNumericData::F64(values),
            _ => {
                return Err(anyhow!(
                    "imfilter kernel violated native floating output contract"
                ))
            }
        };
        self.allocate_numeric_data(data, shape, GpuTensorStorage::Real)
    }
}
