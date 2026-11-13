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
            // Authenticate every operand before reporting a numerical capability gap.
            let image_data = guard
                .get(image)
                .ok_or_else(|| anyhow!("imfilter: image is not a live owned allocation"))?;
            let kernel_data = guard
                .get(kernel)
                .ok_or_else(|| anyhow!("imfilter: kernel is not a live owned allocation"))?;
            let snapshot =
                |handle: &GpuTensorHandle, data: &InProcessNumericData| -> Result<Tensor> {
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
            (snapshot(image, image_data)?, snapshot(kernel, kernel_data)?)
        };
        let result = runmat_runtime::builtins::image::filters::kernel::apply_imfilter_native(
            &image_tensor,
            &kernel_tensor,
            options,
            runmat_accelerate_api::handle_is_explicit(image),
            "imfilter",
        )?;
        let shape = result.shape().to_vec();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn native_single_imfilter_retains_class_shape_and_allocation_lifetimes() {
        let provider = InProcessProvider::new();
        let image = provider
            .allocate_numeric_data(
                InProcessNumericData::F32(vec![2.0]),
                vec![1, 1],
                GpuTensorStorage::Real,
            )
            .expect("single image");
        let kernel = provider
            .allocate_numeric_data(
                InProcessNumericData::F32(vec![3.0]),
                vec![1, 1],
                GpuTensorStorage::Real,
            )
            .expect("single kernel");
        let output = futures::executor::block_on(provider.imfilter(
            &image,
            &kernel,
            &ImfilterOptions::default(),
        ))
        .expect("native imfilter");
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        let downloaded = futures::executor::block_on(provider.download_numeric(&output))
            .expect("download imfilter output");
        assert_eq!(downloaded.shape, vec![1, 1]);
        assert_eq!(downloaded.data, HostNumericDataOwned::F32(vec![6.0]));
        output.release().expect("release imfilter output");
        image.release().expect("release image");
        kernel.release().expect("release kernel");
    }

    #[test]
    fn unsupported_image_class_does_not_mask_foreign_kernel_owner() {
        let provider = InProcessProvider::new();
        let foreign = InProcessProvider::new();
        let kernel = foreign
            .allocate_numeric_data(
                InProcessNumericData::F64(vec![1.0]),
                vec![1, 1],
                GpuTensorStorage::Real,
            )
            .unwrap();
        let images = [
            provider
                .allocate_integer_tensor(HostIntegerDataOwned::I32(vec![1]), vec![1, 1])
                .unwrap(),
            provider
                .allocate_logical_tensor(vec![1.0], vec![1, 1])
                .unwrap(),
            provider
                .allocate_numeric_data(
                    InProcessNumericData::F64(vec![1.0, 2.0]),
                    vec![1, 1],
                    GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap(),
        ];
        for image in images {
            let options = ImfilterOptions::default();
            let error = futures::executor::block_on(provider.imfilter(&image, &kernel, &options))
                .unwrap_err();
            assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
                &error
            ));
            assert!(error
                .to_string()
                .contains("kernel is not a live owned allocation"));
            assert!(image.allocation_lease().validate_live().is_ok());
            assert!(kernel.allocation_lease().validate_live().is_ok());
        }
    }
}
