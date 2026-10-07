//! Typed dot operands over the existing allocation-map authority.
use super::*;

impl InProcessProvider {
    pub(super) fn native_dot(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let (lhs_tensor, rhs_tensor) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            // Authenticate both before interpreting unsupported classes/layouts.
            let left = guard
                .get(lhs)
                .ok_or_else(|| anyhow!("dot left input is not a live owned allocation"))?;
            let right = guard
                .get(rhs)
                .ok_or_else(|| anyhow!("dot right input is not a live owned allocation"))?;
            let snapshot =
                |handle: &GpuTensorHandle, data: &InProcessNumericData| -> Result<Tensor> {
                    if handle.physical().storage() != GpuTensorStorage::Real
                        || runmat_accelerate_api::handle_is_logical(handle)
                        || runmat_accelerate_api::handle_is_transposed(handle)
                        || matches!(data, InProcessNumericData::Integer(_))
                    {
                        return Err(runmat_accelerate_api::unsupported_provider_operation(
                            "dot reference hook requires contiguous real native floating operands",
                        ));
                    }
                    ensure!(
                        data.len() == logical_len_for_shape("dot", handle.shape())?,
                        "dot payload extent mismatch"
                    );
                    match data {
                        InProcessNumericData::F32(values) => {
                            Tensor::from_f32(values.clone(), handle.shape().to_vec())
                                .map_err(anyhow::Error::msg)
                        }
                        InProcessNumericData::F64(values) => {
                            Tensor::new(values.clone(), handle.shape().to_vec())
                                .map_err(anyhow::Error::msg)
                        }
                        InProcessNumericData::Integer(_) => unreachable!("integer rejected above"),
                    }
                };
            (snapshot(lhs, left)?, snapshot(rhs, right)?)
        };
        ensure!(dim != Some(0), "dot dimension must be positive");
        // The operation-owned host kernel defines mixed-class arithmetic and
        // final rounding. Do not widen the authoritative Single payloads.
        let output = dot_host_real_for_provider(&lhs_tensor, &rhs_tensor, dim)?;
        let shape = output.shape().to_vec();
        let data = match output.into_numeric_storage().map_err(anyhow::Error::msg)? {
            runmat_value::NumericStorage::F32(values) => InProcessNumericData::F32(values),
            runmat_value::NumericStorage::F64(values) => InProcessNumericData::F64(values),
            _ => return Err(anyhow!("dot floating kernel returned nonfloating storage")),
        };
        self.allocate_numeric_data(data, shape, GpuTensorStorage::Real)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn native_single_dot_preserves_low_rank_owner_and_output_contract() {
        let provider = InProcessProvider::new();
        for (shape, lhs_values, rhs_values, expected_shape, expected_values) in [
            (
                Vec::new(),
                vec![2.0_f32],
                vec![3.0_f32],
                vec![1, 0],
                Vec::new(),
            ),
            (
                vec![2],
                vec![2.0_f32, 3.0_f32],
                vec![4.0_f32, 5.0_f32],
                vec![1, 1],
                vec![23.0_f32],
            ),
        ] {
            let lhs = provider
                .allocate_numeric_data(
                    InProcessNumericData::F32(lhs_values),
                    shape.clone(),
                    GpuTensorStorage::Real,
                )
                .expect("single left input");
            let rhs = provider
                .allocate_numeric_data(
                    InProcessNumericData::F32(rhs_values),
                    shape,
                    GpuTensorStorage::Real,
                )
                .expect("single right input");
            let output =
                futures::executor::block_on(provider.dot(&lhs, &rhs, None)).expect("native dot");
            assert_eq!(output.physical().element_type(), NumericElementType::F32);
            let downloaded = futures::executor::block_on(provider.download_numeric(&output))
                .expect("download dot output");
            assert_eq!(downloaded.shape, expected_shape);
            assert_eq!(downloaded.data, HostNumericDataOwned::F32(expected_values));
            output.release().expect("release dot output");
            lhs.release().expect("release left input");
            rhs.release().expect("release right input");
        }
    }
}
