//! Deterministic, typed reference implementation of resident selection.
use super::*;
use runmat_accelerate_api::ResidentSelectPlan;

impl InProcessProvider {
    pub(super) fn select_impl(
        &self,
        mask: &GpuTensorHandle,
        then_value: &GpuTensorHandle,
        else_value: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let plan = ResidentSelectPlan::new(mask, then_value, else_value)?;
        let output = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("select allocation owner is poisoned"))?;
            let mask_data = guard
                .get(mask)
                .ok_or_else(|| anyhow!("select mask is not a live owned allocation"))?;
            let then_data = guard
                .get(then_value)
                .ok_or_else(|| anyhow!("select then branch is not a live owned allocation"))?;
            let else_data = guard
                .get(else_value)
                .ok_or_else(|| anyhow!("select else branch is not a live owned allocation"))?;
            ensure!(
                guard.producer_is_logical(mask) == Some(true),
                "select mask is not a logical producer"
            );
            for (handle, data) in [
                (mask, mask_data),
                (then_value, then_data),
                (else_value, else_data),
            ] {
                let count = handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, &extent| n.checked_mul(extent))
                    .ok_or_else(|| anyhow!("select input size overflow"))?;
                ensure!(data.len() == count, "select input payload size mismatch");
            }
            let mask_stride = runtime_compute_strides(mask.shape());
            let then_stride = runtime_compute_strides(then_value.shape());
            let else_stride = runtime_compute_strides(else_value.shape());
            match (then_data, else_data) {
                (InProcessNumericData::F32(then_data), InProcessNumericData::F32(else_data)) => {
                    let mut result = Vec::new();
                    result.try_reserve_exact(plan.len())?;
                    for index in 0..plan.len() {
                        let choice = runtime_broadcast_index(
                            index,
                            plan.shape(),
                            mask.shape(),
                            &mask_stride,
                        );
                        let chosen = if nonzero_mask(mask_data, choice)? {
                            let source = runtime_broadcast_index(
                                index,
                                plan.shape(),
                                then_value.shape(),
                                &then_stride,
                            );
                            then_data[source]
                        } else {
                            let source = runtime_broadcast_index(
                                index,
                                plan.shape(),
                                else_value.shape(),
                                &else_stride,
                            );
                            else_data[source]
                        };
                        result.push(chosen);
                    }
                    InProcessNumericData::F32(result)
                }
                (InProcessNumericData::F64(then_data), InProcessNumericData::F64(else_data)) => {
                    let mut result = Vec::new();
                    result.try_reserve_exact(plan.len())?;
                    for index in 0..plan.len() {
                        let choice = runtime_broadcast_index(
                            index,
                            plan.shape(),
                            mask.shape(),
                            &mask_stride,
                        );
                        let chosen = if nonzero_mask(mask_data, choice)? {
                            let source = runtime_broadcast_index(
                                index,
                                plan.shape(),
                                then_value.shape(),
                                &then_stride,
                            );
                            then_data[source]
                        } else {
                            let source = runtime_broadcast_index(
                                index,
                                plan.shape(),
                                else_value.shape(),
                                &else_stride,
                            );
                            else_data[source]
                        };
                        result.push(chosen);
                    }
                    InProcessNumericData::F64(result)
                }
                _ => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "select requires matching native real floating branches",
                    ))
                }
            }
        };
        let result =
            self.allocate_numeric_data(output, plan.into_shape(), GpuTensorStorage::Real)?;
        ensure!(
            ![mask, then_value, else_value]
                .iter()
                .any(|input| result.shares_allocation_with(input)),
            "select returned an aliased allocation"
        );
        Ok(result)
    }
}

fn nonzero_mask(data: &InProcessNumericData, index: usize) -> Result<bool> {
    match data {
        InProcessNumericData::F32(values) => Ok(values[index] != 0.0),
        InProcessNumericData::F64(values) => Ok(values[index] != 0.0),
        InProcessNumericData::Integer(_) => {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "select logical mask requires floating physical storage",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostLogicalTensorView;

    #[test]
    fn selects_exact_real_bits_with_three_way_broadcast_and_distinct_owner() {
        let provider = InProcessProvider::new();
        let mask = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[1, 0],
                shape: &[2, 1],
            })
            .unwrap();
        let then_value = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[f32::NAN, f32::INFINITY, -0.0]),
                shape: &[1, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let else_value = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[-f32::INFINITY, 7.0, 0.0]),
                shape: &[1, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = block_on(provider.select(&mask, &then_value, &else_value)).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert!(!output.view().is_logical());
        assert!(!output.shares_allocation_with(&mask));
        assert!(!output.shares_allocation_with(&then_value));
        assert!(!output.shares_allocation_with(&else_value));
        let HostNumericDataOwned::F32(actual) =
            block_on(provider.download_numeric(&output)).unwrap().data
        else {
            panic!("select changed the output class");
        };
        assert!(actual[0].is_nan());
        assert_eq!(actual[1], f32::NEG_INFINITY);
        assert_eq!(actual[2], f32::INFINITY);
        assert_eq!(actual[3], 7.0);
        assert_eq!(actual[4].to_bits(), (-0.0f32).to_bits());
        assert_eq!(actual[5].to_bits(), 0.0f32.to_bits());
        drop((mask, then_value, else_value));
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().shape,
            vec![2, 3]
        );
    }

    #[test]
    fn rejects_nonlogical_mixed_class_foreign_and_incompatible_shape() {
        let provider = InProcessProvider::new();
        let mask = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[1, 0],
                shape: &[2, 1],
            })
            .unwrap();
        let f64_values = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .unwrap();
        let f32_values = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0, 2.0]),
                shape: &[2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert!(block_on(provider.select(&f64_values, &f64_values, &f64_values)).is_err());
        assert!(block_on(provider.select(&mask, &f64_values, &f32_values)).is_err());
        let mismatch = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[3, 1],
            })
            .unwrap();
        assert!(block_on(provider.select(&mask, &f64_values, &mismatch)).is_err());
        let foreign = InProcessProvider::new()
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        assert!(block_on(provider.select(&mask, &f64_values, &foreign)).is_err());
    }

    #[test]
    fn empty_output_keeps_shape_and_native_class() {
        let provider = InProcessProvider::new();
        let mask = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[],
                shape: &[0, 2],
            })
            .unwrap();
        let branch = provider
            .upload(&HostTensorView {
                data: &[3.0, 4.0],
                shape: &[1, 2],
            })
            .unwrap();
        let output = block_on(provider.select(&mask, &branch, &branch)).unwrap();
        assert_eq!(output.shape(), &[0, 2]);
        assert_eq!(
            block_on(provider.download(&output)).unwrap().data,
            Vec::<f64>::new()
        );
    }
}
