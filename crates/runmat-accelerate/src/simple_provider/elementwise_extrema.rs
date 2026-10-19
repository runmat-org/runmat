//! Native real elementwise extrema over authenticated borrowed views.
use super::*;

trait FloatingMaximum: Copy + PartialOrd {
    fn is_nan(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn positive_zero() -> Self;
}

impl FloatingMaximum for f32 {
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }
    fn positive_zero() -> Self {
        0.0
    }
}

impl FloatingMaximum for f64 {
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }
    fn positive_zero() -> Self {
        0.0
    }
}

fn maximum<T: FloatingMaximum>(left: T, right: T) -> T {
    if left.is_nan() {
        left
    } else if right.is_nan() {
        right
    } else if left > right {
        left
    } else if right > left {
        right
    } else if left == T::positive_zero() {
        if left.is_sign_negative() && right.is_sign_negative() {
            left
        } else {
            T::positive_zero()
        }
    } else {
        left
    }
}

fn evaluate<T: FloatingMaximum>(
    left: &[T],
    right: &[T],
    left_shape: &[usize],
    right_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    let count = logical_len_for_shape("elementwise maximum", output_shape)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let left_strides = runtime_compute_strides(left_shape);
    let right_strides = runtime_compute_strides(right_shape);
    let mut output = Vec::new();
    output.try_reserve_exact(count)?;
    for index in 0..count {
        let li = runtime_broadcast_index(index, output_shape, left_shape, &left_strides);
        let ri = runtime_broadcast_index(index, output_shape, right_shape, &right_strides);
        output.push(maximum(left[li], right[ri]));
    }
    Ok(output)
}

impl InProcessProvider {
    pub(super) fn native_elem_max(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let (result, shape) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("elementwise maximum allocation owner is poisoned"))?;
            let left_data = guard.get(left).ok_or_else(|| {
                anyhow!("elementwise maximum left allocation is not live and owned")
            })?;
            let right_data = guard.get(right).ok_or_else(|| {
                anyhow!("elementwise maximum right allocation is not live and owned")
            })?;
            if left.physical().storage() != GpuTensorStorage::Real
                || right.physical().storage() != GpuTensorStorage::Real
                || left.view().is_logical()
                || right.view().is_logical()
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "elementwise maximum requires real numeric input",
                ));
            }
            ensure!(
                left_data.len() == logical_len_for_shape("elementwise maximum", left.shape())?,
                "elementwise maximum left payload extent mismatch"
            );
            ensure!(
                right_data.len() == logical_len_for_shape("elementwise maximum", right.shape())?,
                "elementwise maximum right payload extent mismatch"
            );
            let shape =
                runtime_broadcast_shapes("elementwise maximum", left.shape(), right.shape())
                    .map_err(|error| anyhow!(error))?;
            let result = match (
                left_data,
                right_data,
                left.physical().element_type(),
                right.physical().element_type(),
            ) {
                (
                    InProcessNumericData::F32(a),
                    InProcessNumericData::F32(b),
                    NumericElementType::F32,
                    NumericElementType::F32,
                ) => {
                    InProcessNumericData::F32(evaluate(a, b, left.shape(), right.shape(), &shape)?)
                }
                (
                    InProcessNumericData::F64(a),
                    InProcessNumericData::F64(b),
                    NumericElementType::F64,
                    NumericElementType::F64,
                ) => {
                    InProcessNumericData::F64(evaluate(a, b, left.shape(), right.shape(), &shape)?)
                }
                _ => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "elementwise maximum requires matching native floating inputs",
                    ))
                }
            };
            (result, shape)
        };
        self.allocate_numeric_data(result, shape, GpuTensorStorage::Real)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn maximum_keeps_native_class_nan_signed_zero_broadcast_and_owner() {
        let provider = InProcessProvider::new();
        for single in [true, false] {
            let left = if single {
                HostNumericDataView::F32(&[f32::NAN, -0.0, 3.0, f32::INFINITY])
            } else {
                HostNumericDataView::F64(&[f64::NAN, -0.0, 3.0, f64::INFINITY])
            };
            let right = if single {
                HostNumericDataView::F32(&[1.0, 0.0])
            } else {
                HostNumericDataView::F64(&[1.0, 0.0])
            };
            let a = provider
                .upload_numeric(&HostNumericTensorView {
                    data: left,
                    shape: &[2, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let b = provider
                .upload_numeric(&HostNumericTensorView {
                    data: right,
                    shape: &[2, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            provider.reset_telemetry();
            let result = block_on(provider.elem_max(&a, &b)).unwrap();
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(result.shape(), &[2, 2]);
            assert_eq!(
                result.physical().element_type(),
                a.physical().element_type()
            );
            assert!(!result.shares_allocation_with(&a));
            assert!(!result.shares_allocation_with(&b));
            let values = block_on(provider.download_numeric(&result)).unwrap().data;
            match values {
                HostNumericDataOwned::F32(values) => {
                    assert!(single);
                    assert!(values[0].is_nan());
                    assert_eq!(values[1].to_bits(), 0.0f32.to_bits());
                    assert_eq!(values[2], 3.0);
                    assert_eq!(values[3], f32::INFINITY);
                }
                HostNumericDataOwned::F64(values) => {
                    assert!(!single);
                    assert!(values[0].is_nan());
                    assert_eq!(values[1].to_bits(), 0.0f64.to_bits());
                    assert_eq!(values[2], 3.0);
                    assert_eq!(values[3], f64::INFINITY);
                }
                other => panic!("maximum changed numeric class: {other:?}"),
            }
            assert!(a.allocation_lease().validate_live().is_ok());
        }
    }

    #[test]
    fn maximum_rejects_foreign_and_nonfloating_without_touching_borrowed() {
        let provider = InProcessProvider::new();
        let own = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let foreign = InProcessProvider::new()
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[2.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert!(block_on(provider.elem_max(&own, &foreign)).is_err());
        let integer = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::I32(&[1]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert!(block_on(provider.elem_max(&own, &integer))
            .unwrap_err()
            .is_unsupported());
        assert!(own.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn maximum_preserves_negative_zero_when_both_operands_are_negative_zero() {
        assert_eq!(maximum(-0.0_f32, -0.0_f32).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(maximum(-0.0_f64, -0.0_f64).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(maximum(-0.0_f32, 0.0_f32).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn maximum_uses_materialized_transpose_and_admitted_contiguous_reshape() {
        use runmat_accelerate_api::{GpuTensorLayout, GpuTensorView, TransposeInfo};

        let provider = InProcessProvider::new();
        let source = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                shape: &[2, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert!(source
            .with_view(GpuTensorView::contiguous(vec![3, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3
                })
            ))
            .is_err());
        let physically_transposed = provider.transpose(&source).unwrap();
        let zero = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let result = block_on(provider.elem_max(&physically_transposed, &zero)).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(
            block_on(provider.download_numeric(&result)).unwrap().data,
            HostNumericDataOwned::F32(vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0])
        );
        let reshaped = source
            .with_view(GpuTensorView::contiguous(vec![3, 2]))
            .unwrap();
        let result = block_on(provider.elem_max(&reshaped, &zero)).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(
            block_on(provider.download_numeric(&result)).unwrap().data,
            HostNumericDataOwned::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        );
    }

    #[test]
    fn maximum_keeps_empty_shape_without_allocating_by_other_extents() {
        let provider = InProcessProvider::new();
        let empty = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[]),
                shape: &[0, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let scalar = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let result = block_on(provider.elem_max(&empty, &scalar)).unwrap();
        assert_eq!(result.shape(), &[0, 3]);
        assert_eq!(
            block_on(provider.download_numeric(&result)).unwrap().data,
            HostNumericDataOwned::F32(Vec::new())
        );
    }
}
