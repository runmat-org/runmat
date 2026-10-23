use super::*;
pub(super) fn cumtrapz_tensor(
    tensor: &Tensor,
    spacing: &SpacingSpec,
    dim: usize,
    single_output: bool,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(cumtrapz_error_with_detail(
            &CUMTRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let shape = pad_shape_for_dim(&canonical_shape_tensor(tensor), dim);
    let dim_index = (dim - 1).min(shape.len());
    let len_dim = shape.get(dim_index).copied().unwrap_or(1);
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(shape.get(dim..).unwrap_or(&[]));
    let block = stride_before * len_dim;
    let mut output = vec![0.0f64; tensor.len()];

    if len_dim > 0 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let first_idx = base + before;
                output[first_idx] = 0.0;
                let mut acc = 0.0f64;
                for k in 0..len_dim.saturating_sub(1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    acc += 0.5
                        * width
                        * (crate::builtins::common::tensor::tensor_value_f64(tensor, idx0)
                            + crate::builtins::common::tensor::tensor_value_f64(tensor, idx1));
                    output[idx1] = acc;
                }
            }
        }
    }

    if single_output {
        Tensor::from_f32(
            output.into_iter().map(|value| value as f32).collect(),
            shape,
        )
        .map_err(|err| cumtrapz_internal_error(&err))
    } else {
        Tensor::new(output, shape).map_err(|err| cumtrapz_internal_error(&err))
    }
}
pub(super) fn cumtrapz_complex_tensor(
    tensor: &ComplexTensor,
    spacing: &SpacingSpec,
    dim: usize,
    single_output: bool,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(cumtrapz_error_with_detail(
            &CUMTRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let at = |index| {
        let (real, imaginary) = tensor
            .numeric_value_at(index)
            .expect("validated complex sample index");
        (real.materialize_f64(), imaginary.materialize_f64())
    };
    let shape = pad_shape_for_dim(&canonical_shape_complex(tensor), dim);
    let dim_index = (dim - 1).min(shape.len());
    let len_dim = shape.get(dim_index).copied().unwrap_or(1);
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(shape.get(dim..).unwrap_or(&[]));
    let block = stride_before * len_dim;
    let mut output = vec![(0.0f64, 0.0f64); tensor.len()];

    if len_dim > 0 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let first_idx = base + before;
                output[first_idx] = (0.0, 0.0);
                let mut acc = (0.0f64, 0.0f64);
                for k in 0..len_dim.saturating_sub(1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    let (re0, im0) = at(idx0);
                    let (re1, im1) = at(idx1);
                    acc.0 += 0.5 * width * (re0 + re1);
                    acc.1 += 0.5 * width * (im0 + im1);
                    output[idx1] = acc;
                }
            }
        }
    }

    if single_output {
        ComplexTensor::from_f32(
            output
                .into_iter()
                .map(|(real, imaginary)| (real as f32, imaginary as f32))
                .collect(),
            shape,
        )
        .map_err(|err| cumtrapz_internal_error(&err))
    } else {
        ComplexTensor::new(output, shape).map_err(|err| cumtrapz_internal_error(&err))
    }
}
