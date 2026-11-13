use super::*;
pub(crate) fn trapz_tensor(
    tensor: &Tensor,
    spacing: &SpacingSpec,
    dim: usize,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(trapz_error_with_detail(
            &TRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let shape = pad_shape_for_dim(&canonical_shape_tensor(tensor), dim);
    let dim_index = (dim - 1).min(shape.len());
    let len_dim = shape.get(dim_index).copied().unwrap_or(1);
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(shape.get(dim..).unwrap_or(&[]));
    let block = stride_before * len_dim;
    let mut output = vec![0.0f64; stride_before * stride_after];

    if len_dim > 1 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let mut acc = 0.0f64;
                for k in 0..(len_dim - 1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    acc += 0.5
                        * width
                        * (crate::builtins::common::tensor::tensor_value_f64(tensor, idx0)
                            + crate::builtins::common::tensor::tensor_value_f64(tensor, idx1));
                }
                output[after * stride_before + before] = acc;
            }
        }
    }

    let mut out_shape = shape;
    if dim_index < out_shape.len() {
        out_shape[dim_index] = 1;
    }
    if tensor.numeric_dtype() == NumericDType::F32 {
        Tensor::from_f32(
            output.into_iter().map(|value| value as f32).collect(),
            out_shape,
        )
        .map_err(|err| trapz_internal_error(&err))
    } else {
        Tensor::new(output, out_shape).map_err(|err| trapz_internal_error(&err))
    }
}
pub(super) fn trapz_complex_tensor(
    tensor: &ComplexTensor,
    spacing: &SpacingSpec,
    dim: usize,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(trapz_error_with_detail(
            &TRAPZ_ERROR_INVALID_ARGUMENT,
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
    let mut output = vec![(0.0f64, 0.0f64); stride_before * stride_after];

    if len_dim > 1 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let mut acc = (0.0f64, 0.0f64);
                for k in 0..(len_dim - 1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    let (re0, im0) = at(idx0);
                    let (re1, im1) = at(idx1);
                    acc.0 += 0.5 * width * (re0 + re1);
                    acc.1 += 0.5 * width * (im0 + im1);
                }
                output[after * stride_before + before] = acc;
            }
        }
    }

    let mut out_shape = shape;
    if dim_index < out_shape.len() {
        out_shape[dim_index] = 1;
    }
    if tensor.numeric_dtype() == NumericDType::F32 {
        ComplexTensor::from_f32(
            output
                .into_iter()
                .map(|(real, imaginary)| (real as f32, imaginary as f32))
                .collect(),
            out_shape,
        )
        .map_err(|err| trapz_internal_error(&err))
    } else {
        ComplexTensor::new(output, out_shape).map_err(|err| trapz_internal_error(&err))
    }
}
