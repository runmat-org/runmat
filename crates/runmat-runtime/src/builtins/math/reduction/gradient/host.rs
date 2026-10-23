use super::*;

pub fn gradient_real_tensor_host(
    tensor: Tensor,
    dim: usize,
    spacing: f64,
) -> GradientKernelResult<Tensor> {
    let spacing = GradientSpacing::Scalar(spacing);
    gradient_real_tensor_host_with_spacing(tensor, dim, &spacing)
}

pub fn gradient_real_tensor_host_with_coordinates(
    tensor: Tensor,
    dim: usize,
    coordinates: Vec<f64>,
) -> GradientKernelResult<Tensor> {
    let spacing = GradientSpacing::Coordinates(coordinates);
    gradient_real_tensor_host_with_spacing(tensor, dim, &spacing)
}

pub(super) fn gradient_real_tensor_host_with_spacing(
    tensor: Tensor,
    dim: usize,
    spacing: &GradientSpacing,
) -> GradientKernelResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let output_dtype = if tensor.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let at = |index| tensor::tensor_value_f64(&tensor, index);
    let dim_index = dim.saturating_sub(1);
    let mut shape = matlab_gradient_shape(&shape, tensor.len());

    if tensor.is_empty() {
        // Return early before the `push(1)` padding loop: that loop would give a
        // shape like [1] or [1,1] whose product is 1 ≠ 0, violating Tensor's
        // invariant. Use the normalised shape directly, falling back to [0,0] if
        // matlab_gradient_shape returned an empty vec (untyped empty tensor).
        let empty_shape = if shape.is_empty() { vec![0, 0] } else { shape };
        return Tensor::new_with_dtype(Vec::new(), empty_shape, output_dtype)
            .map_err(|e| GradientKernelError::new(format!("gradient: {e}")));
    }

    while shape.len() <= dim_index {
        shape.push(1);
    }

    let mut ext_shape = shape.clone();
    while ext_shape.len() <= dim_index {
        ext_shape.push(1);
    }
    let len_dim = ext_shape[dim_index];
    let stride_before = if dim_index == 0 {
        1usize
    } else {
        product(&ext_shape[..dim_index]).max(1)
    };
    let stride_after = if dim_index + 1 >= ext_shape.len() {
        1usize
    } else {
        product(&ext_shape[dim_index + 1..]).max(1)
    };

    let mut out = vec![0.0; tensor.len()];
    if len_dim > 1 {
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| GradientKernelError::new("gradient: block size overflow"))?;
        for after in 0..stride_after {
            let base = after
                .checked_mul(block)
                .ok_or_else(|| GradientKernelError::new("gradient: indexing overflow"))?;
            for before in 0..stride_before {
                for k in 0..len_dim {
                    let idx = base + before + k * stride_before;
                    out[idx] = if k == 0 {
                        (at(idx + stride_before) - at(idx))
                            / spacing_denominator(spacing, k, len_dim)
                    } else if k + 1 == len_dim {
                        (at(idx) - at(idx - stride_before))
                            / spacing_denominator(spacing, k, len_dim)
                    } else {
                        (at(idx + stride_before) - at(idx - stride_before))
                            / spacing_denominator(spacing, k, len_dim)
                    };
                }
            }
        }
    }

    Tensor::new_with_dtype(out, shape, output_dtype)
        .map_err(|e| GradientKernelError::new(format!("gradient: {e}")))
}

pub fn gradient_complex_tensor_host(
    tensor: ComplexTensor,
    dim: usize,
    spacing: f64,
) -> GradientKernelResult<ComplexTensor> {
    let spacing = GradientSpacing::Scalar(spacing);
    gradient_complex_tensor_host_with_spacing(tensor, dim, &spacing)
}

pub fn gradient_complex_tensor_host_with_coordinates(
    tensor: ComplexTensor,
    dim: usize,
    coordinates: Vec<f64>,
) -> GradientKernelResult<ComplexTensor> {
    let spacing = GradientSpacing::Coordinates(coordinates);
    gradient_complex_tensor_host_with_spacing(tensor, dim, &spacing)
}

pub(super) fn gradient_complex_tensor_host_with_spacing(
    tensor: ComplexTensor,
    dim: usize,
    spacing: &GradientSpacing,
) -> GradientKernelResult<ComplexTensor> {
    let output_dtype = if tensor.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let shape = tensor.shape().to_vec();
    let at = |index| {
        let (real, imaginary) = tensor
            .numeric_value_at(index)
            .expect("validated complex gradient index");
        (real.materialize_f64(), imaginary.materialize_f64())
    };
    let dim_index = dim.saturating_sub(1);
    let mut shape = matlab_gradient_shape(&shape, tensor.len());

    if tensor.is_empty() {
        // Same fix as gradient_real_tensor_host: avoid padding the shape with 1s
        // before the early return, which would produce product ≠ 0 for empty data.
        let empty_shape = if shape.is_empty() { vec![0, 0] } else { shape };
        return ComplexTensor::from_f64_values_with_dtype(Vec::new(), empty_shape, output_dtype)
            .map_err(|e| GradientKernelError::new(format!("gradient: {e}")));
    }

    while shape.len() <= dim_index {
        shape.push(1);
    }

    let mut ext_shape = shape.clone();
    while ext_shape.len() <= dim_index {
        ext_shape.push(1);
    }
    let len_dim = ext_shape[dim_index];
    let stride_before = if dim_index == 0 {
        1usize
    } else {
        product(&ext_shape[..dim_index]).max(1)
    };
    let stride_after = if dim_index + 1 >= ext_shape.len() {
        1usize
    } else {
        product(&ext_shape[dim_index + 1..]).max(1)
    };

    let mut out = vec![(0.0, 0.0); tensor.len()];
    if len_dim > 1 {
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| GradientKernelError::new("gradient: block size overflow"))?;
        for after in 0..stride_after {
            let base = after
                .checked_mul(block)
                .ok_or_else(|| GradientKernelError::new("gradient: indexing overflow"))?;
            for before in 0..stride_before {
                for k in 0..len_dim {
                    let idx = base + before + k * stride_before;
                    out[idx] = if k == 0 {
                        scale_complex(
                            sub_complex(at(idx + stride_before), at(idx)),
                            1.0 / spacing_denominator(spacing, k, len_dim),
                        )
                    } else if k + 1 == len_dim {
                        scale_complex(
                            sub_complex(at(idx), at(idx - stride_before)),
                            1.0 / spacing_denominator(spacing, k, len_dim),
                        )
                    } else {
                        scale_complex(
                            sub_complex(at(idx + stride_before), at(idx - stride_before)),
                            1.0 / spacing_denominator(spacing, k, len_dim),
                        )
                    };
                }
            }
        }
    }

    ComplexTensor::from_f64_values_with_dtype(out, shape, output_dtype)
        .map_err(|e| GradientKernelError::new(format!("gradient: {e}")))
}

pub(super) fn spacing_denominator(spacing: &GradientSpacing, k: usize, len_dim: usize) -> f64 {
    match spacing {
        GradientSpacing::Scalar(spacing) => {
            if k == 0 || k + 1 == len_dim {
                *spacing
            } else {
                2.0 * spacing
            }
        }
        GradientSpacing::Coordinates(coords) => {
            if k == 0 {
                coords[1] - coords[0]
            } else if k + 1 == len_dim {
                coords[len_dim - 1] - coords[len_dim - 2]
            } else {
                coords[k + 1] - coords[k - 1]
            }
        }
    }
}

pub(super) fn sub_complex(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

pub(super) fn scale_complex(value: (f64, f64), scale: f64) -> (f64, f64) {
    (value.0 * scale, value.1 * scale)
}
