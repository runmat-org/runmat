use super::*;

pub(super) fn diff_char_array(
    chars: CharArray,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<Value> {
    if order == 0 {
        return Ok(Value::CharArray(chars));
    }
    chars.validate_layout().map_err(diff_invalid_input)?;
    let shape = chars.shape().to_vec();
    let dimension = dim.unwrap_or_else(|| default_dimension(&shape));
    let first = diff_mapped_tensor_once(
        chars.data(),
        shape,
        dimension,
        |value| f64::from(value.raw()),
        NumericStorage::F64,
    )?;
    diff_tensor_host(first, order - 1, Some(dimension)).map(tensor::tensor_into_value)
}

pub(super) fn diff_logical_array(
    logical: runmat_value::LogicalArray,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<Value> {
    if order == 0 {
        return Ok(Value::LogicalArray(logical));
    }
    let shape = logical.shape.to_vec();
    let dimension = dim.unwrap_or_else(|| default_dimension(&shape));
    let first = diff_mapped_tensor_once(
        &logical.data,
        shape,
        dimension,
        |value| if value == 0 { 0.0 } else { 1.0 },
        NumericStorage::F64,
    )?;
    diff_tensor_host(first, order - 1, Some(dimension)).map(tensor::tensor_into_value)
}

pub fn diff_tensor_host(tensor: Tensor, order: usize, dim: Option<usize>) -> BuiltinResult<Tensor> {
    let mut current = tensor;
    let working_dim = dim.unwrap_or_else(|| default_dimension(current.shape()));
    for step in 0..order {
        current = diff_tensor_once(current, working_dim)?;
        if tensor::tensor_element_len(&current) == 0 {
            let mut shape = current.shape().to_vec();
            shape[working_dim - 1] = shape[working_dim - 1].saturating_sub(order - step - 1);
            current
                .reshape_in_place(shape)
                .map_err(diff_internal_error)?;
            break;
        }
    }
    Ok(current)
}

pub(super) fn diff_complex_tensor(
    tensor: ComplexTensor,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    let mut current = tensor;
    let working_dim = dim.unwrap_or_else(|| default_dimension(current.shape()));
    for step in 0..order {
        current = diff_complex_tensor_once(current, working_dim)?;
        if current.is_empty() {
            let mut shape = current.shape().to_vec();
            shape[working_dim - 1] = shape[working_dim - 1].saturating_sub(order - step - 1);
            current
                .reshape_in_place(shape)
                .map_err(diff_internal_error)?;
            break;
        }
    }
    Ok(current)
}

pub(super) fn diff_tensor_once(tensor: Tensor, dim: usize) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    if let Some(values) = tensor.as_f64_slice() {
        diff_mapped_tensor_once(values, shape, dim, |value| value, NumericStorage::F64)
    } else if let Some(values) = tensor.as_f32_slice() {
        diff_mapped_tensor_once(values, shape, dim, |value| value, NumericStorage::F32)
    } else {
        diff_integer_tensor_once(
            shape,
            tensor.integer_storage().expect("integer numeric storage"),
            dim,
        )
    }
}

pub(super) fn diff_mapped_tensor_once<T, R, C, F>(
    data: &[T],
    mut shape: Vec<usize>,
    dim: usize,
    convert: C,
    wrap: F,
) -> BuiltinResult<Tensor>
where
    T: Copy,
    R: std::ops::Sub<Output = R>,
    C: Fn(T) -> R,
    F: FnOnce(Vec<R>) -> NumericStorage,
{
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return Tensor::from_numeric_storage(wrap(Vec::new()), output_shape)
            .map_err(|e| diff_internal_error(format!("diff: {e}")));
    }
    output_shape[dim_index] = len_dim - 1;
    let plan = DifferenceGeometry::new(&shape, 1, Some(dim))?;
    let output_len = product(&output_shape);
    let mut out = Vec::with_capacity(output_len);
    for row in 0..output_len {
        let (idx0, idx1) = plan
            .pair(0, row)
            .ok_or_else(|| diff_internal_error("difference stage index overflow"))?;
        out.push(convert(data[idx1]) - convert(data[idx0]));
    }

    Tensor::from_numeric_storage(wrap(out), output_shape)
        .map_err(|e| diff_internal_error(format!("diff: {e}")))
}

pub(super) fn diff_integer_tensor_once(
    mut shape: Vec<usize>,
    storage: &IntegerStorage,
    dim: usize,
) -> BuiltinResult<Tensor> {
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || storage.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return Tensor::new_integer(storage.zeros_like(0), output_shape)
            .map_err(|e| diff_internal_error(format!("diff: {e}")));
    }

    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let mut values = Vec::with_capacity(stride_before * (len_dim - 1) * stride_after);
    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for k in 0..(len_dim - 1) {
            for before in 0..stride_before {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                values.push(same_class_saturating_subtract(
                    storage
                        .value_at(idx1)
                        .expect("validated integer difference index"),
                    storage
                        .value_at(idx0)
                        .expect("validated integer difference index"),
                ));
            }
        }
    }
    Tensor::new_integer(
        storage
            .from_same_class_values(values)
            .map_err(diff_internal_error)?,
        output_shape,
    )
    .map_err(|e| diff_internal_error(format!("diff: {e}")))
}

pub(super) fn diff_complex_tensor_once(
    tensor: ComplexTensor,
    dim: usize,
) -> BuiltinResult<ComplexTensor> {
    let mut shape = tensor.shape().to_vec();
    let axis = dim.saturating_sub(1);
    while shape.len() <= axis {
        shape.push(1);
    }
    let count = shape[axis];
    let before = product(&shape[..axis]);
    let after = product(&shape[axis + 1..]);
    shape[axis] = count.saturating_sub(1);
    fn values<T: Copy + std::ops::Sub<Output = T>>(
        data: &[runmat_value::ComplexElement<T>],
        count: usize,
        before: usize,
        after: usize,
    ) -> Vec<(T, T)> {
        let mut out = Vec::with_capacity(before * count.saturating_sub(1) * after);
        for page in 0..after {
            for k in 0..count.saturating_sub(1) {
                for lane in 0..before {
                    let first = page * before * count + k * before + lane;
                    let second = first + before;
                    out.push((
                        data[second].0 - data[first].0,
                        data[second].1 - data[first].1,
                    ));
                }
            }
        }
        out
    }
    match tensor.complex_storage() {
        runmat_value::ComplexStorage::F32(data) => {
            ComplexTensor::from_f32(values(data, count, before, after), shape)
        }
        runmat_value::ComplexStorage::F64(data) => {
            ComplexTensor::new(values(data, count, before, after), shape)
        }
        runmat_value::ComplexStorage::Integer(_) => {
            return Err(diff_invalid_input(
                "complex integer differences are unsupported",
            ))
        }
    }
    .map_err(diff_internal_error)
}
