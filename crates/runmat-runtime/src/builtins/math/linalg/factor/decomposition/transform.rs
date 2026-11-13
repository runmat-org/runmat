use super::*;

#[cfg(test)]
pub(super) fn conjugate_transpose(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let mut indices = vec![0; tensor.len()];
            for row in 0..rows {
                for col in 0..cols {
                    indices[col + row * cols] = row + col * rows;
                }
            }
            let storage = tensor
                .into_numeric_storage()
                .map_err(|e| internal(format!("decomposition: {e}")))?
                .reorder(&indices)
                .map_err(|e| internal(format!("decomposition: {e}")))?;
            Ok(Value::Tensor(
                Tensor::from_numeric_storage(storage, vec![cols, rows])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::ComplexTensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let dtype = tensor.numeric_dtype();
            let shape = vec![cols, rows];
            let mut indices = vec![0; tensor.len()];
            for row in 0..rows {
                for col in 0..cols {
                    indices[col + row * cols] = row + col * rows;
                }
            }
            if let Some(storage) = tensor.integer_storage().cloned() {
                let real = storage
                    .real()
                    .from_exact_values_like(
                        indices
                            .iter()
                            .map(|&index| {
                                storage
                                    .real()
                                    .value_at(index)
                                    .expect("transpose index is in bounds")
                            })
                            .collect(),
                    )
                    .map_err(internal)?;
                let imag_values = storage.imag().exact_values();
                let imag = storage
                    .imag()
                    .from_exact_values_like(
                        indices
                            .iter()
                            .map(|&index| negate_int_saturating(&imag_values[index]))
                            .collect(),
                    )
                    .map_err(internal)?;
                return Ok(Value::ComplexTensor(
                    ComplexTensor::new_integer(
                        IntegerComplexStorage::new(real, imag).map_err(internal)?,
                        shape,
                    )
                    .map_err(internal)?,
                ));
            }
            let values = tensor::complex_tensor_into_values_complex64(tensor);
            let data = indices
                .into_iter()
                .map(|index| {
                    let value = values[index];
                    (value.re, -value.im)
                })
                .collect();
            Ok(Value::ComplexTensor(
                ComplexTensor::from_f64_values_with_dtype(data, shape, dtype)
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::LogicalArray(array) => {
            let rows = logical_rows(&array);
            let cols = logical_cols(&array);
            let mut data = vec![0; array.data.len()];
            for row in 0..rows {
                for col in 0..cols {
                    data[col + row * cols] = array.data[row + col * rows];
                }
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![cols, rows]).map_err(internal)?,
            ))
        }
        other => Err(internal(format!(
            "decomposition: unsupported stored matrix {other:?}"
        ))),
    }
}

#[cfg(test)]
pub(super) fn scale_matrix(value: Value, scale: (f64, f64)) -> BuiltinResult<Value> {
    if scale == (1.0, 0.0) {
        return Ok(value);
    }
    match value {
        Value::Tensor(tensor) if scale.1 == 0.0 => {
            let shape = tensor.shape().to_vec();
            let dtype = floating_scale_dtype(tensor.numeric_dtype());
            Ok(Value::Tensor(
                Tensor::new_with_dtype(
                    tensor::tensor_into_values_f64(tensor)
                        .into_iter()
                        .map(|value| value * scale.0)
                        .collect(),
                    shape,
                    dtype,
                )
                .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let dtype = floating_scale_dtype(tensor.numeric_dtype());
            Ok(Value::ComplexTensor(
                ComplexTensor::from_f64_values_with_dtype(
                    tensor::tensor_into_values_f64(tensor)
                        .into_iter()
                        .map(|value| complex_mul((value, 0.0), scale))
                        .collect(),
                    shape,
                    dtype,
                )
                .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let dtype = floating_scale_dtype(tensor.numeric_dtype());
            let data = tensor::complex_tensor_into_values_complex64(tensor)
                .into_iter()
                .map(|value| complex_mul((value.re, value.im), scale))
                .collect();
            Ok(Value::ComplexTensor(
                ComplexTensor::from_f64_values_with_dtype(data, shape, dtype)
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        other => Err(internal(format!(
            "decomposition: unsupported stored matrix {other:?}"
        ))),
    }
}

#[cfg(test)]
pub(super) fn floating_scale_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

pub(super) fn matrix_datatype(matrix: &Value) -> &'static str {
    match matrix {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => "single",
        Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => "single",
        Value::SparseTensor(matrix) if matrix.numeric_dtype() == Some(NumericDType::F32) => {
            "single"
        }
        _ => "double",
    }
}

#[cfg(test)]
pub(super) fn matrix_is_real(matrix: &Value) -> bool {
    match matrix {
        Value::ComplexTensor(tensor) => {
            let values = complex_tensor_values_pair_view(tensor);
            (0..values.len()).all(|index| values.get(index).1 == 0.0)
        }
        _ => true,
    }
}

#[cfg(test)]
pub(super) enum ComplexPairView<'a> {
    Borrowed(&'a [ComplexElement<f64>]),
    Materialized(Vec<(f64, f64)>),
}

#[cfg(test)]
impl ComplexPairView<'_> {
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Borrowed(values) => values.len(),
            Self::Materialized(values) => values.len(),
        }
    }

    #[cfg(test)]
    pub(super) fn get(&self, index: usize) -> (f64, f64) {
        match self {
            Self::Borrowed(values) => values[index].into(),
            Self::Materialized(values) => values[index],
        }
    }
}

#[cfg(test)]
pub(super) fn complex_tensor_values_pair_view(tensor: &ComplexTensor) -> ComplexPairView<'_> {
    if let Some(values) = tensor.as_f64_slice() {
        ComplexPairView::Borrowed(values)
    } else {
        ComplexPairView::Materialized(
            tensor::complex_tensor_values_complex64(tensor)
                .into_iter()
                .map(|value| (value.re, value.im))
                .collect(),
        )
    }
}
