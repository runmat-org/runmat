use super::*;

pub(super) fn abs_host_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Int(value) => Ok(Value::Int(abs_integer_scalar(value))),
        Value::Complex(re, im) => Ok(Value::Num(complex_magnitude(re, im))),
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "abs")?;
            abs_complex_tensor(ct)
        }
        other => abs_real(other),
    }
}

pub(super) fn abs_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("abs", value)
        .map_err(|err| OPERATION.error(&ABS_ERROR_INVALID_INPUT, err))?;
    Ok(tensor::tensor_into_value(abs_tensor(tensor)?))
}

pub(super) fn abs_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| OPERATION.error(&ABS_ERROR_INTERNAL, e))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64::abs).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32::abs).collect())
        }
        integer => NumericStorage::from_integer_storage(abs_integer_storage(
            &integer
                .into_integer_storage()
                .expect("integer NumericStorage variant"),
        )),
    };
    Tensor::from_numeric_storage(output, shape).map_err(|e| OPERATION.error(&ABS_ERROR_INTERNAL, e))
}

pub(super) fn abs_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let rows = sparse.rows;
    let cols = sparse.cols;
    let col_ptrs = sparse.col_ptrs.clone();
    let row_indices = sparse.row_indices.clone();
    let output = if sparse.is_logical() {
        SparseTensor::new(rows, cols, col_ptrs, row_indices, vec![1.0; sparse.nnz()])
    } else if let Some(values) = sparse.as_f64_slice() {
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values.iter().copied().map(f64::abs).collect(),
        )
    } else if let Some(values) = sparse.as_f32_slice() {
        SparseTensor::new_f32(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values.iter().copied().map(f32::abs).collect(),
        )
    } else if let Some(values) = sparse.integer_storage() {
        SparseTensor::new_integer(
            rows,
            cols,
            col_ptrs,
            row_indices,
            abs_integer_storage(values),
        )
    } else {
        return Err(OPERATION.error(&ABS_ERROR_INTERNAL, "unsupported sparse storage"));
    }
    .map_err(|err| OPERATION.error(&ABS_ERROR_INTERNAL, err))?;
    Ok(Value::SparseTensor(output))
}

pub(super) fn abs_integer_scalar(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value.saturating_abs()),
        IntValue::I16(value) => IntValue::I16(value.saturating_abs()),
        IntValue::I32(value) => IntValue::I32(value.saturating_abs()),
        IntValue::I64(value) => IntValue::I64(value.saturating_abs()),
        IntValue::U8(value) => IntValue::U8(value),
        IntValue::U16(value) => IntValue::U16(value),
        IntValue::U32(value) => IntValue::U32(value),
        IntValue::U64(value) => IntValue::U64(value),
    }
}

pub(super) fn abs_integer_storage(storage: &IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(values.clone()),
        IntegerStorage::U16(values) => IntegerStorage::U16(values.clone()),
        IntegerStorage::U32(values) => IntegerStorage::U32(values.clone()),
        IntegerStorage::U64(values) => IntegerStorage::U64(values.clone()),
    }
}

pub(super) fn abs_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| complex_magnitude(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| real.hypot(imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(OPERATION.error(
                &ABS_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| OPERATION.error(&ABS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

pub(super) fn abs_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| OPERATION.error(&ABS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
pub(super) fn complex_magnitude(re: f64, im: f64) -> f64 {
    re.hypot(im)
}
