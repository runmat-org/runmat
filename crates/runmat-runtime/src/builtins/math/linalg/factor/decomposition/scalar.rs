use super::*;

pub(super) async fn decomposition_scale_scalar_async(
    value: &Value,
    label: &str,
) -> BuiltinResult<(f64, f64)> {
    ensure_scale_operand_extensions_enabled(value)?;
    let host = crate::dispatcher::gather_if_needed_async(value)
        .await
        .map_err(|err| {
            build_runtime_error(err.message())
                .with_builtin(NAME)
                .with_source(err)
                .build()
        })?;
    ensure_exact_integer_scalar_boundary(&host, label)?;
    complex_scalar(&host, label)
}

pub(super) fn ensure_exact_integer_scalar_boundary(
    value: &Value,
    label: &str,
) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let exact_values: Vec<IntValue> = match value {
        Value::Int(value) => vec![value.clone()],
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map_or_else(Vec::new, IntegerStorage::exact_values),
        Value::ComplexTensor(tensor) => tensor.integer_storage().map_or_else(Vec::new, |storage| {
            let mut values = storage.real.exact_values();
            values.extend(storage.imag.exact_values());
            values
        }),
        _ => Vec::new(),
    };
    if exact_values.into_iter().any(|value| {
        let exact = match value {
            IntValue::I8(value) => i128::from(value),
            IntValue::I16(value) => i128::from(value),
            IntValue::I32(value) => i128::from(value),
            IntValue::I64(value) => i128::from(value),
            IntValue::U8(value) => i128::from(value),
            IntValue::U16(value) => i128::from(value),
            IntValue::U32(value) => i128::from(value),
            IntValue::U64(value) => i128::from(value),
        };
        !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact)
    }) {
        return Err(invalid(format!(
            "decomposition: integer {label} must be exactly representable as double"
        )));
    }
    Ok(())
}

pub(super) fn complex_scalar(value: &Value, label: &str) -> BuiltinResult<(f64, f64)> {
    match value {
        Value::Num(n) => Ok((*n, 0.0)),
        Value::Int(i) => Ok((i.to_f64(), 0.0)),
        Value::Bool(b) => Ok((if *b { 1.0 } else { 0.0 }, 0.0)),
        Value::Complex(re, im) => Ok((*re, *im)),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => {
            Ok((tensor::tensor_value_f64(tensor, 0), 0.0))
        }
        Value::ComplexTensor(tensor) if complex_tensor_element_len(tensor) == 1 => {
            if let Some(storage) = tensor.integer_storage() {
                let re = storage
                    .real
                    .value_at(0)
                    .expect("one-element complex integer real storage")
                    .to_f64();
                let im = storage
                    .imag
                    .value_at(0)
                    .expect("one-element complex integer imaginary storage")
                    .to_f64();
                return Ok((re, im));
            }
            Ok(tensor.materialize_f64()[0])
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok((if array.data[0] != 0 { 1.0 } else { 0.0 }, 0.0))
        }
        other => Err(invalid(format!(
            "decomposition: {label} must be scalar, got {other:?}"
        ))),
    }
}

pub(super) fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

pub(super) fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_storage()
        .as_ref()
        .map_or(tensor.materialize_f64().len(), |storage| storage.real.len())
}

pub(super) fn real_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    let (re, im) = complex_scalar(value, label)?;
    if im != 0.0 {
        return Err(invalid(format!(
            "decomposition: {label} must be a real scalar"
        )));
    }
    Ok(re)
}

pub(super) fn bool_scalar(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        _ => {
            let scalar = real_scalar(value, label)?;
            Ok(scalar != 0.0)
        }
    }
}

pub(super) fn text_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

pub(super) fn complex_scalar_to_value(value: (f64, f64)) -> Value {
    if value.1 == 0.0 && !value.1.is_nan() {
        Value::Num(value.0)
    } else {
        Value::Complex(value.0, value.1)
    }
}

pub(super) fn complex_conj(value: (f64, f64)) -> (f64, f64) {
    (value.0, -value.1)
}

pub(super) fn logical_rows(array: &LogicalArray) -> usize {
    array.shape.first().copied().unwrap_or(1)
}

pub(super) fn logical_cols(array: &LogicalArray) -> usize {
    array.shape.get(1).copied().unwrap_or(1)
}

pub(super) fn zero_int_like(storage: &IntegerStorage) -> IntValue {
    storage.cast_f64_assignment(0.0)
}

pub(super) fn negate_int_saturating(value: &IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value.saturating_neg()),
        IntValue::I16(value) => IntValue::I16(value.saturating_neg()),
        IntValue::I32(value) => IntValue::I32(value.saturating_neg()),
        IntValue::I64(value) => IntValue::I64(value.saturating_neg()),
        IntValue::U8(_) => IntValue::U8(0),
        IntValue::U16(_) => IntValue::U16(0),
        IntValue::U32(_) => IntValue::U32(0),
        IntValue::U64(_) => IntValue::U64(0),
    }
}

pub(super) fn complex_mul(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (
        lhs.0.mul_add(rhs.0, -(lhs.1 * rhs.1)),
        lhs.0.mul_add(rhs.1, lhs.1 * rhs.0),
    )
}

pub(super) fn complex_div(lhs: (f64, f64), rhs: (f64, f64)) -> BuiltinResult<(f64, f64)> {
    if rhs.0 == 0.0 && rhs.1 == 0.0 {
        return Err(invalid("decomposition: division by zero scale factor"));
    }
    let result = Complex64::new(lhs.0, lhs.1) / Complex64::new(rhs.0, rhs.1);
    Ok((result.re, result.im))
}
