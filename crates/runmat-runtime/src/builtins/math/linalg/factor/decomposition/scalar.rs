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
    let inexact = |value: IntValue| value.to_exact_f64().is_none();
    let rejected = match value {
        Value::Int(value) => inexact(value.clone()),
        Value::Tensor(t) if t.len() == 1 => t
            .numeric_value_at(0)
            .and_then(runmat_value::NumericScalar::into_int_value)
            .is_some_and(inexact),
        Value::ComplexTensor(t) if t.len() == 1 => t.numeric_value_at(0).is_some_and(|(re, im)| {
            re.into_int_value().is_some_and(inexact) || im.into_int_value().is_some_and(inexact)
        }),
        _ => false,
    };
    if rejected {
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
            let (real, imag) = tensor.numeric_value_at(0).expect("one initialized scalar");
            Ok((native::checked_double(real)?, native::checked_double(imag)?))
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
    tensor.len()
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

pub(super) fn text_from_value(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            sa.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(ca) if ca.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(ca.data()).map_err(|source| {
                let mut error = invalid(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
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

#[cfg(test)]
pub(super) fn zero_int_like(storage: &IntegerStorage) -> IntValue {
    storage.cast_f64_assignment(0.0)
}

#[cfg(test)]
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
