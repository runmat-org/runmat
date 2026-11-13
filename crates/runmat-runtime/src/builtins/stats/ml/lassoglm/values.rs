use super::*;

pub(super) fn value_to_real_tensor(label: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, label)?;
            tensor::integer_tensor_to_f64(tensor)
                .map_err(|err| invalid(format!("lassoglm: {label}: {err}")))
        }
        Value::LogicalArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            Tensor::new(
                array
                    .data
                    .into_iter()
                    .map(|flag| if flag == 0 { 0.0 } else { 1.0 })
                    .collect(),
                shape,
            )
            .map_err(|err| invalid(format!("lassoglm: {label}: {err}")))
        }
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|err| invalid(format!("lassoglm: {label}: {err}"))),
        Value::Int(i) => {
            ensure_exact_integer_value(&i, label)?;
            Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|err| invalid(format!("lassoglm: {label}: {err}")))
        }
        other => Err(invalid(format!(
            "lassoglm: {label} must be real numeric, got {other:?}"
        ))),
    }
}

pub(super) fn ensure_exact_integer_value(
    value: &runmat_value::IntValue,
    label: &str,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(())
    } else {
        Err(invalid(format!(
            "lassoglm: integer {label} values must be exactly representable as double"
        )))
    }
}

pub(super) fn ensure_exact_integer_tensor(value: &Tensor, label: &str) -> BuiltinResult<()> {
    if let Some(storage) = value.integer_storage() {
        for integer in storage.exact_values() {
            ensure_exact_integer_value(&integer, label)?;
        }
    }
    Ok(())
}

pub(super) fn scalar_text<'a>(
    value: &'a Value,
    label: &str,
) -> BuiltinResult<std::borrow::Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(std::borrow::Cow::Borrowed(text.as_str()))
        }
        Value::CharArray(chars) => Ok(std::borrow::Cow::Owned(
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid(format!("lassoglm: {label} must be valid UTF-16 text"))
            })?,
        )),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(std::borrow::Cow::Borrowed(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        other => Err(invalid(format!(
            "lassoglm: {label} must be a string scalar, got {other:?}"
        ))),
    }
}

pub(super) fn scalar_f64(value: &Value, label: &str) -> BuiltinResult<f64> {
    let number = match value {
        Value::Num(n) => *n,
        Value::Int(i) => {
            ensure_exact_integer_value(i, label)?;
            i.to_f64()
        }
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, label)?;
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(invalid(format!(
                "lassoglm: {label} must be a numeric scalar, got {other:?}"
            )))
        }
    };
    if !number.is_finite() {
        return Err(invalid(format!("lassoglm: {label} must be finite")));
    }
    Ok(number)
}

pub(super) fn scalar_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid(format!("lassoglm: {label} must be logical scalar"))),
        };
    }
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::String(runmat_value::StringElement::Text(text))
            if text.eq_ignore_ascii_case("true")
                || text.eq_ignore_ascii_case("false")
                || text.eq_ignore_ascii_case("on")
                || text.eq_ignore_ascii_case("off") =>
        {
            Ok(text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("on"))
        }
        Value::CharArray(chars) => {
            let text = crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid("lassoglm: CV must be valid UTF-16 text")
            })?;
            if text.eq_ignore_ascii_case("true")
                || text.eq_ignore_ascii_case("false")
                || text.eq_ignore_ascii_case("on")
                || text.eq_ignore_ascii_case("off")
            {
                Ok(text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("on"))
            } else {
                Err(invalid(format!("lassoglm: {label} must be logical scalar")))
            }
        }
        _ => Err(invalid(format!("lassoglm: {label} must be logical scalar"))),
    }
}

pub(super) fn positive_usize(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid(format!(
                    "lassoglm: {label} must be a positive integer scalar"
                ))
            });
    }
    let raw = scalar_f64(value, label)?;
    if raw < 1.0
        || raw.fract() != 0.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(invalid(format!(
            "lassoglm: {label} must be a positive integer scalar"
        )));
    }
    Ok(raw as usize)
}

pub(super) fn bounded_usize(value: &Value, label: &str, max: usize) -> BuiltinResult<usize> {
    let parsed = positive_usize(value, label)?;
    if parsed > max {
        return Err(invalid(format!(
            "lassoglm: {label} must be no greater than {max}"
        )));
    }
    Ok(parsed)
}

pub(super) fn numeric_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Tensor(tensor) => {
            if !is_vector_shape(tensor.shape()) {
                return Err(invalid(format!(
                    "lassoglm: {label} must be a numeric vector"
                )));
            }
            ensure_exact_integer_tensor(tensor, label)?;
            tensor::tensor_values_f64(tensor)
        }
        Value::LogicalArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            if !is_vector_shape(&shape) {
                return Err(invalid(format!(
                    "lassoglm: {label} must be a numeric vector"
                )));
            }
            array
                .data
                .iter()
                .map(|flag| if *flag == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        Value::Num(n) => vec![*n],
        Value::Int(i) => {
            ensure_exact_integer_value(i, label)?;
            vec![i.to_f64()]
        }
        other => {
            return Err(invalid(format!(
                "lassoglm: {label} must be a numeric vector, got {other:?}"
            )))
        }
    };
    if data.is_empty() || data.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "lassoglm: {label} must contain finite values"
        )));
    }
    Ok(data)
}

pub(super) fn nonnegative_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value < 0.0) {
        return Err(invalid(format!(
            "lassoglm: {label} values must be nonnegative"
        )));
    }
    Ok(data)
}

pub(super) fn positive_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value <= 0.0) {
        return Err(invalid(format!(
            "lassoglm: {label} values must be positive"
        )));
    }
    Ok(data)
}

pub(super) fn nonnegative_weight_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value < 0.0) || data.iter().all(|value| *value == 0.0) {
        return Err(invalid(format!(
            "lassoglm: {label} must contain nonnegative values with positive total weight"
        )));
    }
    Ok(data)
}

pub(super) fn parse_cv(value: &Value) -> BuiltinResult<CvSpec> {
    match value {
        Value::String(runmat_value::StringElement::Text(text))
            if text.eq_ignore_ascii_case("resubstitution") =>
        {
            Ok(CvSpec::Resubstitution)
        }
        Value::CharArray(chars) => {
            let text = crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid("lassoglm: CV must be valid UTF-16 text")
            })?;
            if text.eq_ignore_ascii_case("resubstitution") {
                Ok(CvSpec::Resubstitution)
            } else {
                Err(invalid(
                    "lassoglm: CV supports 'resubstitution' or a positive integer",
                ))
            }
        }
        _ => {
            let k = positive_usize(value, "CV")?;
            if k < 2 {
                return Err(invalid("lassoglm: CV fold count must be at least 2"));
            }
            Ok(CvSpec::KFold(k))
        }
    }
}

pub(super) fn string_list(value: &Value, label: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(vec![text.as_str().to_owned()]),
        Value::CharArray(chars) => Ok(vec![crate::builtins::stats::decode_utf16_with_error(
            chars.data(),
            || invalid(format!("lassoglm: {label} must contain valid UTF-16 text")),
        )?]),
        Value::StringArray(array) => array.data().iter().map(|text| text.as_str().map(str::to_owned).ok_or_else(|| invalid(format!("lassoglm: {label} entries must be nonmissing text")))).collect(),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|item| scalar_text(item, label).map(std::borrow::Cow::into_owned))
            .collect(),
        other => Err(invalid(format!(
            "lassoglm: {label} must be a string array or cell array of character vectors, got {other:?}"
        ))),
    }
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}
