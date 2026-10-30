use super::*;
use std::borrow::Cow;

pub(super) fn finite_vector(value: Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Num(n) => vec![n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => tensor::tensor_into_values_f64(tensor),
        Value::LogicalArray(logical) => logical
            .data
            .iter()
            .map(|value| if *value != 0 { 1.0 } else { 0.0 })
            .collect(),
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: {label} must be numeric or logical, got {other:?}"),
                &ERROR_OBJECTIVE,
            ))
        }
    };
    if data.iter().any(|value| !value.is_finite()) {
        return Err(bayesopt_error(
            format!("bayesopt: {label} must contain finite values"),
            &ERROR_OBJECTIVE,
        ));
    }
    Ok(data)
}

pub(super) fn value_to_text<'a>(value: &'a Value, label: &str) -> BuiltinResult<Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(Cow::Borrowed(text.as_str())),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Cow::Borrowed(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            runmat_value::decode_utf16_code_units(chars.data())
                .map(Cow::Owned)
                .map_err(|error| {
                    bayesopt_error(
                        format!("bayesopt: {label} contains {error}"),
                        &ERROR_INVALID_ARGUMENT,
                    )
                })
        }
        other => Err(bayesopt_error(
            format!("bayesopt: {label} must be a text scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

pub(super) fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(bayesopt_error(
                format!("bayesopt: option {name} must be logical scalar, got {value:?}"),
                &ERROR_INVALID_ARGUMENT,
            )),
        };
    }
    let parsed = match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(logical) if logical.data.len() == 1 => Ok(logical.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let parsed = tensor::tensor_value_f64(tensor, 0);
            if parsed == 0.0 || parsed == 1.0 {
                Ok(parsed != 0.0)
            } else {
                Err(bayesopt_error(
                    format!("bayesopt: option {name} must be logical scalar, got {value:?}"),
                    &ERROR_INVALID_ARGUMENT,
                ))
            }
        }
        other => Err(bayesopt_error(
            format!("bayesopt: option {name} must be logical scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }?;
    Ok(parsed)
}

pub(super) fn option_usize(
    value: &Value,
    name: &str,
    min: usize,
    max: usize,
) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let parsed = integer.try_to_usize();
        return match parsed.filter(|parsed| *parsed >= min && *parsed <= max) {
            Some(parsed) => Ok(parsed),
            None => Err(bayesopt_error(
                format!("bayesopt: option {name} must be an integer in [{min}, {max}]"),
                &ERROR_INVALID_ARGUMENT,
            )),
        };
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: option {name} must be numeric scalar, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if !parsed.is_finite() || parsed.fract() != 0.0 || parsed < min as f64 || parsed > max as f64 {
        return Err(bayesopt_error(
            format!("bayesopt: option {name} must be an integer in [{min}, {max}]"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(parsed as usize)
}

pub(super) fn option_unit_f64(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(0.0),
            Some(1) => Ok(1.0),
            _ => Err(bayesopt_error(
                format!("bayesopt: option {name} must be in [0, 1]"),
                &ERROR_INVALID_ARGUMENT,
            )),
        };
    }
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: option {name} must be numeric scalar, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err(bayesopt_error(
            format!("bayesopt: option {name} must be in [0, 1]"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(parsed)
}

pub(super) fn is_empty_option(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::String(runmat_value::StringElement::Text(text)) => {
            text.is_empty() || text.eq_ignore_ascii_case("none")
        }
        Value::StringArray(array) => array.data().is_empty(),
        _ => false,
    }
}

pub(super) fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}
