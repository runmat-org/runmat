use super::*;

pub(super) fn canonical_option_name(name: &str) -> BuiltinResult<StatOptionField> {
    let Some(canonical) = StatOptionField::resolve(name) else {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: unknown option '{name}'"),
        ));
    };
    Ok(canonical)
}

pub(super) fn validate_option_value(field: StatOptionField, value: &Value) -> BuiltinResult<Value> {
    let name = field.name();
    match field.value_kind() {
        StatOptionValueKind::Display => one_of_text(name, value, &["off", "final", "iter"]),
        StatOptionValueKind::OnOff => one_of_text(name, value, &["off", "on"]),
        StatOptionValueKind::ToleranceType => one_of_text(name, value, &["abs", "rel"]),
        StatOptionValueKind::RobustWeight => validate_robust_weight(value),
        StatOptionValueKind::PositiveCount => positive_integer_value(name, value),
        StatOptionValueKind::PositiveScalar => positive_scalar_value(name, value),
        StatOptionValueKind::PositiveArray => positive_numeric_value(name, value),
        StatOptionValueKind::Boolean => bool_or_on_off_value(name, value),
        StatOptionValueKind::Opaque => Ok(value.clone()),
    }
}

fn validate_robust_weight(value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value)
        || matches!(
            value,
            Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::ScopedFunctionHandle(_)
        )
    {
        return Ok(value.clone());
    }
    one_of_text(
        "RobustWgtFun",
        value,
        &[
            "andrews", "bisquare", "cauchy", "fair", "huber", "logistic", "talwar", "welsch",
        ],
    )
}

fn one_of_text(field: &str, value: &Value, allowed: &[&str]) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    let text = text_scalar(value)?;
    let lower = text.to_ascii_lowercase();
    if allowed.contains(&lower.as_str()) {
        Ok(Value::from(lower))
    } else {
        Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be one of {}", allowed.join(", ")),
        ))
    }
}

pub(super) fn positive_integer_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let integer = integer
            .try_to_usize()
            .filter(|integer| *integer >= 1 && *integer <= MAX_OPTION_INTEGER)
            .ok_or_else(|| {
                statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {field} must be a positive integer"),
                )
            })?;
        return Ok(Value::Num(integer as f64));
    }
    let scalar = numeric_scalar(field, value)?;
    if scalar < 1.0 || scalar.fract() != 0.0 || scalar > MAX_OPTION_INTEGER as f64 {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be a positive integer"),
        ));
    }
    Ok(Value::Num(scalar))
}

fn positive_scalar_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    let scalar = numeric_scalar(field, value)?;
    if scalar <= 0.0 {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be a positive scalar"),
        ));
    }
    Ok(Value::Num(scalar))
}

fn positive_numeric_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => positive_scalar_value(field, value),
        Value::LogicalArray(array) if array.data.len() == 1 => positive_scalar_value(field, value),
        Value::Tensor(tensor) => {
            if tensor::tensor_values_f64(tensor)
                .into_iter()
                .all(|entry| entry.is_finite() && entry > 0.0)
            {
                Ok(value.clone())
            } else {
                Err(statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {field} must contain positive finite values"),
                ))
            }
        }
        other => Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be numeric, got {other:?}"),
        )),
    }
}

pub(super) fn bool_or_on_off_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(Value::Bool(false)),
            Some(1) => Ok(Value::Bool(true)),
            _ => Err(statset_error(
                &STATSET_ERROR_INVALID_OPTION,
                format!("statset: {field} must be logical or 'on'/'off'"),
            )),
        };
    }
    match value {
        Value::Bool(flag) => Ok(Value::Bool(*flag)),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(Value::Bool(array.data[0] != 0)),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(Value::Bool(*n != 0.0)),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let value = tensor::tensor_value_f64(tensor, 0);
            if value == 0.0 || value == 1.0 {
                Ok(Value::Bool(value != 0.0))
            } else {
                Err(statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {field} must be logical or 'on'/'off'"),
                ))
            }
        }
        _ => {
            let text = text_scalar(value)?;
            match text.to_ascii_lowercase().as_str() {
                "on" | "true" => Ok(Value::Bool(true)),
                "off" | "false" => Ok(Value::Bool(false)),
                _ => Err(statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {field} must be logical or 'on'/'off'"),
                )),
            }
        }
    }
}

fn numeric_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(integer.to_f64());
    }
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::LogicalArray(array) if array.data.len() == 1 => f64::from(array.data[0] != 0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(statset_error(
                &STATSET_ERROR_INVALID_OPTION,
                format!("statset: {field} must be a numeric scalar, got {other:?}"),
            ))
        }
    };
    if !scalar.is_finite() {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be finite"),
        ));
    }
    Ok(scalar)
}
