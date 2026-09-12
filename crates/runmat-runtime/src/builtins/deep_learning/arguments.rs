use super::*;

pub(in crate::builtins) fn scalar_text(
    value: &Value,
    function: &'static str,
) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(deep_learning_error(
            function,
            format!("{function}: expected text scalar, got {other:?}"),
        )),
    }
}

pub(in crate::builtins) fn numeric_scalar(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t)
            if crate::builtins::common::tensor::is_scalar_tensor(t)
                && crate::builtins::common::tensor::tensor_value_f64(t, 0).is_finite() =>
        {
            Ok(crate::builtins::common::tensor::tensor_value_f64(t, 0))
        }
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a finite numeric scalar, got {other:?}"),
        )),
    }
}

/// Parse a scalar flag without consulting an integer tensor's compatibility
/// `f64` mirror.  Structural options use this rather than treating an integer
/// value as ordinary numeric data.
pub(in crate::builtins) fn logical_scalar(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<bool> {
    if let Value::Bool(flag) = value {
        return Ok(*flag);
    }
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return match integer.try_to_i64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(deep_learning_error(
                function,
                format!("{function}: {label} must be logical scalar true or false"),
            )),
        };
    }
    let number = match value {
        Value::Num(number) => *number,
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            crate::builtins::common::tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(deep_learning_error(
                function,
                format!("{function}: {label} must be logical scalar true or false, got {other:?}"),
            ));
        }
    };
    match number {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be logical scalar true or false"),
        )),
    }
}

pub(in crate::builtins) fn positive_i64(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<i64> {
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return integer
            .try_to_i64()
            .filter(|value| *value >= 1)
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {label} must be a positive integer scalar"),
                )
            });
    }
    let number = numeric_scalar(value, function, label)?;
    if number.fract().abs() > f64::EPSILON || number < 1.0 || number >= i64::MAX as f64 {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a positive integer scalar"),
        ));
    }
    Ok(number as i64)
}

pub(in crate::builtins) fn positive_usize(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) => value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {label} must be a positive integer"),
                )
            }),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return value
                    .try_to_usize()
                    .filter(|value| *value >= 1)
                    .ok_or_else(|| {
                        deep_learning_error(
                            function,
                            format!("{function}: {label} must be a positive integer"),
                        )
                    });
            }
            let n = crate::builtins::common::tensor::tensor_value_f64(tensor, 0);
            positive_usize_from_f64(n, function, label)
        }
        _ => {
            let n = numeric_scalar(value, function, label)?;
            positive_usize_from_f64(n, function, label)
        }
    }
}

pub(in crate::builtins) fn nonnegative_usize(
    value: &Value,
    function: &'static str,
    label: &str,
) -> Option<usize> {
    match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return value.try_to_usize();
            }
            nonnegative_usize_from_f64(crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
        }
        Value::Num(n) => nonnegative_usize_from_f64(*n),
        _ => {
            let _ = (function, label);
            None
        }
    }
}

pub(super) fn positive_usize_from_f64(
    n: f64,
    function: &'static str,
    label: &str,
) -> BuiltinResult<usize> {
    if !n.is_finite()
        || n.fract().abs() > f64::EPSILON
        || n < 1.0
        || n > usize::MAX as f64
        || (usize::BITS == 64 && n == usize::MAX as f64)
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a positive integer"),
        ));
    }
    Ok(n as usize)
}

pub(super) fn nonnegative_usize_from_f64(n: f64) -> Option<usize> {
    if n.is_finite()
        && n >= 0.0
        && n.fract() == 0.0
        && (n < usize::MAX as f64 || (usize::BITS < 64 && n == usize::MAX as f64))
    {
        Some(n as usize)
    } else {
        None
    }
}

pub(in crate::builtins) fn numeric_vector(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(value) => value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .map(|value| vec![value])
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {label} must contain positive integers"),
                )
            }),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let storage = tensor.integer_storage().expect("checked integer storage");
            let mut out = Vec::with_capacity(storage.len());
            for index in 0..storage.len() {
                let Some(value) = storage
                    .value_at(index)
                    .and_then(|value| value.try_to_usize())
                    .filter(|value| *value >= 1)
                else {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: {label} must contain positive integers"),
                    ));
                };
                out.push(value);
            }
            Ok(out)
        }
        Value::Num(_) | Value::Tensor(_) => {
            let values = numeric_values(value, function, label)?;
            let mut out = Vec::with_capacity(values.len());
            for item in values {
                if !item.is_finite()
                    || item.fract().abs() > f64::EPSILON
                    || item < 1.0
                    || item > usize::MAX as f64
                    || (usize::BITS == 64 && item == usize::MAX as f64)
                {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: {label} must contain positive integers"),
                    ));
                }
                out.push(item as usize);
            }
            Ok(out)
        }
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be numeric, got {other:?}"),
        )),
    }
}

pub(in crate::builtins) fn numeric_values(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Tensor(t) => Ok(crate::builtins::common::tensor::tensor_values_f64(t)),
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be numeric, got {other:?}"),
        )),
    }
}

pub(in crate::builtins) fn text_or_missing(
    value: Option<&Value>,
    default: &str,
    function: &'static str,
) -> BuiltinResult<String> {
    match value {
        Some(v) => scalar_text(v, function),
        None => Ok(default.to_string()),
    }
}

pub(in crate::builtins) fn string_array(
    values: Vec<String>,
    shape: Vec<usize>,
    function: &'static str,
) -> BuiltinResult<Value> {
    StringArray::new(values, shape)
        .map(Value::StringArray)
        .map_err(|err| deep_learning_error(function, err))
}

pub(in crate::builtins) fn tensor_value(
    data: Vec<f64>,
    shape: Vec<usize>,
    function: &'static str,
) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| deep_learning_error(function, err))
}

pub(in crate::builtins) fn object<K, I>(class_name: &str, properties: I) -> Value
where
    K: Into<String>,
    I: IntoIterator<Item = (K, Value)>,
{
    let mut object = ObjectInstance::new(class_name.to_string());
    for (name, value) in properties {
        object.properties.insert(name.into(), value);
    }
    Value::Object(object)
}

pub(in crate::builtins) fn object_with_identity<K, I>(
    class_name: runmat_types::StaticClassIdentity,
    properties: I,
) -> Value
where
    K: Into<String>,
    I: IntoIterator<Item = (K, Value)>,
{
    let mut object = ObjectInstance::new(class_name);
    for (name, value) in properties {
        object.properties.insert(name.into(), value);
    }
    Value::Object(object)
}
