use super::*;

pub(in crate::builtins) fn scalar_text(
    value: &Value,
    function: &'static str,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.to_string()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(runmat_value::decode_utf16_code_units(
            chars.data(),
        )
        .map_err(|source| {
            let mut error = deep_learning_error(function, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .as_str()
                .expect("nonmissing scalar text")
                .to_owned())
        }
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
    if let Value::LogicalArray(array) = value {
        if array.data.len() == 1 {
            return Ok(array.data[0] != 0);
        }
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

pub(in crate::builtins) fn object<K, I>(class_name: &str, properties: I) -> BuiltinResult<Value>
where
    K: AsRef<str>,
    I: AsRef<[(K, Value)]>,
{
    let properties = properties.as_ref();
    let context = crate::context::legacy::active();
    crate::execution::object::construct_object_from_borrowed_fields(
        context.as_ref(),
        class_name,
        properties
            .iter()
            .map(|(name, value)| (name.as_ref(), value)),
    )
    .map(|value| value.with_owned(|value| value))
}

pub(in crate::builtins) fn object_with_identity<K, I>(
    class_name: runmat_types::StaticClassIdentity,
    properties: I,
) -> BuiltinResult<Value>
where
    K: AsRef<str>,
    I: AsRef<[(K, Value)]>,
{
    object(class_name.display_name(), properties)
}

pub(in crate::builtins::deep_learning) fn replace_property(
    object: &mut ObjectInstance,
    name: &str,
    value: Value,
) -> BuiltinResult<Option<Value>> {
    let context = crate::context::legacy::active();
    let _source = crate::execution::structure::replacement::lease(
        &Value::Object(object.clone()),
        context.as_ref(),
    )?;
    let value = crate::execution::structure::replacement::root(value, context.as_ref())?;
    value.with_owned(|value| {
        crate::execution::object::replacement::replace_property(
            context.as_ref(),
            object,
            name,
            value,
            |message, source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    context.as_ref(),
                    "RunMat:deepLearning:ObjectAllocation",
                    message,
                    source,
                )
            },
            |message| deep_learning_error("deepLearning", message),
        )
    })
}

/// Domain diagnostic adapter; physical replacement belongs to the execution
/// owner. Method syntax reborrows owned and borrowed layer receivers equally.
pub(in crate::builtins::deep_learning) trait DeepLearningObjectEdit {
    fn replace_dl_property(&mut self, name: &str, value: Value) -> BuiltinResult<Option<Value>>;
    fn remove_dl_property(&mut self, name: &str) -> BuiltinResult<Option<Value>>;
    #[cfg(test)]
    fn fixture_dl_property_mut(&mut self, name: &str) -> Option<&mut Value>;
}

impl DeepLearningObjectEdit for ObjectInstance {
    #[cfg(test)]
    fn fixture_dl_property_mut(&mut self, name: &str) -> Option<&mut Value> {
        let value = self.property(name)?.clone();
        replace_property(self, name, value).expect("fixture property destination");
        self.try_property_mut_unique(name)
    }
    fn replace_dl_property(&mut self, name: &str, value: Value) -> BuiltinResult<Option<Value>> {
        replace_property(self, name, value)
    }

    fn remove_dl_property(&mut self, name: &str) -> BuiltinResult<Option<Value>> {
        if self.property(name).is_none() {
            return Ok(None);
        }
        let context = crate::context::legacy::active();
        let _source = crate::execution::structure::replacement::lease(
            &Value::Object(self.clone()),
            context.as_ref(),
        )?;
        crate::execution::object::replacement::remove_property(
            context.as_ref(),
            self,
            name,
            |message, source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    context.as_ref(),
                    "RunMat:deepLearning:ObjectAllocation",
                    message,
                    source,
                )
            },
            |message| deep_learning_error("deepLearning", message),
        )
        .map(Some)
    }
}
