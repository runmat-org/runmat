use super::*;

pub(super) fn duration_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

pub(super) fn ensure_duration_class_registered() {
    DURATION_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        properties.insert(
            FORMAT_FIELD.into(),
            crate::class_registry::RuntimeProperty {
                name: FORMAT_FIELD.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: Some(Value::String(DEFAULT_DURATION_FORMAT.to_string())),
            },
        );

        let mut methods = HashMap::new();
        for name in [
            OBJECT_SUBSREF_METHOD,
            OBJECT_SUBSASGN_METHOD,
            runmat_types::StaticMethodName::new("plus"),
            runmat_types::StaticMethodName::new("minus"),
            runmat_types::StaticMethodName::new("eq"),
            runmat_types::StaticMethodName::new("ne"),
            runmat_types::StaticMethodName::new("lt"),
            runmat_types::StaticMethodName::new("le"),
            runmat_types::StaticMethodName::new("gt"),
            runmat_types::StaticMethodName::new("ge"),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{DURATION_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: DURATION_CLASS.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

pub fn is_duration_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(DURATION_CLASS))
}

pub(super) async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gather_if_needed_async(arg)
                .await
                .map_err(|err| duration_error(format!("duration: {}", err.message())))?,
        );
    }
    Ok(out)
}

pub(super) fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(duration_error(format!(
            "duration: {context} must be a string scalar or character vector"
        ))),
    }
}

pub(super) fn parse_trailing_format(args: &[Value]) -> BuiltinResult<(usize, Option<String>)> {
    let mut positional_end = args.len();
    let mut format = None;

    while positional_end >= 2 {
        let name = match scalar_text(&args[positional_end - 2], "option name") {
            Ok(text) => text,
            Err(_) => break,
        };
        if !name.trim().eq_ignore_ascii_case("format") {
            break;
        }
        format = Some(scalar_text(&args[positional_end - 1], "Format option")?);
        positional_end -= 2;
    }

    Ok((positional_end, format))
}

pub(super) fn tensor_from_numeric(value: Value, context: &str) -> BuiltinResult<Tensor> {
    tensor::value_into_tensor_for(context, value)
        .map_err(|message| duration_error(format!("duration: {message}")))
}

pub(super) fn component_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor.len());
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| duration_error(format!("duration: {err}")))
}

pub(super) fn format_for_object(obj: &ObjectInstance) -> String {
    match obj.properties.get(FORMAT_FIELD) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::StringArray(array)) if array.data.len() == 1 => array.data[0].clone(),
        Some(Value::CharArray(array)) if array.rows == 1 => array.data.iter().collect(),
        _ => DEFAULT_DURATION_FORMAT.to_string(),
    }
}

pub(crate) fn duration_tensor_from_duration_value(value: &Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(obj) if obj.is_class(DURATION_CLASS) => {
            match obj.properties.get(DAYS_FIELD) {
                Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
                Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
                    .map_err(|err| duration_error(format!("duration: {err}"))),
                Some(other) => Err(duration_error(format!(
                    "duration: invalid internal day storage {other:?}"
                ))),
                None => Err(duration_error("duration: missing internal day storage")),
            }
        }
        _ => Err(duration_error("duration: expected a duration value")),
    }
}

pub(crate) fn duration_format_from_value(value: &Value) -> String {
    match value {
        Value::Object(obj) if obj.is_class(DURATION_CLASS) => format_for_object(obj),
        _ => DEFAULT_DURATION_FORMAT.to_string(),
    }
}

pub(crate) fn duration_object_from_days_tensor(
    days: Tensor,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    ensure_duration_class_registered();
    let mut object = ObjectInstance::new(DURATION_CLASS.to_string());
    object
        .properties
        .insert(DAYS_FIELD.to_string(), Value::Tensor(days));
    object
        .properties
        .insert(FORMAT_FIELD.to_string(), Value::String(format.into()));
    Ok(Value::Object(object))
}

pub(super) fn duration_object_from_days(
    days: Vec<f64>,
    shape: Vec<usize>,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(days, shape).map_err(|err| duration_error(format!("duration: {err}")))?;
    duration_object_from_days_tensor(tensor, format)
}

pub(super) async fn duration_unit_value(
    value: Value,
    unit_name: &str,
    days_per_unit: f64,
) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| duration_error(format!("{unit_name}: {}", err.message())))?;
    if is_duration_object(&value) {
        let days = duration_tensor_from_duration_value(&value)?;
        let day_values = tensor::tensor_values_f64_cow(&days);
        let data = day_values
            .iter()
            .map(|day| day / days_per_unit)
            .collect::<Vec<_>>();
        return if data.len() == 1 {
            Ok(Value::Num(data[0]))
        } else {
            Ok(Value::Tensor(
                Tensor::new(
                    data,
                    tensor::default_shape_for(&days.shape, day_values.len()),
                )
                .map_err(|err| duration_error(format!("{unit_name}: {err}")))?,
            ))
        };
    }
    let numeric = component_tensor(value, unit_name)?;
    let shape = tensor::default_shape_for(&numeric.shape, numeric.len());
    let values = tensor::tensor_into_values_f64(numeric);
    let days = values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                Err(duration_error(format!(
                    "{unit_name}: values must be finite"
                )))
            } else {
                let days = value * days_per_unit;
                if days.is_finite() {
                    Ok(days)
                } else {
                    Err(duration_error(format!(
                        "{unit_name}: resulting duration is outside supported range"
                    )))
                }
            }
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    duration_object_from_days(days, shape, DEFAULT_DURATION_FORMAT)
}

pub(super) fn broadcast_component_data(
    arrays: &[Tensor],
    labels: &[&str],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut target_shape = vec![1, 1];
    let mut target_len = 1usize;

    for array in arrays {
        let len = array.len();
        if len > 1 {
            let shape = tensor::default_shape_for(&array.shape, len);
            if target_len == 1 {
                target_len = len;
                target_shape = shape;
            } else if len != target_len || shape != target_shape {
                return Err(duration_error(
                    "duration: non-scalar component inputs must have matching sizes",
                ));
            }
        }
    }

    let mut broadcasted = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        let values = array
            .as_f64_slice()
            .expect("duration components are normalized to double storage");
        if values.len() == 1 {
            broadcasted.push(vec![values[0]; target_len]);
        } else if values.len() == target_len {
            broadcasted.push(values.to_vec());
        } else {
            return Err(duration_error(format!(
                "duration: {} input size does not match the other components",
                labels[idx]
            )));
        }
    }

    Ok((broadcasted, target_shape))
}

pub(super) fn build_from_components(
    args: Vec<Value>,
    format: Option<String>,
) -> BuiltinResult<Value> {
    let labels = ["hours", "minutes", "seconds", "milliseconds"];
    let mut arrays = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 4 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }

    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut days = Vec::with_capacity(len);
    for idx in 0..len {
        let total_seconds = broadcasted[0][idx] * 3600.0
            + broadcasted[1][idx] * 60.0
            + broadcasted[2][idx]
            + broadcasted[3][idx] / 1000.0;
        days.push(total_seconds / SECONDS_PER_DAY);
    }

    duration_object_from_days(
        days,
        shape,
        format.unwrap_or_else(|| DEFAULT_DURATION_FORMAT.to_string()),
    )
}

pub(super) fn is_public_duration_matrix(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.shape.len() <= 2 && tensor.cols() == 3,
        Value::GpuTensor(handle) => {
            handle.shape.len() <= 2 && handle.shape.get(1).copied().unwrap_or(1) == 3
        }
        _ => false,
    }
}

pub(super) fn is_numeric_duration_input(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

pub(super) fn build_from_matrix(value: Value, format: Option<String>) -> BuiltinResult<Value> {
    let matrix = component_tensor(value, "X")?;
    if matrix.shape.len() > 2 || matrix.cols() != 3 {
        return Err(duration_error(
            "duration: X must be a numeric matrix with exactly three columns",
        ));
    }
    let rows = matrix.rows();
    let values = matrix
        .as_f64_slice()
        .expect("duration matrix is normalized to double storage");
    let mut days = Vec::with_capacity(rows);
    for row in 0..rows {
        let total_seconds =
            values[row] * 3600.0 + values[row + rows] * 60.0 + values[row + 2 * rows];
        days.push(total_seconds / SECONDS_PER_DAY);
    }
    duration_object_from_days(
        days,
        vec![rows, 1],
        format.unwrap_or_else(|| DEFAULT_DURATION_FORMAT.to_string()),
    )
}
