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

pub(super) fn validate_duration_operand(value: &Value, numeric_allowed: bool) -> BuiltinResult<()> {
    match value {
        Value::Object(object) if object.is_class(DURATION_CLASS) => {
            match object.properties.get(DAYS_FIELD) {
                Some(Value::Num(_) | Value::Tensor(_)) => Ok(()),
                Some(Value::GpuTensor(handle))
                    if handle.physical().storage()
                        == runmat_accelerate_api::GpuTensorStorage::Real =>
                {
                    Ok(())
                }
                Some(_) => Err(duration_error("duration: invalid internal day storage")),
                None => Err(duration_error("duration: missing internal day storage")),
            }
        }
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
            if numeric_allowed =>
        {
            Ok(())
        }
        Value::GpuTensor(handle)
            if numeric_allowed
                && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            Ok(())
        }
        _ => Err(duration_error(
            "duration: expected a duration or real numeric value",
        )),
    }
}

pub(super) async fn duration_days_operand(
    value: &Value,
    numeric_allowed: bool,
) -> BuiltinResult<Tensor> {
    validate_duration_operand(value, numeric_allowed)?;
    match value {
        Value::Object(object) if object.is_class(DURATION_CLASS) => {
            let days = object
                .properties
                .get(DAYS_FIELD)
                .expect("validated duration day storage");
            if matches!(days, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &DURATION_GPU_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let gathered = gather_if_needed_async(days).await?;
                match gathered {
                    Value::Tensor(tensor) => Ok(tensor),
                    _ => Err(duration_error("duration: invalid gathered day storage")),
                }
            } else {
                duration_tensor_from_duration_value(value)
            }
        }
        Value::GpuTensor(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DURATION_GPU_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
            let gathered = gather_if_needed_async(value).await?;
            component_tensor(gathered, "numeric day count")
        }
        _ => component_tensor(value.clone(), "numeric day count"),
    }
}

pub(super) fn broadcast_duration_days(
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let plan = crate::builtins::common::broadcast::BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(duration_error)?;
    let lhs_values = tensor::tensor_values_f64_cow(lhs);
    let rhs_values = tensor::tensor_values_f64_cow(rhs);
    let mut left = Vec::with_capacity(plan.len());
    let mut right = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        left.push(lhs_values[lhs_index]);
        right.push(rhs_values[rhs_index]);
    }
    Ok((left, right, plan.output_shape().to_vec()))
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

pub(super) fn validate_duration_format(format: &str) -> BuiltinResult<()> {
    if matches!(format, "hh:mm:ss" | "hh:mm" | "mm:ss" | "s" | "ss") {
        Ok(())
    } else {
        Err(duration_error(format!(
            "duration: unsupported Format value '{format}'"
        )))
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
        let value = scalar_text(&args[positional_end - 1], "Format option")?;
        validate_duration_format(&value)?;
        format = Some(value);
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
    let shape = if tensor.shape.is_empty() {
        vec![1, 1]
    } else {
        tensor.shape.clone()
    };
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
    if is_duration_object(&value) {
        let days = duration_days_operand(&value, false).await?;
        let day_values = tensor::tensor_values_f64_cow(&days);
        let data = day_values
            .iter()
            .map(|day| day / days_per_unit)
            .collect::<Vec<_>>();
        return if days.shape.iter().product::<usize>() == 1 {
            Ok(Value::Num(data[0]))
        } else {
            Ok(Value::Tensor(
                Tensor::new(data, days.shape.clone())
                    .map_err(|err| duration_error(format!("{unit_name}: {err}")))?,
            ))
        };
    }
    let numeric = duration_days_operand(&value, true).await?;
    let shape = numeric.shape.clone();
    let values = tensor::tensor_into_values_f64(numeric);
    let days = values
        .iter()
        .map(|value| {
            let days = value * days_per_unit;
            if value.is_finite() && !days.is_finite() {
                Err(duration_error(format!(
                    "{unit_name}: resulting duration is outside supported range"
                )))
            } else {
                Ok(days)
            }
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    duration_object_from_days(days, shape, DEFAULT_DURATION_FORMAT)
}

pub(super) fn broadcast_component_data(
    arrays: &[Tensor],
    labels: &[&str],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut target_shape = None;

    for array in arrays {
        let len = array.len();
        if len != 1 {
            let shape = array.shape.clone();
            if let Some(existing) = &target_shape {
                if existing != &shape {
                    return Err(duration_error(
                        "duration: non-scalar component inputs must have matching sizes",
                    ));
                }
            } else {
                target_shape = Some(shape);
            }
        }
    }

    let target_shape = target_shape.unwrap_or_else(|| vec![1, 1]);
    let target_len = target_shape.iter().product::<usize>();
    let mut broadcasted = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        let values = array
            .as_f64_slice()
            .expect("duration components are normalized to double storage");
        if values.len() == 1 {
            broadcasted.push(vec![values[0]; target_len]);
        } else if values.len() == target_len && array.shape == target_shape {
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
            handle.shape().len() <= 2 && handle.shape().get(1).copied().unwrap_or(1) == 3
        }
        _ => false,
    }
}

pub(super) fn is_numeric_duration_input(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
        || matches!(value, Value::GpuTensor(handle) if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real)
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
