use super::capabilities::{BUILTIN_NAME, DATETIME_CLASS, FORMAT_FIELD};
use super::*;

#[runmat_macros::runtime_builtin(
    name = "datetick",
    builtin_path = "crate::builtins::datetime",
    category = "plotting",
    summary = "Accept MATLAB date-axis formatting calls for compatibility.",
    keywords = "datetick,plot,date axis"
)]
pub(super) async fn datetick_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let _args = gather_args(&args).await?;
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(
    name = "datetime.subsref",
    descriptor(crate::builtins::datetime::DATETIME_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_subsref(obj: Value, subscript: Value) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, datetime_read_step, None).await
}

pub(super) async fn datetime_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<Value> {
    let payload = step.selector_value()?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => datetime_indexing(obj, payload).await,
        crate::object::indexing::ObjectIndexKind::Member => {
            let Value::Object(object) = obj else {
                return Err(datetime_error(
                    "datetime.subsref: receiver must be a datetime object",
                ));
            };
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => Ok(Value::String(format_for_object(&object))),
                _ => Err(datetime_error(format!(
                    "datetime.subsref: unsupported datetime property '{field}'"
                ))),
            }
        }
        crate::object::indexing::ObjectIndexKind::Brace => Err(datetime_error(
            "datetime.subsref: brace indexing is not supported",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "datetime.subsasgn",
    descriptor(crate::builtins::datetime::DATETIME_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        datetime_read_step,
        |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| datetime_error("datetime.subsasgn: assignment value is missing"))?;
            datetime_write_step(obj, step, rhs).await
        },
        None,
    )
    .await
}

pub(super) async fn datetime_write_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let Value::Object(mut object) = obj else {
        return Err(datetime_error(
            "datetime.subsasgn: receiver must be a datetime object",
        ));
    };
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    let text = scalar_text(&rhs, "Format value")?;
                    object
                        .properties
                        .insert(FORMAT_FIELD.to_string(), Value::String(text));
                    Ok(Value::Object(object))
                }
                _ => Err(datetime_error(format!(
                    "datetime.subsasgn: unsupported datetime property '{field}'"
                ))),
            }
        }
        _ => Err(datetime_error(
            "datetime.subsasgn: only member assignment is supported",
        )),
    }
}

pub(super) fn datetime_binary_serials(
    lhs: Value,
    rhs: Value,
    context: &str,
) -> BuiltinResult<(Tensor, Tensor, Vec<usize>, String)> {
    let lhs_serials = serials_from_datetime_value(&lhs)?;
    let rhs_serials = match &rhs {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
        _ => serial_tensor_from_value(rhs, context)?,
    };
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_serials, &rhs_serials, context, BUILTIN_NAME)?;
    let left_tensor = Tensor::new(left, shape.clone())
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let right_tensor = Tensor::new(right, shape.clone())
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    Ok((
        left_tensor,
        right_tensor,
        shape,
        datetime_format_from_value(&lhs),
    ))
}

pub(super) fn compare_datetime(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    let (left, right, shape, _) = datetime_binary_serials(lhs, rhs, op)?;
    let left_values = tensor::tensor_values_f64_cow(&left);
    let right_values = tensor::tensor_values_f64_cow(&right);
    let out = left_values
        .iter()
        .zip(right_values.iter())
        .map(|(a, b)| if cmp(*a, *b) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.eq",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "eq", |a, b| (a - b).abs() <= 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.ne",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "ne", |a, b| (a - b).abs() > 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.lt",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_lt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "lt", |a, b| a < b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.le",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "le", |a, b| a <= b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.gt",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "gt", |a, b| a > b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.ge",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "ge", |a, b| a >= b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.plus",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_serials = serials_from_datetime_value(&lhs)?;
    if is_calendar_duration_object(&rhs) {
        let (months, days) = calendar_duration_tensors_from_value(&rhs)?;
        let (serials, shape) =
            apply_calendar_duration_to_serials(&lhs_serials, &months, &days, 1.0, "plus")?;
        return datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs));
    }
    let rhs_numeric = if crate::builtins::duration::is_duration_object(&rhs) {
        crate::builtins::duration::duration_tensor_from_duration_value(&rhs)?
    } else {
        serial_tensor_from_value(rhs, "plus")?
    };
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_serials, &rhs_numeric, "plus", BUILTIN_NAME)?;
    let serials = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
}

#[runmat_macros::runtime_builtin(
    name = "datetime.minus",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_serials = serials_from_datetime_value(&lhs)?;
    match &rhs {
        _ if is_calendar_duration_object(&rhs) => {
            let (months, days) = calendar_duration_tensors_from_value(&rhs)?;
            let (serials, shape) =
                apply_calendar_duration_to_serials(&lhs_serials, &months, &days, -1.0, "minus")?;
            datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
        }
        _ if crate::builtins::duration::is_duration_object(&rhs) => {
            let rhs_days = crate::builtins::duration::duration_tensor_from_duration_value(&rhs)?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_days, "minus", BUILTIN_NAME)?;
            let serials = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
        }
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => {
            let rhs_serials = serial_tensor_for_object(obj)?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_serials, "minus", BUILTIN_NAME)?;
            let deltas = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            tensor_or_scalar(deltas, shape)
        }
        _ => {
            let rhs_numeric = serial_tensor_from_value(rhs, "minus")?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_numeric, "minus", BUILTIN_NAME)?;
            let serials = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
        }
    }
}

pub(super) fn combine_calendar_durations(
    lhs: &Value,
    rhs: &Value,
    sign: f64,
    context: &str,
) -> BuiltinResult<Value> {
    let (lhs_months, lhs_days) = calendar_duration_tensors_from_value(lhs)?;
    let (rhs_months, rhs_days) = calendar_duration_tensors_from_value(rhs)?;
    let (left_months, right_months, shape) =
        tensor::binary_numeric_tensors(&lhs_months, &rhs_months, context, BUILTIN_NAME)?;
    let lhs_days_shape =
        tensor::default_shape_for(&lhs_days.shape, tensor::tensor_element_len(&lhs_days));
    let rhs_days_shape =
        tensor::default_shape_for(&rhs_days.shape, tensor::tensor_element_len(&rhs_days));
    let lhs_day_tensor = Tensor::new(tensor::tensor_into_values_f64(lhs_days), lhs_days_shape)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let rhs_day_tensor = Tensor::new(tensor::tensor_into_values_f64(rhs_days), rhs_days_shape)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let (left_days, right_days, day_shape) =
        tensor::binary_numeric_tensors(&lhs_day_tensor, &rhs_day_tensor, context, BUILTIN_NAME)?;
    if day_shape != shape {
        return Err(datetime_error(format!(
            "{context}: calendarDuration operands must have matching component sizes"
        )));
    }
    let months = left_months
        .iter()
        .zip(right_months.iter())
        .map(|(left, right)| left + sign * right)
        .collect::<Vec<_>>();
    let days = left_days
        .iter()
        .zip(right_days.iter())
        .map(|(left, right)| left + sign * right)
        .collect::<Vec<_>>();
    calendar_duration_object_from_components(months, days, shape)
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.plus",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if is_datetime_object(&rhs) {
        let (months, days) = calendar_duration_tensors_from_value(&lhs)?;
        let rhs_serials = serials_from_datetime_value(&rhs)?;
        let (serials, shape) =
            apply_calendar_duration_to_serials(&rhs_serials, &months, &days, 1.0, "plus")?;
        return datetime_object_from_serials(serials, shape, datetime_format_from_value(&rhs));
    }
    combine_calendar_durations(&lhs, &rhs, 1.0, "plus")
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.minus",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    combine_calendar_durations(&lhs, &rhs, -1.0, "minus")
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.eq",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs_months, lhs_days) = calendar_duration_tensors_from_value(&lhs)?;
    let (rhs_months, rhs_days) = calendar_duration_tensors_from_value(&rhs)?;
    let (left_months, right_months, shape) =
        tensor::binary_numeric_tensors(&lhs_months, &rhs_months, "eq", BUILTIN_NAME)?;
    let (left_days, right_days, day_shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, "eq", BUILTIN_NAME)?;
    if day_shape != shape {
        return Err(datetime_error(
            "eq: calendarDuration operands must have matching component sizes",
        ));
    }
    let out = left_months
        .iter()
        .zip(right_months.iter())
        .zip(left_days.iter().zip(right_days.iter()))
        .map(|((lm, rm), (ld, rd))| {
            if (lm - rm).abs() <= 1e-12 && (ld - rd).abs() <= 1e-12 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.ne",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let eq = calendar_duration_eq(lhs, rhs).await?;
    match eq {
        Value::Num(value) => Ok(Value::Num(if value == 0.0 { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor);
            Ok(Value::Tensor(
                Tensor::new(
                    values
                        .into_iter()
                        .map(|value| if value == 0.0 { 1.0 } else { 0.0 })
                        .collect(),
                    shape,
                )
                .map_err(|err| datetime_error(format!("ne: {err}")))?,
            ))
        }
        other => Ok(other),
    }
}
