use super::*;

pub(super) fn compare_duration(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, op, BUILTIN_NAME)?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| if cmp(*a, *b) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        Ok(Value::Tensor(Tensor::new(out, shape).map_err(|err| {
            duration_error(format!("duration: {err}"))
        })?))
    }
}

pub(super) async fn duration_indexing(obj: Value, payload: Value) -> BuiltinResult<Value> {
    let Value::Object(object) = obj else {
        return Err(duration_error(
            "duration.subsref: receiver must be a duration object",
        ));
    };
    let format = format_for_object(&object);
    let days = duration_tensor_from_duration_value(&Value::Object(object.clone()))?;

    let Value::Cell(cell) = payload else {
        return Err(duration_error(
            "duration.subsref: indexing payload must be a cell array",
        ));
    };
    if cell.data.is_empty() {
        return duration_object_from_days_tensor(days, format);
    }
    if cell.data.len() != 1 {
        return Err(duration_error(
            "duration.subsref: only linear duration indexing is currently supported",
        ));
    }
    let selector = cell.data[0].clone();
    let selector = match selector {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::Int(value) => {
            Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|err| duration_error(format!("duration.subsref: {err}")))?
        }
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        other => {
            return Err(duration_error(format!(
                "duration.subsref: unsupported index value {other:?}"
            )))
        }
    };
    let indexed =
        crate::perform_indexing(&Value::Tensor(days), &tensor::tensor_values_f64(&selector))
            .await
            .map_err(|err| duration_error(format!("duration.subsref: {}", err.message())))?;
    let indexed_days = match indexed {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::Tensor(tensor) => tensor,
        other => {
            return Err(duration_error(format!(
                "duration.subsref: unexpected indexing result {other:?}"
            )))
        }
    };
    duration_object_from_days_tensor(indexed_days, format)
}

#[runmat_macros::runtime_builtin(
    name = "duration",
    descriptor(crate::builtins::duration::DURATION_DESCRIPTOR),
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration arrays from hour, minute, and second components.",
    keywords = "duration,time span,elapsed time,Format",
    related = "datetime,string,char,disp",
    examples = "t = duration(1, 30, 45);",
    extensions(DURATION_EXTENSIONS),
    integer_capabilities(DURATION_INTEGER_CAPABILITIES)
)]
pub(super) async fn duration_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_duration_class_registered();
    let (raw_positional_end, _) = parse_trailing_format(&args)?;
    let raw_positional = &args[..raw_positional_end];
    if args.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DURATION_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let short_numeric_form = match raw_positional {
        [value] => is_numeric_duration_input(value) && !is_public_duration_matrix(value),
        [first, second] => is_numeric_duration_input(first) && is_numeric_duration_input(second),
        _ => false,
    };
    if short_numeric_form {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DURATION_SHORT_COMPONENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let args = gather_args(&args).await?;
    let (positional_end, format) = parse_trailing_format(&args)?;
    let mut positional = args[..positional_end].to_vec();

    match positional.len() {
        1 if is_public_duration_matrix(&positional[0]) => {
            build_from_matrix(positional.remove(0), format)
        }
        1..=4 if positional.iter().all(is_numeric_duration_input) => {
            build_from_components(positional, format)
        }
        _ => Err(duration_error(
            "duration: unsupported argument pattern; use X or H/MI/S/MS numeric inputs",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "days",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from days or convert duration values to day counts.",
    keywords = "days,duration,datetime"
)]
pub(super) async fn days_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "days", 1.0).await
}

#[runmat_macros::runtime_builtin(
    name = "hours",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from hours or convert duration values to hour counts.",
    keywords = "hours,duration,datetime"
)]
pub(super) async fn hours_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "hours", 1.0 / 24.0).await
}

#[runmat_macros::runtime_builtin(
    name = "minutes",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from minutes or convert duration values to minute counts.",
    keywords = "minutes,duration,datetime"
)]
pub(super) async fn minutes_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "minutes", 1.0 / (24.0 * 60.0)).await
}

#[runmat_macros::runtime_builtin(
    name = "seconds",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from seconds or convert duration values to second counts.",
    keywords = "seconds,duration,datetime"
)]
pub(super) async fn seconds_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "seconds", 1.0 / SECONDS_PER_DAY).await
}

#[runmat_macros::runtime_builtin(
    name = "milliseconds",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from milliseconds or convert duration values to millisecond counts.",
    keywords = "milliseconds,duration,datetime"
)]
pub(super) async fn milliseconds_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "milliseconds", 1.0 / (SECONDS_PER_DAY * 1000.0)).await
}

#[runmat_macros::runtime_builtin(
    name = "years",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create fixed-length duration values from years or convert durations to fixed-length years.",
    keywords = "years,duration,datetime"
)]
pub(super) async fn years_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "years", 365.2425).await
}

#[runmat_macros::runtime_builtin(
    name = "isduration",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Return true for duration values.",
    keywords = "isduration,duration,predicate"
)]
pub(super) fn isduration_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_duration_object(&value)))
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsref",
    descriptor(crate::builtins::duration::DURATION_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_subsref(obj: Value, subscript: Value) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, duration_read_step, None).await
}

pub(super) async fn duration_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<Value> {
    let payload = step.selector_value()?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => duration_indexing(obj, payload).await,
        crate::object::indexing::ObjectIndexKind::Member => {
            let Value::Object(object) = obj else {
                return Err(duration_error(
                    "duration.subsref: receiver must be a duration object",
                ));
            };
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => Ok(Value::String(format_for_object(&object))),
                _ => Err(duration_error(format!(
                    "duration.subsref: unsupported duration property '{field}'"
                ))),
            }
        }
        crate::object::indexing::ObjectIndexKind::Brace => Err(duration_error(
            "duration.subsref: brace indexing is not supported",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsasgn",
    descriptor(crate::builtins::duration::DURATION_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        duration_read_step,
        |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| duration_error("duration.subsasgn: assignment value is missing"))?;
            duration_write_step(obj, step, rhs).await
        },
        None,
    )
    .await
}

pub(super) async fn duration_write_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let Value::Object(mut object) = obj else {
        return Err(duration_error(
            "duration.subsasgn: receiver must be a duration object",
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
                _ => Err(duration_error(format!(
                    "duration.subsasgn: unsupported duration property '{field}'"
                ))),
            }
        }
        _ => Err(duration_error(
            "duration.subsasgn: only member assignment is supported",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.eq",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "eq", |a, b| (a - b).abs() <= 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "duration.ne",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ne", |a, b| (a - b).abs() > 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "duration.lt",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_lt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "lt", |a, b| a < b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.le",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "le", |a, b| a <= b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.gt",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "gt", |a, b| a > b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.ge",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ge", |a, b| a >= b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.plus",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    if crate::builtins::datetime::is_datetime_object(&rhs) {
        let rhs_serials = crate::builtins::datetime::serials_from_datetime_value(&rhs)?;
        let (left, right, shape) =
            tensor::binary_numeric_tensors(&lhs_days, &rhs_serials, "plus", BUILTIN_NAME)?;
        let serials = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let tensor =
            Tensor::new(serials, shape).map_err(|err| duration_error(format!("plus: {err}")))?;
        return crate::builtins::datetime::datetime_object_from_serial_tensor(
            tensor,
            crate::builtins::datetime::datetime_format_from_value(&rhs),
        );
    }

    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, "plus", BUILTIN_NAME)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}

#[runmat_macros::runtime_builtin(
    name = "duration.minus",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, "minus", BUILTIN_NAME)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}
