use super::*;

pub(super) async fn compare_duration(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    if !is_duration_object(&lhs) && !is_duration_object(&rhs) {
        return Err(duration_error(format!(
            "duration.{op}: at least one operand must be a duration"
        )));
    }
    validate_duration_operand(&lhs, true)?;
    validate_duration_operand(&rhs, true)?;
    preflight_operand_shapes(&lhs, &rhs, op)?;
    let lhs_days = duration_days_operand(&lhs, true).await?;
    let rhs_days = duration_days_operand(&rhs, true).await?;
    let (left, right, shape) = broadcast_duration_days(&lhs_days, &rhs_days)
        .map_err(|err| duration_error(format!("duration.{op}: {}", err.message())))?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| u8::from(cmp(*a, *b)))
        .collect::<Vec<_>>();
    if shape.iter().product::<usize>() == 1 {
        Ok(Value::Bool(out[0] != 0))
    } else {
        Ok(Value::LogicalArray(
            runmat_value::LogicalArray::new(out, shape)
                .map_err(|err| duration_error(format!("duration: {err}")))?,
        ))
    }
}

pub(super) async fn duration_indexing(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<Value> {
    let Value::Object(object) = obj else {
        return Err(duration_error(
            "duration.subsref: receiver must be a duration object",
        ));
    };
    let format = format_for_object(&object);
    let days = duration_tensor_from_duration_value(&Value::Object(object.clone()))?;

    let crate::object::indexing::ObjectIndexSelector::IndexValues { components } = step.selector()
    else {
        return Err(duration_error("duration.subsref: invalid index selector"));
    };
    if components.is_empty() {
        return duration_object_from_days_tensor(days, format);
    }
    for component in components {
        if let crate::object::indexing::ObjectIndexComponent::Value(value) = component {
            if crate::value_contains_gpu(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &DURATION_GPU_INPUT_EXTENSION,
                    "duration.subsref",
                )?;
            }
        }
    }
    let selectors =
        crate::indexing::selectors::build_component_selectors(components, &days.shape, false)
            .await
            .map_err(|err| duration_error(format!("duration.subsref: {}", err.message())))?;
    let plan = crate::indexing::plan::build_index_plan(&selectors, components.len(), &days.shape)
        .map_err(|err| duration_error(format!("duration.subsref: {}", err.message())))?;
    let indexed = crate::indexing::value::read_with_plan(Value::Tensor(days), &plan)
        .map_err(|err| duration_error(format!("duration.subsref: {}", err.message())))?;
    let indexed_days = tensor_from_numeric(indexed, "duration index")?;
    duration_object_from_days_tensor(indexed_days, format)
}

#[runmat_macros::runtime_builtin(
    name = "duration",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_duration_class_registered();
    let (raw_positional_end, _) = parse_trailing_format(&args)?;
    let raw_positional = &args[..raw_positional_end];
    if !(1..=4).contains(&raw_positional.len())
        || !raw_positional.iter().all(is_numeric_duration_input)
    {
        return Err(duration_error(
            "duration: expected a numeric three-column matrix or H/MI/S[/MS] components",
        ));
    }
    if raw_positional.len() > 1 {
        let mut component_shape: Option<&[usize]> = None;
        for value in raw_positional {
            let shape = match value {
                Value::Tensor(tensor) => tensor.shape.as_slice(),
                Value::LogicalArray(array) => array.shape.as_slice(),
                Value::GpuTensor(handle) => handle.shape(),
                _ => continue,
            };
            let count = shape
                .iter()
                .try_fold(1usize, |count, dim| count.checked_mul(*dim))
                .ok_or_else(|| duration_error("duration: component shape is too large"))?;
            if count == 1 {
                continue;
            }
            if let Some(existing) = component_shape {
                if existing != shape {
                    return Err(duration_error(
                        "duration: non-scalar component inputs must have matching sizes",
                    ));
                }
            } else {
                component_shape = Some(shape);
            }
        }
    }
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
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn days_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "days", 1.0).await
}

#[runmat_macros::runtime_builtin(
    name = "hours",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn hours_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "hours", 1.0 / 24.0).await
}

#[runmat_macros::runtime_builtin(
    name = "minutes",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn minutes_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "minutes", 1.0 / (24.0 * 60.0)).await
}

#[runmat_macros::runtime_builtin(
    name = "seconds",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn seconds_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "seconds", 1.0 / SECONDS_PER_DAY).await
}

#[runmat_macros::runtime_builtin(
    name = "milliseconds",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn milliseconds_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "milliseconds", 1.0 / (SECONDS_PER_DAY * 1000.0)).await
}

#[runmat_macros::runtime_builtin(
    name = "years",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn years_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "years", 365.2425).await
}

#[runmat_macros::runtime_builtin(
    name = "isduration",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) fn isduration_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_duration_object(&value)))
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsref",
    binding_variant = "default",
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
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => duration_indexing(obj, step).await,
        crate::object::indexing::ObjectIndexKind::Member => {
            let Value::Object(object) = obj else {
                return Err(duration_error(
                    "duration.subsref: receiver must be a duration object",
                ));
            };
            if !object.is_class(DURATION_CLASS) {
                return Err(duration_error(
                    "duration.subsref: receiver must be a duration object",
                ));
            }
            let field = scalar_text(&step.selector_value()?, "field selector")?;
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
    binding_variant = "default",
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
    if !object.is_class(DURATION_CLASS) {
        return Err(duration_error(
            "duration.subsasgn: receiver must be a duration object",
        ));
    }
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    let text = scalar_text(&rhs, "Format value")?;
                    validate_duration_format(&text)?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "eq", |a, b| a == b).await
}

#[runmat_macros::runtime_builtin(
    name = "duration.ne",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ne", |a, b| a != b).await
}

#[runmat_macros::runtime_builtin(
    name = "duration.lt",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_lt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "lt", |a, b| a < b).await
}

#[runmat_macros::runtime_builtin(
    name = "duration.le",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "le", |a, b| a <= b).await
}

#[runmat_macros::runtime_builtin(
    name = "duration.gt",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "gt", |a, b| a > b).await
}

#[runmat_macros::runtime_builtin(
    name = "duration.ge",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ge", |a, b| a >= b).await
}

#[runmat_macros::runtime_builtin(
    name = "duration.plus",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    validate_duration_operand(&lhs, false)?;
    let rhs_serials = if crate::builtins::datetime::is_datetime_object(&rhs) {
        Some(crate::builtins::datetime::serials_from_datetime_value(
            &rhs,
        )?)
    } else {
        validate_duration_operand(&rhs, true)?;
        None
    };
    preflight_operand_shapes(&lhs, &rhs, "plus")?;
    let lhs_days = duration_days_operand(&lhs, false).await?;
    if let Some(rhs_serials) = rhs_serials {
        let (left, right, shape) = broadcast_duration_days(&lhs_days, &rhs_serials)?;
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

    let rhs_days = duration_days_operand(&rhs, true).await?;
    let (left, right, shape) = broadcast_duration_days(&lhs_days, &rhs_days)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}

#[runmat_macros::runtime_builtin(
    name = "duration.minus",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    validate_duration_operand(&lhs, false)?;
    validate_duration_operand(&rhs, true)?;
    preflight_operand_shapes(&lhs, &rhs, "minus")?;
    let lhs_days = duration_days_operand(&lhs, false).await?;
    let rhs_days = duration_days_operand(&rhs, true).await?;
    let (left, right, shape) = broadcast_duration_days(&lhs_days, &rhs_days)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}
