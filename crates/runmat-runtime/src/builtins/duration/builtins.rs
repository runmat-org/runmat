use super::*;

pub(super) async fn compare_duration(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    if !is_duration_object(&lhs) && !is_duration_object(&rhs) {
        return Err(duration_error(
            DurationErrorKind::InvalidArgument,
            format!("duration.{op}: at least one operand must be a duration"),
        ));
    }
    validate_duration_operand(&lhs, true)?;
    validate_duration_operand(&rhs, true)?;
    preflight_operand_shapes(&lhs, &rhs, op)?;
    let lhs_days = duration_days_operand(&lhs, true).await?;
    let rhs_days = duration_days_operand(&rhs, true).await?;
    let (left, right, shape) = broadcast_duration_days(&lhs_days, &rhs_days).map_err(|err| {
        duration_error_from(
            DurationErrorKind::ShapeMismatch,
            &format!("duration.{op}: "),
            err,
        )
    })?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| u8::from(cmp(*a, *b)))
        .collect::<Vec<_>>();
    if shape.iter().product::<usize>() == 1 {
        Ok(Value::Bool(out[0] != 0))
    } else {
        Ok(Value::LogicalArray(
            runmat_value::LogicalArray::new(out, shape).map_err(|err| {
                duration_error(DurationErrorKind::Internal, format!("duration: {err}"))
            })?,
        ))
    }
}

pub(super) async fn duration_indexing(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<crate::execution::RootedValue> {
    let Value::Object(object) = obj else {
        return Err(duration_error(
            DurationErrorKind::InvalidArgument,
            "duration.subsref: receiver must be a duration object",
        ));
    };
    let format = format_for_object(&object)?;
    let days = duration_tensor_from_duration_value(&Value::Object(object.clone()))?;

    let crate::object::indexing::ObjectIndexSelector::IndexValues { components } = step.selector()
    else {
        return Err(duration_error(
            DurationErrorKind::InvalidIndex,
            "duration.subsref: invalid index selector",
        ));
    };
    if components.is_empty() {
        return crate::execution::RootedValue::new(duration_object_from_days_tensor(
            None, days, format,
        )?)
        .map_err(|error| error.into_runtime_error());
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
        crate::indexing::selectors::build_component_selectors(components, days.shape(), false)
            .await
            .map_err(|err| {
                duration_error_from(DurationErrorKind::InvalidIndex, "duration.subsref: ", err)
            })?;
    let plan = crate::indexing::plan::build_index_plan(&selectors, components.len(), days.shape())
        .map_err(|err| {
            duration_error_from(DurationErrorKind::InvalidIndex, "duration.subsref: ", err)
        })?;
    let indexed =
        crate::indexing::value::read_with_plan(Value::Tensor(days), &plan).map_err(|err| {
            duration_error_from(DurationErrorKind::InvalidIndex, "duration.subsref: ", err)
        })?;
    let indexed_days = tensor_from_numeric(indexed, "duration index")?;
    crate::execution::RootedValue::new(duration_object_from_days_tensor(
        None,
        indexed_days,
        format,
    )?)
    .map_err(|error| error.into_runtime_error())
}

#[runmat_macros::runtime_builtin(
    name = "duration",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration",
    context = true
)]
pub(super) async fn duration_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ensure_duration_class_registered();
    let (raw_positional_end, _) = parse_trailing_format(args.view())?;
    let raw_positional = args.view().range(..raw_positional_end)?;
    if !(1..=4).contains(&raw_positional.len())
        || !raw_positional.iter().all(is_numeric_duration_input)
    {
        return Err(duration_error(
            DurationErrorKind::InvalidArgument,
            "duration: expected a numeric three-column matrix or H/MI/S[/MS] components",
        ));
    }
    if raw_positional.len() > 1 {
        let mut component_shape: Option<&[usize]> = None;
        for value in raw_positional {
            let shape = match value {
                Value::Tensor(tensor) => tensor.shape(),
                Value::LogicalArray(array) => array.shape.as_slice(),
                Value::GpuTensor(handle) => handle.shape(),
                _ => continue,
            };
            let count = shape
                .iter()
                .try_fold(1usize, |count, dim| count.checked_mul(*dim))
                .ok_or_else(|| {
                    duration_error(
                        DurationErrorKind::ShapeMismatch,
                        "duration: component shape is too large",
                    )
                })?;
            if count == 1 {
                continue;
            }
            if let Some(existing) = component_shape {
                if existing != shape {
                    return Err(duration_error(
                        DurationErrorKind::ShapeMismatch,
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
    let short_numeric_form = match raw_positional.len() {
        1 => {
            is_numeric_duration_input(&raw_positional[0])
                && !is_public_duration_matrix(&raw_positional[0])
        }
        2 => {
            is_numeric_duration_input(&raw_positional[0])
                && is_numeric_duration_input(&raw_positional[1])
        }
        _ => false,
    };
    if short_numeric_form {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DURATION_SHORT_COMPONENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let args = gather_retained_args(args, runtime.as_ref()).await?;
    let (positional_end, format) = parse_trailing_format(args.view())?;
    let mut positional = args.into_range(..positional_end)?;

    match positional.len() {
        1 if is_public_duration_matrix(&positional[0]) => build_from_matrix(
            runtime.as_ref(),
            positional.pop_front().expect("validated duration matrix"),
            format,
        ),
        1..=4 if positional.iter().all(is_numeric_duration_input) => {
            build_from_components(runtime.as_ref(), positional, format)
        }
        _ => Err(duration_error(
            DurationErrorKind::InvalidArgument,
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
pub(super) async fn duration_subsref(
    obj: Value,
    subscript: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, duration_read_step, None).await
}

pub(super) async fn duration_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => duration_indexing(obj, step).await,
        crate::object::indexing::ObjectIndexKind::Member => {
            let Value::Object(object) = obj else {
                return Err(duration_error(
                    DurationErrorKind::InvalidArgument,
                    "duration.subsref: receiver must be a duration object",
                ));
            };
            if !object.is_class(DURATION_CLASS) {
                return Err(duration_error(
                    DurationErrorKind::InvalidArgument,
                    "duration.subsref: receiver must be a duration object",
                ));
            }
            let field = scalar_text(&step.selector_value()?, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    crate::execution::RootedValue::new(Value::string(format_for_object(&object)?))
                        .map_err(|error| error.into_runtime_error())
                }
                _ => Err(duration_error(
                    DurationErrorKind::InvalidIndex,
                    format!("duration.subsref: unsupported duration property '{field}'"),
                )),
            }
        }
        crate::object::indexing::ObjectIndexKind::Brace => Err(duration_error(
            DurationErrorKind::InvalidIndex,
            "duration.subsref: brace indexing is not supported",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::duration"
)]
pub(super) async fn duration_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn_with_context(
        runtime.as_ref(),
        obj,
        path,
        vec![rhs],
        duration_read_step,
        |obj, step, mut values| {
            let runtime = runtime.as_ref();
            async move {
                let rhs = values.pop().ok_or_else(|| {
                    duration_error(
                        DurationErrorKind::InvalidArgument,
                        "duration.subsasgn: assignment value is missing",
                    )
                })?;
                duration_write_step(runtime, obj, step, rhs).await
            }
        },
        None,
    )
    .await
}

pub(super) async fn duration_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let payload = step.selector_value()?;
    let Value::Object(mut object) = obj else {
        return Err(duration_error(
            DurationErrorKind::InvalidArgument,
            "duration.subsasgn: receiver must be a duration object",
        ));
    };
    if !object.is_class(DURATION_CLASS) {
        return Err(duration_error(
            DurationErrorKind::InvalidArgument,
            "duration.subsasgn: receiver must be a duration object",
        ));
    }
    let value = match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    let text = scalar_text(&rhs, "Format value")?;
                    validate_duration_format(&text)?;
                    let text = crate::call::argument_text::copy_str(&text, runtime)?;
                    crate::execution::object::replacement::replace_property(
                        runtime,
                        &mut object,
                        FORMAT_FIELD,
                        Value::String(text.into()),
                        |message, source| {
                            crate::context::host_allocation::host_allocation_runtime_error(
                                runtime,
                                "RunMat:duration:PropertyAllocation",
                                message,
                                source,
                            )
                        },
                        |message| duration_error(DurationErrorKind::InvalidArgument, message),
                    )?;
                    Ok(Value::Object(object))
                }
                _ => Err(duration_error(
                    DurationErrorKind::InvalidIndex,
                    format!("duration.subsasgn: unsupported duration property '{field}'"),
                )),
            }
        }
        _ => Err(duration_error(
            DurationErrorKind::InvalidIndex,
            "duration.subsasgn: only member assignment is supported",
        )),
    }?;
    crate::execution::RootedValue::in_optional_context(value, runtime)
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
    builtin_path = "crate::builtins::duration",
    context = true
)]
pub(super) async fn duration_plus(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
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
        let tensor = Tensor::new(serials, shape)
            .map_err(|err| duration_error(DurationErrorKind::Internal, format!("plus: {err}")))?;
        return crate::builtins::datetime::datetime_object_from_serial_tensor(
            runtime.as_ref(),
            tensor,
            crate::builtins::datetime::datetime_format_from_value(&rhs)?,
        );
    }

    let rhs_days = duration_days_operand(&rhs, true).await?;
    let (left, right, shape) = broadcast_duration_days(&lhs_days, &rhs_days)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    duration_object_from_days(
        runtime.as_ref(),
        days,
        shape,
        duration_format_from_value(&lhs)?,
    )
}

#[runmat_macros::runtime_builtin(
    name = "duration.minus",
    binding_variant = "default",
    builtin_path = "crate::builtins::duration",
    context = true
)]
pub(super) async fn duration_minus(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
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
    duration_object_from_days(
        runtime.as_ref(),
        days,
        shape,
        duration_format_from_value(&lhs)?,
    )
}
#[cfg(test)]
pub(crate) async fn duration_builtin_source(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let retained = crate::call::BuiltinArguments::from_borrowed(&args, runtime.as_ref())?;
    duration_builtin(runtime, retained).await
}
