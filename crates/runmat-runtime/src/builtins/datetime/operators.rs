use super::capabilities::{BUILTIN_NAME, FORMAT_FIELD};
use super::*;
use crate::execution::RootedValue;

#[runmat_macros::runtime_builtin(
    name = "datetick",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetick_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    remaining::tick(runtime.as_ref(), args).await
}

#[runmat_macros::runtime_builtin(
    name = "datetime.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_subsref(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> crate::BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref_with_context(
        runtime.as_ref(),
        obj,
        path,
        |obj, step| datetime_read_step(runtime.as_ref(), obj, step),
        None,
    )
    .await
}

pub(super) async fn datetime_read_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => {
            datetime_indexing(runtime, obj, payload).await
        }
        crate::object::indexing::ObjectIndexKind::Member => {
            let Value::Object(object) = obj else {
                return Err(datetime_error(
                    "datetime.subsref: receiver must be a datetime object",
                ));
            };
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => RootedValue::new(Value::string(format_for_object(&object)?))
                    .map_err(|error| error.into_runtime_error()),
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let runtime_ref = runtime.as_ref();
    crate::object::protocol::execute_owned_subsasgn_with_context(
        runtime_ref,
        obj,
        path,
        vec![rhs],
        |obj, step| datetime_read_step(runtime_ref, obj, step),
        |obj, step, mut values| {
            let runtime = runtime_ref;
            async move {
                let rhs = values.pop().ok_or_else(|| {
                    datetime_error("datetime.subsasgn: assignment value is missing")
                })?;
                datetime_write_step(runtime, obj, step, rhs).await
            }
        },
        None,
    )
    .await
}

pub(super) async fn datetime_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    let Value::Object(mut object) = obj else {
        return Err(datetime_error(
            "datetime.subsasgn: receiver must be a datetime object",
        ));
    };
    let value = match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    let text = scalar_text(&rhs, "Format value")?;
                    let text = crate::call::argument_text::copy_str(&text, runtime)?;
                    crate::execution::object::replacement::replace_property(
                        runtime,
                        &mut object,
                        FORMAT_FIELD,
                        Value::String(text.into()),
                        |message, source| {
                            crate::context::host_allocation::host_allocation_runtime_error(
                                runtime,
                                "RunMat:datetime:PropertyAllocation",
                                message,
                                source,
                            )
                        },
                        datetime_error,
                    )?;
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
    }?;
    RootedValue::in_optional_context(value, runtime)
}

#[path = "operators/binary.rs"]
mod binary;

// Shared stored-component comparison: exact infinities compare equal, NaN
// never does, and finite components retain the existing calendar tolerance.
fn calendar_component_equal(left: f64, right: f64) -> bool {
    left == right || (left - right).abs() <= 1e-12
}
#[runmat_macros::runtime_builtin(
    name = "datetime.eq",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_eq(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Eq,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.ne",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_ne(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Ne,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.lt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_lt(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Lt,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.le",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_le(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Le,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.gt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_gt(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Gt,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.ge",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_ge(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Ge,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.plus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_plus(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Plus,
    )
}

#[runmat_macros::runtime_builtin(
    name = "datetime.minus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_minus(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    binary::execute(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::DatetimeBinaryOperation::Minus,
    )
}

pub(super) fn combine_calendar_durations(
    runtime: Option<&crate::context::RuntimeContext>,
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
        tensor::default_shape_for(lhs_days.shape(), tensor::tensor_element_len(&lhs_days));
    let rhs_days_shape =
        tensor::default_shape_for(rhs_days.shape(), tensor::tensor_element_len(&rhs_days));
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
    calendar_duration_object_from_components(runtime, months, days, shape)
}

#[path = "operators/remaining.rs"]
pub(super) mod remaining;

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.plus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_plus(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    remaining::calendar(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::CalendarResidualOperation::Plus,
    )
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.minus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_minus(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    remaining::calendar(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::CalendarResidualOperation::Minus,
    )
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.eq",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_eq(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    remaining::calendar(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::CalendarResidualOperation::Eq,
    )
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.ne",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calendar_duration_ne(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    remaining::calendar(
        runtime.as_ref(),
        args.view(),
        runmat_builtins::CalendarResidualOperation::Ne,
    )
}
