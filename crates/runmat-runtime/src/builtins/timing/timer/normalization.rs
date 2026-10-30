//! Typed public-property decoding and value normalization.

use super::*;

pub(super) fn canonical_property_name(name: &str) -> BuiltinResult<TimerProperty> {
    let trimmed = name.trim();
    TimerProperty::parse(trimmed).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: unknown property '{trimmed}'"),
        )
    })
}

pub(super) fn normalize_property_value(name: TimerProperty, value: Value) -> BuiltinResult<Value> {
    match name {
        TimerProperty::Period => {
            let seconds = numeric_scalar(&value, name.name())?;
            if !seconds.is_finite() || seconds <= 0.001 {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: Period must be a finite scalar greater than 0.001",
                ));
            }
            Ok(Value::Num(seconds))
        }
        TimerProperty::StartDelay => {
            let seconds = numeric_scalar(&value, name.name())?;
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: StartDelay must be a finite non-negative scalar",
                ));
            }
            Ok(Value::Num(seconds))
        }
        TimerProperty::TasksToExecute => {
            parse_tasks_to_execute_value(&value)?;
            if let Some(integer) = tensor::scalar_integer_value(&value) {
                Ok(Value::Int(integer))
            } else {
                Ok(Value::Num(numeric_scalar(&value, name.name())?.round()))
            }
        }
        TimerProperty::BusyMode => {
            parse_choice(&value, name, TimerBusyMode::parse, TimerBusyMode::spelling)
        }
        TimerProperty::ExecutionMode => parse_choice(
            &value,
            name,
            TimerExecutionMode::parse,
            TimerExecutionMode::spelling,
        ),
        TimerProperty::ObjectVisibility => parse_choice(
            &value,
            name,
            TimerVisibility::parse,
            TimerVisibility::spelling,
        ),
        TimerProperty::Name | TimerProperty::Tag => Ok(Value::String(
            runmat_value::StringElement::Text(value_to_string(&value)?),
        )),
        TimerProperty::TimerFcn
        | TimerProperty::StartFcn
        | TimerProperty::StopFcn
        | TimerProperty::ErrorFcn => normalize_callback(value, name.name()),
        TimerProperty::UserData => Ok(value),
        _ => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{}' cannot be assigned", name.name()),
        )),
    }
}

fn normalize_callback(value: Value, name: &str) -> BuiltinResult<Value> {
    match value {
        Value::String(_)
        | Value::CharArray(runmat_value::CharArray { rows: 1, .. })
        | Value::ScopedFunctionHandle(_) | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => Ok(value),
        Value::Cell(ref cell)
            if cell.data().is_empty() || cell.data().first().is_some_and(is_callback_function) =>
        {
            Ok(value)
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!(
                "timer: {name} must be text, a function handle, or a callback cell array, got {other:?}"
            ),
        )),
    }
}

fn is_callback_function(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

fn parse_choice<T: Copy>(
    value: &Value,
    name: TimerProperty,
    parse: fn(&str) -> Option<T>,
    spelling: fn(T) -> &'static str,
) -> BuiltinResult<Value> {
    let text = value_to_string(value)?;
    parse(&text)
        .map(|choice| Value::string(spelling(choice).to_string()))
        .ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: invalid {} value '{text}'", name.name()),
            )
        })
}

pub(super) fn value_to_string(value: &Value) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => {
            crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: property text is not valid UTF-16",
                )
            })
            .map(Into::into)
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("timer: expected string scalar or character row, got {other:?}"),
        )),
    }
}

pub(super) fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let exact = match integer {
            IntValue::I8(value) => i128::from(value),
            IntValue::I16(value) => i128::from(value),
            IntValue::I32(value) => i128::from(value),
            IntValue::I64(value) => i128::from(value),
            IntValue::U8(value) => i128::from(value),
            IntValue::U16(value) => i128::from(value),
            IntValue::U32(value) => i128::from(value),
            IntValue::U64(value) => i128::from(value),
        };
        const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: {name} must be exactly representable as double seconds"),
            ));
        }
        return Ok(exact as f64);
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            Ok(tensor::tensor_value_f64(value, 0))
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("timer: {name} must be a numeric scalar, got {other:?}"),
        )),
    }
}

pub(super) fn parse_tasks_to_execute_value(value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_tasks_to_execute_integer(&integer);
    }
    let count = numeric_scalar(value, "TasksToExecute")?;
    if !count.is_finite() || count < 1.0 || count.fract() != 0.0 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        ));
    }
    if !fits_platform_usize(count) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute exceeds platform limits",
        ));
    }
    Ok(count as usize)
}

fn parse_tasks_to_execute_integer(value: &IntValue) -> BuiltinResult<usize> {
    let count = value.try_to_usize().ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        )
    })?;
    if count == 0 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        ));
    }
    Ok(count)
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}
