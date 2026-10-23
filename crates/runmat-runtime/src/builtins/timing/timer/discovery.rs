//! Session-scoped timer discovery and exact property comparison.

use super::*;
use runmat_value::{CellArray, NumericScalar};

pub(super) fn find_timers(
    runtime: &RuntimeContext,
    args: Vec<PreparedTimerPair>,
    include_invisible: bool,
) -> BuiltinResult<Value> {
    ensure_timer_class_registered();
    let filters: Vec<_> = args
        .into_iter()
        .map(|pair| normalize_find_value(pair.value).map(|value| (pair.property, value)))
        .collect::<BuiltinResult<_>>()?;
    let mut matches = Vec::new();
    let handles = runtime.timer_session().borrow().handles();
    for handle in handles
        .iter()
        .filter(|handle| crate::is_handle_valid(handle))
    {
        let visibility = string_property(handle, "ObjectVisibility")
            .ok()
            .and_then(|value| TimerVisibility::parse(&value));
        if let Some(visibility) = visibility {
            if let Some(entry) = runtime.timer_session().borrow_mut().record_mut(handle) {
                entry.visibility = visibility;
            }
        }
        let admitted_visibility = runtime
            .timer_session()
            .borrow()
            .record(handle)
            .map(|entry| entry.visibility);
        if !include_invisible && admitted_visibility != Some(TimerVisibility::On) {
            continue;
        }
        if filters_match(handle, &filters).unwrap_or(false) {
            matches.push(Value::HandleObject(handle.clone()));
        }
    }
    handles_to_result(matches)
}

fn normalize_find_value(value: Value) -> BuiltinResult<Value> {
    if let Some(integer) = tensor::scalar_integer_value(&value) {
        return Ok(Value::Int(integer));
    }
    match value {
        Value::CharArray(chars) if chars.rows == 1 => {
            crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: find value is not valid UTF-16",
                )
            })
            .map(Value::string)
        }
        other => Ok(other),
    }
}

fn filters_match(handle: &HandleRef, filters: &[(TimerProperty, Value)]) -> BuiltinResult<bool> {
    for (name, expected) in filters {
        let actual = normalize_find_value(property(handle, name.name())?)?;
        if !timer_values_equal(&actual, expected) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn timer_values_equal(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Num(a), Value::Num(b)) if a.is_nan() && b.is_nan() => true,
        (Value::Num(a), Value::Num(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Int(a), Value::Num(b)) | (Value::Num(b), Value::Int(a)) => {
            crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact(
                NumericScalar::from(a.clone()),
                NumericScalar::F64(*b),
            ) == Some(std::cmp::Ordering::Equal)
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => lhs == rhs,
    }
}

fn handles_to_result(handles: Vec<Value>) -> BuiltinResult<Value> {
    if handles.len() == 1 {
        Ok(handles.into_iter().next().unwrap())
    } else {
        let len = handles.len();
        CellArray::new(handles, 1, len)
            .map(Value::Cell)
            .map_err(|err| timer_error(&TIMER_ERROR_INVALID_INPUT, err))
    }
}
