//! Session-valid handle access and object presentation updates.

use super::*;

pub(super) fn property(handle: &HandleRef, name: &str) -> BuiltinResult<Value> {
    ensure_valid_timer(handle)?;
    runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        object.properties.get(name).cloned()
    })
    .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?
    .ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: missing property '{name}'"),
        )
    })
}

pub(super) fn set_property(handle: &HandleRef, name: &str, value: Value) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    let updated = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(object) = target else {
            return false;
        };
        object.properties.insert(name.to_string(), value);
        true
    })
    .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    if updated {
        Ok(())
    } else {
        Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: handle target is not an object",
        ))
    }
}

pub(super) fn string_property(handle: &HandleRef, name: &str) -> BuiltinResult<String> {
    match property(handle, name)? {
        Value::String(text) => Ok(text),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{name}' must be text, got {other:?}"),
        )),
    }
}

pub(super) fn numeric_property(handle: &HandleRef, name: &str) -> BuiltinResult<f64> {
    numeric_scalar(&property(handle, name)?, name)
}

pub(super) fn running_is_on(object: &ObjectInstance) -> bool {
    matches!(
        object.properties.get("Running"),
        Some(Value::String(text)) if text.eq_ignore_ascii_case("on")
    )
}

pub(super) fn is_timer_handle(handle: &HandleRef) -> bool {
    handle.class_name.is(TIMER_CLASS)
}

pub(super) fn ensure_valid_timer(handle: &HandleRef) -> BuiltinResult<()> {
    if is_timer_handle(handle) && crate::is_handle_valid(handle) {
        Ok(())
    } else {
        Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: invalid or deleted timer handle",
        ))
    }
}
