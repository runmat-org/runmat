//! Session-valid handle access and object presentation updates.

use super::*;

pub(super) fn property(handle: &HandleRef, name: &str) -> BuiltinResult<Value> {
    ensure_valid_timer(handle)?;
    runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        object.properties().get(name).cloned()
    })
    .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?
    .ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: missing property '{name}'"),
        )
    })
}

pub(super) fn set_property(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    name: &str,
    value: Value,
) -> BuiltinResult<()> {
    set_properties(runtime, handle, &[(name, value)])
}

pub(super) fn set_properties(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    updates: &[(&str, Value)],
) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    let snapshot = runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        Some(object.clone())
    })
    .map_err(|error| timer_error(&TIMER_ERROR_GC, format!("timer: {error}")))?
    .ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: handle target is not an object",
        )
    })?;
    let mut replacement = snapshot.clone();
    crate::execution::object::replacement::replace_properties(
        Some(runtime),
        &mut replacement,
        updates.iter().map(|(name, value)| (*name, value)),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:timer:Allocation",
                message,
                source,
            )
        },
        |message| timer_error(&TIMER_ERROR_INVALID_PROPERTY, message),
    )?;
    let replacement = Value::Object(replacement);
    let old = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(live) = target else {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_HANDLE,
                "timer: handle target is not an object",
            ));
        };
        if !live.shares_generation_with(&snapshot) {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                "timer: properties changed before publication",
            ));
        }
        runmat_gc::gc_record_handle_write(&handle.target, &replacement);
        Ok(std::mem::replace(target, replacement))
    })
    .map_err(|error| timer_error(&TIMER_ERROR_GC, format!("timer: {error}")))??;
    // Old children can retire arbitrary owners; never drop them under GC's
    // mutable target borrow.
    drop(old);
    Ok(())
}

pub(super) fn string_property(
    handle: &HandleRef,
    name: &str,
) -> BuiltinResult<runmat_value::Utf8Text> {
    match property(handle, name)? {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text),
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
        object.properties().get("Running"),
        Some(Value::String(runmat_value::StringElement::Text(text))) if text.eq_ignore_ascii_case("on")
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
