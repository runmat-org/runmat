//! Typed timer setter admission and in-place handle publication.

use super::*;

pub(super) async fn set_timer_property(
    runtime: Option<RuntimeContext>,
    receiver: Value,
    property: TimerProperty,
    value: Value,
) -> BuiltinResult<Value> {
    let Value::HandleObject(handle) = &receiver else {
        return setter_bindings::set_timer_property_object(receiver, property, value).await;
    };
    let runtime = timer_runtime(runtime)?;
    admit_handle(&runtime, handle, property)?;
    let _roots = crate::execution::RootedValueHandles::for_values([&receiver, &value].into_iter())
        .map_err(|error| {
            crate::build_runtime_error("cannot retain timer setter arguments")
                .with_source(error)
                .build()
        })?;
    let prepared = arguments::prepare_timer_setter_value(property, value).await?;
    let normalized = normalize_property_value(property, prepared.value)?;
    // Preparation may suspend or reenter the host. Validate the live handle
    // and its current running policy before publishing only this property.
    admit_handle(&runtime, handle, property)?;
    let old = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(object) = target else {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_HANDLE,
                "timer: handle target is not an object",
            ));
        };
        runmat_gc::gc_record_handle_write(&handle.target, &normalized);
        Ok(object
            .properties
            .insert(property.name().to_string(), normalized))
    })
    .map_err(|error| {
        crate::build_runtime_error(format!("timer: cannot update handle: {error}"))
            .with_source(error)
            .build()
    })??;
    // Destruction can reenter other owners; the managed borrow has ended.
    drop(old);
    Ok(receiver)
}

fn admit_handle(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    property: TimerProperty,
) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    if runtime.timer_session().borrow().record(handle).is_none() {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: handle belongs to another runtime session",
        ));
    }
    if property.frozen_while_running()
        && string_property(handle, "Running")?.eq_ignore_ascii_case("on")
    {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!(
                "timer: property '{}' cannot be changed while Running is on",
                property.name()
            ),
        ));
    }
    Ok(())
}
