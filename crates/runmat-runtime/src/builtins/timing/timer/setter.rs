//! Typed timer setter admission and in-place handle publication.

use super::*;

pub(super) async fn set_timer_property(
    runtime: Option<RuntimeContext>,
    receiver: Value,
    property: TimerProperty,
    value: Value,
) -> BuiltinResult<Value> {
    let Value::HandleObject(handle) = &receiver else {
        return set_timer_property_object(runtime, receiver, property, value).await;
    };
    let runtime = timer_runtime(runtime)?;
    admit_handle(&runtime, handle, property)?;
    let _roots = crate::execution::RootedValueHandles::for_values([&receiver, &value].into_iter())
        .map_err(|error| {
            crate::build_runtime_error("cannot retain timer setter arguments")
                .with_source(error)
                .build()
        })?;
    let prepared = arguments::prepare_timer_setter_value(Some(&runtime), property, value).await?;
    let normalized = normalize_property_value(property, prepared.value)?;
    // Preparation may suspend or reenter the host. Validate the live handle
    // and its current running policy before publishing only this property.
    admit_handle(&runtime, handle, property)?;
    set_property(&runtime, handle, property.name(), normalized)?;
    Ok(receiver)
}

pub(super) async fn set_timer_property_object(
    runtime: Option<RuntimeContext>,
    obj: Value,
    property: TimerProperty,
    value: Value,
) -> BuiltinResult<Value> {
    let _roots = crate::execution::RootedValueHandles::for_values([&obj, &value].into_iter())
        .map_err(|error| {
            crate::build_runtime_error("cannot retain timer setter arguments")
                .with_source(error)
                .build()
        })?;
    let Value::Object(mut object) = obj else {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!(
                "timer: set.{} requires timer object receiver",
                property.name()
            ),
        ));
    };
    if !object.class_name().is(TIMER_CLASS) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!(
                "timer: set.{} requires timer object receiver, got '{}'",
                property.name(),
                object.class_name()
            ),
        ));
    }
    if running_is_on(&object) && property.frozen_while_running() {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!(
                "timer: property '{}' cannot be changed while Running is on",
                property.name()
            ),
        ));
    }
    let prepared = arguments::prepare_timer_setter_value(runtime.as_ref(), property, value).await?;
    apply_name_value_pairs(runtime.as_ref(), &mut object, &[prepared], false)?;
    Ok(Value::Object(object))
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
