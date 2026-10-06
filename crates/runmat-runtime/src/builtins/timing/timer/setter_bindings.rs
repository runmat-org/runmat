//! Canonical hidden setter bindings carry typed properties into admission.

use super::*;

macro_rules! timer_setter_builtin {
    ($fn_name:ident, $builtin_name:literal, $property:expr) => {
        #[runtime_builtin(name = $builtin_name, binding_variant = "default", builtin_path = "crate::builtins::timing::timer")]
        pub async fn $fn_name(obj: Value, value: Value) -> BuiltinResult<Value> {
            set_timer_property_object(obj, $property, value).await
        }
    };
}

timer_setter_builtin!(
    timer_set_timer_fcn_builtin,
    "set.TimerFcn",
    TimerProperty::TimerFcn
);
timer_setter_builtin!(
    timer_set_start_fcn_builtin,
    "set.StartFcn",
    TimerProperty::StartFcn
);
timer_setter_builtin!(
    timer_set_stop_fcn_builtin,
    "set.StopFcn",
    TimerProperty::StopFcn
);
timer_setter_builtin!(
    timer_set_error_fcn_builtin,
    "set.ErrorFcn",
    TimerProperty::ErrorFcn
);
timer_setter_builtin!(
    timer_set_period_builtin,
    "set.Period",
    TimerProperty::Period
);
timer_setter_builtin!(
    timer_set_start_delay_builtin,
    "set.StartDelay",
    TimerProperty::StartDelay
);
timer_setter_builtin!(
    timer_set_tasks_to_execute_builtin,
    "set.TasksToExecute",
    TimerProperty::TasksToExecute
);
timer_setter_builtin!(
    timer_set_busy_mode_builtin,
    "set.BusyMode",
    TimerProperty::BusyMode
);
timer_setter_builtin!(
    timer_set_execution_mode_builtin,
    "set.ExecutionMode",
    TimerProperty::ExecutionMode
);
timer_setter_builtin!(timer_set_name_builtin, "set.Name", TimerProperty::Name);
timer_setter_builtin!(timer_set_tag_builtin, "set.Tag", TimerProperty::Tag);
timer_setter_builtin!(
    timer_set_object_visibility_builtin,
    "set.ObjectVisibility",
    TimerProperty::ObjectVisibility
);
timer_setter_builtin!(
    timer_set_user_data_builtin,
    "set.UserData",
    TimerProperty::UserData
);

async fn set_timer_property_object(
    obj: Value,
    property: TimerProperty,
    value: Value,
) -> BuiltinResult<Value> {
    let Value::Object(mut object) = obj else {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!(
                "timer: set.{} requires timer object receiver",
                property.name()
            ),
        ));
    };
    if !object.class_name.is(TIMER_CLASS) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!(
                "timer: set.{} requires timer object receiver, got '{}'",
                property.name(),
                object.class_name
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
    let prepared = arguments::prepare_timer_setter_value(property, value).await?;
    apply_name_value_pairs(&mut object, &[prepared], false)?;
    Ok(Value::Object(object))
}
