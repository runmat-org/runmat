//! Canonical hidden setter bindings carry typed properties into admission.

use super::*;

macro_rules! timer_setter_builtin {
    ($fn_name:ident, $builtin_name:literal, $property:expr) => {
        #[runtime_builtin(name = $builtin_name, binding_variant = "default", context = true, builtin_path = "crate::builtins::timing::timer")]
        pub async fn $fn_name(runtime: Option<RuntimeContext>, obj: Value, value: Value) -> BuiltinResult<Value> {
            super::setter::set_timer_property(runtime, obj, $property, value).await
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
