//! Property-setter identities; property meaning remains in the typed owner.

use super::{contract, documentation, inference};
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

macro_rules! timer_setter_entry {
    ($id:ident, $name:literal, $infer:expr) => {
        timer_entry!(
            $id,
            $name,
            documentation::INTERNAL_SETTER,
            &contract::TIMER_SETTER_DESCRIPTOR,
            $infer,
            ExecutionStackRequirement::Any,
            &[],
            &[],
            false
        );
    };
}

timer_setter_entry!(SET_TIMER_FCN, "set.TimerFcn", inference::set_timer_fcn);
timer_setter_entry!(SET_START_FCN, "set.StartFcn", inference::set_start_fcn);
timer_setter_entry!(SET_STOP_FCN, "set.StopFcn", inference::set_stop_fcn);
timer_setter_entry!(SET_ERROR_FCN, "set.ErrorFcn", inference::set_error_fcn);
timer_setter_entry!(SET_PERIOD, "set.Period", inference::set_period);
timer_setter_entry!(
    SET_START_DELAY,
    "set.StartDelay",
    inference::set_start_delay
);
timer_setter_entry!(
    SET_TASKS_TO_EXECUTE,
    "set.TasksToExecute",
    inference::set_tasks_to_execute
);
timer_setter_entry!(SET_BUSY_MODE, "set.BusyMode", inference::set_busy_mode);
timer_setter_entry!(
    SET_EXECUTION_MODE,
    "set.ExecutionMode",
    inference::set_execution_mode
);
timer_setter_entry!(SET_NAME, "set.Name", inference::set_name);
timer_setter_entry!(SET_TAG, "set.Tag", inference::set_tag);
timer_setter_entry!(
    SET_OBJECT_VISIBILITY,
    "set.ObjectVisibility",
    inference::set_object_visibility
);
timer_setter_entry!(SET_USER_DATA, "set.UserData", inference::set_user_data);
