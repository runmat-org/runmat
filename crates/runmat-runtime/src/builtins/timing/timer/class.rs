//! Timer class schema and constructor presentation defaults.

use super::*;
use runmat_types::MemberAccess;
use std::collections::HashMap;

pub(super) const TIMER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("timer");
const TIMER_CALLABLE_START: &str = "__runmat_timer_start";
const TIMER_CALLABLE_STARTAT: &str = "__runmat_timer_startat";
const TIMER_CALLABLE_STOP: &str = "__runmat_timer_stop";
const TIMER_CALLABLE_WAIT: &str = "__runmat_timer_wait";
const TIMER_CALLABLE_DELETE: &str = "__runmat_timer_delete";

pub(super) const CALLBACK_PROPS: [&str; 4] = ["TimerFcn", "StartFcn", "StopFcn", "ErrorFcn"];

pub(super) fn ensure_timer_class_registered() {
    if crate::class_registry::get_class(&TIMER_CLASS.owned()).is_some() {
        return;
    }

    let mut methods = HashMap::new();
    for (name, function_name) in [
        (
            runmat_types::StaticMethodName::new("start"),
            TIMER_CALLABLE_START,
        ),
        (
            runmat_types::StaticMethodName::new("startat"),
            TIMER_CALLABLE_STARTAT,
        ),
        (
            runmat_types::StaticMethodName::new("stop"),
            TIMER_CALLABLE_STOP,
        ),
        (
            runmat_types::StaticMethodName::new("wait"),
            TIMER_CALLABLE_WAIT,
        ),
        (
            runmat_types::StaticMethodName::new("delete"),
            TIMER_CALLABLE_DELETE,
        ),
    ] {
        methods.insert(
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
            },
        );
    }

    let mut properties = HashMap::new();
    for property in TimerProperty::ALL {
        let name = property.name();
        properties.insert(
            name.into(),
            crate::class_registry::RuntimeProperty {
                name: name.into(),
                is_static: false,
                is_constant: false,
                is_dependent: !property.is_readonly(),
                get_access: MemberAccess::Public,
                set_access: if property.is_readonly() {
                    MemberAccess::Private
                } else {
                    MemberAccess::Public
                },
                default_value: None,
            },
        );
    }

    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: TIMER_CLASS.into(),
        parent: Some(runmat_types::standard::HANDLE.into()),
        properties,
        methods,
    });
}

pub(super) fn default_timer_object(name: String) -> ObjectInstance {
    let mut object = ObjectInstance::new(TIMER_CLASS.to_string());
    for prop in CALLBACK_PROPS {
        object
            .properties
            .insert(prop.to_string(), Value::String(String::new()));
    }
    for (property, value) in [
        ("Period", Value::Num(1.0)),
        ("StartDelay", Value::Num(0.0)),
        ("TasksToExecute", Value::Num(1.0)),
        ("BusyMode", Value::String("drop".into())),
        ("ExecutionMode", Value::String("singleShot".into())),
        ("Name", Value::String(name)),
        ("Tag", Value::String(String::new())),
        ("ObjectVisibility", Value::String("on".into())),
        ("UserData", Value::Num(0.0)),
        ("AveragePeriod", Value::Num(f64::NAN)),
        ("InstantPeriod", Value::Num(f64::NAN)),
        ("Running", Value::String("off".into())),
        ("TasksExecuted", Value::Num(0.0)),
        ("Type", Value::String(TIMER_CLASS.to_string())),
        (crate::HANDLE_VALID_FLAG_PROPERTY, Value::Bool(true)),
    ] {
        object.properties.insert(property.to_string(), value);
    }
    object
}
