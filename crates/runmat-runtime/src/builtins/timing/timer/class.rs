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
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
    }

    let mut properties = HashMap::new();
    for property in TimerProperty::ALL {
        let name = property.name();
        if !property.is_readonly() {
            let setter = crate::object_property_setter_name(name);
            methods.insert(
                setter.as_str().into(),
                crate::class_registry::RuntimeMethod {
                    name: setter.as_str().into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: setter,
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }
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

    crate::class_registry::register_builtin_class(
        TIMER_CLASS,
        crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: TIMER_CLASS.into(),
            parent: Some(runmat_types::standard::HANDLE.into()),
            properties,
            methods,
        },
    );
}

fn default_timer_fields(name: String) -> [(&'static str, Value); 19] {
    [
        ("TimerFcn", Value::string(String::new())),
        ("StartFcn", Value::string(String::new())),
        ("StopFcn", Value::string(String::new())),
        ("ErrorFcn", Value::string(String::new())),
        ("Period", Value::Num(1.0)),
        ("StartDelay", Value::Num(0.0)),
        ("TasksToExecute", Value::Num(1.0)),
        ("BusyMode", Value::String("drop".into())),
        ("ExecutionMode", Value::String("singleShot".into())),
        ("Name", Value::string(name)),
        ("Tag", Value::string(String::new())),
        ("ObjectVisibility", Value::String("on".into())),
        ("UserData", Value::Num(0.0)),
        ("AveragePeriod", Value::Num(f64::NAN)),
        ("InstantPeriod", Value::Num(f64::NAN)),
        ("Running", Value::String("off".into())),
        ("TasksExecuted", Value::Num(0.0)),
        ("Type", Value::string(TIMER_CLASS.to_string())),
        (crate::HANDLE_VALID_FLAG_PROPERTY, Value::Bool(true)),
    ]
}

#[cfg(test)]
pub(super) fn default_timer_object(name: String) -> ObjectInstance {
    let fields = default_timer_fields(name);
    let mut builder = runmat_value::UnboundedObjectBuilder::with_property_capacity(
        TIMER_CLASS.to_string(),
        fields.len(),
    );
    for (property, value) in fields {
        builder.insert(property.to_owned(), value);
    }
    builder.finish()
}

pub(super) fn prepare_default_timer_object(
    runtime: &RuntimeContext,
    name: String,
    overrides: &[PreparedTimerPair],
) -> BuiltinResult<crate::execution::RootedValue> {
    let mut fields = default_timer_fields(name);
    for pair in overrides {
        let (_, value) = fields
            .iter_mut()
            .find(|(property, _)| *property == pair.property.name())
            .expect("every writable timer property has a default slot");
        *value = pair.value.clone();
    }
    crate::execution::object::construct_object_from_borrowed_fields(
        Some(runtime),
        TIMER_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}
