//! MATLAB-compatible timer handle objects.
//!
//! Timer objects are handle objects with MATLAB timer properties. The runtime
//! session owns discovery and lifecycle state; the host's local task driver
//! polls callbacks after `start` returns.

pub(crate) mod property;
mod api;
mod arguments;
mod callback;
mod catalog;
mod discovery;
mod dispatch;
mod lifecycle;
pub use api::*;
use arguments::{apply_name_value_pairs, prepare_timer_property_pairs, PreparedTimerPair};
use callback::{
    callback_is_empty, run_callback, run_error_callback, text_callback_request,
    update_average_period,
};
use catalog::*;
pub use catalog::{
    FUSION_SPEC, GPU_SPEC, TIMERFINDALL_DESCRIPTOR, TIMERFIND_DESCRIPTOR, TIMERFIND_EXTENSIONS,
    TIMERFIND_INTEGER_CAPABILITIES, TIMER_DESCRIPTOR, TIMER_EXTENSIONS, TIMER_INTEGER_CAPABILITIES,
    TIMER_METHOD_DESCRIPTOR, TIMER_SETTER_DESCRIPTOR,
};
use discovery::find_timers;
use dispatch::{delete_timer_values, start_timer_values, stop_timer_values, wait_timer_values};
use lifecycle::{start_one_timer, stop_one_timer, wait_one_timer};
use property::{TimerBusyMode, TimerExecutionMode, TimerProperty, TimerVisibility};

use runmat_types::MemberAccess;
use std::collections::HashMap;

#[cfg(all(test, not(target_arch = "wasm32")))]
use once_cell::sync::Lazy;
#[cfg(all(test, not(target_arch = "wasm32")))]
use std::sync::Mutex;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, IntValue, ObjectInstance, Value};

use crate::builtins::common::tensor;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const TIMER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("timer");
const BUILTIN_TIMER: &str = "timer";
const TIMER_CALLABLE_START: &str = "__runmat_timer_start";
const TIMER_CALLABLE_STARTAT: &str = "__runmat_timer_startat";
const TIMER_CALLABLE_STOP: &str = "__runmat_timer_stop";
const TIMER_CALLABLE_WAIT: &str = "__runmat_timer_wait";
const TIMER_CALLABLE_DELETE: &str = "__runmat_timer_delete";

const CALLBACK_PROPS: [&str; 4] = ["TimerFcn", "StartFcn", "StopFcn", "ErrorFcn"];

#[cfg(all(test, not(target_arch = "wasm32")))]
static TIMER_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

macro_rules! timer_setter_builtin {
    ($fn_name:ident, $builtin_name:literal, $property:literal) => {
        #[runtime_builtin(
                            name = $builtin_name,
                            category = "timing",
                            summary = "Validate and assign a timer property.",
                            keywords = "timer,set,property",
                            descriptor(crate::builtins::timing::timer::TIMER_SETTER_DESCRIPTOR),
                            builtin_path = "crate::builtins::timing::timer"
                        )]
        pub async fn $fn_name(obj: Value, value: Value) -> BuiltinResult<Value> {
            set_timer_property_object(obj, $property, value).await
        }
    };
}

timer_setter_builtin!(timer_set_timer_fcn_builtin, "set.TimerFcn", "TimerFcn");
timer_setter_builtin!(timer_set_start_fcn_builtin, "set.StartFcn", "StartFcn");
timer_setter_builtin!(timer_set_stop_fcn_builtin, "set.StopFcn", "StopFcn");
timer_setter_builtin!(timer_set_error_fcn_builtin, "set.ErrorFcn", "ErrorFcn");
timer_setter_builtin!(timer_set_period_builtin, "set.Period", "Period");
timer_setter_builtin!(
    timer_set_start_delay_builtin,
    "set.StartDelay",
    "StartDelay"
);
timer_setter_builtin!(
    timer_set_tasks_to_execute_builtin,
    "set.TasksToExecute",
    "TasksToExecute"
);
timer_setter_builtin!(timer_set_busy_mode_builtin, "set.BusyMode", "BusyMode");
timer_setter_builtin!(
    timer_set_execution_mode_builtin,
    "set.ExecutionMode",
    "ExecutionMode"
);
timer_setter_builtin!(timer_set_name_builtin, "set.Name", "Name");
timer_setter_builtin!(timer_set_tag_builtin, "set.Tag", "Tag");
timer_setter_builtin!(
    timer_set_object_visibility_builtin,
    "set.ObjectVisibility",
    "ObjectVisibility"
);
timer_setter_builtin!(timer_set_user_data_builtin, "set.UserData", "UserData");

fn ensure_timer_class_registered() {
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

fn default_timer_object(name: String) -> ObjectInstance {
    let mut object = ObjectInstance::new(TIMER_CLASS.to_string());
    for prop in CALLBACK_PROPS {
        object
            .properties
            .insert(prop.to_string(), Value::String(String::new()));
    }
    object
        .properties
        .insert("Period".to_string(), Value::Num(1.0));
    object
        .properties
        .insert("StartDelay".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("TasksToExecute".to_string(), Value::Num(1.0));
    object
        .properties
        .insert("BusyMode".to_string(), Value::String("drop".to_string()));
    object.properties.insert(
        "ExecutionMode".to_string(),
        Value::String("singleShot".to_string()),
    );
    object
        .properties
        .insert("Name".to_string(), Value::String(name));
    object
        .properties
        .insert("Tag".to_string(), Value::String(String::new()));
    object.properties.insert(
        "ObjectVisibility".to_string(),
        Value::String("on".to_string()),
    );
    object
        .properties
        .insert("UserData".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("AveragePeriod".to_string(), Value::Num(f64::NAN));
    object
        .properties
        .insert("InstantPeriod".to_string(), Value::Num(f64::NAN));
    object
        .properties
        .insert("Running".to_string(), Value::String("off".to_string()));
    object
        .properties
        .insert("TasksExecuted".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("Type".to_string(), Value::String(TIMER_CLASS.to_string()));
    object.properties.insert(
        crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
        Value::Bool(true),
    );
    object
}

async fn set_timer_property_object(
    obj: Value,
    property_name: &'static str,
    value: Value,
) -> BuiltinResult<Value> {
    let Value::Object(mut object) = obj else {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: set.{property_name} requires timer object receiver"),
        ));
    };
    if !object.class_name.is(TIMER_CLASS) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!(
                "timer: set.{property_name} requires timer object receiver, got '{}'",
                object.class_name
            ),
        ));
    }
    let args =
        prepare_timer_property_pairs(&[Value::String(property_name.to_string()), value], false)
            .await?;
    apply_name_value_pairs(&mut object, &args, false)?;
    Ok(Value::Object(object))
}

fn canonical_property_name(name: &str) -> BuiltinResult<TimerProperty> {
    let trimmed = name.trim();
    TimerProperty::parse(trimmed).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: unknown property '{trimmed}'"),
        )
    })
}

fn normalize_property_value(name: TimerProperty, value: Value) -> BuiltinResult<Value> {
    match name {
        TimerProperty::Period => {
            let seconds = numeric_scalar(&value, name.name())?;
            if !seconds.is_finite() || seconds <= 0.001 {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: Period must be a finite scalar greater than 0.001",
                ));
            }
            Ok(Value::Num(seconds))
        }
        TimerProperty::StartDelay => {
            let seconds = numeric_scalar(&value, name.name())?;
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: StartDelay must be a finite non-negative scalar",
                ));
            }
            Ok(Value::Num(seconds))
        }
        TimerProperty::TasksToExecute => {
            parse_tasks_to_execute_value(&value)?;
            if let Some(integer) = tensor::scalar_integer_value(&value) {
                Ok(Value::Int(integer))
            } else {
                Ok(Value::Num(numeric_scalar(&value, name.name())?.round()))
            }
        }
        TimerProperty::BusyMode => {
            parse_choice(&value, name, TimerBusyMode::parse, TimerBusyMode::spelling)
        }
        TimerProperty::ExecutionMode => parse_choice(
            &value,
            name,
            TimerExecutionMode::parse,
            TimerExecutionMode::spelling,
        ),
        TimerProperty::ObjectVisibility => parse_choice(
            &value,
            name,
            TimerVisibility::parse,
            TimerVisibility::spelling,
        ),
        TimerProperty::Name | TimerProperty::Tag => Ok(Value::String(value_to_string(&value)?)),
        TimerProperty::TimerFcn
        | TimerProperty::StartFcn
        | TimerProperty::StopFcn
        | TimerProperty::ErrorFcn => normalize_callback(value, name.name()),
        TimerProperty::UserData => Ok(value),
        _ => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{}' cannot be assigned", name.name()),
        )),
    }
}

fn normalize_callback(value: Value, name: &str) -> BuiltinResult<Value> {
    match value {
        Value::String(_)
        | Value::CharArray(runmat_value::CharArray { rows: 1, .. })
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Ok(value),
        Value::Cell(ref cell)
            if cell.data.is_empty()
                || cell.data.first().is_some_and(is_callback_function) => Ok(value),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!(
                "timer: {name} must be text, a function handle, or a callback cell array, got {other:?}"
            ),
        )),
    }
}

fn is_callback_function(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn parse_choice<T: Copy>(
    value: &Value,
    name: TimerProperty,
    parse: fn(&str) -> Option<T>,
    spelling: fn(T) -> &'static str,
) -> BuiltinResult<Value> {
    let text = value_to_string(value)?;
    parse(&text)
        .map(|choice| Value::String(spelling(choice).to_string()))
        .ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: invalid {} value '{text}'", name.name()),
            )
        })
}

fn value_to_string(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("timer: expected string scalar or character row, got {other:?}"),
        )),
    }
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let exact = match integer {
            IntValue::I8(value) => i128::from(value),
            IntValue::I16(value) => i128::from(value),
            IntValue::I32(value) => i128::from(value),
            IntValue::I64(value) => i128::from(value),
            IntValue::U8(value) => i128::from(value),
            IntValue::U16(value) => i128::from(value),
            IntValue::U32(value) => i128::from(value),
            IntValue::U64(value) => i128::from(value),
        };
        const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: {name} must be exactly representable as double seconds"),
            ));
        }
        return Ok(exact as f64);
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            Ok(tensor::tensor_value_f64(value, 0))
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("timer: {name} must be a numeric scalar, got {other:?}"),
        )),
    }
}

fn parse_tasks_to_execute_value(value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_tasks_to_execute_integer(&integer);
    }

    let count = numeric_scalar(value, "TasksToExecute")?;
    if !count.is_finite() || count < 1.0 || count.fract() != 0.0 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        ));
    }
    if !fits_platform_usize(count) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute exceeds platform limits",
        ));
    }
    Ok(count as usize)
}

fn parse_tasks_to_execute_integer(value: &IntValue) -> BuiltinResult<usize> {
    let count = value.try_to_usize().ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        )
    })?;
    if count == 0 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        ));
    }
    Ok(count)
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

fn property(handle: &HandleRef, name: &str) -> BuiltinResult<Value> {
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

fn set_property(handle: &HandleRef, name: &str, value: Value) -> BuiltinResult<()> {
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

fn string_property(handle: &HandleRef, name: &str) -> BuiltinResult<String> {
    match property(handle, name)? {
        Value::String(text) => Ok(text),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{name}' must be text, got {other:?}"),
        )),
    }
}

fn numeric_property(handle: &HandleRef, name: &str) -> BuiltinResult<f64> {
    numeric_scalar(&property(handle, name)?, name)
}

fn running_is_on(object: &ObjectInstance) -> bool {
    matches!(
        object.properties.get("Running"),
        Some(Value::String(text)) if text.eq_ignore_ascii_case("on")
    )
}

fn is_timer_handle(handle: &HandleRef) -> bool {
    handle.class_name == TIMER_CLASS
}

fn ensure_valid_timer(handle: &HandleRef) -> BuiltinResult<()> {
    if is_timer_handle(handle) && crate::is_handle_valid(handle) {
        Ok(())
    } else {
        Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: invalid or deleted timer handle",
        ))
    }
}

fn timer_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        detail.to_string()
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_TIMER);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn timer_runtime(runtime: Option<RuntimeContext>) -> BuiltinResult<RuntimeContext> {
    runtime.ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "timer: runtime session is required for timer ownership",
        )
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
