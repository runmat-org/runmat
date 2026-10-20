use super::errors;
use crate::{make_cell_with_shape, BuiltinResult};
use runmat_builtins::{GETFIELD_ERROR_INVALID_HANDLE, GETFIELD_ERROR_MISSING_FIELD};
use runmat_types::ExceptionProperty;
use runmat_value::{Listener, MException, Value};

pub(super) fn listener_field(listener: &Listener, name: &str) -> BuiltinResult<Value> {
    match name {
        "Enabled" | "enabled" => Ok(Value::Bool(listener.enabled)),
        "Valid" | "valid" => Ok(Value::Bool(listener.valid)),
        "EventName" | "event_name" => Ok(Value::string(listener.event_name.clone())),
        "Callback" | "callback" => read_listener_value(listener, true),
        "Target" | "target" => read_listener_value(listener, false),
        "Id" | "id" => Ok(Value::Int(runmat_value::IntValue::U64(listener.id))),
        other => Err(errors::with_message(
            format!("getfield: unknown field '{}' on listener object", other),
            &GETFIELD_ERROR_MISSING_FIELD,
        )),
    }
}

fn read_listener_value(listener: &Listener, callback: bool) -> BuiltinResult<Value> {
    if !listener.valid {
        return Err(errors::with_message(
            "getfield: listener is invalid or deleted",
            &GETFIELD_ERROR_INVALID_HANDLE,
        ));
    }
    let source = if callback {
        &listener.callback
    } else {
        &listener.target
    };
    runmat_gc::gc_clone_value(source).map_err(|error| {
        errors::with_message(
            format!("getfield: invalid listener value: {error}"),
            &GETFIELD_ERROR_INVALID_HANDLE,
        )
    })
}

pub(super) fn exception_field(exception: &MException, name: &str) -> BuiltinResult<Value> {
    match ExceptionProperty::from_name(name) {
        Some(ExceptionProperty::Message) => Ok(crate::object::resolve::exception_text_value(
            &exception.message,
        )),
        Some(ExceptionProperty::Identifier) => Ok(crate::object::resolve::exception_text_value(
            &exception.identifier,
        )),
        Some(ExceptionProperty::Stack) => exception_stack_to_value(&exception.stack),
        None => Err(errors::with_message(
            format!("Reference to non-existent field '{}'.", name),
            &GETFIELD_ERROR_MISSING_FIELD,
        )),
    }
}

fn exception_stack_to_value(stack: &[runmat_value::Utf8Text]) -> BuiltinResult<Value> {
    let values = stack.iter().cloned().map(Value::string).collect();
    make_cell_with_shape(values, vec![stack.len(), 1])
        .map_err(|error| errors::internal(format!("getfield: {error}")))
}
