//! Borrowed admission and selective host preparation before provider effects.
use super::*;

pub(super) fn resident_input_handle(value: &Value) -> Option<&GpuTensorHandle> {
    match forward_data(value) {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }
}

pub(in crate::builtins::deep_learning) fn forward_data(value: &Value) -> &Value {
    if let Value::Object(object) = value {
        if object.class_name().is(super::DLARRAY_CLASS) {
            return object.properties().get("Data").unwrap_or(value);
        }
    }
    value
}

pub(super) fn require_network_object<'a>(
    value: &'a Value,
    function: &'static str,
) -> BuiltinResult<&'a ObjectInstance> {
    match value {
        Value::Object(object) if is_deep_learning_network_object(object) => Ok(object),
        other => Err(deep_learning_error(
            function,
            format!("{function}: expected a dlnetwork or trained network object, got {other:?}"),
        )),
    }
}
