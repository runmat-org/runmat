mod objects;
mod structures;

use runmat_value::Value;

pub(super) fn collect(value: &Value) -> crate::BuiltinResult<Vec<String>> {
    match value {
        Value::Struct(structure) => Ok(structures::scalar(structure)),
        Value::StructArray(array) => Ok(structures::array(array)),
        Value::Object(object) => Ok(objects::object(object)),
        Value::HandleObject(handle) => objects::handle(handle),
        Value::Listener(listener) => Ok(objects::listener(listener)),
        other => Err(super::error::invalid_target(format!(
            "{} (got {other:?})",
            runmat_builtins::FIELDNAMES_ERROR_INVALID_TARGET.message
        ))),
    }
}
