//! Whitespace classification for host text.

use runmat_builtins::{
    ISSPACE_DESCRIPTOR, ISSPACE_ERROR_INVALID_STRING_ARRAY, ISSPACE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::core::classification::classify_text_value;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

fn invalid_string_array_error() -> RuntimeError {
    let descriptor = &ISSPACE_ERROR_INVALID_STRING_ARRAY;
    let mut builder = build_runtime_error(descriptor.message).with_builtin("isspace");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "isspace",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::isspace"
)]
pub(crate) async fn isspace_builtin(text: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ISSPACE_DESCRIPTOR,
        &ISSPACE_ERROR_TOO_MANY_OUTPUTS,
        "isspace",
    )?;
    let text = match text {
        Value::StringArray(array) if array.data().len() == 1 => {
            return crate::builtins::strings::core::classification::classify_string_element(
                &array.data()[0],
                "isspace",
                |ch| ch.is_whitespace(),
            );
        }
        Value::StringArray(_) => return Err(invalid_string_array_error()),
        Value::String(_) | Value::CharArray(_) => text,
        _ => return Ok(Value::Bool(false)),
    };
    let text = gather_if_needed_async(&text)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "isspace"))?;
    classify_text_value(text, "isspace", |ch| ch.is_whitespace())
}
