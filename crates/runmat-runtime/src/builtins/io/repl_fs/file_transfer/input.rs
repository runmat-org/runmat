use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::Value;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

pub(super) async fn gather(args: &[Value], builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(|error| map_gather_error(error, builtin))?,
        );
    }
    Ok(gathered)
}

pub(super) fn text(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(descriptor_error(error, builtin)),
    }
}

pub(super) fn force_flag(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> BuiltinResult<bool> {
    if text(value, error, builtin)?.eq_ignore_ascii_case("f") {
        Ok(true)
    } else {
        Err(descriptor_error(error, builtin))
    }
}

pub(super) fn descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn message_identifier(error: &'static BuiltinErrorDescriptor) -> &'static str {
    error.identifier.unwrap_or("")
}

fn map_gather_error(error: RuntimeError, builtin: &'static str) -> RuntimeError {
    let identifier = error.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{builtin}: {}", error.message()))
        .with_builtin(builtin)
        .with_source(error);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
