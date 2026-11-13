//! MATLAB-compatible `jsondecode` builtin for deserialising JSON text into RunMat values.

mod conversion;
pub(crate) use conversion::value_from_json;

use runmat_builtins::{
    BuiltinErrorDescriptor, JSONDECODE_ERROR_INPUT_TYPE, JSONDECODE_ERROR_INTERNAL,
    JSONDECODE_ERROR_NUMERIC_LITERAL, JSONDECODE_ERROR_PARSE,
};
#[cfg(test)]
use runmat_builtins::{JSONDECODE_DESCRIPTOR, JSONDECODE_INTEGER_AUDIT};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, Tensor, Value};
use serde_json::Value as JsonValue;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "jsondecode";

fn jsondecode_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    jsondecode_error_with(error, error.message)
}

fn jsondecode_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn jsondecode_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn jsondecode_flow_with_context(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string()).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    builder.with_source(err).build()
}

#[runtime_builtin(
    name = "jsondecode",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::json::jsondecode"
)]
async fn jsondecode_builtin(text: Value) -> crate::BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(jsondecode_error(&JSONDECODE_ERROR_INPUT_TYPE));
    }
    let gathered = gather_if_needed_async(&text)
        .await
        .map_err(jsondecode_flow_with_context)?;
    let source = extract_text(gathered)?;
    let parsed: JsonValue = serde_json::from_str(&source).map_err(|err| {
        jsondecode_error_with_source(
            &JSONDECODE_ERROR_PARSE,
            format!("{} ({err})", JSONDECODE_ERROR_PARSE.message),
            err,
        )
    })?;
    value_from_json(&parsed)
}

pub(crate) fn decode_json_text(text: &str) -> BuiltinResult<Value> {
    let parsed: JsonValue = serde_json::from_str(text).map_err(|err| {
        jsondecode_error_with_source(
            &JSONDECODE_ERROR_PARSE,
            format!("{} ({err})", JSONDECODE_ERROR_PARSE.message),
            err,
        )
    })?;
    value_from_json(&parsed)
}

fn extract_text(value: Value) -> BuiltinResult<String> {
    match value {
        Value::CharArray(array) => {
            if array.rows > 1 {
                return Err(jsondecode_error(&JSONDECODE_ERROR_INPUT_TYPE));
            }
            crate::builtins::common::text::decode_utf16_with_error(array.data(), || {
                jsondecode_error(&JSONDECODE_ERROR_INPUT_TYPE)
            })
        }
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.into_string_unbounded()),
        Value::StringArray(sa) => {
            if sa.data().len() == 1 && !sa.data()[0].is_missing() {
                Ok(sa.data()[0]
                    .as_str()
                    .expect("checked nonmissing string")
                    .to_owned())
            } else {
                Err(jsondecode_error(&JSONDECODE_ERROR_INPUT_TYPE))
            }
        }
        _other => Err(jsondecode_error(&JSONDECODE_ERROR_INPUT_TYPE)),
    }
}

#[cfg(test)]
pub(crate) mod tests;
