//! Side-effect-free admission for values passed as `feval` targets.

use runmat_builtins::{
    FEVAL_AT_PREFIXED_TEXT_EXTENSION, FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
    FEVAL_ERROR_HANDLE_NAME_INVALID, FEVAL_ERROR_HANDLE_SHAPE_INVALID,
    FEVAL_OBJECT_RECEIVER_EXTENSION,
};
use runmat_value::Value;

use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail, BuiltinResult};

pub(crate) fn text_target_name(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    (!name.is_empty()).then_some(name)
}

/// Admit the target without resolving or invoking it. This is the same
/// boundary used by `feval` itself and by callers that must validate a
/// callback before materializing other operands.
pub(crate) fn admit_feval_target(value: &Value) -> BuiltinResult<()> {
    let text = match value {
        Value::String(text) => Some(text.as_str()),
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            return admit_text_target(&text);
        }
        Value::StringArray(strings) if strings.data.len() == 1 => Some(strings.data[0].as_str()),
        Value::CharArray(_) => {
            return Err(runtime_descriptor_error_with_detail(
                "feval",
                &FEVAL_ERROR_HANDLE_SHAPE_INVALID,
                "char array must be a row vector",
            ));
        }
        Value::StringArray(_) => {
            return Err(runtime_descriptor_error_with_detail(
                "feval",
                &FEVAL_ERROR_HANDLE_SHAPE_INVALID,
                "string array must be scalar",
            ));
        }
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => {
            if name.trim().is_empty() {
                return Err(runtime_descriptor_error(
                    "feval",
                    &FEVAL_ERROR_HANDLE_NAME_INVALID,
                ));
            }
            None
        }
        Value::BoundFunctionHandle(_) | Value::Closure(_) => None,
        Value::Object(_) | Value::HandleObject(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEVAL_OBJECT_RECEIVER_EXTENSION,
                "feval",
            )?;
            None
        }
        other => {
            return Err(runtime_descriptor_error_with_detail(
                "feval",
                &FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
                format!("{other:?}"),
            ));
        }
    };
    if let Some(text) = text {
        admit_text_target(text)?;
    }
    Ok(())
}

fn admit_text_target(text: &str) -> BuiltinResult<()> {
    if text.trim().starts_with('@') {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FEVAL_AT_PREFIXED_TEXT_EXTENSION,
            "feval",
        )?;
    }
    if text_target_name(text).is_none() {
        return Err(runtime_descriptor_error(
            "feval",
            &FEVAL_ERROR_HANDLE_NAME_INVALID,
        ));
    }
    Ok(())
}
