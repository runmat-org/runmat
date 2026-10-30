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
        Value::String(text) => {
            Some(text.as_str().ok_or_else(|| {
                runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID)
            })?)
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let units = chars.data().iter().map(|unit| unit.raw());
            for decoded in char::decode_utf16(units.clone()) {
                if let Err(source) = decoded {
                    let mut error = runtime_descriptor_error_with_detail(
                        "feval",
                        &FEVAL_ERROR_HANDLE_NAME_INVALID,
                        "char target contains malformed UTF-16",
                    );
                    error.source = Some(Box::new(source));
                    return Err(error);
                }
            }
            return admit_target_characters(
                char::decode_utf16(units).map(|decoded| decoded.expect("validated UTF-16 target")),
            );
        }
        Value::StringArray(strings) if strings.data().len() == 1 => {
            Some(strings.data()[0].as_str().ok_or_else(|| {
                runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID)
            })?)
        }
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
        Value::ScopedFunctionHandle(_) | Value::BoundFunctionHandle(_) | Value::Closure(_) => None,
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
    admit_target_characters(text.chars())
}

fn admit_target_characters(characters: impl Iterator<Item = char>) -> BuiltinResult<()> {
    let mut characters = characters.skip_while(|character| character.is_whitespace());
    let first = characters.next();
    if first == Some('@') {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FEVAL_AT_PREFIXED_TEXT_EXTENSION,
            "feval",
        )?;
    }
    if first.is_none() || (first == Some('@') && characters.all(char::is_whitespace)) {
        return Err(runtime_descriptor_error(
            "feval",
            &FEVAL_ERROR_HANDLE_NAME_INVALID,
        ));
    }
    Ok(())
}
