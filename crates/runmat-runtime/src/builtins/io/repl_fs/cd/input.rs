use std::path::PathBuf;

use runmat_value::Value;

use crate::{runtime_descriptor_error, BuiltinResult};

const BUILTIN_NAME: &str = "cd";

pub(super) fn path(value: Value) -> BuiltinResult<(String, PathBuf)> {
    let raw = match value {
        Value::String(runmat_value::StringElement::Text(text)) => text.into_string_unbounded(),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned()
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                runtime_descriptor_error(BUILTIN_NAME, &runmat_builtins::CD_ERROR_INVALID_INPUT)
            })?
        }
        _ => {
            return Err(runtime_descriptor_error(
                BUILTIN_NAME,
                &runmat_builtins::CD_ERROR_INVALID_INPUT,
            ));
        }
    };
    if raw.is_empty() {
        return Err(runtime_descriptor_error(
            BUILTIN_NAME,
            &runmat_builtins::CD_ERROR_EMPTY_FOLDER,
        ));
    }
    let expanded =
        crate::builtins::common::fs::expand_user_path(&raw, BUILTIN_NAME).map_err(|detail| {
            crate::runtime_descriptor_error_with_detail(
                BUILTIN_NAME,
                &runmat_builtins::CD_ERROR_CHANGE_FAILED,
                detail.strip_prefix("cd: ").unwrap_or(&detail),
            )
        })?;
    Ok((raw, PathBuf::from(expanded)))
}
