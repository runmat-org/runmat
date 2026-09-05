use std::path::PathBuf;

use runmat_value::Value;

use crate::{runtime_descriptor_error, BuiltinResult};

const BUILTIN_NAME: &str = "cd";

pub(super) fn path(value: Value) -> BuiltinResult<(String, PathBuf)> {
    let raw = match value {
        Value::String(text) => text,
        Value::StringArray(array) if array.data.len() == 1 => array.data[0].clone(),
        Value::CharArray(chars) if chars.rows == 1 => chars.data.iter().collect(),
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
