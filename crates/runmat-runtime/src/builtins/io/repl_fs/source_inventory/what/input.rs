use std::path::PathBuf;

use runmat_value::Value;

use crate::builtins::common::fs::expand_user_path;
use crate::BuiltinResult;

pub(super) fn folder(args: &[Value]) -> BuiltinResult<PathBuf> {
    match args {
        [] => runmat_filesystem::current_dir().map_err(|error| {
            super::error::message(
                &runmat_builtins::WHAT_ERROR_FILESYSTEM,
                format!("what: unable to read current folder ({error})"),
            )
        }),
        [value] => path(value),
        _ => Err(super::error::contract(&runmat_builtins::WHAT_ERROR_ARITY)),
    }
}

fn path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::StringArray(array) if array.data.len() == 1 => array.data[0].clone(),
        Value::CharArray(chars) if chars.rows == 1 => chars
            .data
            .iter()
            .take(chars.cols)
            .collect::<String>()
            .trim_end()
            .to_owned(),
        _ => return Err(super::error::contract(&runmat_builtins::WHAT_ERROR_FOLDER)),
    };
    expand_user_path(text.trim(), "what")
        .map(PathBuf::from)
        .map_err(|error| super::error::message(&runmat_builtins::WHAT_ERROR_FOLDER, error))
}
