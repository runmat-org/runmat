use std::path::PathBuf;

use runmat_value::Value;

use crate::builtins::common::fs::expand_user_path;
use crate::BuiltinResult;

pub(super) fn folder(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<PathBuf> {
    match args.len() {
        0 => runmat_filesystem::current_dir().map_err(|error| {
            super::error::message(
                &runmat_builtins::WHAT_ERROR_FILESYSTEM,
                format!("what: unable to read current folder ({error})"),
            )
        }),
        1 => path(&args[0]),
        _ => Err(super::error::contract(&runmat_builtins::WHAT_ERROR_ARITY)),
    }
}

fn path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = match value {
        Value::String(runmat_value::StringElement::Text(text)) => text.as_str().to_owned(),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned()
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                super::error::contract(&runmat_builtins::WHAT_ERROR_FOLDER)
            })?
            .trim_end()
            .to_owned()
        }
        _ => return Err(super::error::contract(&runmat_builtins::WHAT_ERROR_FOLDER)),
    };
    expand_user_path(text.trim(), "what")
        .map(PathBuf::from)
        .map_err(|error| super::error::message(&runmat_builtins::WHAT_ERROR_FOLDER, error))
}
