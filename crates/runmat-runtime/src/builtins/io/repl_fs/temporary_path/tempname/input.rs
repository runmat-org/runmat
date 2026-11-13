use std::path::PathBuf;

use runmat_builtins::{
    TEMPNAME_ERROR_FOLDER_EMPTY, TEMPNAME_ERROR_FOLDER_RESOLVE, TEMPNAME_ERROR_FOLDER_TYPE,
};
use runmat_value::Value;

use super::super::error;

const IDENTITY: &str = "tempname";

pub(super) fn folder(value: &Value) -> crate::BuiltinResult<PathBuf> {
    let text = match value {
        Value::String(runmat_value::StringElement::Text(text)) => text.as_str().to_owned(),
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                error::builtin(IDENTITY, &TEMPNAME_ERROR_FOLDER_TYPE)
            })?
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned()
        }
        _ => return Err(error::builtin(IDENTITY, &TEMPNAME_ERROR_FOLDER_TYPE)),
    };
    if text.is_empty() {
        return Err(error::builtin(IDENTITY, &TEMPNAME_ERROR_FOLDER_EMPTY));
    }
    let expanded =
        crate::builtins::common::fs::expand_user_path(&text, IDENTITY).map_err(|detail| {
            error::message(
                IDENTITY,
                &TEMPNAME_ERROR_FOLDER_RESOLVE,
                format!("{}: {detail}", TEMPNAME_ERROR_FOLDER_RESOLVE.message),
            )
        })?;
    if expanded.is_empty() {
        Err(error::builtin(IDENTITY, &TEMPNAME_ERROR_FOLDER_EMPTY))
    } else {
        Ok(PathBuf::from(expanded))
    }
}
