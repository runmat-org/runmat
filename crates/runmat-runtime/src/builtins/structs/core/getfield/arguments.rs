use super::errors::{self, BUILTIN_NAME};
use crate::builtins::structs::core::field_path::{self, FieldPath, PathErrorKind};
use crate::BuiltinResult;
use runmat_builtins::{
    GETFIELD_ERROR_FIELD_EXPECTED, GETFIELD_ERROR_FIELD_NAME_TYPE, GETFIELD_ERROR_INDEX_INVALID,
    GETFIELD_TEXTUAL_INDEX_EXTENSION,
};
use runmat_value::Value;

pub(super) fn parse(arguments: Vec<Value>) -> BuiltinResult<FieldPath> {
    let path = field_path::parse(arguments, BUILTIN_NAME).map_err(|error| {
        let descriptor = match error.kind {
            PathErrorKind::MissingPath => &GETFIELD_ERROR_FIELD_EXPECTED,
            PathErrorKind::FieldName => &GETFIELD_ERROR_FIELD_NAME_TYPE,
            PathErrorKind::EmptySelector | PathErrorKind::InvalidIndex => {
                &GETFIELD_ERROR_INDEX_INVALID
            }
        };
        errors::with_message(format!("getfield: {}", error.detail), descriptor)
    })?;
    if path.uses_textual_index {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GETFIELD_TEXTUAL_INDEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(path)
}
