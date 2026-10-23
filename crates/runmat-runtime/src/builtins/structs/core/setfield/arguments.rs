use super::errors::{self, BUILTIN_NAME};
use crate::builtins::structs::core::field_path::{self, FieldPath, PathErrorKind};
use crate::BuiltinResult;
use runmat_builtins::{
    SETFIELD_ERROR_FIELD_EXPECTED, SETFIELD_ERROR_FIELD_NAME_TYPE, SETFIELD_ERROR_INDEX_INVALID,
    SETFIELD_TEXTUAL_INDEX_EXTENSION,
};

pub(super) fn parse(arguments: crate::call::BuiltinArguments) -> BuiltinResult<FieldPath> {
    let path = field_path::parse(arguments, BUILTIN_NAME).map_err(|error| {
        let descriptor = match error.kind {
            PathErrorKind::MissingPath => &SETFIELD_ERROR_FIELD_EXPECTED,
            PathErrorKind::FieldName => &SETFIELD_ERROR_FIELD_NAME_TYPE,
            PathErrorKind::EmptySelector | PathErrorKind::InvalidIndex => {
                &SETFIELD_ERROR_INDEX_INVALID
            }
        };
        let mut runtime = errors::with_message(format!("setfield: {}", error.detail), descriptor);
        if let Some(source) = error.source {
            runtime.source = Some(Box::new(source));
        }
        runtime
    })?;
    if path.uses_textual_index {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SETFIELD_TEXTUAL_INDEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(path)
}
