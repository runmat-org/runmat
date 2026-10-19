use runmat_builtins::{
    ACCUMARRAY_ERROR_CALLBACK, ACCUMARRAY_ERROR_INVALID_INPUT, ACCUMARRAY_ERROR_TOO_LARGE,
};

use crate::{build_runtime_error, RuntimeError};

fn identified(message: impl Into<String>, identifier: Option<&'static str>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("accumarray");
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    identified(message, ACCUMARRAY_ERROR_INVALID_INPUT.identifier)
}

pub(super) fn callback(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("accumarray");
    if let Some(identifier) = ACCUMARRAY_ERROR_CALLBACK.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(source).build()
}

pub(super) fn too_large(message: impl Into<String>) -> RuntimeError {
    identified(message, ACCUMARRAY_ERROR_TOO_LARGE.identifier)
}
