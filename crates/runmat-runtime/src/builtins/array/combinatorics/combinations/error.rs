use runmat_builtins::BuiltinErrorDescriptor;

use crate::{build_runtime_error, RuntimeError};

pub(super) fn with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("combinations");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    with_message(
        &runmat_builtins::COMBINATIONS_ERROR_INTERNAL,
        format!("combinations: {detail}"),
    )
}
