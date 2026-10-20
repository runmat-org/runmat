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

/// Preserve the operational gather failure under the existing presentation.
/// Argument destination admission failures bypass this domain mapper entirely.
pub(super) fn gathered(source: RuntimeError) -> RuntimeError {
    let descriptor = &runmat_builtins::COMBINATIONS_ERROR_INTERNAL;
    let mut builder = build_runtime_error(format!("combinations: {source}"))
        .with_builtin("combinations")
        .with_source(source);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
