use runmat_builtins::BuiltinErrorDescriptor;

use crate::{build_runtime_error, RuntimeError};

pub(super) fn persistence_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn persistence_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
