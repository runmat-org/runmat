use super::*;
use runmat_builtins::BuiltinErrorDescriptor;

const BUILTIN_NAME: &str = "delete";

pub(super) fn delete_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn delete_io_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    cause: std::io::Error,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(cause);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
