use crate::{build_runtime_error, RuntimeError};

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    build(runmat_builtins::GROUPCOUNTS_ERROR_INVALID_INPUT, message)
}

pub(super) fn too_large(message: impl Into<String>) -> RuntimeError {
    build(runmat_builtins::GROUPCOUNTS_ERROR_TOO_LARGE, message)
}

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    build(runmat_builtins::GROUPCOUNTS_ERROR_INTERNAL, message)
}

fn build(
    descriptor: runmat_builtins::BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("groupcounts");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
