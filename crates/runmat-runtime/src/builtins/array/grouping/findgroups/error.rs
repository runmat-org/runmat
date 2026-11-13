use crate::{build_runtime_error, RuntimeError};

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    let descriptor = &runmat_builtins::FINDGROUPS_ERROR_INVALID_INPUT;
    let mut builder = build_runtime_error(message).with_builtin("findgroups");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    let descriptor = &runmat_builtins::FINDGROUPS_ERROR_INTERNAL;
    let mut builder = build_runtime_error(message).with_builtin("findgroups");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
