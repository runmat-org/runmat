use crate::{build_runtime_error, RuntimeError};

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    build(message, runmat_builtins::SPLITAPPLY_ERROR_INVALID_INPUT)
}

pub(super) fn callback(message: impl Into<String>, source: Option<RuntimeError>) -> RuntimeError {
    let mut error = build_runtime_error(message).with_builtin("splitapply");
    if let Some(identifier) = runmat_builtins::SPLITAPPLY_ERROR_CALLBACK.identifier {
        error = error.with_identifier(identifier);
    }
    if let Some(source) = source {
        error = error.with_source(source);
    }
    error.build()
}

pub(super) fn output(message: impl Into<String>, source: Option<RuntimeError>) -> RuntimeError {
    let mut error = build_runtime_error(message).with_builtin("splitapply");
    if let Some(identifier) = runmat_builtins::SPLITAPPLY_ERROR_OUTPUT.identifier {
        error = error.with_identifier(identifier);
    }
    if let Some(source) = source {
        error = error.with_source(source);
    }
    error.build()
}

fn build(
    message: impl Into<String>,
    descriptor: runmat_builtins::BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut error = build_runtime_error(message).with_builtin("splitapply");
    if let Some(identifier) = descriptor.identifier {
        error = error.with_identifier(identifier);
    }
    error.build()
}
