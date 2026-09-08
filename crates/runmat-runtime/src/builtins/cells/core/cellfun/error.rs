use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, CELLFUN_ERROR_INTERNAL, CELLFUN_ERROR_INVALID_INPUT,
    CELLFUN_ERROR_UNDEFINED_FUNCTION, CELLFUN_ERROR_UNIFORM_OUTPUT,
};

use super::BUILTIN_NAME;

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    with_message(message, &CELLFUN_ERROR_INVALID_INPUT)
}

pub(super) fn uniform(message: impl Into<String>) -> RuntimeError {
    with_message(message, &CELLFUN_ERROR_UNIFORM_OUTPUT)
}

pub(super) fn undefined(message: impl Into<String>) -> RuntimeError {
    with_message(message, &CELLFUN_ERROR_UNDEFINED_FUNCTION)
}

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    with_message(message, &CELLFUN_ERROR_INTERNAL)
}

fn with_message(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
