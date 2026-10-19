use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;

use super::BUILTIN_NAME;

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    build(message, &runmat_builtins::STRUCTFUN_ERROR_INVALID_INPUT)
}
pub(super) fn not_scalar(message: impl Into<String>) -> RuntimeError {
    build(message, &runmat_builtins::STRUCTFUN_ERROR_NOT_SCALAR_STRUCT)
}
pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    build(message, &runmat_builtins::STRUCTFUN_ERROR_INTERNAL)
}
pub(super) fn uniform(message: impl Into<String>) -> RuntimeError {
    build(message, &runmat_builtins::STRUCTFUN_ERROR_UNIFORM_OUTPUT)
}
pub(super) fn function(message: impl Into<String>) -> RuntimeError {
    build(message, &runmat_builtins::STRUCTFUN_ERROR_FUNCTION_ERROR)
}
pub(super) fn undefined(message: impl Into<String>) -> RuntimeError {
    build(
        message,
        &runmat_builtins::STRUCTFUN_ERROR_UNDEFINED_FUNCTION,
    )
}

fn build(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
