use runmat_builtins::{
    DISCRETIZE_ERROR_INTERNAL, DISCRETIZE_ERROR_INVALID_INPUT, DISCRETIZE_ERROR_TOO_LARGE,
};

use crate::{build_runtime_error, RuntimeError};

fn described(
    descriptor: &runmat_builtins::BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("discretize");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    described(&DISCRETIZE_ERROR_INVALID_INPUT, message)
}

pub(super) fn too_large(message: impl Into<String>) -> RuntimeError {
    described(&DISCRETIZE_ERROR_TOO_LARGE, message)
}

pub(super) fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    described(&DISCRETIZE_ERROR_INTERNAL, format!("discretize: {detail}"))
}
