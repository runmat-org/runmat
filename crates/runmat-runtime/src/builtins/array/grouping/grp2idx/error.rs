use runmat_builtins::{GRP2IDX_ERROR_INTERNAL, GRP2IDX_ERROR_INVALID_INPUT};

use crate::{build_runtime_error, RuntimeError};

pub(super) fn invalid(message: impl Into<String>) -> RuntimeError {
    described(message, &GRP2IDX_ERROR_INVALID_INPUT)
}

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    described(message, &GRP2IDX_ERROR_INTERNAL)
}

fn described(
    message: impl Into<String>,
    descriptor: &runmat_builtins::BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("grp2idx");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
