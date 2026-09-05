use runmat_builtins::{
    NUMERIC_LIMIT_ERROR_INTERNAL, NUMERIC_LIMIT_ERROR_INVALID_CLASS,
    NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
};

use crate::{build_runtime_error, RuntimeError};

pub(super) fn invalid_integer_prototype(builtin: &'static str) -> RuntimeError {
    class(
        builtin,
        "like prototype must be an integer variable of class int8, int16, int32, int64, uint8, uint16, uint32, or uint64",
    )
}

pub(super) fn invalid_floating_prototype(builtin: &'static str) -> RuntimeError {
    class(builtin, "like prototype must have class double or single")
}

pub(super) fn syntax(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = NUMERIC_LIMIT_ERROR_INVALID_SYNTAX.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn class(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = NUMERIC_LIMIT_ERROR_INVALID_CLASS.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn internal(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = NUMERIC_LIMIT_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
