use runmat_builtins::{
    BuiltinErrorDescriptor, BSXFUN_ERROR_FUNCTION_ERROR, BSXFUN_ERROR_INTERNAL,
    BSXFUN_ERROR_SIZE_MISMATCH,
};

use crate::{build_runtime_error, RuntimeError};

pub(super) fn from_descriptor(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    detail(error, None)
}

pub(super) fn detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl Into<Option<String>>,
) -> RuntimeError {
    let message = detail.into().map_or_else(
        || error.message.to_string(),
        |detail| format!("{}: {detail}", error.message),
    );
    let mut builder = build_runtime_error(message).with_builtin(super::BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn callback(detail: impl Into<String>) -> RuntimeError {
    self::detail(&BSXFUN_ERROR_FUNCTION_ERROR, Some(detail.into()))
}

pub(super) fn internal(detail: impl Into<String>) -> RuntimeError {
    self::detail(&BSXFUN_ERROR_INTERNAL, Some(detail.into()))
}

pub(super) fn size(detail: impl Into<String>) -> RuntimeError {
    self::detail(&BSXFUN_ERROR_SIZE_MISMATCH, Some(detail.into()))
}
