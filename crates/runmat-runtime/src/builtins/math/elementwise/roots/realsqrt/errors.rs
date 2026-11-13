use runmat_builtins::{BuiltinErrorDescriptor, REALSQRT_ERROR_INTERNAL};

use crate::{build_runtime_error, RuntimeError};

use super::BUILTIN_NAME;

pub(super) fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    with_detail(&REALSQRT_ERROR_INTERNAL, detail)
}

pub(super) fn with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
