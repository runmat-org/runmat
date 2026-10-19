use runmat_builtins::BuiltinErrorDescriptor;

use crate::{build_runtime_error, RuntimeError};

use super::BUILTIN_NAME;

pub(super) fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("sqrt: {detail}"))
        .with_builtin(BUILTIN_NAME)
        .build()
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
