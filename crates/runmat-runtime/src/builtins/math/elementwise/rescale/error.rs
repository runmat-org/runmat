use runmat_builtins::{
    BuiltinErrorDescriptor, RESCALE_ERROR_INTERNAL, RESCALE_ERROR_INVALID_ARGUMENT,
    RESCALE_ERROR_INVALID_INPUT, RESCALE_ERROR_SIZE_MISMATCH,
};

use crate::{build_runtime_error, RuntimeError};

use super::BUILTIN_NAME;

fn build(descriptor: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    build(&RESCALE_ERROR_INVALID_ARGUMENT, detail)
}

pub(super) fn invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    build(&RESCALE_ERROR_INVALID_INPUT, detail)
}

pub(super) fn size_mismatch(detail: impl AsRef<str>) -> RuntimeError {
    build(&RESCALE_ERROR_SIZE_MISMATCH, detail)
}

pub(super) fn internal(detail: impl AsRef<str>) -> RuntimeError {
    build(&RESCALE_ERROR_INTERNAL, detail)
}
