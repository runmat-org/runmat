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

pub(super) fn transfer(source: RuntimeError) -> RuntimeError {
    build_runtime_error(format!("discretize: {source}"))
        .with_builtin("discretize")
        .with_identifier(
            DISCRETIZE_ERROR_INTERNAL
                .identifier
                .expect("discretize error identity"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
