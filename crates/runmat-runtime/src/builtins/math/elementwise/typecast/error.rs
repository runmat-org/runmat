use runmat_builtins::{BuiltinErrorDescriptor, TYPECAST_ERROR_GPU_UNSUPPORTED};

use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

use super::BUILTIN_NAME;

pub(super) fn build(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn terminal_gpu(detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!(
        "{}: {detail}",
        TYPECAST_ERROR_GPU_UNSUPPORTED.message
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier(
        TYPECAST_ERROR_GPU_UNSUPPORTED
            .identifier
            .expect("GPU error identifier"),
    )
    .with_gpu_gather_retry(GpuGatherRetry::Never)
    .build()
}
