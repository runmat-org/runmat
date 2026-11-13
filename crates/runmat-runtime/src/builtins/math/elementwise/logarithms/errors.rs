use super::operation::LogarithmOperation;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn with_detail(
    operation: LogarithmOperation,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(operation.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
pub(super) fn invalid(operation: LogarithmOperation, detail: impl AsRef<str>) -> RuntimeError {
    with_detail(operation, operation.invalid_input(), detail)
}
pub(super) fn internal(operation: LogarithmOperation, detail: impl AsRef<str>) -> RuntimeError {
    with_detail(operation, operation.internal_error(), detail)
}

pub(super) fn missing_provider(operation: LogarithmOperation) -> RuntimeError {
    if operation == LogarithmOperation::Binary {
        let descriptor = &runmat_builtins::LOG2_ERROR_PROVIDER_OWNERSHIP;
        let mut builder = crate::build_runtime_error(descriptor.message)
            .with_builtin(operation.name())
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        return builder.build();
    }
    internal(operation, "GPU provider unavailable for input")
}

pub(super) fn explicit_complex_unsupported(operation: LogarithmOperation) -> RuntimeError {
    debug_assert_eq!(operation, LogarithmOperation::Binary);
    let descriptor = &runmat_builtins::LOG2_ERROR_GPU_COMPLEX_INPUT;
    let mut builder = crate::build_runtime_error(descriptor.message)
        .with_builtin(operation.name())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
