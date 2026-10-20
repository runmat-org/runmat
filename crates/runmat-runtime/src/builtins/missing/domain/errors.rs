use super::metadata::core::{
    MISSING_ERROR_INTERNAL, MISSING_ERROR_INVALID_ARGUMENT, MISSING_ERROR_UNSUPPORTED_TYPE,
};
use super::{build_runtime_error, BuiltinErrorDescriptor, RuntimeError};
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::RuntimeErrorBuilder;

fn missing_error_builder(
    descriptor: &'static BuiltinErrorDescriptor,
    builtin: &str,
    detail: impl Into<String>,
) -> RuntimeErrorBuilder {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.into()))
        .with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
}

pub(super) fn invalid_argument(detail: impl Into<String>) -> RuntimeError {
    missing_error_builder(&MISSING_ERROR_INVALID_ARGUMENT, "missing", detail).build()
}

pub(super) fn unsupported_type(detail: impl Into<String>) -> RuntimeError {
    missing_error_builder(&MISSING_ERROR_UNSUPPORTED_TYPE, "missing", detail).build()
}

pub(super) fn internal_error(detail: impl Into<String>) -> RuntimeError {
    missing_error_builder(&MISSING_ERROR_INTERNAL, "missing", detail).build()
}

pub(super) fn allocation_error(
    runtime: &RuntimeContext,
    builtin: &'static str,
    cause: HostAllocationError,
) -> RuntimeError {
    build_runtime_error(format!("{builtin}: host text allocation: {cause}"))
        .with_builtin(builtin)
        .with_identifier(format!(
            "{}:{}",
            RuntimeContext::error_namespace(runtime),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

/// Keep the established missing-data presentation and the delegated failure.
/// Retry admission belongs to the failing operation, not diagnostic formatting.
pub(super) fn gather_error(builtin: &str, operand: &str, source: RuntimeError) -> RuntimeError {
    let retry = source.gpu_gather_retry();
    missing_error_builder(
        &MISSING_ERROR_INVALID_ARGUMENT,
        builtin,
        format!("{builtin}: failed to gather {operand}: {source}"),
    )
    .with_gpu_gather_retry(retry)
    .with_source(source)
    .build()
}
