use runmat_runtime::runtime_error::{semantic_error, semantic_error_with_source, GpuGatherRetry};
use runmat_runtime::RuntimeError;

pub(super) fn unavailable(message: impl Into<String>) -> RuntimeError {
    semantic_error("DynamicWorkspaceUnavailable", message)
}

/// The workspace mutation is already visible. Retain the exact failing cell
/// error and prohibit a gather retry from replaying that mutation.
pub(super) fn committed(identifier: &str, message: String, source: RuntimeError) -> RuntimeError {
    let failure_origin = source.failure_origin;
    let mut failure = semantic_error_with_source(identifier, message, source);
    failure.failure_origin = failure_origin;
    failure.gpu_gather_retry = GpuGatherRetry::Never;
    failure
}
