use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

pub(super) fn execution(builtin: &str, detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("{builtin}: {detail}"))
        .with_builtin(builtin)
        .with_identifier("RunMat:gpu:ProviderExecutionFailed")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

pub(super) fn payload(builtin: &str, detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("{builtin}: {detail}"))
        .with_builtin(builtin)
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}
