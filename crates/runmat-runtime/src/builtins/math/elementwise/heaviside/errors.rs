use runmat_builtins::{
    BuiltinErrorDescriptor, HEAVISIDE_ERROR_INTERNAL, HEAVISIDE_ERROR_INVALID_INPUT,
    HEAVISIDE_ERROR_PROVIDER_FAILED,
};

use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

use super::BUILTIN_NAME;

pub(super) fn invalid_input(detail: impl std::fmt::Display) -> RuntimeError {
    described(&HEAVISIDE_ERROR_INVALID_INPUT, detail, false)
}

pub(super) fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    described(&HEAVISIDE_ERROR_INTERNAL, detail, true)
}

pub(super) fn provider(error: anyhow::Error) -> RuntimeError {
    let message = error.to_string();
    build_runtime_error(format!(
        "{}: {message}",
        HEAVISIDE_ERROR_PROVIDER_FAILED.message
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier(
        HEAVISIDE_ERROR_PROVIDER_FAILED
            .identifier
            .expect("provider failure identifier"),
    )
    .with_source(ProviderFailure(format!("{error:?}")))
    .with_gpu_gather_retry(GpuGatherRetry::Never)
    .build()
}

fn described(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
    terminal_gpu: bool,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    if terminal_gpu {
        builder = builder.with_gpu_gather_retry(GpuGatherRetry::Never);
    }
    builder.build()
}

#[derive(Debug)]
struct ProviderFailure(String);

impl std::fmt::Display for ProviderFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ProviderFailure {}
