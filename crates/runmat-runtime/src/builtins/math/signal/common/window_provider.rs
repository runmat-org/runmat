//! Admission of newly generated cosine-sum windows from the active provider.
//! A typed Unsupported result alone permits the host reference path. An
//! operational failure or a malformed result is terminal and retains its cause.
use runmat_accelerate_api::{
    self as api, AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
    NumericElementType, ProviderExecutionError, ProviderPrecision,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, BLACKMAN_ERROR_INTERNAL, BLACKMAN_ERROR_PROVIDER_FAILED,
    HAMMING_ERROR_INTERNAL, HAMMING_ERROR_PROVIDER_FAILED, HANN_ERROR_INTERNAL,
    HANN_ERROR_PROVIDER_FAILED,
};
use runmat_builtins::{WindowOutputType, WindowSampling};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

use super::WindowOptions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CosineWindow {
    Blackman,
    Hamming,
    Hann,
}

impl CosineWindow {
    fn name(self) -> &'static str {
        match self {
            Self::Blackman => "blackman",
            Self::Hamming => "hamming",
            Self::Hann => "hann",
        }
    }

    fn internal(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Blackman => &BLACKMAN_ERROR_INTERNAL,
            Self::Hamming => &HAMMING_ERROR_INTERNAL,
            Self::Hann => &HANN_ERROR_INTERNAL,
        }
    }

    fn provider_failed(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Blackman => &BLACKMAN_ERROR_PROVIDER_FAILED,
            Self::Hamming => &HAMMING_ERROR_PROVIDER_FAILED,
            Self::Hann => &HANN_ERROR_PROVIDER_FAILED,
        }
    }

    fn execute(
        self,
        provider: &dyn AccelProvider,
        len: usize,
        periodic: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self {
            Self::Blackman => provider.blackman_window(len, periodic),
            Self::Hamming => provider.hamming_window(len, periodic),
            Self::Hann => provider.hann_window(len, periodic),
        }
    }
}

fn error(kind: CosineWindow, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(descriptor.message).with_builtin(kind.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut error = builder.build();
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

fn valid_output(
    handle: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    options: WindowOptions,
) -> bool {
    let element_type = match options.output_type {
        WindowOutputType::Double => NumericElementType::F64,
        WindowOutputType::Single => NumericElementType::F32,
    };
    handle.shape() == [options.len, 1]
        && handle.device_id() == provider.device_id()
        && api::provider_for_handle(handle)
            .is_some_and(|owner| api::same_provider_instance(owner, provider))
        && gpu_helpers::numeric_descriptor_matches(handle, element_type, GpuTensorStorage::Real)
        && api::handle_integer_type(handle).is_none()
        && !api::handle_is_logical(handle)
        && !matches!(
            api::handle_provenance(handle),
            Some(GpuHandleProvenance::Explicit)
        )
}

/// A provider hook returns an owning result reference with the requested
/// physical class. This boundary validates metadata and lease liveness; it
/// cannot prove that the underlying allocation is physically exclusive.
/// Rejection consumes only the returned reference, leaving other owners live.
pub(crate) fn try_generate(
    kind: CosineWindow,
    options: WindowOptions,
) -> BuiltinResult<Option<Value>> {
    if options.len <= 1 {
        return Ok(None);
    }
    let Some(provider) = api::provider() else {
        return Ok(None);
    };
    let precision = match options.output_type {
        WindowOutputType::Double => ProviderPrecision::F64,
        WindowOutputType::Single => ProviderPrecision::F32,
    };
    if provider.precision() != precision {
        return Ok(None);
    }
    let periodic = matches!(options.sampling, WindowSampling::Periodic);
    let handle = match kind.execute(provider, options.len, periodic) {
        Ok(handle) => handle,
        Err(cause)
            if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(&cause) =>
        {
            return Ok(None)
        }
        Err(cause) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                error(kind, kind.provider_failed()),
                cause.into(),
            ));
        }
    };
    if let Err(cause) = handle.allocation_lease().validate_live() {
        let failure = crate::runtime_error::with_provider_execution_source(
            error(kind, kind.internal()),
            ProviderExecutionError::from_source(cause),
        );
        return Err(gpu_helpers::reject_gpu_output(handle, failure));
    }
    if !valid_output(&handle, provider, options) {
        return Err(gpu_helpers::reject_gpu_output(
            handle,
            error(kind, kind.internal()),
        ));
    }
    gpu_helpers::resident_gpu_value(handle)
        .map_err(|cause| {
            crate::runtime_error::with_provider_execution_source(
                error(kind, kind.internal()),
                ProviderExecutionError::from_source(cause),
            )
        })
        .map(Some)
}

#[cfg(test)]
#[path = "window_provider/tests.rs"]
mod tests;
