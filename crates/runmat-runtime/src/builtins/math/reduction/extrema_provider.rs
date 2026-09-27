//! Execution authority shared by the min/max resident paths.
//!
//! `None` means a typed capability rejection, never a provider execution error.
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ReduceDimResult};
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(super) enum ExtremaKind {
    Min,
    Max,
}

impl ExtremaKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    pub(super) fn contract_error(self, detail: impl std::fmt::Display) -> RuntimeError {
        let descriptor = match self {
            Self::Min => &runmat_builtins::MIN_ERROR_INTERNAL,
            Self::Max => &runmat_builtins::MAX_ERROR_INTERNAL,
        };
        let mut error = build_runtime_error(format!("{}: {detail}", self.name()))
            .with_builtin(self.name())
            .with_gpu_gather_retry(GpuGatherRetry::Never);
        if let Some(identifier) = descriptor.identifier {
            error = error.with_identifier(identifier);
        }
        error.build()
    }

    pub(super) fn owner(
        self,
        input: &GpuTensorHandle,
    ) -> BuiltinResult<&'static dyn AccelProvider> {
        gpu_helpers::exact_provider_for_handle(input)
            .ok_or_else(|| self.contract_error("resident input has no exact owning provider"))
    }

    pub(super) fn optional<T>(
        self,
        operation: &str,
        result: anyhow::Result<T>,
    ) -> BuiltinResult<Option<T>> {
        match result {
            Ok(output) => Ok(Some(output)),
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
            Err(error) => Err(crate::runtime_error::with_provider_execution_source(
                self.contract_error(format!("provider {operation} failed: {error}")),
                error.into(),
            )),
        }
    }

    pub(super) async fn reduce_dim(
        self,
        input: &GpuTensorHandle,
        dimension: usize,
    ) -> BuiltinResult<Option<ReduceDimResult>> {
        let provider = self.owner(input)?;
        let result = match self {
            Self::Min => provider.reduce_min_dim(input, dimension).await,
            Self::Max => provider.reduce_max_dim(input, dimension).await,
        };
        self.optional("dimensional extrema reduction", result)
    }

    pub(super) async fn gather(self, input: &GpuTensorHandle) -> BuiltinResult<Value> {
        // Resolve before gathering: an unrelated default provider is not a
        // capability fallback for an unregistered input owner.
        self.owner(input)?;
        gpu_helpers::gather_value_async(&Value::GpuTensor(input.clone()))
            .await
            .map_err(|mut error| {
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })
    }
}

#[cfg(test)]
mod tests;
