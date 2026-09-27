//! Execution authority shared by the min/max resident paths.
//!
//! `None` means a typed capability rejection, never a provider execution error.
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::{Tensor, Value};

mod demand;
mod elementwise;
mod output;
mod reduction;
pub(super) use demand::{ExtremaDemand, ExtremaResult};

/// Single has no immediate scalar `Value` representation. Its scalar extrema
/// must use the same typed storage selection as its nonscalar extrema, not the
/// double-only scalar shortcut. Mixed precision remains resolved there too.
pub(super) fn needs_typed_scalar_selection(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => value.numeric_dtype() == runmat_value::NumericDType::F32,
        Value::ComplexTensor(value) => value.numeric_dtype() == runmat_value::NumericDType::F32,
        _ => false,
    }
}

/// The legacy scalar hook returns the input's real floating class. Admit it
/// only when that is also the host pairwise result class; integer comparisons
/// need the exact integer domain, not a double scalar extraction.
pub(super) fn scalar_hook_preserves_class(input: &GpuTensorHandle, other: &Value) -> bool {
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
    use runmat_value::NumericDType;
    if input.view().is_logical() || input.physical().storage() != GpuTensorStorage::Real {
        return false;
    }
    let scalar_class = match other {
        Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_) => NumericDType::F64,
        Value::Tensor(tensor) => tensor.numeric_dtype(),
        _ => return false,
    };
    match input.physical().element_type() {
        NumericElementType::F32 => scalar_class == NumericDType::F32,
        NumericElementType::F64 => matches!(scalar_class, NumericDType::F32 | NumericDType::F64),
        _ => false,
    }
}

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
            Self::Min => &super::min::MIN_ERROR_INTERNAL,
            Self::Max => &super::max::MAX_ERROR_INTERNAL,
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
        if let Err(source) = input.allocation_lease().validate_live() {
            let mut error = self.contract_error(format!(
                "resident input allocation is unavailable: {source}"
            ));
            error.source = Some(Box::new(source));
            return Err(error);
        }
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
        demand: ExtremaDemand,
    ) -> BuiltinResult<Option<ExtremaResult>> {
        let provider = self.owner(input)?;
        let result = match self {
            Self::Min => provider.reduce_min_dim(input, dimension).await,
            Self::Max => provider.reduce_max_dim(input, dimension).await,
        };
        let Some(result) = self.optional("dimensional extrema reduction", result)? else {
            return Ok(None);
        };
        self.adopt_reduction_result(provider, input, dimension, result, demand)
            .await
            .map(Some)
    }

    pub(super) fn pair_owner(
        self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> BuiltinResult<Option<&'static dyn AccelProvider>> {
        let left_owner = self.owner(left)?;
        let right_owner = self.owner(right)?;
        // The existing mixed-provider route gathers each input through its
        // own provider. Never dispatch one owner's allocation to the other.
        Ok(std::ptr::eq(left_owner, right_owner).then_some(left_owner))
    }

    pub(super) async fn gather_tensor(self, input: &GpuTensorHandle) -> BuiltinResult<Tensor> {
        self.owner(input)?;
        gpu_helpers::gather_tensor_async(input)
            .await
            .map_err(|mut error| {
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })
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

    pub(super) async fn gather_input(self, value: Value) -> BuiltinResult<Value> {
        match value {
            Value::GpuTensor(input) => self.gather(&input).await,
            value => Ok(value),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod typed_fallback_tests;
