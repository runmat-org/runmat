//! Execution authority shared by the min/max resident paths.
//!
//! `None` means a typed capability rejection, never a provider execution error.
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{Tensor, Value};

mod demand;
mod elementwise;
mod output;
mod placement;
mod reduction;
#[cfg(all(test, feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "extrema_provider/wgpu_required_tests.rs"]
mod wgpu_required_tests;
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
            Self::Min => &runmat_builtins::MIN_ERROR_INTERNAL,
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

    pub(super) fn compute_owner(
        self,
        input: &GpuTensorHandle,
    ) -> BuiltinResult<&'static dyn AccelProvider> {
        gpu_helpers::require_resident_compute_owner(input, || {
            self.contract_error("resident input has no exact owning provider")
        })
    }

    pub(super) fn optional<T, E: Into<runmat_accelerate_api::ProviderExecutionError>>(
        self,
        operation: &str,
        result: Result<T, E>,
    ) -> BuiltinResult<Option<T>> {
        match result {
            Ok(output) => Ok(Some(output)),
            Err(error) => {
                let error = error.into();
                if error.is_unsupported() {
                    return Ok(None);
                }
                Err(crate::runtime_error::with_provider_execution_source(
                    self.contract_error(format!("provider {operation} failed: {error}")),
                    error,
                ))
            }
        }
    }

    pub(super) fn optional_after_effect<
        T,
        E: Into<runmat_accelerate_api::ProviderExecutionError>,
    >(
        self,
        operation: &str,
        result: Result<T, E>,
    ) -> BuiltinResult<Option<T>> {
        self.optional(
            operation,
            result.map_err(|error| error.into().after_effect()),
        )
    }

    pub(super) async fn reduce_dim(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
        input: &GpuTensorHandle,
        dimension: usize,
        demand: ExtremaDemand,
    ) -> BuiltinResult<Option<ExtremaResult>> {
        let provider = self.compute_owner(input)?;
        // The dimensional provider hooks implement real extrema only. Complex
        // magnitude/phase selection belongs to the owning host comparator.
        if input.physical().storage() == GpuTensorStorage::ComplexInterleaved {
            return Ok(None);
        }
        let mut selected = placement::select(runtime, self, provider, input, dimension, demand)?;
        if let Some(selection) = selected.as_ref() {
            if !selection.revalidate(input)? {
                return Err(
                    self.contract_error("dimensional extrema input changed before dispatch")
                );
            }
        }
        let recorder = selected
            .as_ref()
            .map(|selection| selection.placement.recorder());
        let checkpoint = recorder
            .as_ref()
            .map(|recorder| recorder.submission_checkpoint());
        let result =
            runmat_accelerate_api::observation::observe_provider_future(recorder.clone(), async {
                match self {
                    Self::Min => provider.reduce_min_dim(input, dimension).await,
                    Self::Max => provider.reduce_max_dim(input, dimension).await,
                }
            })
            .await;
        if result
            .as_ref()
            .err()
            .is_some_and(|error| error.is_unsupported())
            && recorder
                .as_ref()
                .zip(checkpoint)
                .is_some_and(|(recorder, checkpoint)| recorder.submission_checkpoint() > checkpoint)
        {
            if let Some(selection) = selected {
                selection.placement.finish(Err(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                ));
            }
            let source = result.expect_err("unsupported provider result");
            return Err(crate::runtime_error::with_provider_execution_source(
                self.contract_error(format!(
                    "dimensional extrema provider failed after submission: {source}"
                )),
                source.after_effect(),
            ));
        }
        let result = match self.optional("dimensional extrema reduction", result) {
            Ok(result) => result,
            Err(error) => {
                if let Some(selection) = selected {
                    selection.placement.finish(Err(
                        runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                    ));
                }
                return Err(error);
            }
        };
        let Some(result) = result else {
            if let Some(selection) = selected {
                selection.placement.finish(Ok(false));
            }
            return Ok(None);
        };
        if let Some(selection) = selected.as_mut() {
            let published = match demand {
                ExtremaDemand::ValuesOnly => selection
                    .placement
                    .validate_direct_provider_outputs(&[&result.values]),
                ExtremaDemand::ValuesAndIndices => selection
                    .placement
                    .validate_direct_provider_outputs(&[&result.values, &result.indices]),
            };
            if !published
                || (demand == ExtremaDemand::ValuesAndIndices
                    && provider.precision() == runmat_accelerate_api::ProviderPrecision::F32
                    && !selection
                        .placement
                        .admits_produced_readback(&result.indices))
            {
                let error = self.contract_error(
                    "dimensional extrema output did not match selected physical publication",
                );
                let error = gpu_helpers::reject_gpu_output(
                    result.indices,
                    gpu_helpers::reject_gpu_output(result.values, error),
                );
                selected.expect("selected output").placement.finish(Err(
                    runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed,
                ));
                return Err(error);
            }
        }
        let recorder = selected
            .as_ref()
            .map(|selection| selection.placement.recorder());
        let adopted = runmat_accelerate_api::observation::observe_provider_future(
            recorder,
            self.adopt_reduction_result(provider, input, dimension, result, demand),
        )
        .await;
        if let Some(selection) = selected {
            selection.placement.finish(match adopted.as_ref() {
                Ok(_) => Ok(true),
                Err(_) => Err(runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed),
            });
        }
        adopted.map(Some)
    }

    pub(super) fn pair_owner(
        self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> BuiltinResult<Option<&'static dyn AccelProvider>> {
        gpu_helpers::require_resident_compute_owners(&[left, right], false, || {
            self.contract_error("resident input has no exact owning provider")
        })?;
        let left_owner = self.compute_owner(left)?;
        let right_owner = self.compute_owner(right)?;
        // The existing mixed-provider route gathers each input through its
        // own provider. Never dispatch one owner's allocation to the other.
        Ok(
            runmat_accelerate_api::same_provider_instance(left_owner, right_owner)
                .then_some(left_owner),
        )
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
