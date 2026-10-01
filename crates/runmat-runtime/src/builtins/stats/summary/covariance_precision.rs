//! Output precision shared by the covariance and correlation families.
//! Options and observation weights do not promote the observation data class.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPrecision,
};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::gpu_helpers::{self, GpuOutputAliasPolicy, UnaryGpuOutputContract};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CovariancePrecision {
    Single,
    Double,
}

impl CovariancePrecision {
    pub(super) fn observations(first: &Value, second: Option<&Value>) -> Self {
        if Self::class_of(first) == Self::Single
            && second.is_none_or(|value| Self::class_of(value) == Self::Single)
        {
            Self::Single
        } else {
            Self::Double
        }
    }

    fn class_of(value: &Value) -> Self {
        match value {
            Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => Self::Single,
            Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
                Self::Single
            }
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(handle)
                    && runmat_accelerate_api::handle_precision(handle)
                        == Some(ProviderPrecision::F32) =>
            {
                Self::Single
            }
            _ => Self::Double,
        }
    }

    pub(super) fn dtype(self) -> NumericDType {
        match self {
            Self::Single => NumericDType::F32,
            Self::Double => NumericDType::F64,
        }
    }

    pub(super) fn provider_precision(self) -> ProviderPrecision {
        match self {
            Self::Single => ProviderPrecision::F32,
            Self::Double => ProviderPrecision::F64,
        }
    }

    pub(super) fn cast_host(self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        let error = |detail: String| {
            build_runtime_error(format!("{builtin}: {detail}"))
                .with_builtin(builtin)
                .build()
        };
        match value {
            Value::Tensor(tensor) if tensor.numeric_dtype() != self.dtype() => {
                Tensor::new_with_dtype(tensor.materialize_f64(), tensor.shape.clone(), self.dtype())
                    .map(Value::Tensor)
                    .map_err(error)
            }
            Value::ComplexTensor(tensor) if tensor.numeric_dtype() != self.dtype() => {
                ComplexTensor::from_f64_values_with_dtype(
                    tensor.materialize_f64(),
                    tensor.shape.clone(),
                    self.dtype(),
                )
                .map(Value::ComplexTensor)
                .map_err(error)
            }
            Value::OutputList(values) => values
                .into_iter()
                .map(|value| self.cast_host(value, builtin))
                .collect::<BuiltinResult<Vec<_>>>()
                .map(Value::OutputList),
            other => Ok(other),
        }
    }

    /// Validate a provider result before letting a cast hook read it. Both the
    /// raw result and any converted result must retain the expected geometry,
    /// real floating class, exact owner, live allocation, and distinct storage.
    /// Typed unsupported casts admit the existing host boundary; execution
    /// faults and malformed results never do.
    pub(super) async fn align_provider(
        self,
        provider: &'static dyn AccelProvider,
        result: GpuTensorHandle,
        expected_shape: &[usize],
        protected: &[&GpuTensorHandle],
        invalid_output: impl Fn(&str) -> RuntimeError,
        provider_error: impl Fn(ProviderExecutionError) -> RuntimeError,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        let validate = |output: &GpuTensorHandle, precision: ProviderPrecision| {
            protected.first().is_some_and(|first| {
                gpu_helpers::unary_gpu_output_matches_shape(
                    output,
                    first,
                    provider,
                    expected_shape,
                    UnaryGpuOutputContract {
                        storage: GpuTensorStorage::Real,
                        precision: Some(precision),
                        integer: None,
                        logical: false,
                        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
                    },
                )
            }) && protected
                .iter()
                .all(|input| !output.shares_allocation_with(input))
        };
        let validate_live = |output: &GpuTensorHandle| -> BuiltinResult<()> {
            output.allocation_lease().validate_live().map_err(|source| {
                let mut error = provider_error(ProviderExecutionError::from_source(source));
                error.gpu_gather_retry = GpuGatherRetry::Never;
                gpu_helpers::reject_gpu_output(output.clone(), error)
            })
        };
        validate_live(&result)?;
        let raw_precision = runmat_accelerate_api::handle_precision(&result);
        if !matches!(
            raw_precision,
            Some(ProviderPrecision::F32 | ProviderPrecision::F64)
        ) || !validate(&result, raw_precision.expect("checked floating precision"))
        {
            let mut error = invalid_output("invalid raw provider result");
            error.gpu_gather_retry = GpuGatherRetry::Never;
            return Err(gpu_helpers::reject_gpu_output(result, error));
        }
        if runmat_accelerate_api::handle_precision(&result) == Some(self.provider_precision()) {
            return Ok(Some(result));
        }
        let cast = match self {
            Self::Single => provider.unary_single(&result).await,
            Self::Double => provider.unary_double(&result).await,
        };
        match cast {
            Ok(output) => {
                validate_live(&output)?;
                if !validate(&output, self.provider_precision())
                    || output.shares_allocation_with(&result)
                {
                    let mut error = invalid_output("invalid converted provider result");
                    error.gpu_gather_retry = GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(output, error));
                }
                Ok(Some(output))
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
            Err(error) => Err(provider_error(ProviderExecutionError::from_source(error))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations(dtype: NumericDType) -> Value {
        Value::Tensor(Tensor::new_with_dtype(vec![1.0, 2.0], vec![2, 1], dtype).unwrap())
    }

    #[test]
    fn observation_classes_determine_precision() {
        let single = observations(NumericDType::F32);
        let double = observations(NumericDType::F64);
        assert_eq!(
            CovariancePrecision::observations(&single, None),
            CovariancePrecision::Single
        );
        assert_eq!(
            CovariancePrecision::observations(&single, Some(&single)),
            CovariancePrecision::Single
        );
        assert_eq!(
            CovariancePrecision::observations(&single, Some(&double)),
            CovariancePrecision::Double
        );
        assert_eq!(
            CovariancePrecision::observations(&Value::Num(1.0), None),
            CovariancePrecision::Double
        );
    }

    #[test]
    fn host_cast_preserves_shape_and_all_output_classes() {
        let input = Value::OutputList(vec![
            Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.5, 0.75], vec![2, 1]).unwrap()),
        ]);
        let Value::OutputList(outputs) = CovariancePrecision::Single
            .cast_host(input, "corrcoef")
            .unwrap()
        else {
            panic!("expected output list");
        };
        for (index, value) in outputs.iter().enumerate() {
            let Value::Tensor(tensor) = value else {
                panic!("expected tensor");
            };
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                tensor.shape,
                if index == 0 { vec![1, 1] } else { vec![2, 1] }
            );
        }
    }
}
