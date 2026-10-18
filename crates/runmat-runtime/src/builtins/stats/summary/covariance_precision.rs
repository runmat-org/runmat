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

#[derive(Debug)]
pub(super) enum AlignedCovarianceOutput {
    Resident(GpuTensorHandle),
    Materialized(Value),
}

/// The selected call's output geometry, borrowed allocations, and error owner.
/// Keeping these together makes a class conversion use the same protection and
/// public builtin identity as the producer whose result it adapts.
pub(super) struct CovarianceAlignmentContract<'a, 'b> {
    pub(super) expected_shape: &'a [usize],
    pub(super) protected: &'a [&'b GpuTensorHandle],
    pub(super) builtin: &'static str,
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
                Tensor::new_with_dtype(
                    tensor.materialize_f64(),
                    tensor.shape().to_vec(),
                    self.dtype(),
                )
                .map(Value::Tensor)
                .map_err(error)
            }
            Value::ComplexTensor(tensor) if tensor.numeric_dtype() != self.dtype() => {
                ComplexTensor::from_f64_values_with_dtype(
                    tensor.materialize_f64(),
                    tensor.shape().to_vec(),
                    self.dtype(),
                )
                .map(Value::ComplexTensor)
                .map_err(error)
            }
            Value::OutputList(values) => runmat_value::ValueSequence::from_output_storage(values)
                .map_err(crate::sequence::sequence_error_to_runtime)?
                .try_map(
                    |value| self.cast_host(value, builtin),
                    crate::sequence::sequence_error_to_runtime,
                )
                .map(|sequence| Value::OutputList(sequence.into_output_storage())),
            other => Ok(other),
        }
    }

    /// Validate a provider result before letting a cast hook read it. Both the
    /// raw result and any converted result must retain the expected geometry,
    /// real floating class, exact owner, live allocation, and distinct storage.
    /// The upstream provider result is already produced. Cast failures must
    /// retain that effect and cannot replay the whole covariance operation.
    pub(super) async fn align_provider(
        self,
        provider: &'static dyn AccelProvider,
        result: GpuTensorHandle,
        contract: CovarianceAlignmentContract<'_, '_>,
        invalid_output: impl Fn(&str) -> RuntimeError,
        provider_error: impl Fn(ProviderExecutionError) -> RuntimeError,
    ) -> BuiltinResult<AlignedCovarianceOutput> {
        let CovarianceAlignmentContract {
            expected_shape,
            protected,
            builtin,
        } = contract;
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
        if let Err(source) = result.allocation_lease().validate_live() {
            let mut error = provider_error(ProviderExecutionError::from_source(source));
            error.gpu_gather_retry = GpuGatherRetry::Never;
            return Err(gpu_helpers::reject_gpu_output(result, error));
        }
        let raw_precision = match runmat_accelerate_api::handle_precision(&result) {
            Some(ProviderPrecision::F32) => ProviderPrecision::F32,
            Some(ProviderPrecision::F64) => ProviderPrecision::F64,
            _ => {
                let mut error = invalid_output("invalid raw provider result");
                error.gpu_gather_retry = GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
        };
        if !validate(&result, raw_precision) {
            let mut error = invalid_output("invalid raw provider result");
            error.gpu_gather_retry = GpuGatherRetry::Never;
            return Err(gpu_helpers::reject_gpu_output(result, error));
        }
        if runmat_accelerate_api::handle_precision(&result) == Some(self.provider_precision()) {
            return Ok(AlignedCovarianceOutput::Resident(result));
        }
        let cast = match self {
            Self::Single => provider.unary_single(&result).await,
            Self::Double => provider.unary_double(&result).await,
        };
        match cast {
            Ok(output) => {
                if let Err(source) = output.allocation_lease().validate_live() {
                    let mut error = provider_error(ProviderExecutionError::from_source(source));
                    error.gpu_gather_retry = GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(output, error));
                }
                if !validate(&output, self.provider_precision())
                    || output.shares_allocation_with(&result)
                {
                    let mut error = invalid_output("invalid converted provider result");
                    error.gpu_gather_retry = GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(output, error));
                }
                Ok(AlignedCovarianceOutput::Resident(output))
            }
            Err(error) if error.is_unsupported() => {
                // The covariance/correlation result is already computed. A
                // pre-effect cast refusal may change only its representation;
                // it must never restart the numerical operation on the input.
                let downloaded =
                    gpu_helpers::download_value_preserving_residency_async(provider, &result)
                        .await
                        .map_err(|mut failure| {
                            failure.gpu_gather_retry = GpuGatherRetry::Never;
                            failure
                        })?;
                let materialized = self.cast_host(downloaded, builtin).map_err(|mut failure| {
                    failure.gpu_gather_retry = GpuGatherRetry::Never;
                    failure
                })?;
                Ok(AlignedCovarianceOutput::Materialized(materialized))
            }
            Err(error) => {
                let mut failure = provider_error(error.after_effect());
                failure.gpu_gather_retry = GpuGatherRetry::Never;
                Err(failure)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CastRejectingProvider(
        runmat_accelerate_api::test_support::TestAllocationOwner,
        std::sync::atomic::AtomicUsize,
    );

    impl AccelProvider for CastRejectingProvider {
        fn device_id(&self) -> u32 {
            self.0.device_id()
        }

        fn device_info(&self) -> String {
            "covariance cast rejection fixture".into()
        }

        fn upload(
            &self,
            _: &runmat_accelerate_api::HostTensorView<'_>,
        ) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("unexpected upload")
        }

        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            self.1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { anyhow::bail!("unexpected download") })
        }

        fn unary_single<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
            Box::pin(async {
                Err(ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "cast failed after provider work",
                    ),
                )
                .after_effect())
            })
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn completed_covariance_output_cannot_replay_after_cast_unsupported() {
        let provider: &'static CastRejectingProvider = Box::leak(Box::new(CastRejectingProvider(
            runmat_accelerate_api::test_support::TestAllocationOwner::new(),
            std::sync::atomic::AtomicUsize::new(0),
        )));
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider
            .0
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let produced = provider
            .0
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let result = futures::executor::block_on(CovariancePrecision::Single.align_provider(
            provider,
            produced.handle,
            CovarianceAlignmentContract {
                expected_shape: &[1, 1],
                protected: &[&input.handle],
                builtin: "cov",
            },
            |reason| build_runtime_error(reason).build(),
            |source| {
                crate::runtime_error::with_provider_execution_source(
                    build_runtime_error("covariance provider failure").build(),
                    source,
                )
            },
        ));
        let error = result.expect_err("completed covariance cannot replay");
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert!(error.source.is_some());
        assert_eq!(provider.1.load(std::sync::atomic::Ordering::SeqCst), 0);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while !cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
            cause = cause
                .source()
                .expect("original typed cast rejection retained");
        }
        assert_eq!(produced.probe.retirements(), 1);
        input.handle.allocation_lease().validate_live().unwrap();
    }

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
        let input = Value::output_list(vec![
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
            assert_eq!(tensor.shape(), if index == 0 { &[1, 1] } else { &[2, 1] });
        }
    }
}
