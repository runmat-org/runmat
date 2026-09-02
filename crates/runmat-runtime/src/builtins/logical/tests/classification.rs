use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCatalogEntry, BuiltinErrorDescriptor, NumericClassificationPredicate,
};
use runmat_value::{ComplexTensor, LogicalArray, NumericScalar, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct ClassificationBoundary {
    entry: &'static BuiltinCatalogEntry,
    invalid: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
    too_many_outputs: &'static BuiltinErrorDescriptor,
    predicate: NumericClassificationPredicate,
}

impl ClassificationBoundary {
    pub(super) const fn new(
        entry: &'static BuiltinCatalogEntry,
        invalid: &'static BuiltinErrorDescriptor,
        internal: &'static BuiltinErrorDescriptor,
        too_many_outputs: &'static BuiltinErrorDescriptor,
        predicate: NumericClassificationPredicate,
    ) -> Self {
        Self {
            entry,
            invalid,
            internal,
            too_many_outputs,
            predicate,
        }
    }

    pub(super) fn name(&self) -> &'static str {
        self.entry.identity.name
    }

    pub(super) async fn execute(&self, value: Value) -> BuiltinResult<Value> {
        self.reject_excess_outputs()?;
        match value {
            Value::GpuTensor(handle) => self.execute_resident(handle).await,
            host => self.execute_host(host),
        }
    }

    pub(super) fn classify_tensor(&self, value: Tensor) -> BuiltinResult<Value> {
        let shape = value.shape.clone();
        let bits = (0..value.len())
            .map(|index| {
                value
                    .numeric_value_at(index)
                    .map(|value| u8::from(self.classify_scalar(value)))
                    .ok_or_else(|| self.internal("numeric tensor storage is inconsistent"))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        self.logical_result(bits, shape)
    }

    fn reject_excess_outputs(&self) -> BuiltinResult<()> {
        if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
            return Err(self.error(self.too_many_outputs, "only one output is defined"));
        }
        Ok(())
    }

    async fn execute_resident(&self, handle: GpuTensorHandle) -> BuiltinResult<Value> {
        if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
            return self.resident_integer_mask(&handle);
        }
        let provider = gpu_helpers::exact_provider_for_handle(&handle)
            .ok_or_else(|| self.internal("no acceleration provider owns the input handle"))?;
        let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
        let provider_result = self.invoke_provider(provider, &handle);
        gpu_helpers::restore_handle_metadata(&handle, &metadata);
        match provider_result {
            Ok(mask) if self.valid_provider_mask(&handle, &mask, provider) => {
                Ok(gpu_helpers::logical_gpu_value(mask))
            }
            Ok(mask) => {
                gpu_helpers::free_unprotected_exact_owner(&mask, &[&handle]);
                Err(self.internal("provider returned an invalid logical mask"))
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                let host =
                    gpu_helpers::download_value_preserving_residency_async(provider, &handle)
                        .await
                        .map_err(|error| self.internal(error))?;
                let result = self.execute_host(host)?;
                let restored =
                    gpu_helpers::restore_class_preserving_value(&handle, result, self.name())
                        .map_err(|error| self.internal(error.message()))?;
                if runmat_accelerate_api::handle_is_explicit(&handle)
                    && !matches!(restored, Value::GpuTensor(_))
                {
                    return Err(self
                        .internal("provider cannot preserve explicit gpuArray output residency"));
                }
                Ok(restored)
            }
            Err(error) => Err(self.internal(format!("provider execution failed: {error}"))),
        }
    }

    fn invoke_provider(
        &self,
        provider: &dyn AccelProvider,
        handle: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self.predicate {
            NumericClassificationPredicate::Finite => provider.logical_isfinite(handle),
            NumericClassificationPredicate::Infinite => provider.logical_isinf(handle),
            NumericClassificationPredicate::Nan => provider.logical_isnan(handle),
        }
    }

    fn valid_provider_mask(
        &self,
        input: &GpuTensorHandle,
        output: &GpuTensorHandle,
        provider: &dyn AccelProvider,
    ) -> bool {
        !gpu_helpers::same_gpu_handle(input, output)
            && valid_provider_truth_handle(output, provider, &input.shape)
    }

    fn resident_integer_mask(&self, handle: &GpuTensorHandle) -> BuiltinResult<Value> {
        let integer = runmat_accelerate_api::handle_integer_type(handle)
            .ok_or_else(|| self.internal("resident integer metadata is missing"))?;
        if gpu_helpers::exact_provider_for_handle(handle).is_none()
            || !matches!(
                runmat_accelerate_api::handle_storage(handle),
                GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
            )
            || runmat_accelerate_api::handle_precision(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
            || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
        {
            return Err(self.internal("resident integer metadata is contradictory"));
        }
        let mask = self.logical_fill(
            handle.shape.clone(),
            matches!(self.predicate, NumericClassificationPredicate::Finite),
        )?;
        let restored = gpu_helpers::restore_class_preserving_value(handle, mask, self.name())
            .map_err(|error| self.internal(error.message()))?;
        if runmat_accelerate_api::handle_is_explicit(handle)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(
                self.internal("provider cannot preserve explicit gpuArray output residency")
            );
        }
        Ok(restored)
    }

    fn execute_host(&self, value: Value) -> BuiltinResult<Value> {
        match value {
            Value::Num(value) => Ok(Value::Bool(self.classify_float(value))),
            Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(matches!(
                self.predicate,
                NumericClassificationPredicate::Finite
            ))),
            Value::Complex(real, imaginary) => {
                Ok(Value::Bool(self.classify_complex(real, imaginary)))
            }
            Value::Tensor(value) => self.classify_tensor(value),
            Value::ComplexTensor(value) => self.classify_complex_tensor(value),
            Value::LogicalArray(value) => self.logical_fill(
                value.shape,
                matches!(self.predicate, NumericClassificationPredicate::Finite),
            ),
            Value::CharArray(value) => self.logical_fill(
                vec![value.rows, value.cols],
                matches!(self.predicate, NumericClassificationPredicate::Finite),
            ),
            Value::String(_) => Ok(Value::Bool(false)),
            Value::StringArray(value) => self.logical_fill(value.shape, false),
            _ => Err(self.error(self.invalid, "unsupported input representation")),
        }
    }

    fn classify_complex_tensor(&self, value: ComplexTensor) -> BuiltinResult<Value> {
        let shape = value.shape.clone();
        let bits = (0..value.len())
            .map(|index| {
                value
                    .numeric_value_at(index)
                    .map(|(real, imaginary)| {
                        u8::from(self.classify_scalar_complex(real, imaginary))
                    })
                    .ok_or_else(|| self.internal("complex tensor storage is inconsistent"))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        self.logical_result(bits, shape)
    }

    fn classify_scalar_complex(&self, real: NumericScalar, imaginary: NumericScalar) -> bool {
        match self.predicate {
            NumericClassificationPredicate::Finite => {
                self.classify_scalar(real) && self.classify_scalar(imaginary)
            }
            NumericClassificationPredicate::Infinite | NumericClassificationPredicate::Nan => {
                self.classify_scalar(real) || self.classify_scalar(imaginary)
            }
        }
    }

    fn classify_scalar(&self, value: NumericScalar) -> bool {
        match value {
            NumericScalar::F64(value) => self.classify_float(value),
            NumericScalar::F32(value) => self.classify_float(f64::from(value)),
            _ => matches!(self.predicate, NumericClassificationPredicate::Finite),
        }
    }

    fn classify_complex(&self, real: f64, imaginary: f64) -> bool {
        match self.predicate {
            NumericClassificationPredicate::Finite => {
                self.classify_float(real) && self.classify_float(imaginary)
            }
            NumericClassificationPredicate::Infinite | NumericClassificationPredicate::Nan => {
                self.classify_float(real) || self.classify_float(imaginary)
            }
        }
    }

    fn classify_float(&self, value: f64) -> bool {
        match self.predicate {
            NumericClassificationPredicate::Finite => value.is_finite(),
            NumericClassificationPredicate::Infinite => value.is_infinite(),
            NumericClassificationPredicate::Nan => value.is_nan(),
        }
    }

    fn logical_fill(&self, shape: Vec<usize>, value: bool) -> BuiltinResult<Value> {
        self.logical_result(vec![u8::from(value); tensor::element_count(&shape)], shape)
    }

    fn logical_result(&self, bits: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
        if tensor::element_count(&shape) != bits.len() {
            return Err(self.internal("logical mask length does not match its shape"));
        }
        if bits.len() == 1 {
            return Ok(Value::Bool(bits[0] != 0));
        }
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|error| self.internal(error))
    }

    fn internal(&self, detail: impl std::fmt::Display) -> RuntimeError {
        self.error(self.internal, detail)
    }

    fn error(
        &self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

pub(super) fn valid_provider_truth_handle(
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    expected_shape: &[usize],
) -> bool {
    let precision = runmat_accelerate_api::handle_precision(output);
    let logical = runmat_accelerate_api::handle_is_logical(output);
    output.shape == expected_shape
        && output.device_id == provider.device_id()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && precision.is_some()
        && gpu_helpers::gpu_class_metadata_matches(output, precision, None, logical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{
        AccelDownloadFuture, HostTensorView, ProviderPrecision, ThreadProviderGuard,
    };
    use runmat_builtins::{
        ISNAN_CATALOG_ENTRY, ISNAN_ERROR_INTERNAL, ISNAN_ERROR_INVALID_INPUT,
        ISNAN_ERROR_TOO_MANY_OUTPUTS,
    };
    use std::sync::OnceLock;

    #[derive(Clone, Copy)]
    enum HookResult {
        Unsupported,
        Failure,
    }

    struct ClassificationProvider {
        inner: InProcessProvider,
        result: HookResult,
    }

    impl ClassificationProvider {
        fn new(result: HookResult) -> Self {
            Self {
                inner: InProcessProvider::new(),
                result,
            }
        }
    }

    impl AccelProvider for ClassificationProvider {
        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }

        fn device_info(&self) -> String {
            self.inner.device_info()
        }

        fn precision(&self) -> ProviderPrecision {
            self.inner.precision()
        }

        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(host)
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.inner.download(handle)
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            self.inner.free(handle)
        }

        fn logical_isnan(&self, _input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
            match self.result {
                HookResult::Unsupported => {
                    Err(runmat_accelerate_api::unsupported_provider_operation(
                        "classification hook is unavailable",
                    ))
                }
                HookResult::Failure => anyhow::bail!("injected classification kernel failure"),
            }
        }
    }

    fn boundary() -> ClassificationBoundary {
        ClassificationBoundary::new(
            &ISNAN_CATALOG_ENTRY,
            &ISNAN_ERROR_INVALID_INPUT,
            &ISNAN_ERROR_INTERNAL,
            &ISNAN_ERROR_TOO_MANY_OUTPUTS,
            NumericClassificationPredicate::Nan,
        )
    }

    fn resident_input(provider: &dyn AccelProvider) -> GpuTensorHandle {
        let tensor = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).expect("tensor");
        gpu_helpers::upload_tensor(provider, &tensor).expect("upload")
    }

    fn test_provider(result: HookResult) -> &'static ClassificationProvider {
        static UNSUPPORTED: OnceLock<ClassificationProvider> = OnceLock::new();
        static FAILURE: OnceLock<ClassificationProvider> = OnceLock::new();
        match result {
            HookResult::Unsupported => {
                UNSUPPORTED.get_or_init(|| ClassificationProvider::new(HookResult::Unsupported))
            }
            HookResult::Failure => {
                FAILURE.get_or_init(|| ClassificationProvider::new(HookResult::Failure))
            }
        }
    }

    #[test]
    fn typed_unsupported_hook_uses_class_preserving_fallback() {
        let _state = test_support::accel_test_lock();
        let provider = test_provider(HookResult::Unsupported);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let input = resident_input(provider);

        let output = block_on(boundary().execute(Value::GpuTensor(input)))
            .expect("typed unsupported hook should fall back");
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).expect("gather logical mask");
        assert_eq!(
            gathered
                .into_numeric_storage()
                .expect("logical storage")
                .materialize_f64(),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn provider_failure_is_not_reclassified_as_unsupported() {
        let _state = test_support::accel_test_lock();
        let provider = test_provider(HookResult::Failure);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let input = resident_input(provider);

        let error = block_on(boundary().execute(Value::GpuTensor(input.clone())))
            .expect_err("provider failure must remain visible");
        assert_eq!(error.identifier(), Some("RunMat:isnan:InternalError"));
        assert!(error
            .message()
            .contains("injected classification kernel failure"));
        provider.free(&input).expect("free input");
    }
}
