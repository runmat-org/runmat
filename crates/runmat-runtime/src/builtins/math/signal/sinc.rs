//! MATLAB-compatible normalized `sinc` builtin for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, SINC_ERROR_GATHER_FAILED, SINC_ERROR_INTERNAL,
    SINC_ERROR_INVALID_INPUT, SINC_ERROR_MALFORMED_COMPLEX_BUFFER, SINC_ERROR_PROVIDER_FAILED,
    SINC_NONFLOATING_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, HostComplexBuffer, NumericDType, Tensor, Value,
};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::common::output_precision::SignalPrecision;
use crate::builtins::math::signal::common::{
    admit_resident_waveform_operand, map_real_waveform, prepared_real_integer_tensor,
    waveform_input_tensor, SignalCooperation, WaveformOutputMode,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sinc";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::sinc")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sinc" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute normalized sinc on-device via unary_sinc for real and complex-interleaved gpuArrays; unsupported hooks gather and apply the MATLAB-compatible host implementation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::sinc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            writer.write_formatted(format_args!("select(select(sin((3.141592653589793 * {input})) / (3.141592653589793 * {input}), 0.0, isFinite({input}) && floor(abs({input})) == abs({input})), 1.0, {input} == 0.0)"))
        }),
    }),
    reduction: None,
    emits_nan: true,
    notes:
        "Fusion emits a guarded normalized sinc expression with explicit zero and integer branches.",
};

fn sinc_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sinc_error_with_message(error.message, error)
}

fn sinc_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sinc_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn sinc_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn provider_error(err: anyhow::Error) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        sinc_error_with_detail(&SINC_ERROR_PROVIDER_FAILED, err.to_string()),
        err.into(),
    )
}

#[runtime_builtin(
    name = "sinc",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::sinc"
)]
async fn sinc_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    sinc_in_runtime(runtime.as_ref(), value).await
}

#[cfg(test)]
async fn sinc_builtin(value: Value) -> BuiltinResult<Value> {
    sinc_in_runtime(None, value).await
}

async fn sinc_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME)
        .with_integer_error_descriptor(&SINC_ERROR_INTERNAL);
    cooperation.completed(0).await?;
    if is_supported_nonfloating_input(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINC_NONFLOATING_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    admit_resident_waveform_operand(&value, BUILTIN_NAME, "input", &SINC_ERROR_INTERNAL)?;
    // Only a resident wide integer needs a retained native readback. Host
    // integer sinc has an exact zero/nonzero rule and must not do a separate
    // binary64-exactness scan before its numerical pass.
    let prepared_payload = match &value {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle)
                .is_some_and(|element| element.element_size() > 4) =>
        {
            cooperation
                .prepare_integer_receipt(&value, &SINC_NONFLOATING_INPUT_EXTENSION)
                .await?
                .into_resident_payload()
        }
        _ => None,
    };
    match value {
        Value::GpuTensor(handle) => sinc_gpu(handle, prepared_payload, &mut cooperation).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = sinc_complex_value(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => sinc_complex_tensor(tensor, &mut cooperation).await,
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(sinc_error(&SINC_ERROR_INVALID_INPUT))
        }
        other => sinc_real(other, &mut cooperation).await,
    }
}

fn is_supported_nonfloating_input(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

async fn sinc_gpu(
    handle: GpuTensorHandle,
    prepared_payload: Option<crate::builtins::common::validation::PreparedIntegerPayload>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        let tensor = match prepared_real_integer_tensor(prepared_payload)
            .map_err(|detail| sinc_error_with_detail(&SINC_ERROR_INVALID_INPUT, detail))?
        {
            Some(tensor) => tensor,
            None => gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    let mut error =
                        sinc_error_with_detail(&SINC_ERROR_GATHER_FAILED, source.message());
                    error.source = Some(Box::new(source));
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?,
        };
        let host = tensor::tensor_into_value(
            sinc_tensor_mode(tensor, WaveformOutputMode::Double, cooperation).await?,
        );
        return gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME);
    }
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
        if runmat_accelerate_api::handle_precision(&handle)
            .is_some_and(|precision| precision != provider.precision())
            || runmat_accelerate_api::handle_integer_type(&handle).is_some()
            || runmat_accelerate_api::handle_is_logical(&handle)
        {
            return sinc_gpu_host_fallback(provider, &handle, cooperation).await;
        }
        match provider.unary_sinc(&handle).await {
            Ok(out) => {
                let contract = gpu_helpers::UnaryGpuOutputContract {
                    storage: runmat_accelerate_api::handle_storage(&handle),
                    precision: runmat_accelerate_api::handle_precision(&handle),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
                };
                if !gpu_helpers::unary_gpu_output_matches(&out, &handle, provider, contract) {
                    let mut error = sinc_error_with_detail(
                        &SINC_ERROR_INTERNAL,
                        "provider returned malformed or aliased unary output",
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(out, error));
                }
                let out = gpu_helpers::propagate_owned_output_provenance(out, [&handle])?;
                return gpu_helpers::resident_gpu_value(out);
            }
            Err(err) if gpu_helpers::provider_hook_allows_host_fallback(&err) => {
                return sinc_gpu_host_fallback(provider, &handle, cooperation).await;
            }
            Err(err) => return Err(provider_error(err)),
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    Ok(tensor::tensor_into_value(
        sinc_tensor(tensor, cooperation).await?,
    ))
}

async fn sinc_gpu_host_fallback(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let host = gpu_helpers::download_floating_projection_async(provider, handle)
        .await
        .map_err(|err| {
            let mut failure = sinc_error_with_detail(&SINC_ERROR_GATHER_FAILED, err.to_string());
            failure.source = Some(Box::new(err));
            failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
            failure
        })?;

    let runmat_accelerate_api::HostTensorOwned {
        mut data,
        shape,
        storage,
    } = host;
    let precision = SignalPrecision::of_handle(handle);
    if precision == SignalPrecision::Single {
        for chunk in data.chunks_mut(4096) {
            for value in chunk.iter_mut() {
                *value = (*value as f32) as f64;
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }

    if storage == GpuTensorStorage::ComplexInterleaved {
        let chunks = data.chunks_exact(2);
        if !chunks.remainder().is_empty() {
            return Err(sinc_error(&SINC_ERROR_MALFORMED_COMPLEX_BUFFER));
        }
        let mut complex = Vec::new();
        complex.try_reserve_exact(data.len() / 2).map_err(|cause| {
            crate::builtins::math::signal::common::WaveformMapError::Allocation(cause)
                .into_runtime_error(BUILTIN_NAME, &SINC_ERROR_INTERNAL)
        })?;
        for chunk in data.chunks(4096) {
            for pair in chunk.chunks_exact(2) {
                complex.push((pair[0], pair[1]));
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        let dtype = match precision {
            SignalPrecision::Single => NumericDType::F32,
            SignalPrecision::Double => NumericDType::F64,
        };
        let tensor = ComplexTensor::from_f64_values_with_dtype(complex, shape, dtype)
            .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))?;
        let host = sinc_complex_tensor(tensor, cooperation).await?;
        return gpu_helpers::restore_class_preserving_value(handle, host, BUILTIN_NAME);
    }

    let dtype = match precision {
        SignalPrecision::Single => NumericDType::F32,
        SignalPrecision::Double => NumericDType::F64,
    };
    let tensor = Tensor::new_with_dtype(data, shape, dtype)
        .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))?;
    let host = tensor::tensor_into_value(sinc_tensor(tensor, cooperation).await?);
    gpu_helpers::restore_class_preserving_value(handle, host, BUILTIN_NAME)
}

async fn sinc_real(value: Value, cooperation: &mut SignalCooperation) -> BuiltinResult<Value> {
    let tensor = waveform_input_tensor(value, BUILTIN_NAME, cooperation)
        .await?
        .map_err(|error| {
            error.into_runtime_error(
                BUILTIN_NAME,
                &SINC_ERROR_INVALID_INPUT,
                &SINC_ERROR_INTERNAL,
            )
        })?;
    Ok(tensor::tensor_into_value(
        sinc_tensor(tensor, cooperation).await?,
    ))
}

async fn sinc_tensor(tensor: Tensor, cooperation: &mut SignalCooperation) -> BuiltinResult<Tensor> {
    sinc_tensor_mode(tensor, WaveformOutputMode::PreserveSingle, cooperation).await
}

async fn sinc_tensor_mode(
    tensor: Tensor,
    output_mode: WaveformOutputMode,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Tensor> {
    map_real_waveform(tensor, output_mode, sinc_real_value, cooperation)
        .await?
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &SINC_ERROR_INTERNAL))
}

async fn sinc_complex_tensor(
    tensor: ComplexTensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    cooperation.completed(0).await?;
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            map_complex_values(
                values,
                std::convert::identity,
                std::convert::identity,
                cooperation,
            )
            .await?,
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            map_complex_values(
                values,
                |(re, im)| (f64::from(re), f64::from(im)),
                |(re, im)| (re as f32, im as f32),
                cooperation,
            )
            .await?,
        ),
        ComplexStorage::Integer(_) => return Err(sinc_error(&SINC_ERROR_INVALID_INPUT)),
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))?;
    Ok(complex_tensor_into_value(tensor))
}

async fn map_complex_values<Input: Copy, Output: Copy>(
    values: HostComplexBuffer<Input>,
    into_double: impl Fn((Input, Input)) -> (f64, f64),
    from_double: impl Fn((f64, f64)) -> (Output, Output),
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<HostComplexBuffer<Output>> {
    let mut result = Vec::new();
    result.try_reserve_exact(values.len()).map_err(|cause| {
        crate::builtins::math::signal::common::WaveformMapError::Allocation(cause)
            .into_runtime_error(BUILTIN_NAME, &SINC_ERROR_INTERNAL)
    })?;
    for chunk in values.chunks(4096) {
        for &value in chunk {
            let (re, im) = into_double((value.0, value.1));
            let (re, im) = from_double(sinc_complex_value(re, im));
            result.push(ComplexElement(re, im));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(HostComplexBuffer::from_elements(result))
}

#[inline]
fn sinc_real_value(value: f64) -> f64 {
    if value == 0.0 {
        return 1.0;
    }
    if value.is_finite() && value.fract() == 0.0 {
        return 0.0;
    }
    let scaled = std::f64::consts::PI * value;
    scaled.sin() / scaled
}

fn sinc_complex_value(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        return (sinc_real_value(re), 0.0);
    }

    let scaled_re = std::f64::consts::PI * re;
    let scaled_im = std::f64::consts::PI * im;
    let num_re = scaled_re.sin() * scaled_im.cosh();
    let num_im = scaled_re.cos() * scaled_im.sinh();
    let denom_norm = scaled_re.mul_add(scaled_re, scaled_im * scaled_im);
    (
        (num_re * scaled_re + num_im * scaled_im) / denom_norm,
        (num_im * scaled_re - num_re * scaled_im) / denom_norm,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::spec::FusionExprContext;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorView,
    };
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{CharArray, IntValue, NumericStorage};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(sinc_builtin(value))
    }

    fn assert_close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-12, "got {got}, expected {want}");
    }

    fn assert_complex_close(got: (f64, f64), want: (f64, f64)) {
        assert_close(got.0, want.0);
        assert_close(got.1, want.1);
    }

    #[derive(Default)]
    struct UnsupportedSincProvider(runmat_accelerate::simple_provider::InProcessProvider);

    impl AccelProvider for UnsupportedSincProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.0.upload(host)
        }
        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.0.download(handle)
        }
        fn device_info(&self) -> String {
            "unsupported-sinc-test-provider".to_string()
        }
        fn device_id(&self) -> u32 {
            self.0.device_id()
        }
    }

    #[derive(Default)]
    struct FailingSincProvider(runmat_accelerate::simple_provider::InProcessProvider);

    impl AccelProvider for FailingSincProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.0.upload(host)
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move { Err(anyhow::anyhow!("download should not be called")) })
        }
        fn device_info(&self) -> String {
            "failing-sinc-test-provider".to_string()
        }
        fn device_id(&self) -> u32 {
            self.0.device_id()
        }
        fn unary_sinc<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { Err(anyhow::anyhow!("device lost while running unary_sinc")) })
        }
    }

    #[derive(Default)]
    struct ComplexSincProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
        malformed_download: bool,
    }

    impl AccelProvider for ComplexSincProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(host)
        }
        fn upload_numeric_raw(
            &self,
            host: &runmat_accelerate_api::HostNumericTensorView<'_>,
        ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
            self.inner.upload_numeric_raw(host)
        }
        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                let mut result = self.inner.download(handle).await?;
                if self.malformed_download {
                    result.data.pop();
                }
                Ok(result)
            })
        }
        fn device_info(&self) -> String {
            "complex-sinc-test-provider".to_string()
        }
        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }
        fn unary_sinc<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "unary_sinc",
                ))
            })
        }
    }

    #[test]
    fn sinc_catalog_binding_and_provider_route() {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME).unwrap();
        assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
        assert!(matches!(GPU_SPEC.op_kind, GpuOpKind::Elementwise));
        assert!(FUSION_SPEC.elementwise.is_some());
        let descriptor = entry.descriptor;
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = sinc(X)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.SINC.PROVIDER_FAILED"));
    }

    #[test]
    fn sinc_zero_returns_one() {
        let result = call(Value::Num(0.0)).expect("sinc");
        match result {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sinc_nonzero_integer_inputs_are_exact_zero() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 9_007_199_254_740_992.0],
            vec![1, 6],
        )
        .unwrap();
        let result = call(Value::Tensor(tensor)).expect("sinc");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 6]);
                assert_eq!(out.materialize_f64(), vec![0.0; 6]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = call(Value::Int(IntValue::I32(-3))).expect("sinc int");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar, got {other:?}"),
        }
        let wide = call(Value::Int(IntValue::U64(u64::MAX)))
            .expect("nonzero native integer has exact sinc zero without a binary64 cast");
        assert!(matches!(wide, Value::Num(0.0)));
    }

    #[test]
    fn sinc_fusion_integer_guard_matches_host() {
        let template = FUSION_SPEC.elementwise.expect("fusion template");
        let inputs = [crate::builtins::common::spec::FusionExprInput::text("x")];
        let ctx = FusionExprContext {
            scalar_ty: ScalarType::F64,
            inputs: &inputs,
            constants: &[],
        };
        let expr = template.wgsl_body.render(&ctx).expect("fusion expression");
        assert!(expr.contains("isFinite(x) && floor(abs(x)) == abs(x)"));
        assert!(
            !expr.contains("9.007199254740992e15"),
            "fusion must not cap exact-integer handling below host/unary GPU behavior"
        );
    }

    #[test]
    fn sinc_matches_normalized_values() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 0.5], vec![1, 4]).unwrap();
        let result = call(Value::Tensor(tensor)).expect("sinc");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 4]);
                let expected = [0.0, 1.0, 0.0, 2.0 / std::f64::consts::PI];
                for (got, want) in out.materialize_f64().iter().zip(expected) {
                    assert_close(*got, want);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn sinc_preserves_matrix_shape() {
        let tensor = Tensor::new(vec![0.0, 0.5, 1.5, 2.0], vec![2, 2]).unwrap();
        let result = call(Value::Tensor(tensor)).expect("sinc");
        match result {
            Value::Tensor(out) => assert_eq!(out.shape(), [2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn sinc_preserves_native_single_storage() {
        let input =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 0.5, 1.0]), vec![1, 3])
                .unwrap();
        let result = call(Value::Tensor(input)).expect("single sinc");
        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, (2.0 / std::f32::consts::PI), 0.0])
        );
    }

    #[test]
    fn sinc_logical_inputs_are_numeric() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let zero = call(Value::Bool(false)).expect("sinc false");
        let one = call(Value::Bool(true)).expect("sinc true");
        match (zero, one) {
            (Value::Num(zero), Value::Num(one)) => {
                assert_eq!(zero, 1.0);
                assert_eq!(one, 0.0);
            }
            other => panic!("expected scalar results, got {other:?}"),
        }
    }

    #[test]
    fn integer_and_logical_resident_fallback_keep_double_on_single_provider() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_f32_test_provider(|provider| {
            let integer = Tensor::new_integer(
                runmat_value::IntegerStorage::U64(vec![0, u64::MAX]),
                vec![1, 2],
            )
            .unwrap();
            let integer = gpu_helpers::upload_tensor(provider, &integer)
                .expect("integer upload")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap();
            let integer_result = call(Value::GpuTensor(integer)).expect("integer sinc fallback");
            let Value::Tensor(integer_result) = integer_result else {
                panic!("single provider cannot retain double integer result");
            };
            assert_eq!(integer_result.numeric_dtype(), NumericDType::F64);
            assert_eq!(integer_result.materialize_f64(), vec![1.0, 0.0]);

            let logical = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0, 1],
                    shape: &[1, 2],
                })
                .expect("logical upload")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap();
            let logical_result = call(Value::GpuTensor(logical)).expect("logical sinc fallback");
            let Value::Tensor(logical_result) = logical_result else {
                panic!("single provider cannot retain double logical result");
            };
            assert_eq!(logical_result.numeric_dtype(), NumericDType::F64);
            assert_eq!(logical_result.materialize_f64(), vec![1.0, 0.0]);
        });
    }

    #[test]
    fn sinc_nonfloating_input_follows_compatibility_mode() {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                call(Value::Int(IntValue::I16(1))).expect_err("MATLAB mode rejects integer input");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SincNonfloatingInputExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(Value::Int(IntValue::I16(1))).is_ok());
        }

        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::I16,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(Value::GpuTensor(handle.clone()))
                .expect_err("MATLAB mode rejects resident integer input before dispatch");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SincNonfloatingInputExtension")
            );
        }
    }

    #[test]
    fn sinc_complex_scalar_matches_formula() {
        let result = call(Value::Complex(0.5, 0.25)).expect("sinc complex");
        let expected = sinc_complex_value(0.5, 0.25);
        match result {
            Value::Complex(re, im) => assert_complex_close((re, im), expected),
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn sinc_complex_real_integer_is_exact_zero() {
        let result = call(Value::Complex(2.0, 0.0)).expect("sinc complex integer");
        match result {
            Value::Complex(re, im) => assert_eq!((re, im), (0.0, 0.0)),
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn sinc_complex_tensor_preserves_shape() {
        let tensor = ComplexTensor::new(vec![(0.0, 0.0), (0.5, 0.25)], vec![2, 1]).unwrap();
        let result = call(Value::ComplexTensor(tensor)).expect("sinc complex tensor");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert_complex_close(out.materialize_f64()[0], (1.0, 0.0));
                assert_complex_close(out.materialize_f64()[1], sinc_complex_value(0.5, 0.25));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn sinc_rejects_text_inputs() {
        let err = call(Value::CharArray(CharArray::new_row("abc"))).expect_err("sinc text");
        assert!(err.message().contains("sinc: expected numeric input"));
    }

    #[test]
    fn sinc_gpu_falls_back_only_for_explicit_unsupported_hook() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider =
            Box::leak(Box::new(UnsupportedSincProvider::default()));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = provider
            .upload(&HostTensorView {
                data: &[0.0, 0.5, 1.0],
                shape: &[1, 3],
            })
            .expect("owned input");
        let result = call(Value::GpuTensor(handle)).expect("sinc gpu fallback");
        let Value::GpuTensor(ref output) = result else {
            panic!("supported fallback restoration must remain resident: {result:?}");
        };
        assert_eq!(output.shape(), &[1, 3]);
        let result =
            block_on(gpu_helpers::gather_value_async(&result)).expect("gather restored result");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3]);
                let expected = [1.0, 2.0 / std::f64::consts::PI, 0.0];
                for (got, want) in out.materialize_f64().iter().zip(expected) {
                    assert_close(*got, want);
                }
            }
            other => panic!("expected host tensor fallback, got {other:?}"),
        }
    }

    #[test]
    fn stale_resident_input_rejects_before_sinc_hook_or_download() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Default)]
        struct CountedSincProvider {
            owner: runmat_accelerate_api::test_support::TestAllocationOwner,
            hooks: AtomicUsize,
            downloads: AtomicUsize,
        }
        impl AccelProvider for CountedSincProvider {
            fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
                anyhow::bail!("fixture allocation owner supplies inputs")
            }
            fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
                self.downloads.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { anyhow::bail!("unavailable allocation must not be downloaded") })
            }
            fn device_info(&self) -> String {
                "sinc unavailable allocation fixture".into()
            }
            fn device_id(&self) -> u32 {
                self.owner.device_id()
            }
            fn unary_sinc<'a>(
                &'a self,
                _: &'a GpuTensorHandle,
            ) -> AccelProviderFuture<'a, GpuTensorHandle> {
                self.hooks.fetch_add(1, Ordering::SeqCst);
                Box::pin(async {
                    Err(runmat_accelerate_api::unsupported_provider_operation(
                        "unary_sinc",
                    ))
                })
            }
        }

        let _guard = test_support::accel_test_lock();
        let provider: &'static CountedSincProvider =
            Box::leak(Box::new(CountedSincProvider::default()));
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let allocation = provider
            .owner
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        allocation.probe.mark_unavailable();
        let error = call(Value::GpuTensor(allocation.handle.clone()))
            .expect_err("unavailable input allocation");
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(allocation.probe.retirements(), 0);
    }

    #[test]
    fn sinc_gpu_propagates_provider_execution_errors() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider =
            Box::leak(Box::new(FailingSincProvider::default()));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = provider
            .upload(&HostTensorView {
                data: &[0.0, 0.5, 1.0],
                shape: &[1, 3],
            })
            .expect("owned input");
        let err = call(Value::GpuTensor(handle)).expect_err("provider error should surface");
        assert!(err
            .message()
            .contains("sinc: GPU provider unary_sinc failed"));
        assert!(err
            .message()
            .contains("device lost while running unary_sinc"));
    }

    #[test]
    fn sinc_gpu_complex_interleaved_falls_back_when_hook_unsupported() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider =
            Box::leak(Box::new(ComplexSincProvider::default()));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F64(&[0.0, 0.0, 0.5, 0.25]),
                shape: &[2, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("owned complex input");

        let result = call(Value::GpuTensor(handle)).expect("complex sinc gpu fallback");
        let Value::GpuTensor(ref output) = result else {
            panic!("supported complex restoration must remain resident: {result:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(output),
            GpuTensorStorage::ComplexInterleaved
        );
        let result = block_on(gpu_helpers::gather_value_async(&result))
            .expect("gather restored complex result");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert_complex_close(out.materialize_f64()[0], (1.0, 0.0));
                assert_complex_close(out.materialize_f64()[1], sinc_complex_value(0.5, 0.25));
            }
            other => panic!("expected complex tensor fallback, got {other:?}"),
        }
    }

    #[test]
    fn sinc_gpu_complex_interleaved_rejects_odd_buffer_length() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider = Box::leak(Box::new(ComplexSincProvider {
            malformed_download: true,
            ..Default::default()
        }));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F64(&[0.0, 0.0, 0.5, 0.25]),
                shape: &[2, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("owned complex input");

        let err = call(Value::GpuTensor(handle)).expect_err("odd complex buffer should error");
        assert_eq!(err.identifier(), Some("RunMat:gather:DownloadFailed"));
    }

    #[test]
    fn sinc_gpu_input_stays_resident_when_provider_supports_sinc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(Value::GpuTensor(handle)).expect("sinc gpu");
            assert!(
                matches!(result, Value::GpuTensor(_)),
                "provider-backed sinc should keep gpuArray output resident"
            );
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 3]);
            let expected = [1.0, 2.0 / std::f64::consts::PI, 0.0];
            for (got, want) in gathered.materialize_f64().iter().zip(expected) {
                assert_close(*got, want);
            }
        });
    }

    #[test]
    fn sinc_gpu_complex_input_stays_resident_when_provider_supports_sinc() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(0.0, 0.0), (0.5, 0.25)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = call(Value::GpuTensor(handle)).expect("sinc gpu complex");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                out_handle,
            )))
            .expect("gather");
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected complex tensor");
            };
            assert_eq!(out.shape(), complex.shape());
            assert_complex_close(out.materialize_f64()[0], (1.0, 0.0));
            assert_complex_close(out.materialize_f64()[1], sinc_complex_value(0.5, 0.25));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn sinc_wgpu_matches_cpu_elementwise() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        if runmat_accelerate_api::provider().is_none() {
            return;
        }
        let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.25, 2.0], vec![1, 5]).unwrap();
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let cpu =
            block_on(sinc_real(Value::Tensor(tensor.clone()), &mut cooperation)).expect("cpu sinc");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu = block_on(sinc_gpu(handle, None, &mut cooperation)).expect("gpu sinc");
        let gathered = test_support::gather(gpu).expect("gather");
        let expected = test_support::gather(cpu).expect("gather cpu");
        assert_eq!(gathered.shape(), expected.shape());
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, want) in gathered
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!((got - want).abs() < tol, "|{got} - {want}| >= {tol}");
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn sinc_wgpu_complex_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let complex =
            ComplexTensor::new(vec![(0.0, 0.0), (0.5, 0.25), (1.0, -0.5)], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let gpu = block_on(sinc_gpu(handle, None, &mut cooperation)).expect("gpu sinc");
        let Value::GpuTensor(out_handle) = gpu else {
            panic!("expected resident gpu output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
            out_handle,
        )))
        .expect("gather");
        let Value::ComplexTensor(actual) = gathered else {
            panic!("expected complex tensor");
        };
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let expected =
            match block_on(sinc_complex_tensor(complex, &mut cooperation)).expect("cpu sinc") {
                Value::ComplexTensor(tensor) => tensor,
                other => panic!("expected complex tensor, got {other:?}"),
            };
        assert_eq!(actual.shape(), expected.shape());
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, want) in actual
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!(
                (got.0 - want.0).abs() < tol && (got.1 - want.1).abs() < tol,
                "got {got:?}, expected {want:?}, tol {tol}"
            );
        }
    }
}
