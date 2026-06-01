//! MATLAB-compatible normalized `sinc` builtin for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sinc";

const SINC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise normalized sinc output.",
}];

const SINC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input values.",
}];

const SINC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sinc(X)",
    inputs: &SINC_INPUTS,
    outputs: &SINC_OUTPUT,
}];

const SINC_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.INVALID_INPUT",
    identifier: Some("RunMat:sinc:InvalidInput"),
    when: "Input cannot be interpreted as numeric/complex data.",
    message: "sinc: expected numeric input",
};

const SINC_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.PROVIDER_FAILED",
    identifier: Some("RunMat:sinc:ProviderFailed"),
    when: "Provider unary_sinc dispatch fails with a non-unsupported error.",
    message: "sinc: GPU provider unary_sinc failed",
};

const SINC_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.GATHER_FAILED",
    identifier: Some("RunMat:gather:DownloadFailed"),
    when: "GPU host fallback cannot download source data.",
    message: "gather: download failed",
};

const SINC_ERROR_MALFORMED_COMPLEX_BUFFER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.MALFORMED_COMPLEX_BUFFER",
    identifier: Some("RunMat:sinc:MalformedComplexBuffer"),
    when: "Complex-interleaved fallback buffer has odd element count.",
    message: "sinc: malformed complex buffer, odd length",
};

const SINC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINC.INTERNAL",
    identifier: Some("RunMat:sinc:InternalError"),
    when: "Internal tensor conversion/materialization fails.",
    message: "sinc: internal error",
};

const SINC_ERRORS: [BuiltinErrorDescriptor; 5] = [
    SINC_ERROR_INVALID_INPUT,
    SINC_ERROR_PROVIDER_FAILED,
    SINC_ERROR_GATHER_FAILED,
    SINC_ERROR_MALFORMED_COMPLEX_BUFFER,
    SINC_ERROR_INTERNAL,
];

pub const SINC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SINC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SINC_ERRORS,
};

#[derive(Debug)]
struct ProviderAnyhowError {
    message: String,
}

impl std::fmt::Display for ProviderAnyhowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderAnyhowError {}

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
    notes: "Providers may execute real normalized sinc on-device via unary_sinc; complex-interleaved handles and unsupported hooks gather and apply the MATLAB-compatible host implementation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::sinc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let scaled = format!("(3.141592653589793 * {input})");
            Ok(format!(
                "select(select(sin({scaled}) / {scaled}, 0.0, isFinite({input}) && floor(abs({input})) == abs({input})), 1.0, {input} == 0.0)"
            ))
        },
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

fn sinc_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn provider_error_is_unsupported(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.to_string() == "unary_sinc not supported by provider")
}

fn provider_error(err: anyhow::Error) -> RuntimeError {
    let message = err.to_string();
    let source = ProviderAnyhowError {
        message: format!("{err:?}"),
    };
    sinc_error_with_source(&SINC_ERROR_PROVIDER_FAILED, message, source)
}

#[runtime_builtin(
    name = "sinc",
    category = "math/signal",
    summary = "Compute the normalized sinc function element-wise.",
    keywords = "sinc,normalized sinc,signal processing,elementwise",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::signal::sinc::SINC_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sinc"
)]
async fn sinc_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => sinc_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = sinc_complex_value(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => sinc_complex_tensor(tensor),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(sinc_error(&SINC_ERROR_INVALID_INPUT))
        }
        other => sinc_real(other),
    }
}

async fn sinc_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
            return sinc_gpu_host_fallback(provider, &handle).await;
        }
        match provider.unary_sinc(&handle).await {
            Ok(out) => return Ok(gpu_helpers::resident_gpu_value(out)),
            Err(err) if provider_error_is_unsupported(&err) => {
                return sinc_gpu_host_fallback(provider, &handle).await;
            }
            Err(err) => return Err(provider_error(err)),
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    Ok(tensor::tensor_into_value(sinc_tensor(tensor)?))
}

async fn sinc_gpu_host_fallback(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let host = crate::dispatcher::download_handle_async(provider, handle)
        .await
        .map_err(|err| sinc_error_with_detail(&SINC_ERROR_GATHER_FAILED, err.to_string()))?;

    let runmat_accelerate_api::HostTensorOwned {
        mut data,
        shape,
        storage,
    } = host;
    let precision =
        runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
    if matches!(precision, runmat_accelerate_api::ProviderPrecision::F32) {
        for value in &mut data {
            *value = (*value as f32) as f64;
        }
    }

    if storage == GpuTensorStorage::ComplexInterleaved {
        let chunks = data.chunks_exact(2);
        if !chunks.remainder().is_empty() {
            return Err(sinc_error(&SINC_ERROR_MALFORMED_COMPLEX_BUFFER));
        }
        let complex = chunks.map(|chunk| (chunk[0], chunk[1])).collect::<Vec<_>>();
        let tensor = ComplexTensor::new(complex, shape)
            .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))?;
        return sinc_complex_tensor(tensor);
    }

    let dtype = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
    };
    let tensor = Tensor::new_with_dtype(data, shape, dtype)
        .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))?;
    Ok(tensor::tensor_into_value(sinc_tensor(tensor)?))
}

fn sinc_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INVALID_INPUT, &e))?;
    Ok(tensor::tensor_into_value(sinc_tensor(tensor)?))
}

fn sinc_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| sinc_real_value(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))
}

fn sinc_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| sinc_complex_value(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| sinc_error_with_detail(&SINC_ERROR_INTERNAL, &e))?;
    Ok(complex_tensor_into_value(tensor))
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
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorOwned,
        HostTensorView,
    };
    use runmat_builtins::{
        builtin_function_by_name, AccelTag, CharArray, IntValue, ResolveContext, Type,
    };

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

    struct UnsupportedSincProvider;

    impl AccelProvider for UnsupportedSincProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: 2,
            })
        }

        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0, 0.5, 1.0],
                    shape: vec![1, 3],
                    storage: GpuTensorStorage::Real,
                })
            })
        }

        fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "unsupported-sinc-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            90_001
        }
    }

    struct FailingSincProvider;

    impl AccelProvider for FailingSincProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: 1,
            })
        }

        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move { Err(anyhow::anyhow!("download should not be called")) })
        }

        fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "failing-sinc-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            90_002
        }

        fn unary_sinc<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { Err(anyhow::anyhow!("device lost while running unary_sinc")) })
        }
    }

    struct ComplexSincProvider;

    impl AccelProvider for ComplexSincProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: 3,
            })
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                if handle.buffer_id == 4 {
                    return Ok(HostTensorOwned {
                        data: vec![0.0, 0.0, 0.5],
                        shape: vec![2, 1],
                        storage: GpuTensorStorage::ComplexInterleaved,
                    });
                }
                Ok(HostTensorOwned {
                    data: vec![0.0, 0.0, 0.5, 0.25],
                    shape: vec![2, 1],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
            })
        }

        fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "complex-sinc-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            90_003
        }

        fn unary_sinc<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { Err(anyhow::anyhow!("unary_sinc should not be called")) })
        }
    }

    #[test]
    fn sinc_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn sinc_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn sinc_registers_unary_acceleration_tag() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("registered sinc builtin");
        assert!(
            builtin
                .accel_tags
                .iter()
                .any(|tag| matches!(tag, AccelTag::Unary)),
            "sinc must be tagged unary so native auto-promotion uploads host numeric inputs for unary_sinc"
        );
    }

    #[test]
    fn sinc_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("registered sinc builtin");
        let descriptor = builtin.descriptor.expect("sinc descriptor");
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
                assert_eq!(out.shape, vec![1, 6]);
                assert_eq!(out.data, vec![0.0; 6]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let result = call(Value::Int(IntValue::I32(-3))).expect("sinc int");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sinc_fusion_integer_guard_matches_host() {
        let template = FUSION_SPEC.elementwise.expect("fusion template");
        let inputs = ["x"];
        let ctx = FusionExprContext {
            scalar_ty: ScalarType::F64,
            inputs: &inputs,
            constants: &[],
        };
        let expr = (template.wgsl_body)(&ctx).expect("fusion expression");
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
                assert_eq!(out.shape, vec![1, 4]);
                let expected = [0.0, 1.0, 0.0, 2.0 / std::f64::consts::PI];
                for (got, want) in out.data.iter().zip(expected) {
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
            Value::Tensor(out) => assert_eq!(out.shape, vec![2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn sinc_logical_inputs_are_numeric() {
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
                assert_eq!(out.shape, vec![2, 1]);
                assert_complex_close(out.data[0], (1.0, 0.0));
                assert_complex_close(out.data[1], sinc_complex_value(0.5, 0.25));
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
        let provider: &'static dyn AccelProvider = Box::leak(Box::new(UnsupportedSincProvider));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: provider.device_id(),
            buffer_id: 2,
        };
        let result = call(Value::GpuTensor(handle)).expect("sinc gpu fallback");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                let expected = [1.0, 2.0 / std::f64::consts::PI, 0.0];
                for (got, want) in out.data.iter().zip(expected) {
                    assert_close(*got, want);
                }
            }
            other => panic!("expected host tensor fallback, got {other:?}"),
        }
    }

    #[test]
    fn sinc_gpu_propagates_provider_execution_errors() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider = Box::leak(Box::new(FailingSincProvider));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: provider.device_id(),
            buffer_id: 1,
        };
        let err = call(Value::GpuTensor(handle)).expect_err("provider error should surface");
        assert!(err
            .message()
            .contains("sinc: GPU provider unary_sinc failed"));
        assert!(err
            .message()
            .contains("device lost while running unary_sinc"));
    }

    #[test]
    fn sinc_gpu_complex_interleaved_bypasses_real_unary_provider_hook() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider = Box::leak(Box::new(ComplexSincProvider));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = GpuTensorHandle {
            shape: vec![2, 1],
            device_id: provider.device_id(),
            buffer_id: 3,
        };
        runmat_accelerate_api::set_handle_storage(&handle, GpuTensorStorage::ComplexInterleaved);

        let result = call(Value::GpuTensor(handle)).expect("complex sinc gpu fallback");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_complex_close(out.data[0], (1.0, 0.0));
                assert_complex_close(out.data[1], sinc_complex_value(0.5, 0.25));
            }
            other => panic!("expected complex tensor fallback, got {other:?}"),
        }
    }

    #[test]
    fn sinc_gpu_complex_interleaved_rejects_odd_buffer_length() {
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider = Box::leak(Box::new(ComplexSincProvider));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = GpuTensorHandle {
            shape: vec![2, 1],
            device_id: provider.device_id(),
            buffer_id: 4,
        };
        runmat_accelerate_api::set_handle_storage(&handle, GpuTensorStorage::ComplexInterleaved);

        let err = call(Value::GpuTensor(handle)).expect_err("odd complex buffer should error");
        assert!(err
            .message()
            .contains("sinc: malformed complex buffer, odd length"));
    }

    #[test]
    fn sinc_gpu_input_stays_resident_when_provider_supports_sinc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(Value::GpuTensor(handle)).expect("sinc gpu");
            assert!(
                matches!(result, Value::GpuTensor(_)),
                "provider-backed sinc should keep gpuArray output resident"
            );
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            let expected = [1.0, 2.0 / std::f64::consts::PI, 0.0];
            for (got, want) in gathered.data.iter().zip(expected) {
                assert_close(*got, want);
            }
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
        let cpu = sinc_real(Value::Tensor(tensor.clone())).expect("cpu sinc");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu = block_on(sinc_gpu(handle)).expect("gpu sinc");
        let gathered = test_support::gather(gpu).expect("gather");
        let expected = test_support::gather(cpu).expect("gather cpu");
        assert_eq!(gathered.shape, expected.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, want) in gathered.data.iter().zip(expected.data.iter()) {
            assert!((got - want).abs() < tol, "|{got} - {want}| >= {tol}");
        }
    }
}
