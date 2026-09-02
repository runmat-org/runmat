//! MATLAB-compatible real gamma-function semantics for RunMat.

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, GAMMA_ERROR_INTERNAL, GAMMA_ERROR_INVALID_ARGUMENT,
    GAMMA_ERROR_INVALID_INPUT, GAMMA_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const PI: f64 = std::f64::consts::PI;
const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_5;
const LANCZOS_G: f64 = 7.0;
const EPSILON: f64 = 1e-12;
const BUILTIN_NAME: &str = "gamma";

const LANCZOS_COEFFS: [f64; 8] = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::gamma")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gamma",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_gamma" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute gamma directly on real floating device buffers; the runtime gathers to the host when unary_gamma is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::gamma")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gamma",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently falls back to host evaluation; providers may supply specialised kernels.",
};

#[runtime_builtin(
    name = "gamma",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::gamma"
)]
async fn gamma_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    if !rest.is_empty() {
        return Err(gamma_error_with_detail(
            &GAMMA_ERROR_INVALID_ARGUMENT,
            "gamma accepts exactly one input",
        ));
    }
    match value {
        Value::GpuTensor(handle) => gamma_gpu(handle).await,
        Value::Tensor(tensor) => Ok(tensor::tensor_into_value(gamma_tensor(tensor)?)),
        Value::Num(value) => Ok(Value::Num(gamma_real_scalar(value))),
        _ => Err(gamma_error_with_detail(
            &GAMMA_ERROR_INVALID_INPUT,
            "expected real single or double input",
        )),
    }
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(gamma_error_with_detail(
            &GAMMA_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

async fn gamma_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved
    {
        return Err(gamma_error_with_detail(
            &GAMMA_ERROR_INVALID_INPUT,
            "expected real single or double gpuArray input",
        ));
    }
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        gamma_error_with_detail(&GAMMA_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    match provider.unary_gamma(&handle).await {
        Ok(output) => {
            if !gpu_helpers::unary_gpu_output_matches(
                &output,
                &handle,
                provider,
                gpu_helpers::UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&handle),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                },
            ) {
                gpu_helpers::free_rejected_provider_output(&output, &[&handle], provider);
                return Err(gamma_terminal_error(
                    &GAMMA_ERROR_INTERNAL,
                    "provider unary_gamma returned malformed output",
                ));
            }
            let mut output = output;
            runmat_accelerate_api::set_handle_provenance(
                &mut output,
                runmat_accelerate_api::handle_provenance(&handle)
                    .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
            );
            return Ok(gpu_helpers::resident_gpu_value(output));
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
        Err(error) => {
            return Err(gamma_terminal_error(&GAMMA_ERROR_INTERNAL, error));
        }
    }
    let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = gamma_tensor(gathered)?;
    crate::builtins::math::trigonometry::inverse_helpers::upload_value_like_protected(
        provider,
        tensor::tensor_into_value(output),
        BUILTIN_NAME,
        &handle,
        std::slice::from_ref(&handle),
    )
}

fn gamma_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| gamma_error_with_detail(&GAMMA_ERROR_INTERNAL, detail))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(gamma_real_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| gamma_real_scalar(f64::from(value)) as f32)
                .collect(),
        ),
        _ => {
            return Err(gamma_error_with_detail(
                &GAMMA_ERROR_INVALID_INPUT,
                "expected real single or double input",
            ))
        }
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|detail| gamma_error_with_detail(&GAMMA_ERROR_INTERNAL, detail))
}

fn gamma_real_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_non_positive_integer(x) {
        return f64::INFINITY;
    }
    // Form large positive real results through log-gamma so the Lanczos power
    // and exponential factors cannot overflow separately before their product.
    if x >= 128.0 {
        return super::gammaln::gammaln_nonnegative_scalar(x).exp();
    }
    let result = gamma_complex_scalar(Complex64::new(x, 0.0));
    if result.im.abs() <= EPSILON * result.re.abs().max(1.0) {
        result.re
    } else {
        f64::NAN
    }
}

fn gamma_complex_scalar(z: Complex64) -> Complex64 {
    if z.re.is_nan() || z.im.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if z.im.abs() <= EPSILON && z.re.is_infinite() {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    if is_complex_pole(z) {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    if z.re < 0.5 {
        let sin_term = (Complex64::new(PI, 0.0) * z).sin();
        if sin_term.norm_sqr() <= EPSILON * EPSILON {
            return Complex64::new(f64::INFINITY, 0.0);
        }
        let gamma_one_minus_z = gamma_complex_scalar(Complex64::new(1.0, 0.0) - z);
        return Complex64::new(PI, 0.0) / (sin_term * gamma_one_minus_z);
    }
    lanczos_gamma(z)
}

fn lanczos_gamma(z: Complex64) -> Complex64 {
    let z_minus_one = z - Complex64::new(1.0, 0.0);
    let mut sum = Complex64::new(0.999_999_999_999_809_9, 0.0);
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate() {
        let denom = z_minus_one + Complex64::new((idx + 1) as f64, 0.0);
        sum += Complex64::new(*coeff, 0.0) / denom;
    }
    let t = z_minus_one + Complex64::new(LANCZOS_G + 0.5, 0.0);
    let power = t.powc(z_minus_one + Complex64::new(0.5, 0.0));
    let exponential = (-t).exp();
    Complex64::new(SQRT_TWO_PI, 0.0) * power * exponential * sum
}

fn is_non_positive_integer(x: f64) -> bool {
    x <= 0.0 && is_close_to_integer(x)
}

fn is_complex_pole(z: Complex64) -> bool {
    z.im.abs() <= EPSILON && is_non_positive_integer(z.re)
}

fn is_close_to_integer(x: f64) -> bool {
    if !x.is_finite() {
        return false;
    }
    let nearest = x.round();
    let diff = (x - nearest).abs();
    if nearest == 0.0 {
        diff <= EPSILON * EPSILON
    } else {
        diff <= EPSILON * nearest.abs().max(1.0)
    }
}

fn gamma_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn gamma_terminal_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail))
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{
        CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    };

    fn call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(gamma_builtin(value, rest))
    }

    fn approx_eq(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tol {tolerance})"
        );
    }

    #[test]
    fn rejects_excess_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = call(Value::Num(0.5), Vec::new()).expect_err("second output must reject");
        assert_eq!(error.identifier(), GAMMA_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn scalar_real_values_cover_integer_half_and_negative_inputs() {
        match call(Value::Num(5.0), Vec::new()).unwrap() {
            Value::Num(value) => approx_eq(value, 24.0, 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
        match call(Value::Num(0.5), Vec::new()).unwrap() {
            Value::Num(value) => approx_eq(value, PI.sqrt(), 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
        match call(Value::Num(-0.5), Vec::new()).unwrap() {
            Value::Num(value) => approx_eq(value, -2.0 * PI.sqrt(), 1e-10),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn poles_and_nonfinite_values_follow_real_contract() {
        assert!(matches!(
            call(Value::Num(0.0), Vec::new()).unwrap(),
            Value::Num(value) if value.is_infinite()
        ));
        assert!(matches!(
            call(Value::Num(f64::NAN), Vec::new()).unwrap(),
            Value::Num(value) if value.is_nan()
        ));
        assert!(matches!(
            call(Value::Num(f64::INFINITY), Vec::new()).unwrap(),
            Value::Num(value) if value.is_infinite()
        ));
    }

    #[test]
    fn double_tensor_preserves_shape_and_values() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = call(Value::Tensor(input), Vec::new()).unwrap();
        match output {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                for (actual, expected) in tensor.materialize_f64().iter().zip([1.0, 1.0, 2.0, 6.0])
                {
                    approx_eq(*actual, expected, 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn single_tensor_preserves_native_storage() {
        let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = call(Value::Tensor(input), Vec::new()).unwrap();
        match output {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
                assert_eq!(
                    tensor.into_numeric_storage().unwrap(),
                    NumericStorage::F32(vec![1.0, 1.0, 2.0, 6.0])
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn documented_overflow_thresholds_follow_input_precision() {
        assert!(matches!(
            call(Value::Num(171.0), Vec::new()).unwrap(),
            Value::Num(value) if value.is_finite()
        ));
        assert!(matches!(
            call(Value::Num(172.0), Vec::new()).unwrap(),
            Value::Num(value) if value.is_infinite()
        ));

        let input = Tensor::from_f32(vec![35.0, 36.0], vec![1, 2]).unwrap();
        let Value::Tensor(output) = call(Value::Tensor(input), Vec::new()).unwrap() else {
            panic!("expected tensor output");
        };
        let values = output.materialize_f64();
        assert!(values[0].is_finite());
        assert!(values[1].is_infinite());
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    fn integer_scalar_and_dense_integer_tensor_are_rejected() {
        let scalar = call(Value::Int(IntValue::I32(5)), Vec::new()).unwrap_err();
        assert_eq!(scalar.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);

        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2]).unwrap();
        let tensor = call(Value::Tensor(tensor), Vec::new()).unwrap_err();
        assert_eq!(tensor.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn logical_char_string_and_complex_inputs_are_rejected() {
        for value in [
            Value::Bool(true),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::CharArray(CharArray::new(vec!['A', 'Z'], 1, 2).unwrap()),
            Value::from("text"),
            Value::Complex(1.0, 1.0),
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap()),
        ] {
            let error = call(value, Vec::new()).unwrap_err();
            assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn extra_like_and_other_arguments_are_rejected() {
        let digits = call(Value::Num(1.0), vec![Value::Num(2.0)]).unwrap_err();
        assert_eq!(digits.identifier(), GAMMA_ERROR_INVALID_ARGUMENT.identifier);
        let like = call(Value::Num(1.0), vec![Value::from("like"), Value::Num(0.0)]).unwrap_err();
        assert_eq!(like.identifier(), GAMMA_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn gpu_provider_roundtrip_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let output = call(Value::GpuTensor(handle), Vec::new()).unwrap();
            assert!(matches!(output, Value::GpuTensor(_)));
            let gathered = test_support::gather(output).unwrap();
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            for (actual, expected) in gathered.materialize_f64().iter().zip([1.0, 1.0, 2.0, 6.0]) {
                approx_eq(*actual, expected, 1e-12);
            }
        });
    }

    #[test]
    fn integer_gpu_input_is_rejected_before_provider_gamma() {
        test_support::with_test_provider(|provider| {
            let values = [1u64, u64::MAX];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .unwrap();
            let error = call(Value::GpuTensor(handle), Vec::new()).unwrap_err();
            assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
        });
    }

    #[test]
    fn logical_gpu_input_is_rejected_before_provider_gamma() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let error = call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap_err();
            assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
            assert!(runmat_accelerate_api::provider_for_handle(&handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
        });
    }

    #[test]
    fn complex_gpu_input_is_rejected_before_provider_gamma() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
            let error = call(Value::GpuTensor(handle), Vec::new()).unwrap_err();
            assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_gamma_matches_host_for_real_inputs() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let input = Tensor::new(vec![0.25, 0.5, 1.0, 1.5], vec![2, 2]).unwrap();
        let expected = gamma_tensor(input.clone()).expect("host gamma");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload gamma input");

        let output = call(Value::GpuTensor(handle), Vec::new()).expect("wgpu gamma");
        let actual = test_support::gather(output).expect("gather wgpu gamma");

        assert_eq!(actual.shape, expected.shape);
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
        };
        for (actual, expected) in actual
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {expected}, got {actual} (tol {tolerance})"
            );
        }
    }
}
