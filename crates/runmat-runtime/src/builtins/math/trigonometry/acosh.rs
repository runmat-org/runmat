//! MATLAB-compatible `acosh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic cosine with full complex promotion and GPU fallbacks
//! that mirror MATLAB behaviour for real, logical, character, and complex inputs.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinErrorDescriptor, ACOSH_CHARACTER_INPUT_EXTENSION, ACOSH_ERROR_INTERNAL,
    ACOSH_ERROR_INVALID_INPUT, ACOSH_GPU_REAL_COMPLEX_EXTENSION, ACOSH_INTEGER_INPUT_EXTENSION,
    ACOSH_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{ACOSH_DESCRIPTOR, ACOSH_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "acosh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::acosh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "acosh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_acosh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute acosh directly on device buffers when inputs stay within the real domain (x ≥ 1); otherwise the runtime gathers to the host for complex promotion.",
};

fn acosh_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn acosh_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::acosh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "acosh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real input below one must promote to a complex result instead of producing NaN.",
};

#[runtime_builtin(
    name = "acosh",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::acosh"
)]
async fn acosh_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &ACOSH_INTEGER_INPUT_EXTENSION,
        &ACOSH_LOGICAL_INPUT_EXTENSION,
        &ACOSH_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "acosh")?;
    match value {
        Value::GpuTensor(handle) => acosh_gpu(handle).await,
        Value::Complex(re, im) => Ok(acosh_complex_scalar(re, im)),
        Value::ComplexTensor(ct) => acosh_complex_tensor(ct),
        Value::CharArray(ca) => acosh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(acosh_error(&ACOSH_ERROR_INVALID_INPUT)),
        other => acosh_real(other),
    }
}

async fn acosh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return acosh_gather_compute_restore(handle).await;
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await? {
            Some(false) => match provider.unary_acosh(&handle).await {
                Ok(output) => {
                    return super::inverse_helpers::validate_real_unary_provider_output(
                        provider,
                        &handle,
                        output,
                        BUILTIN_NAME,
                    )
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(acosh_error_with_detail(
                        &ACOSH_ERROR_INTERNAL,
                        format!("provider unary_acosh failed: {error}"),
                    ))
                }
            },
            Some(true) | None => {
                return acosh_gather_compute_restore(handle).await;
            }
        }
    }
    acosh_gather_compute_restore(handle).await
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<bool>> {
    let min_handle = match provider.reduce_min(handle).await {
        Ok(handle) => handle,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => {
            return Err(acosh_error_with_detail(
                &ACOSH_ERROR_INTERNAL,
                format!("reduce_min failed: {error}"),
            ))
        }
    };
    let min_host = gpu_helpers::download_native_values_async(provider, &min_handle)
        .await
        .map_err(|e| {
            let _ = provider.free(&min_handle);
            acosh_error_with_detail(
                &ACOSH_ERROR_INTERNAL,
                format!("reduce_min download failed: {e}"),
            )
        })?;
    let _ = provider.free(&min_handle);
    if min_host.data.iter().any(|value| value.is_nan()) {
        return Ok(None);
    }
    Ok(Some(
        min_host.data.iter().any(|value| value.is_less_than_one()),
    ))
}

async fn acosh_gather_compute_restore(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        acosh_error_with_detail(&ACOSH_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|error| acosh_error_with_detail(&ACOSH_ERROR_INTERNAL, error))?;
    let output = acosh_tensor_real(tensor)?;
    if matches!(output, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ACOSH_GPU_REAL_COMPLEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    super::inverse_helpers::upload_value_like(provider, output, BUILTIN_NAME, &handle)
}

fn acosh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("acosh", value)
        .map_err(|e| acosh_error_with_detail(&ACOSH_ERROR_INVALID_INPUT, e))?;
    acosh_tensor_real(tensor)
}

fn acosh_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    super::inverse_helpers::map_real_tensor_promoting(
        tensor,
        BUILTIN_NAME,
        acosh_real_parts,
        acosh_real_parts_f32,
    )
}

fn acosh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let tensor = super::inverse_helpers::map_complex_tensor(
        ct,
        BUILTIN_NAME,
        |(real, imag)| {
            let result = Complex64::new(real, imag).acosh();
            (result.re, result.im)
        },
        |(real, imag)| {
            let result = num_complex::Complex32::new(real, imag).acosh();
            (result.re, result.im)
        },
    )?;
    Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
}

fn acosh_real_parts(value: f64) -> (f64, f64) {
    if value.is_nan() {
        return (f64::NAN, 0.0);
    }
    if value.is_infinite() && value.is_sign_positive() {
        return (f64::INFINITY, 0.0);
    }
    if value.is_infinite() && value.is_sign_negative() {
        return (f64::INFINITY, std::f64::consts::PI);
    }
    if value >= 1.0 {
        return (value.acosh(), 0.0);
    }
    let result = Complex64::new(value, 0.0).acosh();
    (result.re, result.im)
}

fn acosh_real_parts_f32(value: f32) -> (f32, f32) {
    if value.is_nan() {
        return (f32::NAN, 0.0);
    }
    if value.is_infinite() && value.is_sign_positive() {
        return (f32::INFINITY, 0.0);
    }
    if value.is_infinite() && value.is_sign_negative() {
        return (f32::INFINITY, std::f32::consts::PI);
    }
    if value >= 1.0 {
        return (value.acosh(), 0.0);
    }
    let result = num_complex::Complex32::new(value, 0.0).acosh();
    (result.re, result.im)
}

fn acosh_complex_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).acosh();
    Value::Complex(result.re, result.im)
}

fn acosh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![ca.rows, ca.cols])
            .map_err(|e| acosh_error_with_detail(&ACOSH_ERROR_INTERNAL, e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| acosh_error_with_detail(&ACOSH_ERROR_INTERNAL, e))?;
    acosh_tensor_real(tensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex64;
    use runmat_value::{IntValue, LogicalArray};

    fn acosh_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::acosh_builtin(value))
    }

    #[test]
    fn acosh_extensions_and_output_arity_are_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::acosh_builtin(Value::Int(IntValue::I8(1))))
            .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            ACOSH_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = block_on(super::acosh_builtin(Value::Bool(true)))
            .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            ACOSH_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let character = block_on(super::acosh_builtin(Value::CharArray(chars)))
            .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            ACOSH_CHARACTER_INPUT_EXTENSION.error_identifier
        );
        let _outputs = crate::output_count::push_output_count(Some(2));
        let arity = block_on(super::acosh_builtin(Value::Num(1.0)))
            .expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), ACOSH_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn acosh_preserves_native_single_through_complex_promotion() {
        let input = Tensor::from_f32(vec![0.5, 2.0], vec![2, 1]).unwrap();
        let Value::ComplexTensor(output) =
            acosh_builtin(Value::Tensor(input)).expect("single acosh")
        else {
            panic!("expected complex-single tensor");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn acosh_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = ACOSH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = acosh(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_scalar_real() {
        let value = Value::Num(1.5);
        let result = acosh_builtin(value).expect("acosh");
        match result {
            Value::Num(v) => assert!((v - 0.9624236501192069).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_scalar_complex() {
        let result = acosh_builtin(Value::Num(0.5)).expect("acosh");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - 1.0471975511965976).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_preserves_small_complex_components() {
        let input = Complex64::new(2.0, 1.0e-15);
        let expected = input.acosh();
        let Value::Complex(real, imaginary) =
            acosh_builtin(Value::Complex(input.re, input.im)).expect("complex acosh")
        else {
            panic!("expected complex scalar");
        };
        assert_eq!(real, expected.re);
        assert_eq!(imaginary, expected.im);
        assert_ne!(imaginary, 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_tensor_mixed() {
        let tensor = Tensor::new(vec![0.5, 1.0, 2.0], vec![3, 1]).expect("tensor construction");
        let result = acosh_builtin(Value::Tensor(tensor)).expect("acosh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [
                    (0.0, 1.0471975511965976),
                    (0.0, 0.0),
                    (1.3169578969248166, 0.0),
                ];
                for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual.0 - exp.0).abs() < 1e-12);
                    assert!((actual.1 - exp.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1])
                .expect("integer tensor");

        match acosh_builtin(Value::Tensor(tensor)).expect("acosh") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 2.0f64.acosh(), 3.0f64.acosh()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_below_domain_typed_integer_promotes_from_storage() {
        let tensor = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 2]), vec![1, 2])
            .expect("integer tensor");

        match acosh_builtin(Value::Tensor(tensor)).expect("acosh") {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.materialize_f64()[0], (0.0, std::f64::consts::FRAC_PI_2));
                assert!((out.materialize_f64()[1].0 - 2.0f64.acosh()).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1].1, 0.0);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_logical_array_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).expect("logical array");
        let result = acosh_builtin(Value::LogicalArray(logical)).expect("acosh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (0.0, 0.0),
                    (0.0, std::f64::consts::FRAC_PI_2),
                    (0.0, 0.0),
                    (0.0, std::f64::consts::FRAC_PI_2),
                ];
                for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual.0 - exp.0).abs() < 1e-12);
                    assert!((actual.1 - exp.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_char_array_roundtrip() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).expect("char array");
        let result = acosh_builtin(Value::CharArray(chars)).expect("acosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> =
                    "Az".chars().map(|ch| (ch as u32 as f64).acosh()).collect();
                for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<Complex64> = "Az"
                    .chars()
                    .map(|ch| Complex64::new(ch as u32 as f64, 0.0).acosh())
                    .collect();
                for ((re, im), exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((re - exp.re).abs() < 1e-12);
                    assert!((im - exp.im).abs() < 1e-12);
                }
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_char_array_promotes_to_complex() {
        let chars = CharArray::new(vec!['\0'], 1, 1).expect("char array");
        let result = acosh_builtin(Value::CharArray(chars)).expect("acosh");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                let (re, im) = t.materialize_f64()[0];
                assert!(re.abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-2.0, 0.5)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = acosh_builtin(Value::ComplexTensor(complex)).expect("acosh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.materialize_f64().iter().zip(inputs.iter()) {
                    let expected = input.acosh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_integer_input() {
        let result = acosh_builtin(Value::Int(IntValue::I32(4))).expect("acosh");
        match result {
            Value::Num(v) => assert!((v - 2.0634370688955608).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_bool_inputs() {
        let true_result = acosh_builtin(Value::Bool(true)).expect("acosh");
        match true_result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected real scalar, got {other:?}"),
        }
        let false_result = acosh_builtin(Value::Bool(false)).expect("acosh");
        match false_result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_infinity_inputs() {
        let pos = acosh_builtin(Value::Num(f64::INFINITY)).expect("acosh");
        match pos {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected positive infinity result, got {other:?}"),
        }

        let neg = acosh_builtin(Value::Num(f64::NEG_INFINITY)).expect("acosh");
        match neg {
            Value::Complex(re, im) => {
                assert!(re.is_infinite() && re.is_sign_positive());
                assert!((im - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected complex infinity result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_nan_propagates() {
        let result = acosh_builtin(Value::Num(f64::NAN)).expect("acosh");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_string_errors() {
        let err = acosh_builtin(Value::from("oops")).expect_err("expected error");
        let message = error_message(&err);
        assert!(message.contains("invalid input"));
        assert_eq!(err.identifier(), ACOSH_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![1.0, 2.0, 5.0, 10.0], vec![4, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = acosh_builtin(Value::GpuTensor(handle)).expect("acosh");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            for (actual, expected) in gathered
                .materialize_f64()
                .iter()
                .zip(tensor.materialize_f64().iter())
            {
                let ref_val = expected.acosh();
                assert!((actual - ref_val).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acosh_gpu_falls_back_for_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 2.0], vec![2, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = acosh_builtin(Value::GpuTensor(handle)).expect("acosh");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(&result))
                .expect("gather complex result");
            match gathered {
                Value::ComplexTensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    let expected = [
                        Complex64::new(0.5, 0.0).acosh(),
                        Complex64::new(2.0, 0.0).acosh(),
                    ];
                    for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                        assert!((actual.0 - exp.re).abs() < 1e-12);
                        assert!((actual.1 - exp.im).abs() < 1e-12);
                    }
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn acosh_wgpu_matches_cpu_when_real() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 10.0], vec![3, 1]).unwrap();
        let cpu = acosh_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("provider")
            .upload(&view)
            .expect("upload");
        let gpu = block_on(acosh_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (actual, expected) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(ct.materialize_f64().iter())
                {
                    assert!((actual - expected).abs() < tol);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
