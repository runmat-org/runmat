//! MATLAB-compatible `atanh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic tangent with full complex promotion and GPU fallbacks
//! mirroring MATLAB behaviour across scalars, tensors, logical inputs, and complex numbers.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinErrorDescriptor, ATANH_CHARACTER_INPUT_EXTENSION, ATANH_ERROR_INTERNAL,
    ATANH_ERROR_INVALID_INPUT, ATANH_GPU_REAL_COMPLEX_EXTENSION, ATANH_INTEGER_INPUT_EXTENSION,
    ATANH_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{ATANH_DESCRIPTOR, ATANH_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atanh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atanh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_atanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Keeps tensors on the device when the provider exposes unary_atanh and every element satisfies |x| ≤ 1; otherwise gathers to the host for complex promotion.",
};

fn atanh_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn atanh_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atanh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real input outside the closed unit interval must promote to a complex result instead of producing NaN.",
};

#[runtime_builtin(
    name = "atanh",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::atanh"
)]
async fn atanh_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &ATANH_INTEGER_INPUT_EXTENSION,
        &ATANH_LOGICAL_INPUT_EXTENSION,
        &ATANH_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "atanh")?;
    match value {
        Value::GpuTensor(handle) => atanh_gpu(handle).await,
        Value::Complex(re, im) => Ok(atanh_complex_scalar(re, im)),
        Value::ComplexTensor(ct) => atanh_complex_tensor(ct),
        Value::CharArray(ca) => atanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(atanh_error(&ATANH_ERROR_INVALID_INPUT)),
        other => atanh_real(other),
    }
}

async fn atanh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return atanh_gather_compute_restore(handle).await;
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await? {
            Some(false) => match provider.unary_atanh(&handle).await {
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
                    return Err(atanh_error_with_detail(
                        &ATANH_ERROR_INTERNAL,
                        format!("provider unary_atanh failed: {error}"),
                    ))
                }
            },
            Some(true) | None => {
                return atanh_gather_compute_restore(handle).await;
            }
        }
    }
    atanh_gather_compute_restore(handle).await
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<bool>> {
    if handle.shape.contains(&0) {
        return Ok(None);
    }
    let min_handle = match provider.reduce_min(handle).await {
        Ok(handle) => handle,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => {
            return Err(atanh_error_with_detail(
                &ATANH_ERROR_INTERNAL,
                format!("reduce_min failed: {error}"),
            ))
        }
    };
    let max_handle = match provider.reduce_max(handle).await {
        Ok(handle) => handle,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            let _ = provider.free(&min_handle);
            return Ok(None);
        }
        Err(error) => {
            let _ = provider.free(&min_handle);
            return Err(atanh_error_with_detail(
                &ATANH_ERROR_INTERNAL,
                format!("reduce_max failed: {error}"),
            ));
        }
    };

    let min_host = match gpu_helpers::download_native_values_async(provider, &min_handle).await {
        Ok(values) => values,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(atanh_error_with_detail(
                &ATANH_ERROR_INTERNAL,
                format!("reduce_min download failed: {err}"),
            ));
        }
    };
    let max_host = match gpu_helpers::download_native_values_async(provider, &max_handle).await {
        Ok(values) => values,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(atanh_error_with_detail(
                &ATANH_ERROR_INTERNAL,
                format!("reduce_max download failed: {err}"),
            ));
        }
    };

    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);

    if min_host
        .data
        .iter()
        .chain(&max_host.data)
        .any(|value| value.is_nan())
    {
        return Ok(None);
    }
    Ok(Some(
        min_host
            .data
            .iter()
            .chain(&max_host.data)
            .any(|value| value.is_outside_closed_unit_interval(0.0)),
    ))
}

async fn atanh_gather_compute_restore(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        atanh_error_with_detail(&ATANH_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|error| atanh_error_with_detail(&ATANH_ERROR_INTERNAL, error))?;
    let output = atanh_tensor_real(tensor)?;
    if matches!(output, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ATANH_GPU_REAL_COMPLEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    super::inverse_helpers::upload_value_like(provider, output, BUILTIN_NAME, &handle)
}

fn atanh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("atanh", value)
        .map_err(|e| atanh_error_with_detail(&ATANH_ERROR_INVALID_INPUT, e))?;
    atanh_tensor_real(tensor)
}

fn atanh_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    super::inverse_helpers::map_real_tensor_promoting(
        tensor,
        BUILTIN_NAME,
        atanh_real_parts,
        atanh_real_parts_f32,
    )
}

fn atanh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let tensor = super::inverse_helpers::map_complex_tensor(
        ct,
        BUILTIN_NAME,
        |(real, imag)| {
            let result = Complex64::new(real, imag).atanh();
            (result.re, result.im)
        },
        |(real, imag)| {
            let result = num_complex::Complex32::new(real, imag).atanh();
            (result.re, result.im)
        },
    )?;
    Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
}

fn atanh_real_parts(value: f64) -> (f64, f64) {
    if value.is_nan() {
        return (f64::NAN, 0.0);
    }
    if value.is_finite() && value.abs() <= 1.0 {
        return (value.atanh(), 0.0);
    }
    if value.is_finite() {
        return atanh_real_outside_domain(value);
    }
    (0.0, std::f64::consts::FRAC_PI_2)
}

fn atanh_real_parts_f32(value: f32) -> (f32, f32) {
    if value.is_nan() {
        return (f32::NAN, 0.0);
    }
    if value.is_finite() && value.abs() <= 1.0 {
        return (value.atanh(), 0.0);
    }
    if value.is_finite() {
        let real = 0.5 * ((value + 1.0) / (value - 1.0)).ln();
        return (real, std::f32::consts::FRAC_PI_2);
    }
    (0.0, std::f32::consts::FRAC_PI_2)
}

fn atanh_complex_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).atanh();
    Value::Complex(result.re, result.im)
}

fn atanh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![ca.rows, ca.cols])
            .map_err(|e| atanh_error_with_detail(&ATANH_ERROR_INTERNAL, e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| atanh_error_with_detail(&ATANH_ERROR_INTERNAL, e))?;
    atanh_tensor_real(tensor)
}

fn atanh_real_outside_domain(x: f64) -> (f64, f64) {
    // MATLAB convention: for real x with |x| > 1, atanh returns a complex result
    // with imaginary part always +π/2, regardless of the sign of x.
    // The formula: atanh(x) = 0.5*ln((x+1)/(x-1)) + i*π/2
    // This differs from the standard complex atanh branch cut convention.
    let re = 0.5 * ((x + 1.0) / (x - 1.0)).ln();
    let im = std::f64::consts::FRAC_PI_2;
    (re, im)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex64;
    use runmat_value::{CharArray, IntValue, LogicalArray};

    fn atanh_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::atanh_builtin(value))
    }

    #[test]
    fn atanh_extensions_and_output_arity_are_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::atanh_builtin(Value::Int(
            runmat_value::IntValue::I8(1),
        )))
        .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            ATANH_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = block_on(super::atanh_builtin(Value::Bool(true)))
            .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            ATANH_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let character = block_on(super::atanh_builtin(Value::CharArray(chars)))
            .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            ATANH_CHARACTER_INPUT_EXTENSION.error_identifier
        );
        let _outputs = crate::output_count::push_output_count(Some(2));
        let arity = block_on(super::atanh_builtin(Value::Num(0.0)))
            .expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), ATANH_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn atanh_preserves_native_single_through_complex_promotion() {
        let input = Tensor::from_f32(vec![0.5, 2.0], vec![2, 1]).unwrap();
        let Value::ComplexTensor(output) =
            atanh_builtin(Value::Tensor(input)).expect("single atanh")
        else {
            panic!("expected complex-single tensor");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn atanh_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = ATANH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = atanh(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_scalar_real() {
        let result = atanh_builtin(Value::Num(0.5)).expect("atanh");
        match result {
            Value::Num(v) => assert!((v - 0.5493061443340549).abs() < 1e-12),
            other => panic!("expected scalar real result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_scalar_boundary() {
        let result = atanh_builtin(Value::Num(1.0)).expect("atanh");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected +Inf, got {other:?}"),
        }
        let result = atanh_builtin(Value::Num(-1.0)).expect("atanh");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected -Inf, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_real_nan_and_infinities_follow_the_real_input_branch() {
        let Value::Num(nan) = atanh_builtin(Value::Num(f64::NAN)).expect("atanh NaN") else {
            panic!("expected real NaN")
        };
        assert!(nan.is_nan());

        for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
            let Value::Complex(real, imaginary) =
                atanh_builtin(Value::Num(infinity)).expect("atanh infinity")
            else {
                panic!("expected complex infinity result")
            };
            assert_eq!(real, 0.0);
            assert_eq!(imaginary, std::f64::consts::FRAC_PI_2);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_tensor_real_values() {
        let tensor =
            Tensor::new(vec![0.0, 0.5, -0.5, 0.9], vec![2, 2]).expect("tensor construction");
        let result = atanh_builtin(Value::Tensor(tensor)).expect("atanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0,
                    0.5493061443340549,
                    -0.5493061443340549,
                    1.4722194895832204,
                ];
                for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match atanh_builtin(Value::Tensor(tensor)).expect("atanh") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert!(
                    out.materialize_f64()[0].is_infinite()
                        && out.materialize_f64()[0].is_sign_negative()
                );
                assert_eq!(out.materialize_f64()[1], 0.0);
                assert!(
                    out.materialize_f64()[2].is_infinite()
                        && out.materialize_f64()[2].is_sign_positive()
                );
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_all_integer_scalar_classes_cross_the_double_boundary() {
        for value in [
            IntValue::I8(0),
            IntValue::I16(0),
            IntValue::I32(0),
            IntValue::I64(0),
            IntValue::U8(0),
            IntValue::U16(0),
            IntValue::U32(0),
            IntValue::U64(0),
        ] {
            let Value::Num(result) =
                atanh_builtin(Value::Int(value)).expect("atanh integer scalar")
            else {
                panic!("expected real double scalar")
            };
            assert_eq!(result, 0.0);
        }

        let Value::Complex(real, imaginary) =
            atanh_builtin(Value::Int(IntValue::U64(u64::MAX))).expect("wide unsigned atanh")
        else {
            panic!("expected complex double scalar")
        };
        let expected = atanh_real_outside_domain(u64::MAX as f64);
        assert_eq!(real, expected.0);
        assert_eq!(imaginary, expected.1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_uses_the_exact_closed_unit_interval() {
        let just_above_one = f64::from_bits(1.0f64.to_bits() + 1);
        assert!(matches!(
            atanh_builtin(Value::Num(just_above_one)).expect("atanh"),
            Value::Complex(_, _)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_preserves_small_legitimate_complex_components() {
        let input = Complex64::new(1.0e-14, 0.25);
        let expected = input.atanh();
        let Value::Complex(real, imaginary) =
            atanh_builtin(Value::Complex(input.re, input.im)).expect("complex atanh")
        else {
            panic!("expected complex scalar")
        };
        assert_ne!(real, 0.0);
        assert_eq!(real, expected.re);
        assert_eq!(imaginary, expected.im);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_outside_domain_typed_integer_promotes_from_storage() {
        let tensor = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![2, 0]), vec![1, 2])
            .expect("integer tensor");

        match atanh_builtin(Value::Tensor(tensor)).expect("atanh") {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = atanh_real_outside_domain(2.0);
                assert!((out.materialize_f64()[0].0 - expected.0).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - expected.1).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1], (0.0, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_real_promotes_to_complex() {
        let result = atanh_builtin(Value::Num(2.0)).expect("atanh");
        match result {
            Value::Complex(re, im) => {
                let (exp_re, exp_im) = atanh_real_outside_domain(2.0);
                assert!((re - exp_re).abs() < 1e-12);
                assert!((im - exp_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_tensor_complex_output() {
        let tensor =
            Tensor::new(vec![2.0, -3.0, 0.5, -0.5], vec![2, 2]).expect("tensor construction");
        let result = atanh_builtin(Value::Tensor(tensor)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    atanh_real_outside_domain(2.0),
                    atanh_real_outside_domain(-3.0),
                    (0.5_f64.atanh(), 0.0),
                    ((-0.5_f64).atanh(), 0.0),
                ];
                for ((re, im), (exp_re, exp_im)) in t.materialize_f64().iter().zip(expected.iter())
                {
                    assert!((re - exp_re).abs() < 1e-12);
                    assert!((im - exp_im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.75)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = atanh_builtin(Value::ComplexTensor(complex)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.materialize_f64().iter().zip(inputs.iter()) {
                    let expected = input.atanh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_char_array_promotes_to_complex() {
        let chars = CharArray::new(vec!['A'], 1, 1).expect("char array");
        let result = atanh_builtin(Value::CharArray(chars)).expect("atanh");
        match result {
            Value::Complex(re, im) => {
                let (exp_re, exp_im) = atanh_real_outside_domain('A' as u32 as f64);
                assert!((re - exp_re).abs() < 1e-12);
                assert!((im - exp_im).abs() < 1e-12);
            }
            other => panic!("expected complex scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_string_input_errors() {
        let err = atanh_builtin(Value::from("hello")).expect_err("expected error");
        let message = error_message(&err);
        assert!(message.contains("invalid input"));
        assert_eq!(err.identifier(), ATANH_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_char_arrays() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).expect("chars");
        let result = atanh_builtin(Value::CharArray(chars)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                // 'A' = 65, 'B' = 66 -> complex outputs
                for (idx, (re, im)) in t.materialize_f64().iter().enumerate() {
                    let value = (65 + idx) as f64;
                    let (exp_re, exp_im) = atanh_real_outside_domain(value);
                    assert!((re - exp_re).abs() < 1e-12);
                    assert!((im - exp_im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_logical_array() {
        let logical =
            LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).expect("logical array creation");
        let result = atanh_builtin(Value::LogicalArray(logical)).expect("atanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.materialize_f64()[0] == 0.0);
                assert!(t.materialize_f64()[1].is_infinite());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![-0.5, -0.25, 0.25, 0.5], vec![2, 2]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor
                .materialize_f64()
                .iter()
                .map(|&x| x.atanh())
                .collect();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_keeps_residency_for_real_inputs() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-0.75, -0.25, 0.25, 0.75], vec![2, 2])
                .expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            match result {
                Value::GpuTensor(out_handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(out_handle.clone())).expect("gather");
                    let expected: Vec<f64> = tensor
                        .materialize_f64()
                        .iter()
                        .copied()
                        .map(f64::atanh)
                        .collect();
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for (actual, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                        assert!((actual - exp).abs() < 1e-12);
                    }
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_falls_back_for_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 2.0], vec![2, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            assert!(matches!(result, Value::GpuTensor(_)));
            let result = block_on(crate::dispatcher::gather_if_needed_async(&result))
                .expect("gather complex result");
            // Helper to compute expected values using MATLAB convention
            let matlab_atanh = |x: f64| -> (f64, f64) {
                if x.abs() <= 1.0 {
                    (x.atanh(), 0.0)
                } else {
                    atanh_real_outside_domain(x)
                }
            };
            match result {
                Value::ComplexTensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    let expected: Vec<(f64, f64)> = tensor
                        .materialize_f64()
                        .iter()
                        .map(|&x| matlab_atanh(x))
                        .collect();
                    for ((re, im), (exp_re, exp_im)) in
                        t.materialize_f64().iter().zip(expected.iter())
                    {
                        assert!((re - exp_re).abs() < 1e-12);
                        assert!((im - exp_im).abs() < 1e-12);
                    }
                }
                Value::Complex(re, im) => {
                    let (exp_re, exp_im) = atanh_real_outside_domain(2.0);
                    assert!((re - exp_re).abs() < 1e-12);
                    assert!((im - exp_im).abs() < 1e-12);
                }
                other => panic!("expected complex host result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_real_to_complex_promotion_is_compatibility_gated() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let tensor = Tensor::new(vec![0.5, 2.0], vec![2, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let error = block_on(super::atanh_builtin(Value::GpuTensor(handle)))
                .expect_err("real resident input requiring complex output must be gated");
            assert_eq!(
                error.identifier(),
                ATANH_GPU_REAL_COMPLEX_EXTENSION.error_identifier
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atanh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor =
            Tensor::new(vec![-0.8, -0.4, 0.4, 0.8], vec![2, 2]).expect("tensor construction");
        let expected: Vec<f64> = tensor
            .materialize_f64()
            .iter()
            .map(|&x| x.atanh())
            .collect();

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, tensor.shape);

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 5e-5,
        };

        for (actual, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - exp).abs() < tol, "|{actual} - {exp}| >= {tol}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_accepts_int_inputs() {
        let value = Value::Int(IntValue::I8(0));
        let result = atanh_builtin(value).expect("atanh");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar real result, got {other:?}"),
        }
    }
}
