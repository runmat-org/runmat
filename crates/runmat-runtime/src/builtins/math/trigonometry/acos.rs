//! MATLAB-compatible `acos` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse cosine with the same domain promotion, complex handling, and
//! GPU fallbacks as MATLAB. Real arguments outside `[-1, 1]` promote to complex outputs; the
//! runtime automatically gathers data to the host whenever a GPU provider cannot satisfy those
//! semantics.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinErrorDescriptor, ACOS_CHARACTER_INPUT_EXTENSION, ACOS_ERROR_INTERNAL,
    ACOS_ERROR_INVALID_INPUT, ACOS_GPU_REAL_COMPLEX_EXTENSION, ACOS_INTEGER_INPUT_EXTENSION,
    ACOS_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{ACOS_DESCRIPTOR, ACOS_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "acos";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::acos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "acos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_acos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute acos in-place when inputs stay within [-1, 1]; otherwise the runtime gathers to host to honour MATLAB-compatible complex promotion.",
};

fn acos_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::acos")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "acos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real input outside [-1, 1] must promote to a complex result instead of producing NaN.",
};

#[runtime_builtin(
    name = "acos",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::acos"
)]
async fn acos_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &ACOS_INTEGER_INPUT_EXTENSION,
        &ACOS_LOGICAL_INPUT_EXTENSION,
        &ACOS_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "acos")?;
    match value {
        Value::GpuTensor(handle) => acos_gpu(handle).await,
        Value::Complex(re, im) => Ok(acos_complex_value(re, im)),
        Value::ComplexTensor(ct) => acos_complex_tensor(ct),
        Value::CharArray(ca) => acos_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(acos_error_with_detail(
            &ACOS_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => acos_real(other),
    }
}

async fn acos_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return crate::builtins::common::provider_restore::gather_compute_restore(
            handle,
            BUILTIN_NAME,
            acos_tensor_real,
        )
        .await;
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => match provider.unary_acos(&handle).await {
                Ok(output) => {
                    return crate::builtins::common::provider_restore::validate_real_unary_provider_output(
                        provider,
                        &handle,
                        output,
                        BUILTIN_NAME,
                    )
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(acos_error_with_detail(
                        &ACOS_ERROR_INTERNAL,
                        format!("provider unary_acos failed: {error}"),
                    ))
                }
            },
            Ok(true) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &ACOS_GPU_REAL_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
                return crate::builtins::common::provider_restore::gather_compute_restore(
                    handle,
                    BUILTIN_NAME,
                    acos_tensor_real,
                )
                .await;
            }
            Err(_) => {
                // Fall back to host path below.
            }
        }
    }
    crate::builtins::common::provider_restore::gather_compute_restore(
        handle,
        BUILTIN_NAME,
        acos_tensor_real,
    )
    .await
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider.reduce_min(handle).await.map_err(|e| {
        acos_error_with_detail(&ACOS_ERROR_INTERNAL, format!("reduce_min failed: {e}"))
    })?;
    let max_handle = match provider.reduce_max(handle).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&min_handle);
            return Err(acos_error_with_detail(
                &ACOS_ERROR_INTERNAL,
                format!("reduce_max failed: {err}"),
            ));
        }
    };
    let min_host = match gpu_helpers::download_native_values_async(provider, &min_handle).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(acos_error_with_detail(
                &ACOS_ERROR_INTERNAL,
                format!("reduce_min download failed: {err}"),
            ));
        }
    };
    let max_host = match gpu_helpers::download_native_values_async(provider, &max_handle).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(acos_error_with_detail(
                &ACOS_ERROR_INTERNAL,
                format!("reduce_max download failed: {err}"),
            ));
        }
    };
    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);
    if min_host.data.iter().any(|value| value.is_nan())
        || max_host.data.iter().any(|value| value.is_nan())
    {
        return Err(acos_error_with_detail(
            &ACOS_ERROR_INTERNAL,
            "reduction results contained NaN",
        ));
    }
    Ok(min_host
        .data
        .iter()
        .chain(&max_host.data)
        .any(|value| value.is_outside_closed_unit_interval(0.0)))
}

fn acos_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("acos", value)
        .map_err(|e| acos_error_with_detail(&ACOS_ERROR_INVALID_INPUT, e))?;
    acos_tensor_real(tensor)
}

fn acos_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    super::inverse_helpers::map_real_tensor_promoting(
        tensor,
        BUILTIN_NAME,
        acos_real_matlab,
        acos_real_matlab_f32,
    )
}

/// MATLAB-compatible acos for real values.
///
/// For values within `[-1, 1]`, returns the standard real acos.
/// For values outside this range, MATLAB's principal branch is:
/// - `x > 1`:  `acos(x) = -i * acosh(x)`  → `(0, -acosh(x))`
/// - `x < -1`: `acos(x) = π - i * acosh(-x)` → `(π, -acosh(-x))`
///
/// This differs from `num_complex::Complex64::acos()` which uses a different branch cut.
fn acos_real_matlab(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, 0.0);
    }
    if (-1.0..=1.0).contains(&x) {
        // Within domain: real result
        (x.acos(), 0.0)
    } else if x > 1.0 {
        // x > 1: acos(x) = -i * acosh(x)
        (0.0, -x.acosh())
    } else {
        // x < -1: acos(x) = π - i * acosh(-x)
        (std::f64::consts::PI, -(-x).acosh())
    }
}

fn acos_real_matlab_f32(x: f32) -> (f32, f32) {
    if x.is_nan() {
        return (f32::NAN, 0.0);
    }
    if (-1.0..=1.0).contains(&x) {
        (x.acos(), 0.0)
    } else if x > 1.0 {
        (0.0, -x.acosh())
    } else {
        (std::f32::consts::PI, -(-x).acosh())
    }
}

fn acos_complex_value(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).acos();
    Value::Complex(result.re, result.im)
}

fn acos_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let tensor = super::inverse_helpers::map_complex_tensor(
        ct,
        BUILTIN_NAME,
        |(real, imag)| {
            let result = Complex64::new(real, imag).acos();
            (result.re, result.im)
        },
        |(real, imag)| {
            let result = num_complex::Complex32::new(real, imag).acos();
            (result.re, result.im)
        },
    )?;
    Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
}

fn acos_char_array(ca: CharArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![ca.rows, ca.cols])
            .map_err(|e| acos_error_with_detail(&ACOS_ERROR_INTERNAL, e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| acos_error_with_detail(&ACOS_ERROR_INTERNAL, e))?;
    acos_tensor_real(tensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray};

    fn acos_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::acos_builtin(value))
    }

    #[test]
    fn acos_extensions_and_output_arity_are_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::acos_builtin(Value::Int(runmat_value::IntValue::I8(
            1,
        ))))
        .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            ACOS_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = block_on(super::acos_builtin(Value::Bool(true)))
            .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            ACOS_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let character = block_on(super::acos_builtin(Value::CharArray(chars)))
            .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            ACOS_CHARACTER_INPUT_EXTENSION.error_identifier
        );
        let _outputs = crate::output_count::push_output_count(Some(2));
        let arity =
            block_on(super::acos_builtin(Value::Num(0.0))).expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), ACOS_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn acos_preserves_native_single_through_complex_promotion() {
        let input = Tensor::from_f32(vec![0.0, 2.0], vec![2, 1]).unwrap();
        let Value::ComplexTensor(output) = acos_builtin(Value::Tensor(input)).expect("single acos")
        else {
            panic!("expected complex-single tensor");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn acos_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = ACOS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = acos(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_scalar_within_domain() {
        let result = acos_builtin(Value::Num(0.5)).expect("acos");
        match result {
            Value::Num(v) => assert!((v - 0.5f64.acos()).abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_preserves_real_nan_and_small_complex_components() {
        let Value::Num(result) = acos_builtin(Value::Num(f64::NAN)).expect("NaN acos") else {
            panic!("real NaN input must remain real")
        };
        assert!(result.is_nan());

        let input = Complex64::new(1.0, 1e-14);
        let expected = input.acos();
        let Value::Complex(real, imag) =
            acos_builtin(Value::Complex(input.re, input.im)).expect("complex acos")
        else {
            panic!("complex input must remain complex")
        };
        assert_eq!(real, expected.re);
        assert_eq!(imag, expected.im);
        assert_ne!(imag, 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_scalar_outside_domain_returns_complex() {
        // For x > 1, MATLAB returns acos(x) = -i * acosh(x)
        let result = acos_builtin(Value::Num(1.2)).expect("acos");
        match result {
            Value::Complex(re, im) => {
                // MATLAB: acos(1.2) = 0 - 0.6224i
                assert!(re.abs() < 1e-10, "real part should be ~0, got {}", re);
                let expected_im = -1.2f64.acosh(); // negative imaginary
                assert!(
                    (im - expected_im).abs() < 1e-10,
                    "expected im={}, got im={}",
                    expected_im,
                    im
                );
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_of_two_matches_matlab() {
        // MATLAB: acos(2) = 0 - 1.3170i
        // This is the principal branch: acos(x) = -i * acosh(x) for x > 1
        let result = acos_builtin(Value::Num(2.0)).expect("acos(2)");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-10, "expected re=0, got {}", re);
                // acosh(2) ≈ 1.3169578969248166
                let expected_im = -2.0f64.acosh();
                assert!(
                    (im - expected_im).abs() < 1e-10,
                    "expected im≈{:.4}, got im≈{:.4}",
                    expected_im,
                    im
                );
                // Verify the sign is negative (MATLAB convention)
                assert!(im < 0.0, "imaginary part should be negative, got {}", im);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_negative_outside_domain() {
        // MATLAB: acos(-2) = π - i * acosh(2) ≈ 3.1416 - 1.3170i
        let result = acos_builtin(Value::Num(-2.0)).expect("acos(-2)");
        match result {
            Value::Complex(re, im) => {
                assert!(
                    (re - std::f64::consts::PI).abs() < 1e-10,
                    "expected re=π, got {}",
                    re
                );
                let expected_im = -2.0f64.acosh();
                assert!(
                    (im - expected_im).abs() < 1e-10,
                    "expected im≈{:.4}, got im≈{:.4}",
                    expected_im,
                    im
                );
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_matrix_elementwise() {
        let tensor = Tensor::new(vec![0.0, -0.5, 0.75, 1.0], vec![2, 2]).expect("tensor");
        let result = acos_builtin(Value::Tensor(tensor)).expect("acos matrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0f64.acos(),
                    (-0.5f64).acos(),
                    (0.75f64).acos(),
                    1.0f64.acos(),
                ];
                for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match acos_builtin(Value::Tensor(tensor)).expect("acos") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [std::f64::consts::PI, std::f64::consts::FRAC_PI_2, 0.0];
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
    fn acos_outside_domain_typed_integer_promotes_from_storage() {
        let tensor = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![2, 0]), vec![1, 2])
            .expect("integer tensor");

        match acos_builtin(Value::Tensor(tensor)).expect("acos") {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = acos_real_matlab(2.0);
                assert!((out.materialize_f64()[0].0 - expected.0).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - expected.1).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1], (std::f64::consts::FRAC_PI_2, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_wide_unsigned_integer_tensor_uses_storage_not_mirror() {
        let value = u64::MAX;
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![value]), vec![1, 1])
                .expect("integer tensor");

        match acos_builtin(Value::Tensor(tensor)).expect("acos") {
            Value::Complex(re, im) => {
                let expected = acos_real_matlab(value as f64);
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_logical_array() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical");
        let result = acos_builtin(Value::LogicalArray(logical)).expect("acos logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64().len(), 4);
                assert!((t.materialize_f64()[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
                assert!(t.materialize_f64()[1].abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_char_array_complex_promotion() {
        // 'B' has ASCII code 66, which is > 1, so complex promotion is required.
        // MATLAB: acos(66) = 0 - i * acosh(66)
        let chars = CharArray::new("B".chars().collect(), 1, 1).expect("char");
        let result = acos_builtin(Value::CharArray(chars)).expect("acos char");
        match result {
            Value::Complex(re, im) => {
                let x = 'B' as u32 as f64; // 66.0
                assert!(re.abs() < 1e-10, "expected re=0, got {}", re);
                let expected_im = -x.acosh();
                assert!(
                    (im - expected_im).abs() < 1e-10,
                    "expected im={}, got {}",
                    expected_im,
                    im
                );
            }
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.materialize_f64().len(), 1);
                let (re, im) = ct.materialize_f64()[0];
                let x = 'B' as u32 as f64;
                assert!(re.abs() < 1e-10, "expected re=0, got {}", re);
                let expected_im = -x.acosh();
                assert!(
                    (im - expected_im).abs() < 1e-10,
                    "expected im={}, got {}",
                    expected_im,
                    im
                );
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_string_errors() {
        let err = acos_builtin(Value::from("hello")).expect_err("acos string should error");
        assert_eq!(err.identifier(), ACOS_ERROR_INVALID_INPUT.identifier);
        let message = error_message(err);
        assert!(message.contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_all_integer_scalar_classes_cross_the_double_boundary() {
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            assert_eq!(
                acos_builtin(Value::Int(value)).expect("acos int"),
                Value::Num(0.0)
            );
        }
        let Value::Complex(real, imag) =
            acos_builtin(Value::Int(IntValue::U64(u64::MAX))).expect("wide acos")
        else {
            panic!("wide integer should produce complex double")
        };
        let expected = acos_real_matlab(u64::MAX as f64);
        assert_eq!(real, expected.0);
        assert_eq!(imag, expected.1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_complex_scalar_input() {
        let result = acos_builtin(Value::Complex(1.0, 2.0)).expect("acos complex");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 2.0).acos();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, -0.75, 1.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result_handle) =
                acos_builtin(Value::GpuTensor(handle)).expect("acos gpu")
            else {
                panic!("expected resident result")
            };
            assert_eq!(result_handle.device_id, provider.device_id());
            assert!(runmat_accelerate_api::provider_for_handle(&result_handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            let gathered = test_support::gather(Value::GpuTensor(result_handle)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [
                0.0f64.acos(),
                0.5f64.acos(),
                (-0.75f64).acos(),
                1.0f64.acos(),
            ];
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn acos_gpu_outside_domain_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0 + 5e-13, -1.3], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result_handle) =
                acos_builtin(Value::GpuTensor(handle)).expect("acos gpu complex")
            else {
                panic!("expected resident complex result")
            };
            assert_eq!(result_handle.device_id, provider.device_id());
            assert!(runmat_accelerate_api::provider_for_handle(&result_handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            let result = Value::GpuTensor(result_handle);
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(&result))
                .expect("gather complex result");
            match gathered {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![2, 1]);
                }
                Value::Complex(_, _) => {}
                other => panic!("expected complex result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn acos_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Some(provider) = test_support::wgpu_provider_if_available() else {
            return;
        };
        let t = Tensor::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![5, 1]).unwrap();
        let cpu = acos_real(Value::Tensor(t.clone())).expect("acos cpu");
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = provider.upload(&view).unwrap();
        let gpu = block_on(acos_gpu(h)).expect("acos gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-3,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
