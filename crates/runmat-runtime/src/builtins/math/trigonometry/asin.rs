//! MATLAB-compatible `asin` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse sine for scalars, vectors, matrices, and N-D tensors while
//! matching MATLAB's complex promotion rules. Real arguments outside `[-1, 1]` automatically
//! become complex outputs. GPU execution uses provider hooks when available and falls back to
//! host computation whenever complex promotion is required or the provider lacks a dedicated
//! kernel.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinErrorDescriptor, ASIN_CHARACTER_INPUT_EXTENSION, ASIN_ERROR_INTERNAL,
    ASIN_ERROR_INVALID_INPUT, ASIN_GPU_REAL_COMPLEX_EXTENSION, ASIN_INTEGER_INPUT_EXTENSION,
    ASIN_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{ASIN_DESCRIPTOR, ASIN_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "asin";
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::asin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "asin",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_asin" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute asin in-place when inputs remain within [-1, 1]; the runtime gathers to host when complex promotion is required.",
};

fn asin_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::asin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "asin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real input outside [-1, 1] must promote to a complex result instead of producing NaN.",
};

#[runtime_builtin(
    name = "asin",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::asin"
)]
async fn asin_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &ASIN_INTEGER_INPUT_EXTENSION,
        &ASIN_LOGICAL_INPUT_EXTENSION,
        &ASIN_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "asin")?;
    match value {
        Value::GpuTensor(handle) => asin_gpu(handle).await,
        Value::Complex(re, im) => Ok(asin_complex_value(re, im)),
        Value::ComplexTensor(ct) => asin_complex_tensor(ct),
        Value::CharArray(ca) => asin_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(asin_error_with_detail(
            &ASIN_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => asin_real(other),
    }
}

async fn asin_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return crate::builtins::common::provider_restore::gather_compute_restore(
            handle,
            BUILTIN_NAME,
            asin_tensor_real,
        )
        .await;
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => match provider.unary_asin(&handle).await {
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
                    return Err(asin_error_with_detail(
                        &ASIN_ERROR_INTERNAL,
                        format!("provider unary_asin failed: {error}"),
                    ))
                }
            },
            Ok(true) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &ASIN_GPU_REAL_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
                return crate::builtins::common::provider_restore::gather_compute_restore(
                    handle,
                    BUILTIN_NAME,
                    asin_tensor_real,
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
        asin_tensor_real,
    )
    .await
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider.reduce_min(handle).await.map_err(|e| {
        asin_error_with_detail(&ASIN_ERROR_INTERNAL, format!("reduce_min failed: {e}"))
    })?;
    let max_handle = match provider.reduce_max(handle).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&min_handle);
            return Err(asin_error_with_detail(
                &ASIN_ERROR_INTERNAL,
                format!("reduce_max failed: {err}"),
            ));
        }
    };
    let min_host = match gpu_helpers::download_native_values_async(provider, &min_handle).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(asin_error_with_detail(
                &ASIN_ERROR_INTERNAL,
                format!("reduce_min download failed: {err}"),
            ));
        }
    };
    let max_host = match gpu_helpers::download_native_values_async(provider, &max_handle).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(asin_error_with_detail(
                &ASIN_ERROR_INTERNAL,
                format!("reduce_max download failed: {err}"),
            ));
        }
    };
    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);
    if min_host.data.iter().any(|value| value.is_nan())
        || max_host.data.iter().any(|value| value.is_nan())
    {
        return Err(asin_error_with_detail(
            &ASIN_ERROR_INTERNAL,
            "reduction results contained NaN",
        ));
    }
    Ok(min_host
        .data
        .iter()
        .chain(&max_host.data)
        .any(|value| value.is_outside_closed_unit_interval(0.0)))
}

fn asin_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("asin", value)
        .map_err(|e| asin_error_with_detail(&ASIN_ERROR_INVALID_INPUT, e))?;
    asin_tensor_real(tensor)
}

fn asin_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    super::inverse_helpers::map_real_tensor_promoting(
        tensor,
        BUILTIN_NAME,
        |value| {
            if value.is_nan() || (-1.0..=1.0).contains(&value) {
                return (value.asin(), 0.0);
            }
            let result = Complex64::new(value, 0.0).asin();
            (result.re, result.im)
        },
        |value| {
            if value.is_nan() || (-1.0..=1.0).contains(&value) {
                return (value.asin(), 0.0);
            }
            let result = num_complex::Complex32::new(value, 0.0).asin();
            (result.re, result.im)
        },
    )
}

fn asin_complex_value(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).asin();
    Value::Complex(result.re, result.im)
}

fn asin_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let tensor = super::inverse_helpers::map_complex_tensor(
        ct,
        BUILTIN_NAME,
        |(real, imag)| {
            let result = Complex64::new(real, imag).asin();
            (result.re, result.im)
        },
        |(real, imag)| {
            let result = num_complex::Complex32::new(real, imag).asin();
            (result.re, result.im)
        },
    )?;
    Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
}

fn asin_char_array(ca: CharArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![ca.rows, ca.cols])
            .map_err(|e| asin_error_with_detail(&ASIN_ERROR_INTERNAL, e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| asin_error_with_detail(&ASIN_ERROR_INTERNAL, e))?;
    asin_tensor_real(tensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray};

    fn asin_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::asin_builtin(value))
    }

    #[test]
    fn asin_extensions_and_output_arity_are_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::asin_builtin(Value::Int(runmat_value::IntValue::I8(
            1,
        ))))
        .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            ASIN_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = block_on(super::asin_builtin(Value::Bool(true)))
            .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            ASIN_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let character = block_on(super::asin_builtin(Value::CharArray(chars)))
            .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            ASIN_CHARACTER_INPUT_EXTENSION.error_identifier
        );
        let _outputs = crate::output_count::push_output_count(Some(2));
        let arity =
            block_on(super::asin_builtin(Value::Num(0.0))).expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), ASIN_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn asin_preserves_native_single_through_complex_promotion() {
        let input = Tensor::from_f32(vec![0.0, 2.0], vec![2, 1]).unwrap();
        let Value::ComplexTensor(output) = asin_builtin(Value::Tensor(input)).expect("single asin")
        else {
            panic!("expected complex-single tensor");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn asin_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = ASIN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = asin(X)"));
        assert!(FUSION_SPEC.elementwise.is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_scalar_within_domain() {
        let result = asin_builtin(Value::Num(0.5)).expect("asin");
        match result {
            Value::Num(v) => assert!((v - 0.5f64.asin()).abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_preserves_small_real_values_and_real_nan() {
        let tiny = 1e-13;
        let Value::Num(result) = asin_builtin(Value::Num(tiny)).expect("tiny asin") else {
            panic!("tiny real input must remain real")
        };
        assert!(result != 0.0);
        assert!((result - tiny.asin()).abs() < 1e-28);

        let Value::Num(result) = asin_builtin(Value::Num(f64::NAN)).expect("NaN asin") else {
            panic!("real NaN input must remain real")
        };
        assert!(result.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_scalar_outside_domain_returns_complex() {
        let result = asin_builtin(Value::Num(1.2)).expect("asin");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.2, 0.0).asin();
                assert!((re - expected.re).abs() < 1e-10);
                assert!((im - expected.im).abs() < 1e-10);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_matrix_elementwise() {
        let tensor = Tensor::new(vec![0.0, -0.5, 0.75, 1.0], vec![2, 2]).expect("tensor");
        let result = asin_builtin(Value::Tensor(tensor)).expect("asin matrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [0.0, (-0.5f64).asin(), (0.75f64).asin(), 1.0f64.asin()];
                for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match asin_builtin(Value::Tensor(tensor)).expect("asin") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [
                    -std::f64::consts::FRAC_PI_2,
                    0.0,
                    std::f64::consts::FRAC_PI_2,
                ];
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
    fn asin_outside_domain_typed_integer_promotes_from_storage() {
        let tensor = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![2, 0]), vec![1, 2])
            .expect("integer tensor");

        match asin_builtin(Value::Tensor(tensor)).expect("asin") {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = Complex64::new(2.0, 0.0).asin();
                assert!((out.materialize_f64()[0].0 - expected.re).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - expected.im).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1], (0.0, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_logical_array() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical");
        let result = asin_builtin(Value::LogicalArray(logical)).expect("asin logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64().len(), 4);
                assert!(t.materialize_f64()[0].abs() < 1e-12);
                assert!((t.materialize_f64()[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_char_array_complex_promotion() {
        let chars = CharArray::new("B".chars().collect(), 1, 1).expect("char");
        let result = asin_builtin(Value::CharArray(chars)).expect("asin char");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new('B' as u32 as f64, 0.0).asin();
                assert!((re - expected.re).abs() < 1e-10);
                assert!((im - expected.im).abs() < 1e-10);
            }
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.materialize_f64().len(), 1);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_string_errors() {
        let err = asin_builtin(Value::from("hello")).expect_err("asin string should error");
        assert_eq!(err.identifier(), ASIN_ERROR_INVALID_INPUT.identifier);
        let message = error_message(err);
        assert!(message.contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_all_integer_scalar_classes_cross_the_double_boundary() {
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
            assert_eq!(
                asin_builtin(Value::Int(value)).expect("asin int"),
                Value::Num(0.0)
            );
        }
        let Value::Complex(re, im) =
            asin_builtin(Value::Int(IntValue::U64(u64::MAX))).expect("wide asin")
        else {
            panic!("wide integer should produce complex double")
        };
        let expected = Complex64::new(u64::MAX as f64, 0.0).asin();
        assert!((re - expected.re).abs() < 1e-12);
        assert!((im - expected.im).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_complex_scalar_input() {
        let result = asin_builtin(Value::Complex(1.0, 2.0)).expect("asin complex");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 2.0).asin();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, -0.75, 1.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result_handle) =
                asin_builtin(Value::GpuTensor(handle)).expect("asin gpu")
            else {
                panic!("expected resident result")
            };
            assert_eq!(result_handle.device_id, provider.device_id());
            assert!(runmat_accelerate_api::provider_for_handle(&result_handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            let gathered = test_support::gather(Value::GpuTensor(result_handle)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [0.0, 0.5f64.asin(), (-0.75f64).asin(), 1.0f64.asin()];
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_gpu_outside_domain_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0 + 5e-13, -1.3], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result_handle) =
                asin_builtin(Value::GpuTensor(handle)).expect("asin gpu complex")
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
    fn asin_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Some(provider) = test_support::wgpu_provider_if_available() else {
            return;
        };
        let t = Tensor::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![5, 1]).unwrap();
        let cpu = asin_real(Value::Tensor(t.clone())).expect("asin cpu");
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = provider.upload(&view).unwrap();
        let gpu = block_on(asin_gpu(h)).expect("asin gpu");
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
