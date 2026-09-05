//! MATLAB-compatible `tanh` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::TANH_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, TANH_CHARACTER_INPUT_EXTENSION, TANH_ERROR_INTERNAL,
    TANH_ERROR_INVALID_INPUT, TANH_INTEGER_INPUT_EXTENSION, TANH_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};
use runmat_value::{ComplexStorage, NumericDType};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "tanh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::tanh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute tanh directly on the device; runtimes gather to the host when unary_tanh is unavailable.",
};

fn tanh_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tanh_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::tanh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tanh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL `tanh` calls; providers may override with specialised kernels.",
};

#[runtime_builtin(
    name = "tanh",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::tanh"
)]
async fn tanh_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &TANH_INTEGER_INPUT_EXTENSION,
        &TANH_LOGICAL_INPUT_EXTENSION,
        &TANH_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &value,
        &TANH_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "tanh")?;
    match value {
        Value::GpuTensor(handle) => tanh_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = tanh_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => tanh_complex_tensor(ct),
        Value::CharArray(ca) => tanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(tanh_error(&TANH_ERROR_INVALID_INPUT)),
        other => tanh_real(other),
    }
}

async fn tanh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let exact_fallback = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle)
            != runmat_accelerate_api::GpuTensorStorage::Real;
    if !exact_fallback {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
            match provider.unary_tanh(&handle).await {
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
                    return Err(tanh_error_with_detail(
                        &TANH_ERROR_INTERNAL,
                        format!("provider unary_tanh failed: {error}"),
                    ))
                }
            }
        }
    }
    crate::builtins::common::provider_restore::gather_value_compute_restore(
        handle,
        BUILTIN_NAME,
        |value| match value {
            Value::Complex(re, im) => {
                let (out_re, out_im) = tanh_complex_parts(re, im);
                Ok(Value::Complex(out_re, out_im))
            }
            Value::ComplexTensor(tensor) => tanh_complex_tensor(tensor),
            other => tanh_real(other),
        },
    )
    .await
}

fn tanh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("tanh", value)
        .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INVALID_INPUT, e))?;
    tanh_tensor(tensor).map(tensor::tensor_into_value)
}

fn tanh_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.tanh())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.tanh())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e))
}

fn tanh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| tanh_complex_parts_f32(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| tanh_complex_parts(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

fn tanh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tanh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

fn tanh_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let scale = (-2.0 * re.abs()).exp();
    let cos_im = im.cos();
    let denominator = (1.0 - scale).powi(2) + 4.0 * scale * cos_im.powi(2);
    let real = (-(-4.0 * re.abs()).exp_m1() / denominator).copysign(re);
    let imag = 4.0 * scale * im.sin() * cos_im / denominator;
    (real, imag)
}

fn tanh_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    let scale = (-2.0 * re.abs()).exp();
    let cos_im = im.cos();
    let denominator = (1.0 - scale).powi(2) + 4.0 * scale * cos_im.powi(2);
    let real = (-(-4.0 * re.abs()).exp_m1() / denominator).copysign(re);
    let imag = 4.0 * scale * im.sin() * cos_im / denominator;
    (real, imag)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex64;
    use runmat_value::{CharArray, IntValue, Tensor};

    fn tanh_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::tanh_builtin(value))
    }

    #[test]
    fn tanh_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = TANH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = tanh(X)"));
    }

    #[test]
    fn tanh_extensions_integer_boundary_and_output_arity_are_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::tanh_builtin(Value::Int(IntValue::I8(1))))
            .expect_err("integer extension must be gated");
        assert_eq!(
            integer.identifier(),
            TANH_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(block_on(super::tanh_builtin(Value::Int(IntValue::U64(
            (1_u64 << 53) + 1
        ))))
        .is_err());
        assert!(block_on(super::tanh_builtin(Value::Int(IntValue::U64(1_u64 << 54)))).is_ok());
        let _outputs = crate::output_count::push_output_count(Some(2));
        let arity =
            block_on(super::tanh_builtin(Value::Num(0.0))).expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), Some("RunMat:tanh:TooManyOutputs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_scalar_num() {
        let result = tanh_builtin(Value::Num(1.0)).expect("tanh");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.tanh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn tanh_preserves_native_single_real_and_complex_storage() {
        let real = Tensor::from_f32(vec![0.5, 2.0], vec![2, 1]).unwrap();
        let Value::Tensor(real_output) = tanh_builtin(Value::Tensor(real)).expect("single tanh")
        else {
            panic!("expected single tensor")
        };
        assert_eq!(real_output.numeric_dtype(), NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(0.5, 0.25), (2.0, -1.0)], vec![2, 1]).unwrap();
        let Value::ComplexTensor(complex_output) =
            tanh_builtin(Value::ComplexTensor(complex)).expect("complex-single tanh")
        else {
            panic!("expected complex-single tensor")
        };
        assert_eq!(complex_output.numeric_dtype(), NumericDType::F32);
        for (actual, &(real, imag)) in complex_output
            .materialize_f64()
            .iter()
            .zip([(0.5, 0.25), (2.0, -1.0)].iter())
        {
            let expected = Complex64::new(real, imag).tanh();
            assert!((actual.0 - expected.re).abs() < 1e-6);
            assert!((actual.1 - expected.im).abs() < 1e-6);
        }
    }

    #[test]
    fn tanh_all_integer_scalar_classes_cross_the_double_boundary_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            let Value::Num(result) =
                block_on(super::tanh_builtin(Value::Int(value))).expect("integer tanh")
            else {
                panic!("expected real double scalar")
            };
            assert_eq!(result, 1.0f64.tanh());
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = tanh_builtin(Value::Tensor(tensor)).expect("tanh");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                for (value, expected) in out
                    .materialize_f64()
                    .iter()
                    .zip([-1.0_f64.tanh(), 0.0, 1.0_f64.tanh()].iter())
                {
                    assert!((*value - *expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match tanh_builtin(Value::Tensor(tensor)).expect("tanh") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [-1.0f64.tanh(), 0.0, 1.0f64.tanh()];
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
    fn tanh_complex_scalar() {
        let result = tanh_builtin(Value::Complex(0.5, 1.0)).expect("tanh");
        match result {
            Value::Complex(re, im) => {
                let target = Complex64::new(0.5, 1.0).tanh();
                assert!((re - target.re).abs() < 1e-12);
                assert!((im - target.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn tanh_scaled_complex_formula_matches_reference_away_from_poles() {
        for real in [-5.0, -1.0, -0.125, 0.0, 0.125, 1.0, 5.0] {
            for imag in [-2.0, -0.5, 0.0, 0.5, 2.0] {
                let (actual_real, actual_imag) = tanh_complex_parts(real, imag);
                let expected = Complex64::new(real, imag).tanh();
                assert!((actual_real - expected.re).abs() < 1e-12);
                assert!((actual_imag - expected.im).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn tanh_complex_large_real_part_reaches_its_finite_limit_without_overflow() {
        for (real, expected) in [(1_000.0, 1.0), (-1_000.0, -1.0)] {
            let Value::Complex(output_real, output_imag) =
                tanh_builtin(Value::Complex(real, 0.25)).expect("large complex tanh")
            else {
                panic!("expected complex scalar")
            };
            assert_eq!(output_real, expected);
            assert_eq!(output_imag, 0.0);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_char_array_roundtrip() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = tanh_builtin(Value::CharArray(chars)).expect("tanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Az".chars().map(|c| (c as u32 as f64).tanh()).collect();
                for (value, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*value - *expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_string_errors() {
        let err = tanh_builtin(Value::from("not numeric")).expect_err("expected error");
        assert!(err.message().contains("invalid input"));
        assert_eq!(err.identifier(), TANH_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tanh_builtin(Value::GpuTensor(handle)).expect("tanh");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            for (value, expect) in gathered
                .materialize_f64()
                .iter()
                .zip(tensor.materialize_f64().iter())
            {
                assert!((*value - expect.tanh()).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn tanh_complex_gpu_fallback_restores_through_the_owner() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.25), (1.0, -0.75)], vec![2, 1])
                .expect("complex tensor");
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let resident = tanh_builtin(Value::GpuTensor(handle)).expect("resident tanh");
            let gathered = block_on(gpu_helpers::gather_value_async(&resident)).expect("gather");
            let Value::ComplexTensor(output) = gathered else {
                panic!("expected complex tensor")
            };
            for (actual, &(real, imag)) in output
                .materialize_f64()
                .iter()
                .zip(input.materialize_f64().iter())
            {
                let expected = Complex64::new(real, imag).tanh();
                assert!((actual.0 - expected.re).abs() < 1e-12);
                assert!((actual.1 - expected.im).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn tanh_gpu_fallback_preserves_single_and_source_owner() {
        test_support::with_f32_test_provider(|provider| {
            let input = [0.0, 0.5, 1.0];
            let source = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &input,
                    shape: &[3, 1],
                })
                .expect("upload");
            let source_device = source.device_id;
            let result =
                block_on(super::tanh_builtin(Value::GpuTensor(source))).expect("tanh fallback");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident result")
            };
            assert_eq!(handle.device_id, source_device);
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.shape, vec![3, 1]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tanh_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Some(provider) = test_support::wgpu_provider_if_available() else {
            return;
        };

        let tensor = Tensor::new(vec![-1.25, -0.5, 0.0, 0.75, 1.5], vec![5, 1]).unwrap();
        let cpu_value = tanh_real(Value::Tensor(tensor.clone())).expect("cpu tanh");
        let cpu_tensor = test_support::gather(cpu_value).expect("gather cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(tanh_gpu(handle)).expect("gpu tanh");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, expect) in gpu_tensor
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!(
                (*got - *expect).abs() < tol,
                "tanh mismatch: got {got}, expect {expect}, tol {tol}"
            );
        }
    }
}
