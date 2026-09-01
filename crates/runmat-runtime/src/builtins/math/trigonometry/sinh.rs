//! MATLAB-compatible `sinh` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, SINH_CHARACTER_INPUT_EXTENSION, SINH_ERROR_INTERNAL,
    SINH_ERROR_INVALID_INPUT, SINH_INTEGER_INPUT_EXTENSION, SINH_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{SINH_DESCRIPTOR, SINH_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sinh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::sinh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sinh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sinh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sinh directly on the device; runtimes gather to the host when unary_sinh is unavailable.",
};

fn sinh_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sinh_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::sinh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sinh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sinh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `sinh` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "sinh",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::sinh"
)]
async fn sinh_builtin(value: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        BUILTIN_NAME,
        &SINH_INTEGER_INPUT_EXTENSION,
        &SINH_LOGICAL_INPUT_EXTENSION,
        &SINH_CHARACTER_INPUT_EXTENSION,
    )?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &value,
        &SINH_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "sinh")?;
    match value {
        Value::GpuTensor(handle) => sinh_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(
            sinh_complex_re(re, im),
            sinh_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => sinh_complex_tensor(ct),
        Value::CharArray(ca) => sinh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(sinh_error(&SINH_ERROR_INVALID_INPUT)),
        other => sinh_real(other),
    }
}

async fn sinh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return super::inverse_helpers::gather_compute_restore(handle, BUILTIN_NAME, |tensor| {
            sinh_tensor(tensor).map(tensor::tensor_into_value)
        })
        .await;
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match provider.unary_sinh(&handle).await {
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
                return Err(sinh_error_with_detail(
                    &SINH_ERROR_INTERNAL,
                    format!("provider unary_sinh failed: {error}"),
                ))
            }
        }
    }
    super::inverse_helpers::gather_compute_restore(handle, BUILTIN_NAME, |tensor| {
        sinh_tensor(tensor).map(tensor::tensor_into_value)
    })
    .await
}

fn sinh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sinh", value)
        .map_err(|e| sinh_error_with_detail(&SINH_ERROR_INVALID_INPUT, e))?;
    sinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn sinh_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    super::inverse_helpers::map_real_tensor(tensor, BUILTIN_NAME, f64::sinh, f32::sinh)
}

fn sinh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let tensor = super::inverse_helpers::map_complex_tensor(
        ct,
        BUILTIN_NAME,
        |(real, imag)| (sinh_complex_re(real, imag), sinh_complex_im(real, imag)),
        |(real, imag)| (real.sinh() * imag.cos(), real.cosh() * imag.sin()),
    )?;
    Ok(Value::ComplexTensor(tensor))
}

fn sinh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).sinh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| sinh_error_with_detail(&SINH_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn sinh_complex_re(re: f64, im: f64) -> f64 {
    re.sinh() * im.cos()
}

#[inline]
fn sinh_complex_im(re: f64, im: f64) -> f64 {
    re.cosh() * im.sin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, Tensor};

    use crate::builtins::common::test_support;

    fn sinh_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::sinh_builtin(value))
    }

    #[test]
    fn sinh_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = SINH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = sinh(X)"));
    }

    #[test]
    fn sinh_preserves_native_single_real_and_complex_storage() {
        let real = Tensor::from_f32(vec![0.5, 2.0], vec![2, 1]).unwrap();
        let Value::Tensor(real_output) = sinh_builtin(Value::Tensor(real)).expect("single sinh")
        else {
            panic!("expected single tensor");
        };
        assert_eq!(real_output.numeric_dtype(), runmat_value::NumericDType::F32);

        let complex = ComplexTensor::from_f32(vec![(0.5, 0.25), (2.0, -1.0)], vec![2, 1]).unwrap();
        let Value::ComplexTensor(complex_output) =
            sinh_builtin(Value::ComplexTensor(complex)).expect("complex-single sinh")
        else {
            panic!("expected complex-single tensor");
        };
        assert_eq!(
            complex_output.numeric_dtype(),
            runmat_value::NumericDType::F32
        );
    }

    #[test]
    fn sinh_extensions_and_output_arity_are_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(super::sinh_builtin(Value::Int(IntValue::I8(1))))
            .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            SINH_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = block_on(super::sinh_builtin(Value::Bool(true)))
            .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            SINH_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let character = block_on(super::sinh_builtin(Value::CharArray(chars)))
            .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            SINH_CHARACTER_INPUT_EXTENSION.error_identifier
        );
        let _outputs = crate::output_count::push_output_count(Some(2));
        let arity =
            block_on(super::sinh_builtin(Value::Num(0.0))).expect_err("excess outputs must reject");
        assert_eq!(arity.identifier(), SINH_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_scalar() {
        let value = Value::Num(1.0);
        let result = sinh_builtin(value).expect("sinh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.sinh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = sinh_builtin(Value::Tensor(tensor)).expect("sinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [-1.0f64.sinh(), 0.0, 1.0f64.sinh()];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match sinh_builtin(Value::Tensor(tensor)).expect("sinh") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [-1.0f64.sinh(), 0.0, 1.0f64.sinh()];
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
    fn sinh_int_value_promotes() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Int(IntValue::I32(1));
        let result = sinh_builtin(value).expect("sinh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.sinh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_all_integer_scalar_classes_cross_the_double_boundary_exactly() {
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
            let Value::Num(result) = sinh_builtin(Value::Int(value)).expect("integer sinh") else {
                panic!("expected real double scalar")
            };
            assert_eq!(result, 1.0f64.sinh());
        }

        let error = sinh_builtin(Value::Int(IntValue::U64(u64::MAX)))
            .expect_err("inexact wide integer must reject");
        assert!(error
            .message()
            .contains("must be exactly representable as double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_complex_scalar() {
        let result = sinh_builtin(Value::Complex(1.0, 2.0)).expect("sinh");
        match result {
            Value::Complex(re, im) => {
                assert!((re - sinh_complex_re(1.0, 2.0)).abs() < 1e-12);
                assert!((im - sinh_complex_im(1.0, 2.0)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_char_array_roundtrip() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = sinh_builtin(Value::CharArray(chars)).expect("sinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).sinh();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sinh_builtin(Value::GpuTensor(handle)).expect("sinh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.sinh()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sinh_string_errors() {
        let err = sinh_builtin(Value::from("not numeric")).expect_err("expected error");
        let message = err.message().to_string();
        assert!(message.contains("invalid input"));
        assert_eq!(err.identifier(), SINH_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sinh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 0.25, 0.5, 0.75], vec![4, 1]).unwrap();
        let cpu = sinh_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(sinh_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
