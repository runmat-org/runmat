//! MATLAB-compatible `expm1` builtin with GPU-aware semantics for RunMat.
//!
//! Provides an element-wise `exp(x) - 1` with improved accuracy for tiny magnitudes, covering
//! real, logical, character, and complex inputs. GPU execution uses provider hooks when available
//! and falls back to host computation otherwise, matching MATLAB behaviour.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::expm1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "expm1",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_expm1" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may implement expm1 directly; runtimes gather to host when unary_expm1 is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::expm1")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "expm1",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let one = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("(exp({input}) - {one})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL exp(x) - 1 sequences; providers may override via fused elementwise kernels.",
};

const BUILTIN_NAME: &str = "expm1";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "expm1",
    category = "math/elementwise",
    summary = "Accurate element-wise computation of exp(x) - 1.",
    keywords = "expm1,exp(x)-1,exponential,elementwise,gpu,precision",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::expm1"
)]
async fn expm1_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => expm1_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = expm1_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => expm1_complex_tensor(ct),
        Value::CharArray(ca) => expm1_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("expm1: expected numeric input"))
        }
        other => expm1_real(other),
    }
}

async fn expm1_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_expm1(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(expm1_tensor(tensor)?))
}

fn expm1_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("expm1", value)
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    Ok(tensor::tensor_into_value(expm1_tensor(tensor)?))
}

fn expm1_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&v| v.exp_m1()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("expm1: {e}")))
}

fn expm1_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| expm1_complex_parts(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn expm1_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).exp_m1())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("expm1: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn expm1_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let half = 0.5 * im;
    let sin_half = half.sin();
    let cos_half = half.cos();
    let cos_b_minus_one = -2.0 * sin_half * sin_half;
    let sin_b = 2.0 * sin_half * cos_half;
    let expm1_a = re.exp_m1();
    let exp_a = expm1_a + 1.0;
    let real = expm1_a + exp_a * cos_b_minus_one;
    let imag = exp_a * sin_b;
    (real, imag)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, ResolveContext, Tensor, Type};

    fn expm1_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::expm1_builtin(value))
    }

    #[test]
    fn expm1_type_preserves_tensor_shape() {
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
    fn expm1_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_scalar_zero() {
        let result = expm1_builtin(Value::Num(0.0)).expect("expm1");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_scalar_small_matches_high_precision() {
        let input = 1.0e-16;
        let result = expm1_builtin(Value::Num(input)).expect("expm1");
        match result {
            Value::Num(v) => {
                let naive = input.exp() - 1.0;
                let delta_precise = v - input;
                let delta_naive = naive - input;
                assert!(delta_precise.abs() <= delta_naive.abs());
                assert!(delta_precise.abs() < 1e-28);
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0], vec![3, 1]).unwrap();
        let result = expm1_builtin(Value::Tensor(tensor)).expect("expm1");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.exp_m1(), (-1.0f64).exp_m1()];
                for (out, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((out - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_int_promotes() {
        let result = expm1_builtin(Value::Int(IntValue::I32(1))).expect("expm1");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.exp_m1()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_complex_scalar() {
        let result = expm1_builtin(Value::Complex(1.0, 1.0)).expect("expm1");
        match result {
            Value::Complex(re, im) => {
                let exp_a = 1.0f64.exp();
                let expected_re = exp_a * 1.0f64.cos() - 1.0;
                let expected_im = exp_a * 1.0f64.sin();
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = expm1_builtin(Value::CharArray(chars)).expect("expm1");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).exp_m1();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_string_rejects() {
        let err = expm1_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expm1_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = expm1_builtin(Value::GpuTensor(handle)).expect("expm1");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.exp_m1()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn expm1_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, -0.5, 0.5, 1.0], vec![4, 1]).unwrap();
        let cpu = expm1_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(expm1_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
