//! MATLAB-compatible `exp` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise exponential for real, logical, character, and complex inputs while
//! preserving MATLAB broadcasting semantics. GPU execution uses provider hooks when available
//! and falls back to host computation otherwise.

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::exp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "exp",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_exp" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may evaluate exp directly on device buffers; runtimes gather to host when unary_exp is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::exp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "exp",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `exp` calls; providers can override with fused elementwise kernels.",
};

const BUILTIN_NAME: &str = "exp";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "exp",
    category = "math/elementwise",
    summary = "Element-wise exponential of scalars, vectors, matrices, or N-D tensors.",
    keywords = "exp,exponential,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::exp"
)]
async fn exp_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => exp_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(
            exp_complex_re(re, im),
            exp_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => exp_complex_tensor(ct),
        Value::CharArray(ca) => exp_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("exp: expected numeric input, got string"))
        }
        other => exp_real(other),
    }
}

async fn exp_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_exp(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(exp_tensor(tensor)?))
}

fn exp_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("exp", value)
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    Ok(tensor::tensor_into_value(exp_tensor(tensor)?))
}

fn exp_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = tensor.data.iter().map(|&v| v.exp()).collect();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("exp: {e}")))
}

fn exp_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (exp_complex_re(re, im), exp_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn exp_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| (ch as u32 as f64).exp()).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn exp_complex_re(re: f64, im: f64) -> f64 {
    let exp_re = re.exp();
    exp_re * im.cos()
}

#[inline]
fn exp_complex_im(re: f64, im: f64) -> f64 {
    let exp_re = re.exp();
    exp_re * im.sin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Tensor, Type};

    fn exp_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::exp_builtin(value))
    }

    #[test]
    fn exp_type_preserves_tensor_shape() {
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
    fn exp_type_scalar_tensor_returns_num() {
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
    fn exp_scalar() {
        let result = exp_builtin(Value::Num(1.0)).expect("exp");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::E).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let result = exp_builtin(Value::Tensor(tensor)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected: Vec<f64> = vec![0.0_f64, 1.0_f64, 2.0_f64]
                    .into_iter()
                    .map(|v| v.exp())
                    .collect();
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            Value::Num(_) => panic!("expected tensor result"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_int_value_promotes() {
        let value = Value::Int(IntValue::I32(2));
        let result = exp_builtin(value).expect("exp");
        match result {
            Value::Num(v) => assert!((v - 2.0_f64.exp()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_bool_scalar() {
        let result = exp_builtin(Value::Bool(true)).expect("exp");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::E).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_complex_scalar() {
        let result = exp_builtin(Value::Complex(1.0, 2.0)).expect("exp");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.0f64.exp() * 2.0f64.cos(), 1.0f64.exp() * 2.0f64.sin());
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_complex_tensor_elements() {
        let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 1.0)], vec![2, 1]).unwrap();
        let result = exp_builtin(Value::ComplexTensor(tensor)).expect("exp");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 1.0)]
                    .into_iter()
                    .map(|(re, im)| (exp_complex_re(re, im), exp_complex_im(re, im)))
                    .collect();
                for (idx, (re, im)) in t.data.iter().enumerate() {
                    assert!((re - expected[idx].0).abs() < 1e-12);
                    assert!((im - expected[idx].1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_char_array_roundtrip() {
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).unwrap();
        let result = exp_builtin(Value::CharArray(chars)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Hi".chars().map(|c| (c as u32 as f64).exp()).collect();
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_logical_array_promotes_to_double() {
        let logical =
            LogicalArray::new(vec![1u8, 0u8, 1u8, 0u8], vec![2, 2]).expect("logical array");
        let result = exp_builtin(Value::LogicalArray(logical)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [std::f64::consts::E, 1.0, std::f64::consts::E, 1.0];
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_string_rejected() {
        let err = exp_builtin(Value::from("runmat")).unwrap_err();
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = exp_builtin(Value::GpuTensor(handle)).expect("exp");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.exp()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn exp_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let cpu = exp_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(exp_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }
}
