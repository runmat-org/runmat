//! MATLAB-compatible `imag` builtin with GPU-aware semantics for RunMat.

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::imag")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imag",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_imag" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    two_pass_threshold: None,
    workgroup_size: None,
    nan_mode: ReductionNaN::Include,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_imag to materialise zero tensors in-place; the runtime gathers to the host whenever complex storage or string conversions are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::imag")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imag",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat imag as a zero-producing transform for real tensors; providers can override via fused pipelines to keep tensors resident on the GPU.",
};

const BUILTIN_NAME: &str = "imag";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "imag",
    category = "math/elementwise",
    summary = "Extract the imaginary component of scalars, vectors, matrices, or N-D tensors.",
    keywords = "imag,imaginary,complex,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::imag"
)]
async fn imag_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => imag_gpu(handle).await,
        Value::Complex(_, im) => Ok(Value::Num(im)),
        Value::ComplexTensor(ct) => imag_complex_tensor(ct),
        Value::CharArray(ca) => imag_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("imag: expected numeric input"))
        }
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => imag_real(x),
        other => Err(builtin_error(format!(
            "imag: unsupported input type {:?}; expected numeric, logical, or char input",
            other
        ))),
    }
}

async fn imag_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_imag(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(imag_tensor(tensor)?))
}

fn imag_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("imag", value)
        .map_err(|e| builtin_error(format!("imag: {e}")))?;
    Ok(tensor::tensor_into_value(imag_tensor(tensor)?))
}

fn imag_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    Tensor::new(vec![0.0; tensor.data.len()], tensor.shape.clone())
        .map_err(|e| builtin_error(format!("imag: {e}")))
}

fn imag_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let data = ct.data.iter().map(|&(_, im)| im).collect::<Vec<_>>();
    let tensor =
        Tensor::new(data, ct.shape.clone()).map_err(|e| builtin_error(format!("imag: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn imag_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let zeros = vec![0.0; ca.rows * ca.cols];
    let tensor = Tensor::new(zeros, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("imag: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, StringArray, Type};

    fn imag_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::imag_builtin(value))
    }

    #[test]
    fn imag_type_preserves_tensor_shape() {
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
    fn imag_type_scalar_tensor_returns_num() {
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
    fn imag_scalar_real_zero() {
        let result = imag_builtin(Value::Num(-2.5)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_complex_scalar() {
        let result = imag_builtin(Value::Complex(3.0, 4.0)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_bool_scalar_zero() {
        let result = imag_builtin(Value::Bool(true)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_int_scalar_zero() {
        let result = imag_builtin(Value::Int(IntValue::I32(-42))).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_tensor_real_is_zero() {
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 4.25], vec![4, 1]).unwrap();
        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                assert!(t.data.iter().all(|v| *v == 0.0));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_empty_tensor_zero_length() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_complex_tensor_to_tensor_of_imag_parts() {
        let complex =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).expect("complex tensor");
        let result = imag_builtin(Value::ComplexTensor(complex)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![2.0, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_logical_array_zero() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
        let result = imag_builtin(Value::LogicalArray(logical)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0; 4]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_char_array_zeroes() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).expect("char array");
        let result = imag_builtin(Value::CharArray(chars)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_string_error() {
        let err = imag_builtin(Value::from("hello")).expect_err("imag should error");
        assert!(err.message().contains("expected numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_string_array_error() {
        let arr =
            StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).expect("array");
        let err = imag_builtin(Value::StringArray(arr)).expect_err("imag should error");
        assert!(err.message().contains("expected numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = imag_builtin(Value::GpuTensor(handle)).expect("imag");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert!(gathered.data.iter().all(|v| *v == 0.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn imag_wgpu_matches_cpu_zero() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, -2.5, 4.0], vec![4, 1]).unwrap();
        let cpu = imag_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(imag_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data.len(), cpu_tensor.data.len());
        for (g, c) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((g - c).abs() < 1e-12, "imag mismatch {} vs {}", g, c);
        }
    }
}
