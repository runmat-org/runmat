//! MATLAB-compatible `cosh` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cosh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cosh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cosh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cosh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosh directly on the device; runtimes gather to the host when unary_cosh is unavailable.",
};

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cosh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cosh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cosh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cosh` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cosh",
    category = "math/trigonometry",
    summary = "Hyperbolic cosine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "cosh,hyperbolic cosine,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::trigonometry::cosh"
)]
async fn cosh_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => cosh_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(
            cosh_complex_re(re, im),
            cosh_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cosh_complex_tensor(ct),
        Value::CharArray(ca) => cosh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(runtime_error_for("cosh: expected numeric input"))
        }
        other => cosh_real(other),
    }
}

async fn cosh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_cosh(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    cosh_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cosh", value).map_err(runtime_error_for)?;
    cosh_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosh_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&v| v.cosh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| runtime_error_for(format!("cosh: {e}")))
}

fn cosh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (cosh_complex_re(re, im), cosh_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| runtime_error_for(format!("cosh: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn cosh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cosh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| runtime_error_for(format!("cosh: {e}")))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn cosh_complex_re(re: f64, im: f64) -> f64 {
    re.cosh() * im.cos()
}

#[inline]
fn cosh_complex_im(re: f64, im: f64) -> f64 {
    re.sinh() * im.sin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Tensor, Type};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn cosh_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::cosh_builtin(value))
    }

    #[test]
    fn cosh_type_preserves_tensor_shape() {
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
    fn cosh_type_scalar_tensor_returns_num() {
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
    fn cosh_scalar() {
        let value = Value::Num(2.0);
        let result = cosh_builtin(value).expect("cosh");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.cosh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = cosh_builtin(Value::Tensor(tensor)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [(-1.0f64).cosh(), 1.0, 1.0f64.cosh()];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cosh_builtin(value).expect("cosh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cosh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_complex_scalar() {
        let result = cosh_builtin(Value::Complex(1.0, 2.0)).expect("cosh");
        match result {
            Value::Complex(re, im) => {
                assert!((re - cosh_complex_re(1.0, 2.0)).abs() < 1e-12);
                assert!((im - cosh_complex_im(1.0, 2.0)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_char_array_roundtrip() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = cosh_builtin(Value::CharArray(chars)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (idx, ch) in ['A', 'Z'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cosh();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_logical_array_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let result = cosh_builtin(Value::LogicalArray(logical)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [1.0f64.cosh(), 0.0f64.cosh(), 1.0f64.cosh()];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_string_errors() {
        let err = cosh_builtin(Value::String("runmat".to_string())).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cosh_builtin(Value::GpuTensor(handle)).expect("cosh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cosh()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cosh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 0.25, 0.5, 0.75], vec![4, 1]).unwrap();
        let cpu = cosh_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(cosh_gpu(h)).unwrap();
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
