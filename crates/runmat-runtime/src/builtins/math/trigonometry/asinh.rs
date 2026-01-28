//! MATLAB-compatible `asinh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic sine for scalars, tensors, and complex inputs while
//! matching MATLAB's promotion and residency rules.

use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
const BUILTIN_NAME: &str = "asinh";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::asinh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "asinh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_asinh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute asinh directly on device buffers; runtimes gather to host when unary_asinh is unavailable.",
};

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::asinh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "asinh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("asinh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `asinh` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "asinh",
    category = "math/trigonometry",
    summary = "Inverse hyperbolic sine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "asinh,arcsinh,inverse hyperbolic sine,trigonometry,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::trigonometry::asinh"
)]
async fn asinh_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => asinh_gpu(handle).await,
        Value::Complex(re, im) => Ok(complex_asinh_scalar(re, im)),
        Value::ComplexTensor(ct) => asinh_complex_tensor(ct),
        Value::CharArray(ca) => asinh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(runtime_error_for("asinh: expected numeric input"))
        }
        other => asinh_real(other),
    }
}

async fn asinh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_asinh(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    asinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn asinh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("asinh", value).map_err(runtime_error_for)?;
    asinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn asinh_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&v| v.asinh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| runtime_error_for(format!("asinh: {e}")))
}

fn asinh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| {
            let res = Complex64::new(re, im).asinh();
            (res.re, res.im)
        })
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| runtime_error_for(format!("asinh: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn asinh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).asinh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| runtime_error_for(format!("asinh: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn complex_asinh_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).asinh();
    Value::Complex(result.re, result.im)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex64;
    use runmat_builtins::LogicalArray;

    fn asinh_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::asinh_builtin(value))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_scalar() {
        let value = Value::Num(0.5);
        let result = asinh_builtin(value).expect("asinh");
        match result {
            Value::Num(v) => assert!((v - 0.48121182505960347).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_tensor_values() {
        let tensor =
            Tensor::new(vec![0.0, -0.5, 1.0, 3.0], vec![2, 2]).expect("tensor construction");
        let result = asinh_builtin(Value::Tensor(tensor)).expect("asinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0,
                    -0.48121182505960347,
                    0.881373587019543,
                    1.8184464592320668,
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.75)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = asinh_builtin(Value::ComplexTensor(complex)).expect("asinh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.data.iter().zip(inputs.iter()) {
                    let expected = input.asinh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_char_array_roundtrip() {
        let chars = CharArray::new("az".chars().collect(), 1, 2).expect("char array");
        let result = asinh_builtin(Value::CharArray(chars)).expect("asinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(('a' as u32) as f64).asinh(), (('z' as u32) as f64).asinh()];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![0.0, 0.5, 1.0, 2.0], vec![2, 2]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = asinh_builtin(Value::GpuTensor(handle)).expect("asinh");
            let gathered = test_support::gather(result).expect("gather");
            let expected = tensor.data.iter().map(|&v| v.asinh()).collect::<Vec<_>>();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_logical_array_promotes() {
        let logical =
            LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).expect("logical array construction");
        let result = asinh_builtin(Value::LogicalArray(logical)).expect("asinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    1.0f64.asinh(),
                    0.0f64.asinh(),
                    1.0f64.asinh(),
                    1.0f64.asinh(),
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asinh_string_errors() {
        let err = asinh_builtin(Value::from("not numeric")).expect_err("expected error");
        let message = error_message(err);
        assert!(
            message.contains("expected numeric input"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn asinh_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5, 1]).unwrap();
        let cpu = asinh_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("provider")
            .upload(&view)
            .expect("upload");
        let gpu = block_on(asinh_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (actual, expected) in gathered.data.iter().zip(ct.data.iter()) {
                    assert!(
                        (actual - expected).abs() < tol,
                        "|{} - {}| >= {}",
                        actual,
                        expected,
                        tol
                    );
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
