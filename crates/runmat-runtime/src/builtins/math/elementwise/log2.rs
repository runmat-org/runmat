//! MATLAB-compatible base-2 logarithm (`log2`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise base-2 logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs and GPU execution
//! falls back to the host whenever complex numbers are required or the provider lacks a dedicated
//! kernel.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use super::log::{detect_gpu_requires_complex, log_complex_parts};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const IMAG_EPS: f64 = 1e-12;
const LOG2_E: f64 = std::f64::consts::LOG2_E;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log2" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log2 directly on device buffers; runtimes fall back to the host when complex outputs are required or the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("log2({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log2` calls; providers can override with fused kernels when available.",
};

const BUILTIN_NAME: &str = "log2";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "log2",
    category = "math/elementwise",
    summary = "Base-2 logarithm of scalars, vectors, matrices, or N-D tensors.",
    keywords = "log2,base-2 logarithm,elementwise,gpu,complex",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::log2"
)]
async fn log2_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => log2_gpu(handle).await,
        Value::Complex(re, im) => {
            let (r, i) = log2_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => log2_complex_tensor(ct),
        Value::CharArray(ca) => log2_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("log2: expected numeric input"))
        }
        other => log2_real(other),
    }
}

async fn log2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => {
                if let Ok(out) = provider.unary_log2(&handle).await {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                return log2_tensor(tensor);
            }
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(err);
                }
                // Fall through to host fallback below.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    log2_tensor(tensor)
}

fn log2_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log2", value)
        .map_err(|e| builtin_error(format!("log2: {e}")))?;
    log2_tensor(tensor)
}

fn log2_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let len = tensor.data.len();
    let mut complex_values = Vec::with_capacity(len);
    let mut has_imag = false;

    for &v in &tensor.data {
        let (re_part, im_part) = log2_complex_parts(v, 0.0);
        if im_part != 0.0 {
            has_imag = true;
        }
        complex_values.push((re_part, im_part));
    }

    if has_imag {
        if len == 1 {
            let (re, im) = complex_values[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(complex_values, shape)
                .map_err(|e| builtin_error(format!("log2: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let data: Vec<f64> = complex_values
            .into_iter()
            .map(|(mut re, _)| {
                if re.is_finite() && re.abs() < IMAG_EPS {
                    re = 0.0;
                }
                re
            })
            .collect();
        let tensor = Tensor::new(data, shape).map_err(|e| builtin_error(format!("log2: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log2_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        data.push(log2_complex_parts(re, im));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor = ComplexTensor::new(data, ct.shape.clone())
            .map_err(|e| builtin_error(format!("log2: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn log2_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log2: {e}")))?;
    log2_tensor(tensor)
}

fn log2_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let (real_ln, imag_ln) = log_complex_parts(re, im);
    let mut real_part = real_ln * LOG2_E;
    let mut imag_part = imag_ln * LOG2_E;

    if real_part.is_finite() && real_part.abs() < IMAG_EPS {
        real_part = 0.0;
    }
    if imag_part.abs() < IMAG_EPS {
        imag_part = 0.0;
    }

    (real_part, imag_part)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{LogicalArray, ResolveContext, StringArray, Tensor, Type, Value};

    fn log2_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::log2_builtin(value))
    }

    #[test]
    fn log2_type_preserves_tensor_shape() {
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
    fn log2_type_scalar_tensor_returns_num() {
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
    fn log2_scalar_one() {
        let result = log2_builtin(Value::Num(1.0)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_two() {
        let result = log2_builtin(Value::Num(2.0)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_zero() {
        let result = log2_builtin(Value::Num(0.0)).expect("log2");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_negative() {
        let result = log2_builtin(Value::Num(-1.0)).expect("log2");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_bool_true() {
        let result = log2_builtin(Value::Bool(true)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = log2_builtin(Value::LogicalArray(logical)).expect("log2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!(t.data[1].is_infinite() && t.data[1].is_sign_negative());
                assert!((t.data[2] - 0.0).abs() < 1e-12);
                assert!(t.data[3].is_infinite() && t.data[3].is_sign_negative());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_string_input_errors() {
        let err = log2_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.message().contains("log2: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_string_array_errors() {
        let array = StringArray::new(vec!["hello".to_string()], vec![1, 1]).unwrap();
        let err = log2_builtin(Value::StringArray(array)).unwrap_err();
        assert!(
            err.message().contains("log2: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = log2_builtin(Value::Tensor(tensor)).expect("log2");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.data[0].0 - 0.0).abs() < 1e-12);
                assert!((ct.data[0].1 - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
                assert!((ct.data[1].0 - 0.0).abs() < 1e-12);
                assert!((ct.data[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_complex_scalar() {
        let result = log2_builtin(Value::Complex(1.0, 2.0)).expect("log2");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = log2_complex_parts(1.0, 2.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log2_builtin(Value::CharArray(chars)).expect("log2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (65.0f64).log2()).abs() < 1e-12);
                assert!((t.data[1] - (90.0f64).log2()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log2_builtin(Value::GpuTensor(handle)).expect("log2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.log2()).collect();
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, 2.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log2_builtin(Value::GpuTensor(handle)).expect("log2");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!((ct.data[0].0 - 1.0).abs() < 1e-12);
                    assert!((ct.data[0].1 - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
                    assert!((ct.data[1].0 - 1.0).abs() < 1e-12);
                    assert!((ct.data[1].1).abs() < 1e-12);
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log2_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = log2_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = block_on(log2_gpu(handle)).expect("log2 gpu");
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
