//! MATLAB-compatible `log1p` builtin with GPU-aware semantics for RunMat.
//!
//! Provides an element-wise `log(1 + x)` with improved accuracy for small magnitudes, covering
//! real, logical, character, and complex inputs. GPU execution uses provider hooks when available
//! and falls back to host computation whenever complex results are required or device support is
//! missing, mirroring MATLAB behavior.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const IMAG_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log1p")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log1p",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log1p" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers should supply unary_log1p and reduce_min; runtimes gather to host when complex outputs are required or either hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log1p")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log1p",
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
            Ok(format!("log({input} + {one})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log(x + 1)` sequences; providers may substitute fused kernels when available.",
};

const BUILTIN_NAME: &str = "log1p";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "log1p",
    category = "math/elementwise",
    summary = "Accurate element-wise computation of log(1 + x).",
    keywords = "log1p,log(1+x),natural logarithm,elementwise,gpu,precision",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::log1p"
)]
async fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => log1p_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = log1p_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => log1p_complex_tensor(ct),
        Value::CharArray(ca) => log1p_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("log1p: expected numeric input"))
        }
        other => log1p_real(other),
    }
}

async fn log1p_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        // Fast path: try device op first; if unsupported, fall back to complex-domain check
        if let Ok(out) = provider.unary_log1p(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                return log1p_tensor(tensor);
            }
            Ok(false) => {}
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(builtin_error("interaction pending..."));
                }
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    log1p_tensor(tensor)
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| builtin_error(format!("log1p: reduce_min failed: {e}")))?;
    let download = download_handle_async(provider, &min_handle)
        .await
        .map_err(|e| builtin_error(format!("log1p: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err(builtin_error("log1p: reduce_min result contained NaN"));
    }
    Ok(host.data.iter().any(|&v| v < -1.0))
}

fn log1p_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log1p", value)
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    log1p_tensor(tensor)
}

fn log1p_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let mut entries = Vec::with_capacity(tensor.data.len());
    let mut has_imag = false;

    for &v in &tensor.data {
        let sum = 1.0 + v;
        if sum.is_nan() {
            entries.push((f64::NAN, 0.0));
            continue;
        }
        if sum < 0.0 {
            let (mut real_part, mut imag_part) = log1p_complex_parts(v, 0.0);
            if real_part.abs() < IMAG_EPS {
                real_part = 0.0;
            }
            if imag_part.abs() < IMAG_EPS {
                imag_part = 0.0;
            }
            if imag_part != 0.0 {
                has_imag = true;
            }
            entries.push((real_part, imag_part));
        } else {
            entries.push((v.ln_1p(), 0.0));
        }
    }

    if has_imag {
        if entries.len() == 1 {
            let (re, im) = entries[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(entries, shape)
                .map_err(|e| builtin_error(format!("log1p: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let data: Vec<f64> = entries.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::new(data, shape).map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log1p_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let (mut real_part, mut imag_part) = log1p_complex_parts(re, im);
        if real_part.abs() < IMAG_EPS {
            real_part = 0.0;
        }
        if imag_part.abs() < IMAG_EPS {
            imag_part = 0.0;
        }
        data.push((real_part, imag_part));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor = ComplexTensor::new(data, ct.shape.clone())
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn log1p_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    log1p_tensor(tensor)
}

fn log1p_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let shifted_re = re + 1.0;
    let magnitude = shifted_re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = magnitude.ln();
        let imag_part = im.atan2(shifted_re);
        (real_part, imag_part)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{LogicalArray, Tensor, Type};
    use std::f64::consts::PI;

    fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::log1p_builtin(value))
    }

    #[test]
    fn log1p_type_preserves_tensor_shape() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn log1p_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_zero() {
        let result = log1p_builtin(Value::Num(0.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_negative_one() {
        let result = log1p_builtin(Value::Num(-1.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_less_than_negative_one_complex() {
        let result = log1p_builtin(Value::Num(-2.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_tensor_mixed_values() {
        let tensor = Tensor::new(vec![0.0, -0.5, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                let expected = [
                    (0.0, 0.0),
                    ((0.5f64).ln(), 0.0),
                    (0.0, PI),
                    ((4.0f64).ln(), 0.0),
                ];
                for ((re, im), (er, ei)) in ct.data.iter().zip(expected.iter()) {
                    assert!((re - er).abs() < 1e-12);
                    assert!((im - ei).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_complex_input() {
        let result = log1p_builtin(Value::Complex(0.5, 1.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.5f64.hypot(1.0).ln(), 1.0f64.atan2(1.5));
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_char_array_roundtrip() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = log1p_builtin(Value::CharArray(chars)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['A', 'B', 'C'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).ln_1p();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_string_rejects() {
        let err = log1p_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, -0.25, 0.5, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log1p_builtin(Value::GpuTensor(handle)).expect("log1p");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.ln_1p()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_bool_promotes() {
        let result = log1p_builtin(Value::Bool(true)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_logical_array_converts() {
        let logical = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = log1p_builtin(Value::LogicalArray(logical)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!((t.data[1] - 2.0f64.ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_gpu_complex_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -3.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log1p_builtin(Value::GpuTensor(handle)).expect("log1p");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![2, 1]);
                    let expected = [(0.0, PI), ((2.0f64).ln(), PI)];
                    for ((re, im), (er, ei)) in ct.data.iter().zip(expected.iter()) {
                        assert!((re - er).abs() < 1e-12);
                        assert!((im - ei).abs() < 1e-12);
                    }
                }
                other => panic!("expected complex tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log1p_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, -0.25, 0.25, 1.0], vec![4, 1]).unwrap();
        let cpu = log1p_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(log1p_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected value kinds"),
        }
    }
}
