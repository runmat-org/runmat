//! MATLAB-compatible `asin` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse sine for scalars, vectors, matrices, and N-D tensors while
//! matching MATLAB's complex promotion rules. Real arguments outside `[-1, 1]` automatically
//! become complex outputs. GPU execution uses provider hooks when available and falls back to
//! host computation whenever complex promotion is required or the provider lacks a dedicated
//! kernel.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "asin";
const ZERO_EPS: f64 = 1e-12;
const DOMAIN_TOL: f64 = 1e-12;

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

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::asin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "asin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("asin({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL asin calls; providers can substitute custom kernels when available.",
};

#[runtime_builtin(
    name = "asin",
    category = "math/trigonometry",
    summary = "Element-wise inverse sine with MATLAB-compatible complex promotion.",
    keywords = "asin,inverse sine,arcsin,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::trigonometry::asin"
)]
async fn asin_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => asin_gpu(handle).await,
        Value::Complex(re, im) => Ok(asin_complex_value(re, im)),
        Value::ComplexTensor(ct) => asin_complex_tensor(ct),
        Value::CharArray(ca) => asin_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(runtime_error_for("asin: expected numeric input"))
        }
        other => asin_real(other),
    }
}

async fn asin_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => {
                if let Ok(out) = provider.unary_asin(&handle).await {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
                return asin_tensor_real(tensor);
            }
            Err(_) => {
                // Fall back to host path below.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    asin_tensor_real(tensor)
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| runtime_error_for(format!("asin: reduce_min failed: {e}")))?;
    let max_handle = match provider.reduce_max(handle).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&min_handle);
            return Err(runtime_error_for(format!("asin: reduce_max failed: {err}")));
        }
    };
    let min_host = match download_handle_async(provider, &min_handle).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(runtime_error_for(format!(
                "asin: reduce_min download failed: {err}"
            )));
        }
    };
    let max_host = match download_handle_async(provider, &max_handle).await {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(runtime_error_for(format!(
                "asin: reduce_max download failed: {err}"
            )));
        }
    };
    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);
    if min_host.data.iter().any(|&v| v.is_nan()) || max_host.data.iter().any(|&v| v.is_nan()) {
        return Err(runtime_error_for("asin: reduction results contained NaN"));
    }
    let min_val = min_host.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = max_host
        .data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(min_val < -1.0 - DOMAIN_TOL || max_val > 1.0 + DOMAIN_TOL)
}

fn asin_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("asin", value).map_err(runtime_error_for)?;
    asin_tensor_real(tensor)
}

fn asin_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    let len = tensor.data.len();
    if len == 0 {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let mut requires_complex = false;
    let mut real_data = Vec::with_capacity(len);
    let mut complex_data = Vec::with_capacity(len);
    for &v in &tensor.data {
        let result = Complex64::new(v, 0.0).asin();
        let re = zero_small(result.re);
        let im = zero_small(result.im);
        if im.abs() > ZERO_EPS {
            requires_complex = true;
        }
        real_data.push(re);
        complex_data.push((re, im));
    }

    if requires_complex {
        if len == 1 {
            let (re, im) = complex_data[0];
            return Ok(Value::Complex(re, im));
        }
        let tensor = ComplexTensor::new(complex_data, tensor.shape.clone())
            .map_err(|e| runtime_error_for(format!("asin: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    } else {
        let tensor = Tensor::new(real_data, tensor.shape.clone())
            .map_err(|e| runtime_error_for(format!("asin: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn asin_complex_value(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).asin();
    Value::Complex(zero_small(result.re), zero_small(result.im))
}

fn asin_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    if ct.data.is_empty() {
        return Ok(Value::ComplexTensor(ct));
    }
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let result = Complex64::new(re, im).asin();
        data.push((zero_small(result.re), zero_small(result.im)));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor = ComplexTensor::new(data, ct.shape.clone())
            .map_err(|e| runtime_error_for(format!("asin: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn asin_char_array(ca: CharArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![ca.rows, ca.cols])
            .map_err(|e| runtime_error_for(format!("asin: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| runtime_error_for(format!("asin: {e}")))?;
    asin_tensor_real(tensor)
}

fn zero_small(value: f64) -> f64 {
    if value.abs() < ZERO_EPS {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray};

    fn asin_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::asin_builtin(value))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
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
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_logical_array() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical");
        let result = asin_builtin(Value::LogicalArray(logical)).expect("asin logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data.len(), 4);
                assert!(t.data[0].abs() < 1e-12);
                assert!((t.data[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
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
                assert_eq!(ct.data.len(), 1);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_string_errors() {
        let err = asin_builtin(Value::from("hello")).expect_err("asin string should error");
        let message = error_message(err);
        assert!(message.contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_integer_scalar() {
        let result = asin_builtin(Value::Int(IntValue::I32(0))).expect("asin int");
        assert_eq!(result, Value::Num(0.0));
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
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = asin_builtin(Value::GpuTensor(handle)).expect("asin gpu");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [0.0, 0.5f64.asin(), (-0.75f64).asin(), 1.0f64.asin()];
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn asin_gpu_outside_domain_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.2, -1.3], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = asin_builtin(Value::GpuTensor(handle)).expect("asin gpu complex");
            match result {
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
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![5, 1]).unwrap();
        let cpu = asin_real(Value::Tensor(t.clone())).expect("asin cpu");
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(asin_gpu(h)).expect("asin gpu");
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
