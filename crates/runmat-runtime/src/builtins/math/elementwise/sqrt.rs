//! MATLAB-compatible `sqrt` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise square roots for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs. GPU execution
//! utilises provider hooks when available and falls back to host computation whenever complex
//! results are required or the provider lacks the dedicated kernel.

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

const ZERO_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::sqrt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sqrt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sqrt" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers execute sqrt directly on device buffers when inputs are non-negative; runtime gathers to host when complex promotion is required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::sqrt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sqrt",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sqrt({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL sqrt calls; providers may replace them with fused elementwise kernels.",
};

const BUILTIN_NAME: &str = "sqrt";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "sqrt",
    category = "math/elementwise",
    summary = "Element-wise square root of scalars, vectors, matrices, or N-D tensors.",
    keywords = "sqrt,square root,elementwise,gpu,complex",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::sqrt"
)]
async fn sqrt_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => sqrt_gpu(handle).await,
        Value::Complex(re, im) => Ok(sqrt_complex_value(re, im)),
        Value::ComplexTensor(ct) => sqrt_complex_tensor(ct),
        Value::CharArray(ca) => sqrt_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("sqrt: expected numeric input"))
        }
        other => sqrt_real(other),
    }
}

async fn sqrt_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => {
                if let Ok(out) = provider.unary_sqrt(&handle).await {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                return sqrt_tensor_real(tensor);
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
    sqrt_tensor_real(tensor)
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| builtin_error(format!("sqrt: reduce_min failed: {e}")))?;
    let download = download_handle_async(provider, &min_handle)
        .await
        .map_err(|e| builtin_error(format!("sqrt: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err(builtin_error("sqrt: reduce_min result contained NaN"));
    }
    Ok(host.data.iter().any(|&v| v < 0.0))
}

fn sqrt_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sqrt", value)
        .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
    sqrt_tensor_real(tensor)
}

fn sqrt_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    let len = tensor.data.len();
    let mut requires_complex = false;
    for &v in &tensor.data {
        if v < 0.0 {
            requires_complex = true;
            break;
        }
    }

    if !requires_complex {
        let mut data = Vec::with_capacity(len);
        for &v in &tensor.data {
            let root = zero_small(v.sqrt());
            data.push(root);
        }
        let tensor = Tensor::new(data, tensor.shape.clone())
            .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(len);
        for &v in &tensor.data {
            if v < 0.0 {
                let imag = zero_small((-v).sqrt());
                data.push((0.0, imag));
            } else {
                let real = zero_small(v.sqrt());
                data.push((real, 0.0));
            }
        }
        if len == 1 {
            let (re, im) = data[0];
            if im == 0.0 {
                Ok(Value::Num(re))
            } else {
                Ok(Value::Complex(re, im))
            }
        } else {
            let tensor = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    }
}

fn sqrt_complex_value(re: f64, im: f64) -> Value {
    let (mut real_part, mut imag_part) = sqrt_complex_parts(re, im);
    real_part = zero_small(real_part);
    imag_part = zero_small(imag_part);
    Value::Complex(real_part, imag_part)
}

fn sqrt_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let (mut real_part, mut imag_part) = sqrt_complex_parts(re, im);
        real_part = zero_small(real_part);
        imag_part = zero_small(imag_part);
        data.push((real_part, imag_part));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor = ComplexTensor::new(data, ct.shape.clone())
            .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn sqrt_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for &ch in &ca.data {
        let code = ch as u32 as f64;
        data.push(zero_small(code.sqrt()));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn sqrt_complex_parts(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        if re < 0.0 {
            (0.0, (-re).sqrt())
        } else {
            (re.sqrt(), 0.0)
        }
    } else {
        let magnitude = re.hypot(im);
        if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            let real_part = ((magnitude + re) / 2.0).sqrt();
            let imag_part_raw = ((magnitude - re) / 2.0).sqrt();
            let imag_part = if im >= 0.0 {
                imag_part_raw
            } else {
                -imag_part_raw
            };
            (real_part, imag_part)
        }
    }
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
    use runmat_builtins::{
        CharArray, IntValue, LogicalArray, ResolveContext, Tensor, Type,
    };

    fn sqrt_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::sqrt_builtin(value))
    }

    #[test]
    fn sqrt_type_preserves_tensor_shape() {
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
    fn sqrt_type_scalar_tensor_returns_num() {
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
    fn sqrt_scalar_positive() {
        let result = sqrt_builtin(Value::Num(9.0)).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_scalar_negative() {
        let result = sqrt_builtin(Value::Num(-4.0)).expect("sqrt");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_bool_true() {
        let result = sqrt_builtin(Value::Bool(true)).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = sqrt_builtin(Value::LogicalArray(logical)).expect("sqrt");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.data[0] - 1.0).abs() < 1e-12);
                assert!(t.data[1].abs() < 1e-12);
                assert!((t.data[2] - 1.0).abs() < 1e-12);
                assert!(t.data[3].abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 4.0], vec![1, 2]).unwrap();
        let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!(ct.data[0].0.abs() < 1e-12);
                assert!((ct.data[0].1 - 1.0).abs() < 1e-12);
                assert!((ct.data[1].0 - 2.0).abs() < 1e-12);
                assert!(ct.data[1].1.abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = sqrt_builtin(Value::CharArray(chars)).expect("sqrt");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (65.0f64).sqrt()).abs() < 1e-12);
                assert!((t.data[1] - (90.0f64).sqrt()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_string_input_errors() {
        let err = sqrt_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.message().contains("sqrt: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_complex_scalar() {
        let result = sqrt_builtin(Value::Complex(3.0, 4.0)).expect("sqrt");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_integer_argument() {
        let result = sqrt_builtin(Value::Int(IntValue::I32(9))).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sqrt()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (gpu, cpu) in gathered.data.iter().zip(expected.iter()) {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 9.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!(ct.data[0].0.abs() < 1e-12);
                    assert!((ct.data[0].1 - 1.0).abs() < 1e-12);
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sqrt_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
        let cpu = sqrt_real(Value::Tensor(tensor.clone())).expect("cpu sqrt");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = block_on(sqrt_gpu(handle)).expect("gpu sqrt");
        let gathered = test_support::gather(gpu_value).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                for (gpu, cpu) in gathered.data.iter().zip(ct.data.iter()) {
                    let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                    };
                    assert!((gpu - cpu).abs() < tol, "|{gpu} - {cpu}| >= {tol}");
                }
            }
            Value::Num(_) => panic!("expected tensor result from cpu path"),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }
}
