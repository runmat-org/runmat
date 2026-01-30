//! MATLAB-compatible `atanh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic tangent with full complex promotion and GPU fallbacks
//! mirroring MATLAB behaviour across scalars, tensors, logical inputs, and complex numbers.

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
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atanh";
const ZERO_EPS: f64 = 1.0e-12;
const DOMAIN_EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atanh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_atanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Keeps tensors on the device when the provider exposes unary_atanh and every element satisfies |x| ≤ 1; otherwise gathers to the host for complex promotion.",
};

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atanh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("atanh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `atanh` calls; providers can substitute custom kernels when available.",
};

#[runtime_builtin(
    name = "atanh",
    category = "math/trigonometry",
    summary = "Inverse hyperbolic tangent with MATLAB-compatible complex promotion.",
    keywords = "atanh,inverse hyperbolic tangent,artanh,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::trigonometry::atanh"
)]
async fn atanh_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => atanh_gpu(handle).await,
        Value::Complex(re, im) => Ok(atanh_complex_scalar(re, im)),
        Value::ComplexTensor(ct) => atanh_complex_tensor(ct),
        Value::CharArray(ca) => atanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(runtime_error_for("atanh: expected numeric input"))
        }
        other => atanh_real(other),
    }
}

async fn atanh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match gpu_domain_is_real(provider, &handle).await {
            Ok(true) => {
                if let Ok(out) = provider.unary_atanh(&handle).await {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(false) => {
                // fall back to host below
            }
            Err(_) => {
                // Fall back to host path below.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    atanh_tensor_real(tensor)
}

async fn gpu_domain_is_real(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| runtime_error_for(format!("atanh: reduce_min failed: {e}")))?;
    let max_handle = provider.reduce_max(handle).await.map_err(|e| {
        let _ = provider.free(&min_handle);
        runtime_error_for(format!("atanh: reduce_max failed: {e}"))
    })?;

    let min_host = match download_handle_async(provider, &min_handle).await {
        Ok(values) => values,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(runtime_error_for(format!(
                "atanh: reduce_min download failed: {err}"
            )));
        }
    };
    let max_host = match download_handle_async(provider, &max_handle).await {
        Ok(values) => values,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(runtime_error_for(format!(
                "atanh: reduce_max download failed: {err}"
            )));
        }
    };

    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);

    if min_host.data.is_empty() || max_host.data.is_empty() {
        return Err(runtime_error_for(
            "atanh: reduce_min/reduce_max returned empty result",
        ));
    }

    let min_value = min_host.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = max_host
        .data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if !min_value.is_finite() || !max_value.is_finite() {
        return Ok(false);
    }

    if min_value < -1.0 - DOMAIN_EPS || max_value > 1.0 + DOMAIN_EPS {
        return Ok(false);
    }

    Ok(true)
}

fn atanh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("atanh", value).map_err(runtime_error_for)?;
    atanh_tensor_real(tensor)
}

fn atanh_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    if tensor.data.is_empty() {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let mut requires_complex = false;
    let mut real_values = Vec::with_capacity(tensor.data.len());
    let mut complex_values = Vec::with_capacity(tensor.data.len());

    for &x in &tensor.data {
        if x.is_finite() && x.abs() <= 1.0 {
            let re = zero_small(x.atanh());
            real_values.push(re);
            complex_values.push((re, 0.0));
        } else {
            let result = Complex64::new(x, 0.0).atanh();
            let re = zero_small(result.re);
            let im = zero_small(result.im);
            if im.abs() > ZERO_EPS {
                requires_complex = true;
            }
            real_values.push(re);
            complex_values.push((re, im));
        }
    }

    if requires_complex {
        if complex_values.len() == 1 {
            let (re, im) = complex_values[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(complex_values, tensor.shape.clone())
                .map_err(|e| runtime_error_for(format!("atanh: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let tensor = Tensor::new(real_values, tensor.shape.clone())
            .map_err(|e| runtime_error_for(format!("atanh: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn atanh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    if ct.data.is_empty() {
        return Ok(Value::ComplexTensor(ct));
    }
    let mut mapped = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let result = Complex64::new(re, im).atanh();
        mapped.push((zero_small(result.re), zero_small(result.im)));
    }
    if mapped.len() == 1 {
        let (re, im) = mapped[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor = ComplexTensor::new(mapped, ct.shape.clone())
            .map_err(|e| runtime_error_for(format!("atanh: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn atanh_complex_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).atanh();
    Value::Complex(zero_small(result.re), zero_small(result.im))
}

fn atanh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![ca.rows, ca.cols])
            .map_err(|e| runtime_error_for(format!("atanh: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| runtime_error_for(format!("atanh: {e}")))?;
    atanh_tensor_real(tensor)
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
    use num_complex::Complex64;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, Type};

    fn atanh_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::atanh_builtin(value))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn atanh_type_preserves_tensor_shape() {
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
    fn atanh_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_scalar_real() {
        let result = atanh_builtin(Value::Num(0.5)).expect("atanh");
        match result {
            Value::Num(v) => assert!((v - 0.5493061443340549).abs() < 1e-12),
            other => panic!("expected scalar real result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_scalar_boundary() {
        let result = atanh_builtin(Value::Num(1.0)).expect("atanh");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected +Inf, got {other:?}"),
        }
        let result = atanh_builtin(Value::Num(-1.0)).expect("atanh");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected -Inf, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_tensor_real_values() {
        let tensor =
            Tensor::new(vec![0.0, 0.5, -0.5, 0.9], vec![2, 2]).expect("tensor construction");
        let result = atanh_builtin(Value::Tensor(tensor)).expect("atanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0,
                    0.5493061443340549,
                    -0.5493061443340549,
                    1.4722194895832204,
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
    fn atanh_real_promotes_to_complex() {
        let result = atanh_builtin(Value::Num(2.0)).expect("atanh");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(2.0, 0.0).atanh();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_tensor_complex_output() {
        let tensor =
            Tensor::new(vec![2.0, -3.0, 0.5, -0.5], vec![2, 2]).expect("tensor construction");
        let result = atanh_builtin(Value::Tensor(tensor)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let inputs = [
                    Complex64::new(2.0, 0.0),
                    Complex64::new(-3.0, 0.0),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(-0.5, 0.0),
                ];
                let expected: Vec<Complex64> = inputs.iter().map(|z| z.atanh()).collect();
                for ((re, im), exp) in t.data.iter().zip(expected.iter()) {
                    assert!((re - exp.re).abs() < 1e-12);
                    assert!((im - exp.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.75)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = atanh_builtin(Value::ComplexTensor(complex)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.data.iter().zip(inputs.iter()) {
                    let expected = input.atanh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_char_array_promotes_to_complex() {
        let chars = CharArray::new(vec!['A'], 1, 1).expect("char array");
        let result = atanh_builtin(Value::CharArray(chars)).expect("atanh");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new('A' as u32 as f64, 0.0).atanh();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_string_input_errors() {
        let err = atanh_builtin(Value::from("hello")).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_char_arrays() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).expect("chars");
        let result = atanh_builtin(Value::CharArray(chars)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                // 'A' = 65, 'B' = 66 -> complex outputs
                for (idx, (re, im)) in t.data.iter().enumerate() {
                    let value = (65 + idx) as f64;
                    let expected = Complex64::new(value, 0.0).atanh();
                    assert!((re - expected.re).abs() < 1e-12);
                    assert!((im - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_logical_array() {
        let logical =
            LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).expect("logical array creation");
        let result = atanh_builtin(Value::LogicalArray(logical)).expect("atanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data[0] == 0.0);
                assert!(t.data[1].is_infinite());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![-0.5, -0.25, 0.25, 0.5], vec![2, 2]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&x| x.atanh()).collect();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_keeps_residency_for_real_inputs() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-0.75, -0.25, 0.25, 0.75], vec![2, 2])
                .expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            match result {
                Value::GpuTensor(out_handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(out_handle.clone())).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().copied().map(f64::atanh).collect();
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
                        assert!((actual - exp).abs() < 1e-12);
                    }
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_gpu_falls_back_for_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 2.0], vec![2, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            match result {
                Value::ComplexTensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    let expected: Vec<Complex64> = tensor
                        .data
                        .iter()
                        .map(|&x| Complex64::new(x, 0.0).atanh())
                        .collect();
                    for ((re, im), exp) in t.data.iter().zip(expected.iter()) {
                        assert!((re - exp.re).abs() < 1e-12);
                        assert!((im - exp.im).abs() < 1e-12);
                    }
                }
                Value::Complex(re, im) => {
                    let expected = Complex64::new(2.0, 0.0).atanh();
                    assert!((re - expected.re).abs() < 1e-12);
                    assert!((im - expected.im).abs() < 1e-12);
                }
                other => panic!("expected complex host result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atanh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor =
            Tensor::new(vec![-0.8, -0.4, 0.4, 0.8], vec![2, 2]).expect("tensor construction");
        let expected: Vec<f64> = tensor.data.iter().map(|&x| x.atanh()).collect();

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, tensor.shape);

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 5e-5,
        };

        for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
            assert!((actual - exp).abs() < tol, "|{actual} - {exp}| >= {tol}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atanh_accepts_int_inputs() {
        let value = Value::Int(IntValue::I8(0));
        let result = atanh_builtin(value).expect("atanh");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar real result, got {other:?}"),
        }
    }
}
