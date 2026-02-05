//! MATLAB-compatible `pow2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    broadcast::BroadcastPlan, gpu_helpers, map_control_flow_with_builtin, tensor,
};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const LN_2: f64 = std::f64::consts::LN_2;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::pow2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pow2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_pow2" },
        ProviderHook::Binary {
            name: "pow2_scale",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_pow2 and pow2_scale to keep tensors on-device; the runtime gathers to host when hooks are unavailable or shapes require implicit expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::pow2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pow2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input} * {:.17})", LN_2))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits `exp(x * ln2)` for unary pow2; binary scaling currently falls back to the host when implicit expansion is required.",
};

const BUILTIN_NAME: &str = "pow2";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "pow2",
    category = "math/elementwise",
    summary = "Compute 2.^X or scale mantissas by binary exponents.",
    keywords = "pow2,ldexp,binary scaling,gpu",
    accel = "unary",
    type_resolver(numeric_binary_type),
    builtin_path = "crate::builtins::math::elementwise::pow2"
)]
async fn pow2_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match rest.len() {
        0 => pow2_unary(first).await,
        1 => pow2_binary(first, rest.into_iter().next().unwrap()).await,
        _ => Err(builtin_error("pow2: expected at most two arguments")),
    }
}

async fn pow2_unary(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => pow2_gpu(handle).await,
        Value::Complex(re, im) => {
            let (rr, ii) = pow2_complex(re, im);
            Ok(Value::Complex(rr, ii))
        }
        Value::ComplexTensor(ct) => pow2_complex_tensor(ct),
        Value::CharArray(ca) => pow2_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("pow2: expected numeric input, got string"))
        }
        other => pow2_real(other),
    }
}

async fn pow2_binary(mantissa: Value, exponent: Value) -> BuiltinResult<Value> {
    match (mantissa, exponent) {
        (Value::GpuTensor(mh), Value::GpuTensor(eh)) => pow2_gpu_scale(mh, eh).await,
        (Value::GpuTensor(mh), other) => {
            let gathered = gpu_helpers::gather_tensor_async(&mh)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            pow2_host_scale(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(eh)) => {
            let gathered = gpu_helpers::gather_tensor_async(&eh)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            pow2_host_scale(other, Value::Tensor(gathered))
        }
        (m, e) => pow2_host_scale(m, e),
    }
}

async fn pow2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_pow2(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(pow2_tensor(tensor)?))
}

async fn pow2_gpu_scale(
    mantissa: GpuTensorHandle,
    exponent: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if mantissa.device_id == exponent.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&mantissa) {
            if mantissa.shape == exponent.shape {
                if let Ok(out) = provider.pow2_scale(&mantissa, &exponent) {
                    return Ok(Value::GpuTensor(out));
                }
            }
        }
    }
    let m = gpu_helpers::gather_tensor_async(&mantissa)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let e = gpu_helpers::gather_tensor_async(&exponent)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    pow2_host_scale(Value::Tensor(m), Value::Tensor(e))
}

fn pow2_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("pow2", value)
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    Ok(tensor::tensor_into_value(pow2_tensor(tensor)?))
}

fn pow2_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = tensor.data.iter().map(|&v| v.exp2()).collect();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("pow2: {e}")))
}

fn pow2_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| pow2_complex(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn pow2_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).exp2())
        .collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn pow2_host_scale(mantissa: Value, exponent: Value) -> BuiltinResult<Value> {
    if let Some(result) = scalar_pow2_value(&mantissa, &exponent) {
        return Ok(result);
    }
    let mantissa_array = value_into_numeric_array(mantissa, "pow2")?;
    let exponent_array = value_into_numeric_array(exponent, "pow2")?;
    let plan = BroadcastPlan::new(mantissa_array.shape(), exponent_array.shape())
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    if plan.is_empty() {
        if mantissa_array.is_complex() || exponent_array.is_complex() {
            let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            return Ok(Value::ComplexTensor(tensor));
        } else {
            let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            return Ok(tensor::tensor_into_value(tensor));
        }
    }
    match (mantissa_array, exponent_array) {
        (NumericArray::Real(m), NumericArray::Real(e)) => {
            let mut out = vec![0.0f64; plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let scale = e.data[idx_e].exp2();
                out[idx_out] = m.data[idx_m] * scale;
            }
            let tensor = Tensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(tensor::tensor_into_value(tensor))
        }
        (NumericArray::Real(m), NumericArray::Complex(e)) => {
            let mut out = vec![(0.0f64, 0.0f64); plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let (re_pow, im_pow) = pow2_complex(e.data[idx_e].0, e.data[idx_e].1);
                let scale = m.data[idx_m];
                out[idx_out] = (scale * re_pow, scale * im_pow);
            }
            let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        (NumericArray::Complex(m), NumericArray::Real(e)) => {
            let mut out = vec![(0.0f64, 0.0f64); plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let scale = e.data[idx_e].exp2();
                let (re_m, im_m) = m.data[idx_m];
                out[idx_out] = (re_m * scale, im_m * scale);
            }
            let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        (NumericArray::Complex(m), NumericArray::Complex(e)) => {
            let mut out = vec![(0.0f64, 0.0f64); plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let (re_pow, im_pow) = pow2_complex(e.data[idx_e].0, e.data[idx_e].1);
                let (re_m, im_m) = m.data[idx_m];
                out[idx_out] = complex_mul(re_m, im_m, re_pow, im_pow);
            }
            let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
    }
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => t.data.first().copied(),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        Value::ComplexTensor(ct) if ct.data.len() == 1 => ct.data.first().copied(),
        _ => None,
    }
}

fn scalar_pow2_value(mantissa: &Value, exponent: &Value) -> Option<Value> {
    let base =
        scalar_complex_value(mantissa).or_else(|| scalar_real_value(mantissa).map(|v| (v, 0.0)))?;
    let exp =
        scalar_complex_value(exponent).or_else(|| scalar_real_value(exponent).map(|v| (v, 0.0)))?;
    let (mr, mi) = base;
    let (er, ei) = exp;
    if mi != 0.0 || ei != 0.0 {
        let (re_pow, im_pow) = pow2_complex(er, ei);
        let (re, im) = complex_mul(mr, mi, re_pow, im_pow);
        return Some(Value::Complex(re, im));
    }
    let scale = er.exp2();
    Some(Value::Num(mr * scale))
}

fn pow2_complex(re: f64, im: f64) -> (f64, f64) {
    let scale = (re * LN_2).exp();
    let angle = im * LN_2;
    (scale * angle.cos(), scale * angle.sin())
}

fn complex_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn value_into_numeric_array(value: Value, name: &str) -> BuiltinResult<NumericArray> {
    match value {
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(NumericArray::Complex(ct)),
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Real(tensor))
        }
        Value::String(_) | Value::StringArray(_) => Err(builtin_error(format!(
            "{name}: expected numeric input, got string"
        ))),
        Value::GpuTensor(_) => Err(builtin_error(format!(
            "{name}: internal error converting GPU tensor"
        ))),
        other => {
            let tensor = tensor::value_into_tensor_for(name, other)
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Real(tensor))
        }
    }
}

enum NumericArray {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NumericArray {
    fn shape(&self) -> &[usize] {
        match self {
            NumericArray::Real(t) => &t.shape,
            NumericArray::Complex(t) => &t.shape,
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, NumericArray::Complex(_))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, ResolveContext, Tensor, Type};

    fn pow2_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::pow2_builtin(first, rest))
    }

    #[test]
    fn pow2_type_preserves_tensor_shape() {
        let out = numeric_binary_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
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
    fn pow2_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_scalar_exponent() {
        let result = pow2_builtin(Value::Num(3.0), Vec::new()).expect("pow2");
        match result {
            Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_tensor_exponent() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let result = pow2_builtin(Value::Tensor(tensor), Vec::new()).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [0.5, 1.0, 2.0, 4.0];
                for (a, b) in out.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_binary_scaling() {
        let mantissa = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
        let exponent = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result =
            pow2_builtin(Value::Tensor(mantissa), vec![Value::Tensor(exponent)]).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![4.0, 24.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_complex_exponent_scalar() {
        let result = pow2_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("pow2");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = pow2_complex(1.0, 2.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_complex_mantissa_real_exponent() {
        let mantissa =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -0.5)], vec![2, 1]).expect("complex tensor");
        let exponent = Tensor::new(vec![2.0, -1.0], vec![2, 1]).unwrap();
        let result = pow2_builtin(
            Value::ComplexTensor(mantissa),
            vec![Value::Tensor(exponent)],
        )
        .expect("pow2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                let scale0 = 2.0f64.exp2();
                let scale1 = (-1.0f64).exp2();
                assert!((out.data[0].0 - (1.0 * scale0)).abs() < 1e-12);
                assert!((out.data[0].1 - (1.0 * scale0)).abs() < 1e-12);
                assert!((out.data[1].0 - (2.0 * scale1)).abs() < 1e-12);
                assert!((out.data[1].1 - (-0.5 * scale1)).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_char_array() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = pow2_builtin(Value::CharArray(chars), Vec::new()).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert!((out.data[0] - (65.0f64).exp2()).abs() < 1e-6);
                assert!((out.data[1] - (66.0f64).exp2()).abs() < 1e-6);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_rejects_strings() {
        let err = pow2_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = pow2_builtin(Value::GpuTensor(handle), Vec::new()).expect("pow2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected = vec![1.0, 2.0, 4.0];
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_gpu_scale_roundtrip() {
        test_support::with_test_provider(|provider| {
            let mantissa = Tensor::new(vec![0.5, 1.5], vec![2, 1]).unwrap();
            let exponent = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let m_view = runmat_accelerate_api::HostTensorView {
                data: &mantissa.data,
                shape: &mantissa.shape,
            };
            let e_view = runmat_accelerate_api::HostTensorView {
                data: &exponent.data,
                shape: &exponent.shape,
            };
            let m_handle = provider.upload(&m_view).expect("upload m");
            let e_handle = provider.upload(&e_view).expect("upload e");
            let result = pow2_builtin(Value::GpuTensor(m_handle), vec![Value::GpuTensor(e_handle)])
                .expect("pow2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![4.0, 24.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_binary_broadcast_host() {
        let mantissa = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let exponent = Value::Int(IntValue::I32(2));
        let result = pow2_builtin(Value::Tensor(mantissa), vec![exponent]).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![4.0, 8.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pow2_wgpu_matches_cpu_unary() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.5, -1.0, 0.0, 2.0, 4.25], vec![5, 1]).unwrap();
        let cpu_value = pow2_real(Value::Tensor(tensor.clone())).expect("pow2 cpu");
        let cpu = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result from cpu path, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(pow2_gpu(handle)).expect("pow2 gpu");
        let gpu = test_support::gather(gpu_value).expect("gather gpu result");

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (g, c) in gpu.data.iter().zip(cpu.data.iter()) {
            assert!((g - c).abs() <= tol, "mismatch: gpu={g} cpu={c} tol={tol}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pow2_wgpu_scale_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let mantissa = Tensor::new(vec![0.5, 1.5, 3.0], vec![3, 1]).unwrap();
        let exponent = Tensor::new(vec![3.0, -2.0, 5.5], vec![3, 1]).unwrap();

        let cpu_value = pow2_host_scale(
            Value::Tensor(mantissa.clone()),
            Value::Tensor(exponent.clone()),
        )
        .expect("pow2 host scale");
        let cpu = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor from cpu scale, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let m_view = runmat_accelerate_api::HostTensorView {
            data: &mantissa.data,
            shape: &mantissa.shape,
        };
        let e_view = runmat_accelerate_api::HostTensorView {
            data: &exponent.data,
            shape: &exponent.shape,
        };
        let m_handle = provider.upload(&m_view).expect("upload mantissa");
        let e_handle = provider.upload(&e_view).expect("upload exponent");
        let gpu_value = block_on(pow2_gpu_scale(m_handle, e_handle)).expect("pow2 gpu scale");
        let gpu = test_support::gather(gpu_value).expect("gather gpu scale result");

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (g, c) in gpu.data.iter().zip(cpu.data.iter()) {
            assert!(
                (g - c).abs() <= tol,
                "scale mismatch: gpu={g} cpu={c} tol={tol}"
            );
        }
    }
}
