//! MATLAB-compatible `angle` builtin with GPU-aware semantics for RunMat.

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::angle")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "angle",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_angle" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers implement unary_angle to evaluate atan2(imag(x), real(x)) on device; the runtime gathers to host whenever the hook is unavailable or when complex tensors need host conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::angle")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "angle",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("atan2({zero}, {input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion assumes real-valued inputs (imaginary part zero). Complex tensors are gathered to the host until GPU complex storage lands.",
};

const BUILTIN_NAME: &str = "angle";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "angle",
    category = "math/elementwise",
    summary = "Phase angle of scalars, vectors, matrices, or N-D tensors.",
    keywords = "angle,phase,argument,complex,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::angle"
)]
async fn angle_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => angle_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Num(angle_scalar(re, im))),
        Value::ComplexTensor(ct) => angle_complex_tensor(ct),
        Value::CharArray(ca) => angle_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("angle: expected numeric input"))
        }
        other => angle_real(other),
    }
}

async fn angle_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(device_result) = provider.unary_angle(&handle).await {
            return Ok(Value::GpuTensor(device_result));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(angle_tensor(tensor)?))
}

fn angle_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("angle", value)
        .map_err(|e| builtin_error(format!("angle: {e}")))?;
    Ok(tensor::tensor_into_value(angle_tensor(tensor)?))
}

fn angle_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let Tensor { data, shape, .. } = tensor;
    let mapped: Vec<f64> = data.into_iter().map(|re| angle_scalar(re, 0.0)).collect();
    Tensor::new(mapped, shape).map_err(|e| builtin_error(format!("angle: {e}")))
}

fn angle_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let ComplexTensor { data, shape, .. } = ct;
    let mapped: Vec<f64> = data
        .into_iter()
        .map(|(re, im)| angle_scalar(re, im))
        .collect();
    let tensor = Tensor::new(mapped, shape).map_err(|e| builtin_error(format!("angle: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn angle_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = ca;
    let mapped: Vec<f64> = data
        .into_iter()
        .map(|ch| angle_scalar(ch as u32 as f64, 0.0))
        .collect();
    let tensor =
        Tensor::new(mapped, vec![rows, cols]).map_err(|e| builtin_error(format!("angle: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn angle_scalar(re: f64, im: f64) -> f64 {
    im.atan2(re)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, StringArray, Type};
    use std::f64::consts::PI;

    fn angle_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::angle_builtin(value))
    }

    #[test]
    fn angle_type_preserves_tensor_shape() {
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
    fn angle_type_scalar_tensor_returns_num() {
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
    fn angle_real_positive_negative() {
        let pos = angle_builtin(Value::Num(5.0)).expect("angle");
        assert_eq!(pos, Value::Num(0.0));

        let neg = angle_builtin(Value::Num(-3.0)).expect("angle");
        if let Value::Num(val) = neg {
            assert!((val - PI).abs() < 1e-12);
        } else {
            panic!("expected numeric result, got {neg:?}");
        }

        let zero = angle_builtin(Value::Num(0.0)).expect("angle");
        assert_eq!(zero, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_complex_scalar_matches_atan2() {
        let value = Value::Complex(3.0, -4.0);
        let result = angle_builtin(value).expect("angle");
        if let Value::Num(angle) = result {
            assert!((angle - (-4.0f64).atan2(3.0)).abs() < 1e-12);
        } else {
            panic!("expected numeric result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_tensor_values() {
        let tensor = Tensor::new(vec![1.0, -1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = angle_builtin(Value::Tensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!((out.data[0] - 0.0).abs() < 1e-12);
                assert!((out.data[1] - PI).abs() < 1e-12);
                assert_eq!(out.data[2], 0.0);
                assert_eq!(out.data[3], 0.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_logical_and_char_inputs() {
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let logical_value = Value::LogicalArray(logical);
        let logical_result = angle_builtin(logical_value).expect("angle");
        match logical_result {
            Value::Tensor(out) => assert!(out.data.iter().all(|&v| v == 0.0)),
            other => panic!("expected tensor result, got {other:?}"),
        }

        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let char_result = angle_builtin(Value::CharArray(chars)).expect("angle");
        match char_result {
            Value::Tensor(out) => assert!(out.data.iter().all(|&v| v == 0.0)),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_complex_tensor() {
        let data = vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = angle_builtin(Value::ComplexTensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    (1.0f64).atan2(1.0),
                    (1.0f64).atan2(-1.0),
                    (-1.0f64).atan2(-1.0),
                    (-1.0f64).atan2(1.0),
                ];
                for (actual, target) in out.data.iter().zip(expected.iter()) {
                    assert!((actual - target).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = angle_builtin(Value::GpuTensor(handle)).expect("angle");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| angle_scalar(v, 0.0)).collect();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, target) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - target).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_dimensionless_int_input() {
        let result = angle_builtin(Value::Int(IntValue::I32(-10))).expect("angle");
        if let Value::Num(val) = result {
            assert!((val - PI).abs() < 1e-12);
        } else {
            panic!("expected numeric result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_nan_propagates() {
        let result = angle_builtin(Value::Num(f64::NAN)).expect("angle");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_rejects_strings() {
        let err = angle_builtin(Value::from("hello")).unwrap_err();
        assert!(err.message().contains("angle: expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_rejects_string_arrays() {
        let array = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).unwrap();
        let err = angle_builtin(Value::StringArray(array)).unwrap_err();
        assert!(err.message().contains("angle: expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn angle_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
        let cpu = angle_tensor(tensor.clone()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(angle_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (Value::Tensor(cpu), gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
