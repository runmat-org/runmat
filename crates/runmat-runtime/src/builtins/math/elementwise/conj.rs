//! MATLAB-compatible `conj` builtin with GPU-aware semantics for RunMat.

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conj",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_conj" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute conj in-place for real tensors via unary_conj; complex tensors currently gather to the host for conjugation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conj",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion kernels treat conj as an identity for real tensors; complex tensors fall back to the CPU path until native complex fusion is available.",
};

const BUILTIN_NAME: &str = "conj";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "conj",
    category = "math/elementwise",
    summary = "Compute the complex conjugate of scalars, vectors, matrices, or N-D tensors.",
    keywords = "conj,complex conjugate,complex,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::conj"
)]
async fn conj_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => conj_gpu(handle).await,
        Value::Complex(re, im) => conj_complex_scalar(re, im),
        Value::ComplexTensor(ct) => conj_complex_tensor(ct),
        Value::CharArray(ca) => conj_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("conj: expected numeric input"))
        }
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => conj_real(x),
        other => Err(builtin_error(format!(
            "conj: unsupported input type {:?}; expected numeric, logical, or char data",
            other
        ))),
    }
}

async fn conj_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_conj(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(conj_tensor(tensor)?))
}

fn conj_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("conj", value)
        .map_err(|e| builtin_error(format!("conj: {e}")))?;
    Ok(tensor::tensor_into_value(conj_tensor(tensor)?))
}

fn conj_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    Ok(tensor)
}

fn conj_complex_scalar(re: f64, im: f64) -> BuiltinResult<Value> {
    let imag = -im;
    if imag == 0.0 && !imag.is_nan() {
        Ok(Value::Num(re))
    } else {
        Ok(Value::Complex(re, imag))
    }
}

fn conj_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let ComplexTensor {
        data: ct_data,
        shape,
        ..
    } = ct;

    let mut all_real = true;
    let mut data = Vec::with_capacity(ct_data.len());
    for (re, im) in ct_data {
        let imag = -im;
        if imag != 0.0 || imag.is_nan() {
            all_real = false;
        }
        data.push((re, imag));
    }
    if all_real {
        let real: Vec<f64> = data.into_iter().map(|(re, _)| re).collect();
        let tensor =
            Tensor::new(real, shape.clone()).map_err(|e| builtin_error(format!("conj: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let tensor =
            ComplexTensor::new(data, shape).map_err(|e| builtin_error(format!("conj: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn conj_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("conj: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn conj_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::conj_builtin(value))
    }

    #[test]
    fn conj_type_preserves_tensor_shape() {
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
    fn conj_type_scalar_tensor_returns_num() {
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
    fn conj_scalar_real() {
        let result = conj_builtin(Value::Num(-2.5)).expect("conj");
        match result {
            Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar() {
        let result = conj_builtin(Value::Complex(3.0, 4.0)).expect("conj");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 3.0).abs() < 1e-12);
                assert!((im + 4.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar_zero_imag_returns_real() {
        let result = conj_builtin(Value::Complex(5.0, 0.0)).expect("conj");
        match result {
            Value::Num(n) => assert!((n - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_promotes_logical_to_double() {
        let logical =
            LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array construction");
        let result = conj_builtin(Value::LogicalArray(logical)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_int_promotes_to_double() {
        let result = conj_builtin(Value::Int(IntValue::I32(7))).expect("conj");
        match result {
            Value::Num(n) => assert!((n - 7.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_to_complex_tensor() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.data[0], (1.0, -2.0));
                assert_eq!(ct.data[1], (-3.0, 4.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_realises_real_when_imag_zero() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, -0.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![1.0, 2.0]);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_char_array_returns_double_codes() {
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).expect("char array");
        let result = conj_builtin(Value::CharArray(chars)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![72.0, 105.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_errors_on_string_input() {
        let err = conj_builtin(Value::from("hello")).unwrap_err();
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, tensor.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conj_wgpu_matches_cpu_for_real() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 0.0], vec![4, 1]).unwrap();
        let cpu = conj_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(conj_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                assert_eq!(ct.data, gt.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
