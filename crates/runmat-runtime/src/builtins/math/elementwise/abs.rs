//! MATLAB-compatible `abs` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::abs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "abs",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_abs" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute abs in-place; the runtime gathers to host when unary_abs is unavailable or when complex magnitudes are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::abs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "abs",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("abs({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL abs; providers can swap in specialised kernels.",
};

#[runtime_builtin(
    name = "abs",
    category = "math/elementwise",
    summary = "Absolute value or magnitude of scalars, vectors, matrices, or N-D tensors.",
    keywords = "abs,absolute value,magnitude,complex,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::abs"
)]
fn abs_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => abs_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Num(complex_magnitude(re, im))),
        Value::ComplexTensor(ct) => abs_complex_tensor(ct),
        Value::CharArray(ca) => abs_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("abs: expected numeric input".to_string()),
        other => abs_real(other),
    }
}

fn abs_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_abs(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    abs_tensor(tensor).map(tensor::tensor_into_value)
}

fn abs_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("abs", value)?;
    abs_tensor(tensor).map(tensor::tensor_into_value)
}

fn abs_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.abs()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("abs: {e}"))
}

fn abs_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let data = ct
        .data
        .iter()
        .map(|&(re, im)| complex_magnitude(re, im))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, ct.shape.clone()).map_err(|e| format!("abs: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn abs_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("abs: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn complex_magnitude(re: f64, im: f64) -> f64 {
    re.hypot(im)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_scalar_negative() {
        let result = abs_builtin(Value::Num(-3.5)).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 3.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_int_promotes() {
        let result = abs_builtin(Value::Int(IntValue::I32(-8))).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 8.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let result = abs_builtin(Value::Tensor(tensor)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_scalar() {
        let result = abs_builtin(Value::Complex(3.0, 4.0)).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_tensor_to_real_tensor() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let result = abs_builtin(Value::ComplexTensor(complex)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 5.0).abs() < 1e-12);
                assert!((t.data[1] - (2f64).sqrt()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_char_array_codes() {
        let char_array = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = abs_builtin(Value::CharArray(char_array)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 122.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_string_rejected() {
        let err = abs_builtin(Value::from("hello")).expect_err("should error");
        assert!(err.contains("expected numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -1.0, 0.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![2.0, 1.0, 0.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn abs_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.0, -1.0, 0.5, -0.25], vec![4, 1]).unwrap();
        let cpu = abs_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = abs_gpu(h).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((*a - *b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected result shape"),
        }
    }
}
