//! MATLAB-compatible `tanh` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::tanh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute tanh directly on the device; runtimes gather to the host when unary_tanh is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::tanh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tanh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL `tanh` calls; providers may override with specialised kernels.",
};

#[runtime_builtin(
    name = "tanh",
    category = "math/trigonometry",
    summary = "Hyperbolic tangent of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "tanh,hyperbolic tangent,trigonometry,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::trigonometry::tanh"
)]
fn tanh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => tanh_gpu(handle),
        Value::Complex(re, im) => {
            let (real, imag) = tanh_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => tanh_complex_tensor(ct),
        Value::CharArray(ca) => tanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("tanh: expected numeric input".to_string()),
        other => tanh_real(other),
    }
}

fn tanh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_tanh(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    tanh_tensor(tensor).map(tensor::tensor_into_value)
}

fn tanh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("tanh", value)?;
    tanh_tensor(tensor).map(tensor::tensor_into_value)
}

fn tanh_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.tanh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("tanh: {e}"))
}

fn tanh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| tanh_complex_parts(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("tanh: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn tanh_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tanh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("tanh: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn tanh_complex_parts(re: f64, im: f64) -> (f64, f64) {
    // Use tanh(z) = sinh(z) / cosh(z) with explicit real/imag components.
    let sinh_re = re.sinh() * im.cos();
    let sinh_im = re.cosh() * im.sin();
    let cosh_re = re.cosh() * im.cos();
    let cosh_im = re.sinh() * im.sin();
    let denom = cosh_re * cosh_re + cosh_im * cosh_im;
    // Division by zero yields the expected IEEE infinities/NaNs, matching MATLAB's behaviour at poles.
    let real = (sinh_re * cosh_re + sinh_im * cosh_im) / denom;
    let imag = (sinh_im * cosh_re - sinh_re * cosh_im) / denom;
    (real, imag)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use num_complex::Complex64;
    use runmat_builtins::{CharArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_scalar_num() {
        let result = tanh_builtin(Value::Num(1.0)).expect("tanh");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.tanh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = tanh_builtin(Value::Tensor(tensor)).expect("tanh");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                for (value, expected) in out
                    .data
                    .iter()
                    .zip([-1.0_f64.tanh(), 0.0, 1.0_f64.tanh()].iter())
                {
                    assert!((*value - *expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_complex_scalar() {
        let result = tanh_builtin(Value::Complex(0.5, 1.0)).expect("tanh");
        match result {
            Value::Complex(re, im) => {
                let target = Complex64::new(0.5, 1.0).tanh();
                assert!((re - target.re).abs() < 1e-12);
                assert!((im - target.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_char_array_roundtrip() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = tanh_builtin(Value::CharArray(chars)).expect("tanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Az".chars().map(|c| (c as u32 as f64).tanh()).collect();
                for (value, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((*value - *expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tanh_builtin(Value::GpuTensor(handle)).expect("tanh");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            for (value, expect) in gathered.data.iter().zip(tensor.data.iter()) {
                assert!((*value - expect.tanh()).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tanh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![-1.25, -0.5, 0.0, 0.75, 1.5], vec![5, 1]).unwrap();
        let cpu_value = tanh_real(Value::Tensor(tensor.clone())).expect("cpu tanh");
        let cpu_tensor = test_support::gather(cpu_value).expect("gather cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu_value = tanh_gpu(handle).expect("gpu tanh");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, expect) in gpu_tensor.data.iter().zip(cpu_tensor.data.iter()) {
            assert!(
                (*got - *expect).abs() < tol,
                "tanh mismatch: got {got}, expect {expect}, tol {tol}"
            );
        }
    }
}
