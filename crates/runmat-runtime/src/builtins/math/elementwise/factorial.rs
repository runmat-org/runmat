//! MATLAB-compatible `factorial` builtin with GPU-aware semantics for RunMat.
//!
//! Implements element-wise factorial for numerical inputs, mirroring MATLAB’s
//! restrictions to non-negative integers while providing documented fallbacks
//! when GPU providers lack a dedicated kernel.

use once_cell::sync::Lazy;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const MAX_FACTORIAL_N: usize = 170;

static FACT_TABLE: Lazy<[f64; MAX_FACTORIAL_N + 1]> = Lazy::new(|| {
    let mut table = [1.0f64; MAX_FACTORIAL_N + 1];
    let mut acc = 1.0;
    for (n, slot) in table.iter_mut().enumerate().skip(1) {
        acc *= n as f64;
        *slot = acc;
    }
    table
});

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::factorial")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "factorial",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_factorial",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_factorial; otherwise the runtime gathers to host and mirrors MATLAB overflow/NaN behaviour.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::factorial"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "factorial",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Factorial is evaluated as a scalar helper; fusion currently bypasses it and executes the standalone host or provider kernel.",
};

#[runtime_builtin(
    name = "factorial",
    category = "math/elementwise",
    summary = "Element-wise factorial for non-negative integers.",
    keywords = "factorial,n!,permutation,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::factorial"
)]
fn factorial_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let output = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => factorial_gpu(handle)?,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(
                "factorial: complex inputs are not supported; use gamma(z + 1) instead".to_string(),
            )
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            return Err("factorial: expected numeric or logical input".to_string())
        }
        other => {
            let tensor = tensor::value_into_tensor_for("factorial", other)?;
            factorial_tensor(tensor).map(tensor::tensor_into_value)?
        }
    };
    apply_output_template(base, &output)
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> Result<OutputTemplate, String> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err("factorial: expected prototype after 'like'".to_string())
            } else {
                Err("factorial: unrecognised option; only 'like' is supported".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("factorial: unrecognised option; only 'like' is supported".to_string())
            }
        }
        _ => Err("factorial: too many input arguments".to_string()),
    }
}

fn factorial_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_factorial(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    factorial_tensor(tensor).map(tensor::tensor_into_value)
}

fn factorial_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let mut data = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        data.push(factorial_scalar(value));
    }
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("factorial: {e}"))
}

fn factorial_scalar(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value == 0.0 {
        return 1.0;
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if value < 0.0 {
        return f64::NAN;
    }
    let Some(n) = classify_nonnegative_integer(value) else {
        return f64::NAN;
    };
    if n > MAX_FACTORIAL_N {
        return f64::INFINITY;
    }
    FACT_TABLE[n]
}

fn classify_nonnegative_integer(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    if value < 0.0 {
        return None;
    }
    let rounded = value.round();
    let tol = f64::EPSILON * value.abs().max(1.0);
    if (value - rounded).abs() > tol {
        return None;
    }
    if rounded < 0.0 {
        return None;
    }
    Some(rounded as usize)
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> Result<Value, String> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => {
            let analysis = analyse_like_prototype(proto)?;
            match analysis.device {
                DevicePreference::Host => convert_to_host_like(value),
                DevicePreference::Gpu => convert_to_gpu_like(value),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

struct LikeAnalysis {
    device: DevicePreference,
}

fn analyse_like_prototype(proto: &Value) -> Result<LikeAnalysis, String> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
        }),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
        }),
        Value::LogicalArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(
            "factorial: complex prototypes for 'like' are not supported; results are always real"
                .to_string(),
        ),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err("factorial: prototype must be numeric or a gpuArray".to_string())
        }
        other => {
            let gathered =
                gpu_helpers::gather_value(other).map_err(|e| format!("factorial: {e}"))?;
            analyse_like_prototype(&gathered)
        }
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value(&Value::GpuTensor(handle))
            .map_err(|e| format!("factorial: {e}")),
        other => Ok(other),
    }
}

fn convert_to_gpu_like(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "factorial: GPU output requested via 'like' but no acceleration provider is active"
            .to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => upload_tensor(provider, tensor),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("factorial: {e}"))?;
            upload_tensor(provider, tensor)
        }
        Value::Int(i) => convert_to_gpu_like(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu_like(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            upload_tensor(provider, tensor)
        }
        other => Err(format!(
            "factorial: cannot place value {other:?} on the GPU via 'like'"
        )),
    }
}

fn upload_tensor(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> Result<Value, String> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| format!("factorial: failed to upload GPU result: {e}"))?;
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_scalar_positive() {
        let result = factorial_builtin(Value::Num(5.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(120.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_zero_is_one() {
        let result = factorial_builtin(Value::Num(0.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_vector_inputs() {
        let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![1.0, 1.0, 6.0, 120.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_non_integer_produces_nan() {
        let result = factorial_builtin(Value::Num(2.5), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_negative_produces_nan() {
        let tensor = Tensor::new(vec![-1.0, 3.0], vec![2, 1]).unwrap();
        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert!(out.data[0].is_nan());
                assert_eq!(out.data[1], 6.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_small_positive_non_integer_nan() {
        let result = factorial_builtin(Value::Num(1e-12), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_overflow_returns_inf() {
        let result = factorial_builtin(Value::Num(171.0), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_infinite()),
            other => panic!("expected scalar Inf, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_missing_prototype_errors() {
        let err = factorial_builtin(Value::Num(3.0), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_gpu_prototype_uploads() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = factorial_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("factorial");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.data, vec![6.0, 24.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(Value::GpuTensor(handle), Vec::new()).expect("fact");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 1.0, 6.0, 120.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("factorial");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.data, vec![6.0, 24.0]);
                }
                other => panic!("expected host tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_logical_input_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = factorial_builtin(Value::LogicalArray(logical), Vec::new()).expect("fact");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_int_input_promotes_to_double() {
        let value = Value::Int(IntValue::U16(5));
        let result = factorial_builtin(value, Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(120.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_nan_propagates() {
        let result = factorial_builtin(Value::Num(f64::NAN), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_complex_input_errors() {
        let err = factorial_builtin(Value::Complex(1.0, 0.5), Vec::new())
            .expect_err("expected complex rejection");
        assert!(err.contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_string_input_errors() {
        let err = factorial_builtin(Value::from("hello"), Vec::new())
            .expect_err("expected string rejection");
        assert!(err.contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_complex_prototype_rejected() {
        let err = factorial_builtin(
            Value::Num(3.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected complex prototype rejection");
        assert!(err.contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn factorial_wgpu_matches_cpu_after_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0], vec![3, 1]).unwrap();
        let cpu = factorial_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = factorial_gpu(handle).expect("gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }
}
