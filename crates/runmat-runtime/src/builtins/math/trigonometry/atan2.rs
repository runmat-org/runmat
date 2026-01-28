//! MATLAB-compatible `atan2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atan2";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atan2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_atan2",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can implement elem_atan2 to keep the computation on device; the runtime gathers operands to the host when the hook is unavailable or broadcasting is required.",
};

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atan2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let y = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let x = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("atan2({y}, {x})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits WGSL atan2(y, x); providers may override via elem_atan2 for standalone execution.",
};

#[runtime_builtin(
    name = "atan2",
    category = "math/trigonometry",
    summary = "Quadrant-aware inverse tangent atan2(y, x) with MATLAB-compatible broadcasting.",
    keywords = "atan2,inverse tangent,quadrant,gpu",
    accel = "binary",
    builtin_path = "crate::builtins::math::trigonometry::atan2"
)]
async fn atan2_builtin(y: Value, x: Value) -> BuiltinResult<Value> {
    match (y, x) {
        (Value::GpuTensor(yh), Value::GpuTensor(xh)) => atan2_gpu_pair(yh, xh).await,
        (Value::GpuTensor(yh), other) => {
            let gathered = gpu_helpers::gather_tensor_async(&yh).await?;
            atan2_host(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(xh)) => {
            let gathered = gpu_helpers::gather_tensor_async(&xh).await?;
            atan2_host(other, Value::Tensor(gathered))
        }
        (lhs, rhs) => atan2_host(lhs, rhs),
    }
}

async fn atan2_gpu_pair(y: GpuTensorHandle, x: GpuTensorHandle) -> BuiltinResult<Value> {
    if y.device_id == x.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&y) {
            if y.shape == x.shape {
                if let Ok(handle) = provider.elem_atan2(&y, &x).await {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let host_y = gpu_helpers::gather_tensor_async(&y).await?;
    let host_x = gpu_helpers::gather_tensor_async(&x).await?;
    atan2_host(Value::Tensor(host_y), Value::Tensor(host_x))
}

fn atan2_host(y: Value, x: Value) -> BuiltinResult<Value> {
    if let (Some(y_scalar), Some(x_scalar)) = (scalar_atan2_value(&y), scalar_atan2_value(&x)) {
        return Ok(Value::Num(y_scalar.atan2(x_scalar)));
    }
    let tensor_y = value_into_atan2_tensor(y)?;
    let tensor_x = value_into_atan2_tensor(x)?;
    compute_atan2_tensor(&tensor_y, &tensor_x)
}

fn compute_atan2_tensor(y: &Tensor, x: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&y.shape, &x.shape).map_err(runtime_error_for)?;
    if plan.is_empty() {
        let empty = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| runtime_error_for(format!("atan2: {e}")))?;
        return Ok(tensor::tensor_into_value(empty));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_index, idx_y, idx_x) in plan.iter() {
        out[out_index] = y.data[idx_y].atan2(x.data[idx_x]);
    }
    let tensor = Tensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| runtime_error_for(format!("atan2: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn value_into_atan2_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
            Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| runtime_error_for(format!("atan2: {e}")))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(runtime_error_for("atan2: complex inputs are not supported"))
        }
        Value::GpuTensor(_) => Err(runtime_error_for(
            "atan2: internal error converting GPU tensor",
        )),
        other => tensor::value_into_tensor_for("atan2", other).map_err(runtime_error_for),
    }
}

fn scalar_atan2_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => t.data.first().copied(),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(chars) if chars.rows * chars.cols == 1 => {
            Some(chars.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};
    use std::f64::consts::PI;

    const EPS: f64 = 1e-12;

    fn atan2_builtin(y: Value, x: Value) -> BuiltinResult<Value> {
        block_on(super::atan2_builtin(y, x))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_scalar_pair() {
        let result = atan2_builtin(Value::Num(1.0), Value::Num(1.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - PI / 4.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_quadrant_detection() {
        let result = atan2_builtin(Value::Num(-1.0), Value::Num(-1.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v + 3.0 * PI / 4.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_matrix_vs_scalar_broadcast() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = atan2_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(2.0),
                    (2.0f64).atan2(2.0),
                    (3.0f64).atan2(2.0),
                    (4.0f64).atan2(2.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_row_vector_broadcast() {
        let y = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("row broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(1.0),
                    (-1.0f64).atan2(1.0),
                    (2.0f64).atan2(1.0),
                    (-2.0f64).atan2(1.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_char_input() {
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let result = atan2_builtin(Value::CharArray(chars), Value::Num(100.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - (65.0f64).atan2(100.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_logical_input() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
        let result =
            atan2_builtin(Value::LogicalArray(logical), Value::Tensor(x)).expect("logical atan2");
        match result {
            Value::Tensor(t) => {
                let expected = [
                    1.0f64.atan2(1.0),
                    0.0f64.atan2(1.0),
                    0.0f64.atan2(-1.0),
                    1.0f64.atan2(-1.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_zero_zero_is_zero() {
        let result = atan2_builtin(Value::Num(0.0), Value::Num(0.0)).expect("atan2");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_signed_zero_behaviour() {
        let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
        let Value::Num(pi_case) =
            atan2_builtin(Value::Num(0.0), Value::Num(neg_zero)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert!((pi_case - PI).abs() < EPS, "{pi_case} vs PI");

        let Value::Num(neg_pi_case) =
            atan2_builtin(Value::Num(neg_zero), Value::Num(neg_zero)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert!((neg_pi_case + PI).abs() < EPS, "{neg_pi_case} vs -PI");

        let Value::Num(neg_zero_result) =
            atan2_builtin(Value::Num(neg_zero), Value::Num(0.0)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert_eq!(
            neg_zero_result.to_bits(),
            f64::from_bits(0x8000_0000_0000_0000).to_bits(),
            "expected negative zero, got {neg_zero_result}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_empty_tensor_result() {
        let y = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let x = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_complex_input_errors() {
        let err = atan2_builtin(Value::Complex(1.0, 1.0), Value::Num(1.0)).unwrap_err();
        let message = error_message(err);
        assert!(message.to_ascii_lowercase().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_dimension_mismatch_errors() {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let x = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let err = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).unwrap_err();
        let message = error_message(err);
        assert!(
            message.to_ascii_lowercase().contains("dimension"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
            let x = Tensor::new(vec![1.0, -1.0, 1.0, -1.0], vec![2, 2]).unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.data,
                    shape: &y.shape,
                })
                .expect("upload y");
            let hx = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &x.data,
                    shape: &x.shape,
                })
                .expect("upload x");
            let result =
                atan2_builtin(Value::GpuTensor(hy), Value::GpuTensor(hx)).expect("gpu atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [
                (1.0f64).atan2(1.0),
                (1.0f64).atan2(-1.0),
                (-1.0f64).atan2(1.0),
                (-1.0f64).atan2(-1.0),
            ];
            for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_gpu_host_mix_falls_back() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.data,
                    shape: &y.shape,
                })
                .expect("upload y");
            let result = atan2_builtin(Value::GpuTensor(hy), Value::Num(2.0)).expect("atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected = [(1.0f64).atan2(2.0), (2.0f64).atan2(2.0)];
            for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atan2_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let y = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
        let cpu = atan2_host(Value::Tensor(y.clone()), Value::Tensor(x.clone())).unwrap();
        let hy = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.data,
                shape: &y.shape,
            })
            .unwrap();
        let hx = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            })
            .unwrap();
        let gpu = block_on(atan2_gpu_pair(hy, hx)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (actual, expect) in gathered.data.iter().zip(ct.data.iter()) {
                    assert!((actual - expect).abs() < tol, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
