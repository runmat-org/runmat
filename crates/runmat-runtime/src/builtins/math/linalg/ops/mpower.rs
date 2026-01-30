//! MATLAB-compatible `mpower` builtin (matrix power) with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::type_resolvers::matrix_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "mpower";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::mpower")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mpower",
    op_kind: GpuOpKind::MatMul,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "matmul",
            commutative: false,
        },
        ProviderHook::Custom("eye_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses repeated provider matmul calls via binary exponentiation; falls back to the host implementation when matmul or identity creation is unavailable.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::mpower")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mpower",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion relies on the provider matmul hook; when unavailable the runtime executes the CPU fallback.",
};

#[runtime_builtin(
    name = "mpower",
    category = "math/linalg/ops",
    summary = "Matrix power with MATLAB-compatible semantics.",
    keywords = "mpower,matrix power,linear algebra,gpu",
    accel = "matmul",
    type_resolver(matrix_unary_type),
    builtin_path = "crate::builtins::math::linalg::ops::mpower"
)]
async fn mpower_builtin(base: Value, exponent: Value) -> BuiltinResult<Value> {
    mpower_eval(&base, &exponent).await
}

pub(crate) async fn mpower_eval(base: &Value, exponent: &Value) -> BuiltinResult<Value> {
    if let Some(result) = try_gpu_mpower(base, exponent).await? {
        return Ok(result);
    }

    let base_host = crate::dispatcher::gather_if_needed_async(base)
        .await
        .map_err(map_control_flow)?;
    let exponent_host = crate::dispatcher::gather_if_needed_async(exponent)
        .await
        .map_err(map_control_flow)?;
    let result = crate::elementwise::power(&base_host, &exponent_host).map_err(builtin_error)?;

    if matches!(base, Value::GpuTensor(_)) {
        if let Value::Tensor(tensor) = result {
            if let Some(provider) = runmat_accelerate_api::provider() {
                let view = HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                if let Ok(handle) = provider.upload(&view) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            return Ok(Value::Tensor(tensor));
        }
    }

    Ok(result)
}

async fn try_gpu_mpower(base: &Value, exponent: &Value) -> BuiltinResult<Option<Value>> {
    // Only attempt a GPU path when the base already resides on the GPU.
    let handle = match base {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let exponent_value = match parse_integer_exponent(exponent)? {
        Some(value) => value,
        None => return Ok(None),
    };

    if exponent_value < 0 {
        return Err(builtin_error("Negative matrix powers not supported yet"));
    }
    let shape = handle.shape.clone();
    if shape.len() != 2 {
        return Ok(None);
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows != cols {
        return Err(builtin_error(format!(
            "Matrix must be square for matrix power: {}x{}",
            rows, cols
        )));
    }

    if exponent_value == 0 {
        match gpu_identity_like(provider, handle, rows) {
            Ok(Some(identity)) => return Ok(Some(Value::GpuTensor(identity))),
            Ok(None) => return Ok(None),
            Err(err) => return Err(err),
        }
    }

    if exponent_value == 1 {
        return Ok(Some(Value::GpuTensor(handle.clone())));
    }

    gpu_binary_exponentiation(provider, handle, exponent_value as u32).await
}

fn gpu_identity_like(
    provider: &'static dyn AccelProvider,
    prototype: &GpuTensorHandle,
    size: usize,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    match provider.eye_like(prototype) {
        Ok(handle) => Ok(Some(handle)),
        Err(_) => {
            let eye = crate::matrix::matrix_eye(size);
            let view = HostTensorView {
                data: &eye.data,
                shape: &eye.shape,
            };
            match provider.upload(&view) {
                Ok(handle) => Ok(Some(handle)),
                Err(_) => Ok(None),
            }
        }
    }
}

async fn gpu_binary_exponentiation(
    provider: &'static dyn AccelProvider,
    base: &GpuTensorHandle,
    exponent: u32,
) -> BuiltinResult<Option<Value>> {
    let mut exp = exponent;
    let mut base_state = HandleState::borrowed(base);
    let mut result_state: Option<HandleState> = None;

    while exp > 0 {
        if exp & 1 == 1 {
            if let Some(ref mut current) = result_state {
                match provider.matmul(&current.handle, &base_state.handle).await {
                    Ok(new_handle) => {
                        if current.owned {
                            let _ = provider.free(&current.handle);
                        }
                        current.handle = new_handle;
                        current.owned = true;
                    }
                    Err(_) => {
                        if current.owned {
                            let _ = provider.free(&current.handle);
                        }
                        if base_state.owned {
                            let _ = provider.free(&base_state.handle);
                        }
                        return Ok(None);
                    }
                }
            } else {
                result_state = Some(HandleState::borrowed(&base_state.handle));
            }
        }

        exp >>= 1;
        if exp > 0 {
            match provider
                .matmul(&base_state.handle, &base_state.handle)
                .await
            {
                Ok(new_handle) => {
                    if base_state.owned {
                        let _ = provider.free(&base_state.handle);
                    }
                    base_state.handle = new_handle;
                    base_state.owned = true;
                }
                Err(_) => {
                    if base_state.owned {
                        let _ = provider.free(&base_state.handle);
                    }
                    if let Some(current) = result_state.take() {
                        if current.owned {
                            let _ = provider.free(&current.handle);
                        }
                    }
                    return Ok(None);
                }
            }
        }
    }

    if base_state.owned {
        let _ = provider.free(&base_state.handle);
    }

    let result_state = match result_state {
        Some(state) => state,
        None => return Ok(None),
    };

    Ok(Some(Value::GpuTensor(result_state.handle)))
}

fn parse_integer_exponent(value: &Value) -> BuiltinResult<Option<i32>> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
                return Err(builtin_error(
                    "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)",
                ));
            }
            Ok(Some(raw as i32))
        }
        Value::Num(n) => {
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(builtin_error("Matrix power requires integer exponent"));
            }
            if *n > i32::MAX as f64 || *n < i32::MIN as f64 {
                return Err(builtin_error(
                    "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)",
                ));
            }
            Ok(Some(*n as i32))
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            let scalar = t.data[0];
            if scalar.fract() != 0.0 || !scalar.is_finite() {
                return Err(builtin_error("Matrix power requires integer exponent"));
            }
            if scalar > i32::MAX as f64 || scalar < i32::MIN as f64 {
                return Err(builtin_error(
                    "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)",
                ));
            }
            Ok(Some(scalar as i32))
        }
        _ => Ok(None),
    }
}

#[derive(Clone)]
struct HandleState {
    handle: GpuTensorHandle,
    owned: bool,
}

impl HandleState {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor, Type};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_square_power() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).expect("mpower");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![7.0, 10.0, 15.0, 22.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mpower_type_preserves_matrix_shape() {
        let out = matrix_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(3), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zero_exponent_returns_identity() {
        let matrix = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let result =
            mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(0))).expect("mpower");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_inputs_match_standard_power() {
        let result = mpower_builtin(Value::Num(4.0), Value::Num(0.5)).expect("mpower");
        match result {
            Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_integer_exponent_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = unwrap_error(mpower_builtin(Value::Tensor(matrix), Value::Num(1.5)).unwrap_err());
        assert!(
            err.message()
                .contains("Matrix power requires integer exponent"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_exponent_errors() {
        let matrix = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = unwrap_error(
            mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(-1))).unwrap_err(),
        );
        assert!(
            err.message()
                .contains("Negative matrix powers not supported yet"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_square_matrix_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = unwrap_error(
            mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).unwrap_err(),
        );
        assert!(
            err.message()
                .contains("Matrix must be square for matrix power"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_scalar_power() {
        let result =
            mpower_builtin(Value::Complex(2.0, 1.0), Value::Int(IntValue::I32(3))).expect("mpower");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im - 11.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_matrix_power_roundtrip() {
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mpower_builtin(Value::GpuTensor(handle), Value::Int(IntValue::I32(3)))
                .expect("gpu mpower");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![37.0, 54.0, 81.0, 118.0]);
        });
    }

    fn mpower_builtin(base: Value, exponent: Value) -> BuiltinResult<Value> {
        block_on(super::mpower_builtin(base, exponent))
    }
}
