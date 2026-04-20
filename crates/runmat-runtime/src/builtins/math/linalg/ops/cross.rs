//! MATLAB-compatible `cross` builtin with GPU-aware semantics for RunMat.
//!
//! Implements 3-element vector cross products for row vectors, column vectors,
//! matrices of vectors, and higher-rank tensors. GPU inputs dispatch to a
//! provider-side `cross` hook when available and otherwise fall back to the
//! host implementation with result re-upload for real-valued outputs.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::type_resolvers::cross_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CROSS_NAME: &str = "cross";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::cross")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cross",
    op_kind: GpuOpKind::Custom("cross"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cross")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Dispatches to a provider-side cross implementation when available; otherwise gathers inputs, evaluates on the host, and re-uploads real-valued outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::cross")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cross",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Cross products allocate a fresh tensor and terminate fusion graphs.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(CROSS_NAME)
        .build()
}

async fn parse_dimension_arg(value: &Value) -> Result<usize, String> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            tensor::dimension_from_value_async(value, CROSS_NAME, false)
                .await
                .and_then(|dim| {
                    dim.ok_or_else(|| {
                        format!("{CROSS_NAME}: dimension must be numeric, got {value:?}")
                    })
                })
        }
        _ => Err(format!(
            "{CROSS_NAME}: dimension must be numeric, got {value:?}"
        )),
    }
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(CROSS_NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(CROSS_NAME);
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

#[runtime_builtin(
    name = "cross",
    category = "math/linalg/ops",
    summary = "Cross product of 3-element vectors along a matching dimension.",
    keywords = "cross,vector product,3d vector,gpu,linear algebra",
    accel = "custom",
    type_resolver(cross_type),
    builtin_path = "crate::builtins::math::linalg::ops::cross"
)]
async fn cross_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(builtin_error("cross: too many input arguments"));
    }
    let dim = match rest.first() {
        Some(value) => Some(parse_dimension_arg(value).await.map_err(builtin_error)?),
        None => None,
    };

    if let (Value::GpuTensor(lhs_handle), Value::GpuTensor(rhs_handle)) = (&lhs, &rhs) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            match provider.cross(lhs_handle, rhs_handle, dim) {
                Ok(handle) => return Ok(Value::GpuTensor(handle)),
                Err(err) => {
                    log::trace!("cross: provider cross fallback triggered: {err}");
                }
            }
        }
    }

    let lhs_gpu = matches!(lhs, Value::GpuTensor(_));
    let rhs_gpu = matches!(rhs, Value::GpuTensor(_));

    let lhs_host = gather_if_needed_async(&lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs_host = gather_if_needed_async(&rhs)
        .await
        .map_err(map_control_flow)?;

    let has_complex = value_is_complex(&lhs_host) || value_is_complex(&rhs_host);

    let value = if has_complex {
        let lhs_complex = value_into_complex_tensor(lhs_host)?;
        let rhs_complex = value_into_complex_tensor(rhs_host)?;
        let result = cross_complex_tensor(&lhs_complex, &rhs_complex, dim)?;
        complex_tensor_into_value(result)
    } else {
        let lhs_tensor =
            tensor::value_into_tensor_for(CROSS_NAME, lhs_host).map_err(builtin_error)?;
        let rhs_tensor =
            tensor::value_into_tensor_for(CROSS_NAME, rhs_host).map_err(builtin_error)?;
        let result = cross_real_tensor(&lhs_tensor, &rhs_tensor, dim)?;
        if lhs_gpu || rhs_gpu {
            return promote_real_result_to_gpu(result);
        }
        tensor::tensor_into_value(result)
    };

    Ok(value)
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn value_into_complex_tensor(value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(t) => Ok(t),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}"))),
        Value::Tensor(t) => real_tensor_to_complex(&t),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(builtin_error)?;
            real_tensor_to_complex(&tensor)
        }
        other => Err(builtin_error(format!(
            "{CROSS_NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn real_tensor_to_complex(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let shape = canonical_shape_tensor(tensor);
    let mut data = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        data.push((value, 0.0));
    }
    ComplexTensor::new(data, shape).map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}")))
}

pub fn cross_host_real_for_provider(
    a: &Tensor,
    b: &Tensor,
    dim: Option<usize>,
) -> BuiltinResult<Tensor> {
    cross_real_tensor(a, b, dim)
}

fn cross_real_tensor(a: &Tensor, b: &Tensor, dim: Option<usize>) -> BuiltinResult<Tensor> {
    ensure_same_size(a, b)?;

    let shape = canonical_shape_tensor(a);
    let target_dim = resolve_dimension(&shape, dim)?;
    let dim_index = target_dim - 1;
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let slice_stride = stride_before * 3;
    let mut output = vec![0.0f64; a.data.len()];

    for after in 0..stride_after {
        let slice_base = after * slice_stride;
        for before in 0..stride_before {
            let idx1 = slice_base + before;
            let idx2 = idx1 + stride_before;
            let idx3 = idx2 + stride_before;

            let a1 = a.data[idx1];
            let a2 = a.data[idx2];
            let a3 = a.data[idx3];
            let b1 = b.data[idx1];
            let b2 = b.data[idx2];
            let b3 = b.data[idx3];

            output[idx1] = a2 * b3 - a3 * b2;
            output[idx2] = a3 * b1 - a1 * b3;
            output[idx3] = a1 * b2 - a2 * b1;
        }
    }

    Tensor::new(output, shape).map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}")))
}

fn cross_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    ensure_same_size_complex(a, b)?;

    let shape = canonical_shape_complex(a);
    let target_dim = resolve_dimension(&shape, dim)?;
    let dim_index = target_dim - 1;
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let slice_stride = stride_before * 3;
    let mut output = vec![(0.0f64, 0.0f64); a.data.len()];

    for after in 0..stride_after {
        let slice_base = after * slice_stride;
        for before in 0..stride_before {
            let idx1 = slice_base + before;
            let idx2 = idx1 + stride_before;
            let idx3 = idx2 + stride_before;

            let a1 = a.data[idx1];
            let a2 = a.data[idx2];
            let a3 = a.data[idx3];
            let b1 = b.data[idx1];
            let b2 = b.data[idx2];
            let b3 = b.data[idx3];

            output[idx1] = complex_sub(complex_mul(a2, b3), complex_mul(a3, b2));
            output[idx2] = complex_sub(complex_mul(a3, b1), complex_mul(a1, b3));
            output[idx3] = complex_sub(complex_mul(a1, b2), complex_mul(a2, b1));
        }
    }

    ComplexTensor::new(output, shape).map_err(|e| builtin_error(format!("{CROSS_NAME}: {e}")))
}

fn complex_mul(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn complex_sub(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

fn ensure_same_size(a: &Tensor, b: &Tensor) -> BuiltinResult<()> {
    if a.data.len() != b.data.len() || canonical_shape_tensor(a) != canonical_shape_tensor(b) {
        return Err(builtin_error("cross: A and B must be the same size."));
    }
    Ok(())
}

fn ensure_same_size_complex(a: &ComplexTensor, b: &ComplexTensor) -> BuiltinResult<()> {
    if a.data.len() != b.data.len() || canonical_shape_complex(a) != canonical_shape_complex(b) {
        return Err(builtin_error("cross: A and B must be the same size."));
    }
    Ok(())
}

fn canonical_shape_tensor(t: &Tensor) -> Vec<usize> {
    if t.shape.is_empty() {
        vec![t.rows, t.cols]
    } else {
        t.shape.clone()
    }
}

fn canonical_shape_complex(t: &ComplexTensor) -> Vec<usize> {
    if t.shape.is_empty() {
        vec![t.rows, t.cols]
    } else {
        t.shape.clone()
    }
}

fn resolve_dimension(shape: &[usize], dim: Option<usize>) -> BuiltinResult<usize> {
    match dim {
        Some(target_dim) => {
            if target_dim > shape.len() {
                return Err(builtin_error(format!(
                    "cross: dimension {} exceeds the number of array dimensions ({})",
                    target_dim,
                    shape.len()
                )));
            }
            if shape[target_dim - 1] != 3 {
                return Err(builtin_error(format!(
                    "cross: dimension {} must have length 3",
                    target_dim
                )));
            }
            Ok(target_dim)
        }
        None => shape
            .iter()
            .position(|&extent| extent == 3)
            .map(|idx| idx + 1)
            .ok_or_else(|| builtin_error("cross: inputs must have a dimension of length 3")),
    }
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .expect("cross: internal dimension overflow")
}

fn promote_real_result_to_gpu(tensor: Tensor) -> BuiltinResult<Value> {
    let provider = match runmat_accelerate_api::provider() {
        Some(provider) => provider,
        None => return Ok(tensor::tensor_into_value(tensor)),
    };
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    match provider.upload(&view) {
        Ok(handle) => Ok(Value::GpuTensor(handle)),
        Err(_) => Ok(tensor::tensor_into_value(tensor)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LiteralValue, LogicalArray, ResolveContext, Type};

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[test]
    fn cross_type_preserves_known_shape() {
        let out = cross_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(3)])
            }
        );
    }

    #[test]
    fn cross_type_uses_literal_dim() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
        ]);
        let out = cross_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Int,
            ],
            &ctx,
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_row_vectors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_column_vectors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![3, 1]).unwrap();
        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data, vec![0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_rowwise_dimension_argument() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![2, 3]).unwrap();
        let value = cross_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_nd_along_third_dimension() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![1, 2, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![1, 2, 3]).unwrap();
        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2, 3]);
                assert_eq!(t.data, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_complex_vectors() {
        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], vec![1, 3]).unwrap();
        let rhs =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, -2.0), (0.0, 0.0)], vec![1, 3]).unwrap();
        let value = cross_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("cross");
        match value {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data[0], (0.0, 0.0));
                assert_eq!(t.data[1], (0.0, 0.0));
                assert_eq!(t.data[2], (3.0, -1.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_promotes_logical_inputs() {
        let lhs = LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let value =
            cross_builtin(Value::LogicalArray(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![0.0, 0.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_shapes_mismatch() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let err = unwrap_error(
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect_err("cross"),
        );
        assert!(err.message().contains("A and B must be the same size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_no_dimension_has_length_three() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let err = unwrap_error(
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new())
                .expect_err("expected cross error"),
        );
        assert!(err.message().contains("dimension of length 3"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_dimension_exceeds_rank() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Int(IntValue::I32(3))],
            )
            .expect_err("expected rank error"),
        );
        assert!(err
            .message()
            .contains("exceeds the number of array dimensions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_dimension_length_is_not_three() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Int(IntValue::I32(1))],
            )
            .expect_err("expected length error"),
        );
        assert!(err.message().contains("must have length 3"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_dimension_zero_errors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect_err("expected dimension error"),
        );
        assert!(err.message().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_dimension_non_integer_errors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Num(1.5)],
            )
            .expect_err("expected integer dimension error"),
        );
        assert!(err.message().contains("dimension must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let view_rhs = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
            let value = cross_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::GpuTensor(gpu_rhs),
                Vec::new(),
            )
            .expect("cross");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 3]);
                    assert_eq!(gathered.data, vec![0.0, 0.0, 1.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cross_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![2, 3]).unwrap();
        let cpu = cross_real_tensor(&lhs, &rhs, Some(2)).expect("cpu cross");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_lhs = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
        let gpu_value = cross_builtin(
            Value::GpuTensor(gpu_lhs),
            Value::GpuTensor(gpu_rhs),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("gpu cross");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }

    fn cross_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cross_builtin(lhs, rhs, rest))
    }
}
