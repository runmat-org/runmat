//! MATLAB-compatible `dot` builtin with GPU-aware semantics for RunMat.
//!
//! Implements inner products for real and complex inputs, including dimension-aware
//! reductions that match MathWorks MATLAB behaviour. GPU inputs are gathered when
//! necessary and the result is re-uploaded to the active provider when possible so
//! downstream consumers can remain device-resident.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::type_resolvers::dot_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DOT_NAME: &str = "dot";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::dot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dot",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Reduction { name: "dot" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(1024),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes: "Dispatches to a provider-side dot implementation when available; otherwise gathers operands and re-uploads real outputs.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(DOT_NAME).build()
}

async fn parse_dimension_arg(value: &Value) -> Result<usize, String> {
    match value {
        Value::Int(_) | Value::Num(_) => tensor::dimension_from_value_async(value, DOT_NAME, false)
            .await
            .and_then(|dim| {
                dim.ok_or_else(|| format!("{DOT_NAME}: dimension must be numeric, got {value:?}"))
            }),
        _ => Err(format!(
            "{DOT_NAME}: dimension must be numeric, got {value:?}"
        )),
    }
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(DOT_NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(DOT_NAME);
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::dot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Higher-level fusion currently delegates to dedicated dot kernels or host fallbacks.",
};

#[runtime_builtin(
    name = "dot",
    category = "math/linalg/ops",
    summary = "Dot product (inner product) of matching tensors along a specified dimension.",
    keywords = "dot,inner product,gpu,linear algebra",
    accel = "reduction",
    type_resolver(dot_type),
    builtin_path = "crate::builtins::math::linalg::ops::dot"
)]
async fn dot_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(builtin_error("dot: too many input arguments"));
    }
    let dim = match rest.first() {
        Some(value) => Some(parse_dimension_arg(value).await.map_err(builtin_error)?),
        None => None,
    };

    if let (Value::GpuTensor(lhs_handle), Value::GpuTensor(rhs_handle)) = (&lhs, &rhs) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            match provider.dot(lhs_handle, rhs_handle, dim).await {
                Ok(handle) => return Ok(Value::GpuTensor(handle)),
                Err(err) => {
                    log::trace!("dot: provider dot fallback triggered: {err}");
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
        let result = dot_complex_tensor(&lhs_complex, &rhs_complex, dim)?;
        complex_tensor_into_value(result)
    } else {
        let lhs_tensor =
            tensor::value_into_tensor_for(DOT_NAME, lhs_host).map_err(builtin_error)?;
        let rhs_tensor =
            tensor::value_into_tensor_for(DOT_NAME, rhs_host).map_err(builtin_error)?;
        let result = dot_real_tensor(&lhs_tensor, &rhs_tensor, dim)?;
        tensor::tensor_into_value(result)
    };

    if lhs_gpu || rhs_gpu {
        promote_result_to_gpu(value)
    } else {
        Ok(value)
    }
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn value_into_complex_tensor(value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(t) => Ok(t),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|e| builtin_error(format!("{DOT_NAME}: {e}"))),
        Value::Tensor(t) => real_tensor_to_complex(&t),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| builtin_error(e))?;
            real_tensor_to_complex(&tensor)
        }
        other => Err(builtin_error(format!(
            "{DOT_NAME}: unsupported input type {:?}; expected numeric or logical values",
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
    ComplexTensor::new(data, shape).map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))
}

fn dot_real_tensor(a: &Tensor, b: &Tensor, dim: Option<usize>) -> BuiltinResult<Tensor> {
    ensure_same_size(a, b)?;

    let shape = canonical_shape_tensor(a);
    let target_dim = dim.unwrap_or_else(|| default_dimension(&shape));
    let dim_index = target_dim - 1;

    if dim_index >= shape.len() {
        return elementwise_real_product(a, b);
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let mut output = vec![0.0f64; stride_before * stride_after];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut acc = 0.0;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let prod = a.data[idx] * b.data[idx];
                acc += prod;
            }
            let out_idx = after * stride_before + before;
            output[out_idx] = acc;
        }
    }

    let mut out_shape = shape.clone();
    out_shape[dim_index] = 1;
    Tensor::new(output, out_shape).map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))
}

fn dot_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    ensure_same_size_complex(a, b)?;

    let shape = canonical_shape_complex(a);
    let target_dim = dim.unwrap_or_else(|| default_dimension(&shape));
    let dim_index = target_dim - 1;

    if dim_index >= shape.len() {
        return elementwise_complex_product(a, b);
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let mut output = vec![(0.0f64, 0.0f64); stride_before * stride_after];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let (ar, ai) = a.data[idx];
                let (br, bi) = b.data[idx];
                let real = ar * br + ai * bi;
                let imag = ar * bi - ai * br;
                acc_re += real;
                acc_im += imag;
            }
            let out_idx = after * stride_before + before;
            output[out_idx] = (acc_re, acc_im);
        }
    }

    let mut out_shape = shape.clone();
    out_shape[dim_index] = 1;
    ComplexTensor::new(output, out_shape).map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))
}

pub fn dot_host_real_for_provider(
    a: &Tensor,
    b: &Tensor,
    dim: Option<usize>,
) -> BuiltinResult<Tensor> {
    dot_real_tensor(a, b, dim)
}

pub fn dot_host_complex_for_provider(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    dot_complex_tensor(a, b, dim)
}

fn elementwise_real_product(a: &Tensor, b: &Tensor) -> BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(a.data.len());
    for (x, y) in a.data.iter().zip(&b.data) {
        data.push(x * y);
    }
    let shape = canonical_shape_tensor(a);
    Tensor::new(data, shape).map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))
}

fn elementwise_complex_product(
    a: &ComplexTensor,
    b: &ComplexTensor,
) -> BuiltinResult<ComplexTensor> {
    let mut data = Vec::with_capacity(a.data.len());
    for ((ar, ai), (br, bi)) in a.data.iter().zip(&b.data) {
        let real = ar * br + ai * bi;
        let imag = ar * bi - ai * br;
        data.push((real, imag));
    }
    let shape = canonical_shape_complex(a);
    ComplexTensor::new(data, shape).map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))
}

fn ensure_same_size(a: &Tensor, b: &Tensor) -> BuiltinResult<()> {
    if a.data.len() != b.data.len() {
        return Err(builtin_error(format!(
            "{DOT_NAME}: A and B must be the same size."
        )));
    }
    if canonical_shape_tensor(a) != canonical_shape_tensor(b) {
        return Err(builtin_error(format!(
            "{DOT_NAME}: A and B must be the same size."
        )));
    }
    Ok(())
}

fn ensure_same_size_complex(a: &ComplexTensor, b: &ComplexTensor) -> BuiltinResult<()> {
    if a.data.len() != b.data.len() {
        return Err(builtin_error(format!(
            "{DOT_NAME}: A and B must be the same size."
        )));
    }
    if canonical_shape_complex(a) != canonical_shape_complex(b) {
        return Err(builtin_error(format!(
            "{DOT_NAME}: A and B must be the same size."
        )));
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

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim))
}

fn promote_result_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(value),
    };
    match value {
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            match provider.upload(&view) {
                Ok(handle) => Ok(Value::GpuTensor(handle)),
                Err(_) => Ok(Value::Tensor(tensor)),
            }
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("{DOT_NAME}: {e}")))?;
            promote_result_to_gpu(Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| builtin_error(e))?;
            promote_result_to_gpu(Value::Tensor(tensor))
        }
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        other => Ok(other),
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_row_vectors() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Num(result) => assert_eq!(result, 32.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn dot_type_reduces_first_dimension() {
        let out = dot_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(2)])
            }
        );
    }

    #[test]
    fn dot_type_vector_with_dim_returns_scalar() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
        ]);
        let out = dot_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(4)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(4)]),
                },
                Type::Int,
            ],
            &ctx,
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_column_vectors() {
        let lhs = Tensor::new(vec![1.0, 3.0, 5.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Num(result) => assert_eq!(result, 44.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_with_dimension_argument() {
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], vec![2, 3]).unwrap();
        let cols = dot_builtin(
            Value::Tensor(lhs.clone()),
            Value::Tensor(rhs.clone()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("dot");
        match cols {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![18.0, 20.0, 18.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let rows = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("dot");
        match rows {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![28.0, 28.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_with_dimension() {
        let lhs = ComplexTensor::new(
            vec![(1.0, 1.0), (3.0, -2.0), (2.0, -3.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let rhs = ComplexTensor::new(
            vec![(2.0, -1.0), (1.0, 4.0), (-1.0, 2.0), (3.0, 5.0)],
            vec![2, 2],
        )
        .unwrap();
        let value = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("dot");
        match value {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(-4.0, 11.0), (4.0, 21.0)];
                for (idx, (got, exp)) in t.data.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got.0 - exp.0).abs() < 1e-12,
                        "real mismatch at {idx}: got {}, expected {}",
                        got.0,
                        exp.0
                    );
                    assert!(
                        (got.1 - exp.1).abs() < 1e-12,
                        "imag mismatch at {idx}: got {}, expected {}",
                        got.1,
                        exp.1
                    );
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_uses_conjugate_first_argument() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -3.0), (-1.0, 5.0)], vec![1, 2]).unwrap();
        let value = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("dot");
        match value {
            Value::Complex(re, im) => {
                assert!((re + 27.0).abs() < 1e-12, "expected real -27, got {re}");
                assert!((im - 4.0).abs() < 1e-12, "expected imag 4, got {im}");
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_and_real_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let value =
            dot_builtin(Value::ComplexTensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Complex(re, im) => {
                assert!((re - 11.0).abs() < 1e-12, "expected real 11, got {re}");
                assert!((im - 1.0).abs() < 1e-12, "expected imag 1, got {im}");
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_empty_reduction_returns_zero() {
        let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
                assert_eq!(t.shape, vec![1, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_mismatched_shapes_error() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect_err("dot"),
        );
        assert!(err.message().contains("A and B must be the same size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_zero_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            dot_builtin(
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
    fn dot_dimension_non_integer_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            dot_builtin(
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
    fn dot_promotes_logical_inputs() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = dot_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("dot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 7.0]);
                assert_eq!(t.shape, vec![1, 2]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![1, 4]).unwrap();
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
            let value = dot_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::GpuTensor(gpu_rhs),
                Vec::new(),
            )
            .expect("dot");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 1]);
                    assert_eq!(gathered.data, vec![20.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_mixed_gpu_and_host_returns_gpu() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![1, 4]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let value = dot_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::Tensor(rhs.clone()),
                Vec::new(),
            )
            .expect("dot");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather result");
                    assert_eq!(gathered.shape, vec![1, 1]);
                    assert_eq!(gathered.data, vec![20.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_exceeds_rank_returns_product() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let value = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Num(3.0)],
        )
        .expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![3.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn dot_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![6.0, 3.0, 5.0, 1.0], vec![2, 2]).unwrap();
        let cpu = dot_real_tensor(&lhs, &rhs, Some(1)).expect("cpu dot");
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
        let gpu_value = dot_builtin(
            Value::GpuTensor(gpu_lhs),
            Value::GpuTensor(gpu_rhs),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("gpu dot");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }

    fn dot_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::dot_builtin(lhs, rhs, rest))
    }
}
