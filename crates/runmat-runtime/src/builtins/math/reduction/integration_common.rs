use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone)]
pub(crate) enum SpacingSpec {
    Unit,
    Scalar(f64),
    Vector(Vec<f64>),
    Tensor(Tensor),
}

pub(crate) fn integration_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn canonical_shape_tensor(tensor: &Tensor) -> Vec<usize> {
    if tensor.shape.is_empty() {
        vec![tensor.rows, tensor.cols]
    } else {
        tensor.shape.clone()
    }
}

pub(crate) fn canonical_shape_complex(tensor: &ComplexTensor) -> Vec<usize> {
    if tensor.shape.is_empty() {
        vec![tensor.rows, tensor.cols]
    } else {
        tensor.shape.clone()
    }
}

pub(crate) fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

pub(crate) fn pad_shape_for_dim(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut padded = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    while padded.len() < dim {
        padded.push(1);
    }
    padded
}

pub(crate) fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}

pub(crate) fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(la) => la.data.is_empty(),
        _ => false,
    }
}

pub(crate) fn is_scalar_like(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(la) => la.data.len() == 1,
        Value::ComplexTensor(t) => t.data.len() == 1,
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    }
}

pub(crate) fn is_dimension_candidate(value: &Value) -> bool {
    is_empty_value(value)
        || matches!(value, Value::Int(_) | Value::Num(_))
        || matches!(value, Value::Tensor(t) if t.data.len() == 1)
}

pub(crate) fn parse_optional_dim(name: &str, value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_value(value) {
        return Ok(None);
    }
    match value {
        Value::Int(_) | Value::Num(_) => tensor::parse_dimension(value, name)
            .map(Some)
            .map_err(|err| integration_error(name, err)),
        Value::Tensor(t) if t.data.len() == 1 => {
            tensor::parse_dimension(&Value::Num(t.data[0]), name)
                .map(Some)
                .map_err(|err| integration_error(name, err))
        }
        other => Err(integration_error(
            name,
            format!("{name}: dimension must be a positive integer scalar, got {other:?}"),
        )),
    }
}

pub(crate) fn value_has_gpu_tensor(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

pub(crate) async fn gather_host_value(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        gpu_helpers::gather_value_async(&value).await
    } else {
        Ok(value)
    }
}

pub(crate) fn spacing_from_value(
    name: &str,
    value: Option<Value>,
    y_shape: &[usize],
    dim: usize,
) -> BuiltinResult<SpacingSpec> {
    let Some(value) = value else {
        return Ok(SpacingSpec::Unit);
    };
    if is_empty_value(&value) {
        return Ok(SpacingSpec::Unit);
    }

    if let Some(scalar) = scalar_f64_from_host_value(&value)? {
        return Ok(SpacingSpec::Scalar(scalar));
    }

    let mut tensor_value =
        tensor::value_into_tensor_for(name, value).map_err(|err| integration_error(name, err))?;
    let tensor_shape = canonical_shape_tensor(&tensor_value);
    let padded_y_shape = pad_shape_for_dim(y_shape, dim);

    if shapes_equal_with_trailing_ones(&tensor_shape, &padded_y_shape) {
        tensor_value.shape = padded_y_shape;
        return Ok(SpacingSpec::Tensor(tensor_value));
    }

    if is_vector_shape(&tensor_shape) {
        let expected = padded_y_shape[dim - 1];
        if tensor_value.data.len() != expected {
            return Err(integration_error(
                name,
                format!(
                    "{name}: X must be a scalar, a vector with {} elements, or the same size as Y",
                    expected
                ),
            ));
        }
        return Ok(SpacingSpec::Vector(tensor_value.data));
    }

    Err(integration_error(
        name,
        format!("{name}: X must be a scalar, vector, or the same size as Y"),
    ))
}

pub(crate) fn interval_width(spacing: &SpacingSpec, idx0: usize, idx1: usize, k: usize) -> f64 {
    match spacing {
        SpacingSpec::Unit => 1.0,
        SpacingSpec::Scalar(step) => *step,
        SpacingSpec::Vector(values) => values[k + 1] - values[k],
        SpacingSpec::Tensor(tensor) => tensor.data[idx1] - tensor.data[idx0],
    }
}

pub(crate) fn value_into_complex_tensor(name: &str, value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|err| integration_error(name, format!("{name}: {err}"))),
        other => {
            let tensor = tensor::value_into_tensor_for(name, other)
                .map_err(|err| integration_error(name, err))?;
            real_tensor_to_complex(name, &tensor)
        }
    }
}

pub(crate) fn promote_real_value_to_gpu(name: &str, value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(value);
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
                .map_err(|err| integration_error(name, format!("{name}: {err}")))?;
            promote_real_value_to_gpu(name, Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|err| integration_error(name, err))?;
            promote_real_value_to_gpu(name, Value::Tensor(tensor))
        }
        other => Ok(other),
    }
}

fn scalar_f64_from_host_value(value: &Value) -> Result<Option<f64>, RuntimeError> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok(Some(t.data[0]))
            } else {
                Ok(None)
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == 1 {
                Ok(Some(if la.data[0] != 0 { 1.0 } else { 0.0 }))
            } else {
                Ok(None)
            }
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(build_runtime_error("integration spacing must be real-valued").build())
        }
        _ => Ok(None),
    }
}

fn real_tensor_to_complex(name: &str, tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let shape = canonical_shape_tensor(tensor);
    let data = tensor
        .data
        .iter()
        .copied()
        .map(|value| (value, 0.0))
        .collect();
    ComplexTensor::new(data, shape).map_err(|err| integration_error(name, format!("{name}: {err}")))
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|&&extent| extent > 1).count() <= 1
}

fn shapes_equal_with_trailing_ones(lhs: &[usize], rhs: &[usize]) -> bool {
    let len = lhs.len().max(rhs.len());
    (0..len).all(|idx| lhs.get(idx).copied().unwrap_or(1) == rhs.get(idx).copied().unwrap_or(1))
}
