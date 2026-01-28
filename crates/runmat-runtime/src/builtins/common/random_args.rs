use crate::builtins::common::tensor;
use runmat_builtins::{ComplexTensor, Value};

/// Extract a lowercased keyword from runtime values such as strings or
/// single-row char arrays.
pub(crate) fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

/// Attempt to parse a dimension argument. Returns `Ok(Some(Vec))` when the
/// value encodes dimensions, `Ok(None)` when the value is not a dimension
/// argument, and `Err` when the value is dimension-like but invalid.
pub(crate) async fn extract_dims(value: &Value, label: &str) -> Result<Option<Vec<usize>>, String> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(format!("{label}: {err}"))
            }
        }
    }
}

/// Determine the output shape encoded by a runtime value.
pub(crate) fn shape_from_value(value: &Value, label: &str) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::StringArray(_) => Ok(vec![1, 1]),
        other => Err(format!("{label}: unsupported prototype {other:?}")),
    }
}

/// Convert a complex tensor back into an appropriate runtime value.
pub(crate) fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}
