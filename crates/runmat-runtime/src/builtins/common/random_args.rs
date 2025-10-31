use runmat_builtins::{ComplexTensor, Tensor, Value};

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
pub(crate) fn extract_dims(value: &Value, label: &str) -> Result<Option<Vec<usize>>, String> {
    match value {
        Value::Int(i) => {
            let dim = i.to_i64();
            if dim < 0 {
                return Err(format!("{label}: matrix dimensions must be non-negative"));
            }
            Ok(Some(vec![dim as usize]))
        }
        Value::Num(n) => parse_numeric_dimension(label, *n).map(|d| Some(vec![d])),
        Value::Tensor(t) => dims_from_tensor(label, t),
        Value::LogicalArray(_) => Ok(None),
        _ => Ok(None),
    }
}

/// Parse a numeric dimension, ensuring it aligns with MATLAB semantics.
pub(crate) fn parse_numeric_dimension(label: &str, n: f64) -> Result<usize, String> {
    if !n.is_finite() {
        return Err(format!("{label}: dimensions must be finite"));
    }
    if n < 0.0 {
        return Err(format!("{label}: matrix dimensions must be non-negative"));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(format!("{label}: dimensions must be integers"));
    }
    Ok(rounded as usize)
}

/// Parse dimensions from a tensor representing a size vector.
pub(crate) fn dims_from_tensor(label: &str, tensor: &Tensor) -> Result<Option<Vec<usize>>, String> {
    let is_row = tensor.rows() == 1;
    let is_column = tensor.cols() == 1;
    let is_scalar = tensor.data.len() == 1;
    if !(is_row || is_column || is_scalar || tensor.shape.len() == 1) {
        return Ok(None);
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &v in &tensor.data {
        match parse_numeric_dimension(label, v) {
            Ok(dim) => dims.push(dim),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(dims))
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
