use std::convert::TryFrom;

use runmat_builtins::{LogicalArray, NumericDType, Tensor, Value};

use crate::dispatcher::gather_if_needed_async;

/// Return the total number of elements for a given shape.
pub fn element_count(shape: &[usize]) -> usize {
    let mut acc: u128 = 1;
    for &dim in shape {
        let dim128 = dim as u128;
        acc = acc
            .checked_mul(dim128)
            .expect("tensor::element_count: overflow computing element count");
    }
    usize::try_from(acc).expect("tensor::element_count: overflow converting to usize")
}

/// Construct a zero-filled tensor with the provided shape.
pub fn zeros(shape: &[usize]) -> Result<Tensor, String> {
    Tensor::new(vec![0.0; element_count(shape)], shape.to_vec())
        .map_err(|e| format!("tensor zeros: {e}"))
}

/// Construct an one-filled tensor with the provided shape.
pub fn ones(shape: &[usize]) -> Result<Tensor, String> {
    Tensor::new(vec![1.0; element_count(shape)], shape.to_vec())
        .map_err(|e| format!("tensor ones: {e}"))
}

/// Construct a zero-filled tensor with an explicit dtype flag.
pub fn zeros_with_dtype(shape: &[usize], dtype: NumericDType) -> Result<Tensor, String> {
    Tensor::new_with_dtype(vec![0.0; element_count(shape)], shape.to_vec(), dtype)
        .map_err(|e| format!("tensor zeros: {e}"))
}

/// Construct a one-filled tensor with an explicit dtype flag.
pub fn ones_with_dtype(shape: &[usize], dtype: NumericDType) -> Result<Tensor, String> {
    Tensor::new_with_dtype(vec![1.0; element_count(shape)], shape.to_vec(), dtype)
        .map_err(|e| format!("tensor ones: {e}"))
}

/// Convert a logical array (0/1 bytes) into a numeric tensor.
pub fn logical_to_tensor(logical: &LogicalArray) -> Result<Tensor, String> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone()).map_err(|e| format!("logical->tensor: {e}"))
}

fn value_into_tensor_impl(name: &str, value: Value) -> Result<Tensor, String> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(&logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("tensor: {e}")),
        Value::Int(i) => {
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("tensor: {e}"))
        }
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| format!("tensor: {e}")),
        other => Err(format!(
            "{name}: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

/// Convert a `Value` into an owned `Tensor`, defaulting error messages to `"sum"`.
pub fn value_into_tensor(value: Value) -> Result<Tensor, String> {
    value_into_tensor_impl("sum", value)
}

/// Convert a `Value` into a tensor while customising the builtin name in error messages.
pub fn value_into_tensor_for(name: &str, value: Value) -> Result<Tensor, String> {
    value_into_tensor_impl(name, value)
}

/// Clone a `Value` and coerce it into a tensor.
pub fn value_to_tensor(value: &Value) -> Result<Tensor, String> {
    value_into_tensor(value.clone())
}

/// Convert a `Tensor` back into a runtime value.
///
/// Scalars (exactly one element) become `Value::Num`, all other tensors
/// remain as dense tensor variants.
pub fn tensor_into_value(tensor: Tensor) -> Value {
    if tensor.data.len() == 1 {
        Value::Num(tensor.data[0])
    } else {
        Value::Tensor(tensor)
    }
}

/// Return true when a tensor contains exactly one scalar element.
pub fn is_scalar_tensor(tensor: &Tensor) -> bool {
    tensor.data.len() == 1
}

fn scalar_f64_from_host_value(value: &Value) -> Result<Option<f64>, String> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok(Some(t.data[0]))
            } else {
                Err(format!(
                    "expected scalar tensor, got tensor of size {}",
                    t.data.len()
                ))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == 1 {
                Ok(Some(if la.data[0] != 0 { 1.0 } else { 0.0 }))
            } else {
                Err(format!(
                    "expected scalar logical array, got array of size {}",
                    la.data.len()
                ))
            }
        }
        _ => Ok(None),
    }
}

/// Attempt to extract a scalar f64 from a runtime value asynchronously.
pub async fn scalar_f64_from_value_async(value: &Value) -> Result<Option<f64>, String> {
    match value {
        Value::GpuTensor(handle) => {
            if !handle.shape.is_empty() {
                let len = element_count(&handle.shape);
                if len != 1 {
                    return Err(format!(
                        "expected scalar gpuArray, got array of size {len}"
                    ));
                }
            }
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|e| format!("scalar: {e}"))?;
            scalar_f64_from_host_value(&gathered)
        }
        _ => scalar_f64_from_host_value(value),
    }
}

/// Attempt to parse a dimension index from a scalar-like runtime value.
pub async fn dimension_from_value_async(
    value: &Value,
    name: &str,
    allow_zero: bool,
) -> Result<Option<usize>, String> {
    let Some(raw) = scalar_f64_from_value_async(value).await? else {
        return Ok(None);
    };
    if !raw.is_finite() {
        return Err(format!("{name}: dimension must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-6 {
        return Err(format!("{name}: dimension must be an integer"));
    }
    let min = if allow_zero { 0.0 } else { 1.0 };
    if rounded < min {
        let bound = if allow_zero { 0 } else { 1 };
        return Err(format!("{name}: dimension must be >= {bound}"));
    }
    Ok(Some(rounded as usize))
}

fn parse_numeric_dimension(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("dimensions must be finite".to_string());
    }
    if value < 0.0 {
        return Err("matrix dimensions must be non-negative".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("dimensions must be integers".to_string());
    }
    Ok(rounded as usize)
}

fn dims_from_tensor_values(values: &[f64], shape: &[usize]) -> Result<Option<Vec<usize>>, String> {
    let len = values.len();
    let is_scalar = len == 1;
    let is_row = shape.len() >= 2 && shape[0] == 1;
    let is_column = shape.len() >= 2 && shape[1] == 1;
    if !(is_row || is_column || is_scalar || shape.len() == 1) {
        return Ok(None);
    }
    let mut dims = Vec::with_capacity(len);
    for &value in values {
        dims.push(parse_numeric_dimension(value)?);
    }
    Ok(Some(dims))
}

/// Attempt to parse a dimension vector from a runtime value asynchronously.
pub async fn dims_from_value_async(value: &Value) -> Result<Option<Vec<usize>>, String> {
    match value {
        Value::Num(n) => parse_numeric_dimension(*n).map(|dim| Some(vec![dim])),
        Value::Int(i) => parse_numeric_dimension(i.to_f64()).map(|dim| Some(vec![dim])),
        Value::Tensor(t) => dims_from_tensor_values(&t.data, &t.shape),
        Value::LogicalArray(la) => {
            let values: Vec<f64> = la.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect();
            dims_from_tensor_values(&values, &la.shape)
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|e| format!("dimensions: {e}"))?;
            match gathered {
                Value::Tensor(t) => dims_from_tensor_values(&t.data, &t.shape),
                Value::LogicalArray(la) => {
                    let values: Vec<f64> =
                        la.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect();
                    dims_from_tensor_values(&values, &la.shape)
                }
                Value::Num(n) => parse_numeric_dimension(n).map(|dim| Some(vec![dim])),
                Value::Int(i) => parse_numeric_dimension(i.to_f64()).map(|dim| Some(vec![dim])),
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Convert an argument into a dimension index (1-based) if possible.
pub fn parse_dimension(value: &Value, name: &str) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 1 {
                return Err(format!("{name}: dimension must be >= 1"));
            }
            Ok(raw as usize)
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!("{name}: dimension must be finite"));
            }
            let rounded = n.round();
            // Allow small floating error tolerance when users pass float-typed dims
            if (rounded - n).abs() > 1e-6 {
                return Err(format!("{name}: dimension must be an integer"));
            }
            if rounded < 1.0 {
                return Err(format!("{name}: dimension must be >= 1"));
            }
            Ok(rounded as usize)
        }
        other => Err(format!(
            "{name}: dimension must be numeric, got {:?}",
            other
        )),
    }
}

/// Attempt to extract a string from a runtime value.
pub fn value_to_string(value: &Value) -> Option<String> {
    String::try_from(value).ok()
}
