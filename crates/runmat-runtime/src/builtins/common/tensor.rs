use std::convert::TryFrom;

use runmat_builtins::{LogicalArray, Tensor, Value};

/// Return the total number of elements for a given shape.
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

/// Construct a zero-filled tensor with the provided shape.
pub fn zeros(shape: &[usize]) -> Result<Tensor, String> {
    Tensor::new(vec![0.0; element_count(shape)], shape.to_vec())
        .map_err(|e| format!("tensor zeros: {e}"))
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
            if (rounded - n).abs() > f64::EPSILON {
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
