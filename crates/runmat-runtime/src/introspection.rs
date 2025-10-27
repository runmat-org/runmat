use crate::builtins::common::shape::{value_ndims, value_numel};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[runtime_builtin(name = "numel")]
fn numel_builtin(a: Value) -> Result<f64, String> {
    Ok(value_numel(&a) as f64)
}

#[runtime_builtin(name = "ndims")]
fn ndims_builtin(a: Value) -> Result<f64, String> {
    Ok(value_ndims(&a) as f64)
}

#[runtime_builtin(name = "isempty")]
fn isempty_builtin(a: Value) -> Result<bool, String> {
    Ok(match a {
        Value::Tensor(t) => t.data.is_empty() || t.rows == 0 || t.cols == 0,
        Value::LogicalArray(la) => la.data.is_empty() || la.shape.contains(&0),
        Value::StringArray(sa) => sa.data.is_empty() || sa.rows == 0 || sa.cols == 0,
        Value::CharArray(ca) => ca.rows == 0 || ca.cols == 0,
        Value::Cell(ca) => ca.data.is_empty() || ca.rows == 0 || ca.cols == 0,
        _ => false,
    })
}

#[runtime_builtin(name = "isnumeric")]
fn isnumeric_builtin(a: Value) -> Result<bool, String> {
    Ok(matches!(
        a,
        Value::Num(_)
            | Value::Complex(_, _)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
    ))
}

#[runtime_builtin(name = "ischar")]
fn ischar_builtin(a: Value) -> Result<bool, String> {
    Ok(matches!(a, Value::CharArray(_)))
}

#[runtime_builtin(name = "isstring")]
fn isstring_builtin(a: Value) -> Result<bool, String> {
    Ok(matches!(a, Value::String(_) | Value::StringArray(_)))
}

// ---------------------------
// String predicates and ops
// ---------------------------

fn extract_scalar_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) => Some(ca.data.iter().collect()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Some(sa.data[0].clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

#[runtime_builtin(name = "strcmp")]
fn strcmp_builtin(a: Value, b: Value) -> Result<bool, String> {
    let sa = extract_scalar_string(&a)
        .ok_or_else(|| "strcmp: expected string/char scalar inputs".to_string())?;
    let sb = extract_scalar_string(&b)
        .ok_or_else(|| "strcmp: expected string/char scalar inputs".to_string())?;
    Ok(sa == sb)
}

#[runtime_builtin(name = "strcmpi")]
fn strcmpi_builtin(a: Value, b: Value) -> Result<bool, String> {
    let sa = extract_scalar_string(&a)
        .ok_or_else(|| "strcmpi: expected string/char scalar inputs".to_string())?;
    let sb = extract_scalar_string(&b)
        .ok_or_else(|| "strcmpi: expected string/char scalar inputs".to_string())?;
    Ok(sa.eq_ignore_ascii_case(&sb))
}
