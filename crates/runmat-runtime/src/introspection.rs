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

#[runtime_builtin(name = "class")]
fn class_builtin(a: Value) -> Result<String, String> {
    let s = match &a {
        Value::Num(_) | Value::Tensor(_) => "double",
        Value::ComplexTensor(_) => "double",
        Value::Complex(_, _) => "double", // The MATLAB language reports 'double' though value is complex
        Value::Int(iv) => iv.class_name(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical",
        Value::String(_) | Value::StringArray(_) => "string",
        Value::CharArray(_) => "char",
        Value::Cell(_) => "cell",
        Value::Struct(_) => "struct",
        Value::GpuTensor(_) => "gpuArray",
        Value::FunctionHandle(_) | Value::Closure(_) => "function_handle",
        Value::HandleObject(_) => "handle",
        Value::Listener(_) => "listener",
        Value::Object(o) => o.class_name.as_str(),
        Value::ClassRef(_) => "meta.class",
        Value::MException(_) => "MException",
    };
    Ok(s.to_string())
}

#[runtime_builtin(name = "isa")]
fn isa_builtin(a: Value, type_name: String) -> Result<bool, String> {
    let t = type_name.to_lowercase();
    let is = match &a {
        Value::Num(_) | Value::Tensor(_) => t == "double" || t == "numeric",
        Value::ComplexTensor(_) => t == "double" || t == "numeric",
        Value::Complex(_, _) => t == "double" || t == "numeric",
        Value::Int(iv) => t == iv.class_name() || t == "numeric",
        Value::Bool(_) | Value::LogicalArray(_) => t == "logical",
        Value::String(_) | Value::StringArray(_) => t == "string",
        Value::CharArray(_) => t == "char",
        Value::Cell(_) => t == "cell",
        Value::Struct(_) => t == "struct",
        Value::GpuTensor(_) => t == "gpuarray",
        Value::FunctionHandle(_) | Value::Closure(_) => t == "function_handle",
        Value::HandleObject(_) => t == "handle",
        Value::Listener(_) => t == "listener",
        Value::Object(o) => t == o.class_name.to_lowercase(),
        Value::ClassRef(_) => t == "meta.class",
        Value::MException(_) => t == "mexception",
    };
    Ok(is)
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
