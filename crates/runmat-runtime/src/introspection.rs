use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

fn tensor_from_dims(dims: Vec<usize>) -> Result<Value, String> {
    // Return as a 1xN row vector (double)
    let len = dims.len();
    let data: Vec<f64> = dims.into_iter().map(|d| d as f64).collect();
    Ok(Value::Tensor(Tensor::new_2d(data, 1, len)?))
}

fn dims_of_value(v: &Value) -> Vec<usize> {
    match v {
        Value::Tensor(t) => {
            if t.shape.is_empty() { vec![1, 1] } else if t.shape.len() == 1 { vec![1, t.shape[0]] } else { t.shape.clone() }
        }
        Value::LogicalArray(la) => {
            if la.shape.is_empty() { vec![1,1] } else if la.shape.len()==1 { vec![1, la.shape[0]] } else { la.shape.clone() }
        }
        Value::StringArray(sa) => {
            if sa.shape.is_empty() { vec![1, 1] } else if sa.shape.len() == 1 { vec![1, sa.shape[0]] } else { sa.shape.clone() }
        }
        Value::CharArray(ca) => vec![ca.rows, ca.cols],
        // Scalars and other values treated as 1x1
        _ => vec![1, 1],
    }
}

fn numel_of_value(v: &Value) -> usize {
    match v {
        Value::Tensor(t) => t.data.len(),
        Value::LogicalArray(la) => la.data.len(),
        Value::StringArray(sa) => sa.data.len(),
        Value::CharArray(ca) => ca.rows * ca.cols,
        Value::Cell(ca) => ca.data.len(),
        // Scalars and objects
        _ => 1,
    }
}

fn ndims_of_value(v: &Value) -> usize {
    let dims = dims_of_value(v);
    // MATLAB returns at least 2 for arrays/scalars
    if dims.len() < 2 { 2 } else { dims.len() }
}

#[runtime_builtin(name = "size")]
fn size_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        let dims = dims_of_value(&a);
        return tensor_from_dims(dims);
    }
    let mut dims = dims_of_value(&a);
    let dim: f64 = (&rest[0]).try_into()?;
    let d = if dim < 1.0 { 1usize } else { dim as usize };
    if dims.len() < 2 { dims.resize(2, 1); }
    let out = if d == 0 { 1.0 } else if d <= dims.len() { dims[d - 1] as f64 } else { 1.0 };
    Ok(Value::Num(out))
}

#[runtime_builtin(name = "numel")]
fn numel_builtin(a: Value) -> Result<f64, String> {
    Ok(numel_of_value(&a) as f64)
}

#[runtime_builtin(name = "ndims")]
fn ndims_builtin(a: Value) -> Result<f64, String> {
    Ok(ndims_of_value(&a) as f64)
}

#[runtime_builtin(name = "isempty")]
fn isempty_builtin(a: Value) -> Result<bool, String> {
    Ok(match a {
        Value::Tensor(t) => t.data.is_empty() || t.rows == 0 || t.cols == 0,
        Value::LogicalArray(la) => la.data.is_empty() || la.shape.iter().any(|&d| d==0),
        Value::StringArray(sa) => sa.data.is_empty() || sa.rows == 0 || sa.cols == 0,
        Value::CharArray(ca) => ca.rows == 0 || ca.cols == 0,
        Value::Cell(ca) => ca.data.is_empty() || ca.rows == 0 || ca.cols == 0,
        _ => false,
    })
}

#[runtime_builtin(name = "islogical")]
fn islogical_builtin(a: Value) -> Result<bool, String> {
    Ok(matches!(a, Value::Bool(_) | Value::LogicalArray(_)))
}

#[runtime_builtin(name = "isnumeric")]
fn isnumeric_builtin(a: Value) -> Result<bool, String> {
    Ok(matches!(a, Value::Num(_) | Value::Complex(_,_) | Value::Int(_) | Value::Tensor(_) | Value::ComplexTensor(_)))
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
        Value::Complex(_,_) => "double", // MATLAB class reports 'double' though value is complex
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
        Value::Complex(_,_) => t == "double" || t == "numeric",
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
// Numeric predicates
// ---------------------------

fn map_tensor_to_mask<F: Fn(f64) -> bool>(t: &Tensor, pred: F) -> Result<Value, String> {
    let data: Vec<u8> = t.data.iter().map(|&x| if pred(x) { 1u8 } else { 0u8 }).collect();
    let out = runmat_builtins::LogicalArray::new(data, t.shape.clone())?;
    Ok(Value::LogicalArray(out))
}

#[runtime_builtin(name = "isnan")]
fn isnan_builtin(a: Value) -> Result<Value, String> {
    Ok(match a {
        Value::Num(x) => Value::Bool(x.is_nan()),
        Value::Int(_) => Value::Bool(false),
        Value::Tensor(t) => {
            return map_tensor_to_mask(&t, |x| x.is_nan())
        }
        Value::LogicalArray(la) => Value::LogicalArray(la),
        _ => Value::Bool(false),
    })
}

#[runtime_builtin(name = "isfinite")]
fn isfinite_builtin(a: Value) -> Result<Value, String> {
    Ok(match a {
        Value::Num(x) => Value::Bool(x.is_finite()),
        Value::Int(_) => Value::Bool(true),
        Value::Tensor(t) => {
            return map_tensor_to_mask(&t, |x| x.is_finite())
        }
        Value::LogicalArray(la) => Value::LogicalArray(la),
        _ => Value::Bool(false),
    })
}

#[runtime_builtin(name = "isinf")]
fn isinf_builtin(a: Value) -> Result<Value, String> {
    Ok(match a {
        Value::Num(x) => Value::Bool(x.is_infinite()),
        Value::Int(_) => Value::Bool(false),
        Value::Tensor(t) => {
            return map_tensor_to_mask(&t, |x| x.is_infinite())
        }
        Value::LogicalArray(la) => Value::LogicalArray(la),
        _ => Value::Bool(false),
    })
}

// ---------------------------
// String predicates and ops
// ---------------------------

fn extract_scalar_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) => Some(ca.data.iter().collect()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 { Some(sa.data[0].clone()) } else { None }
        }
        _ => None,
    }
}

#[runtime_builtin(name = "strcmp")]
fn strcmp_builtin(a: Value, b: Value) -> Result<bool, String> {
    let sa = extract_scalar_string(&a).ok_or_else(|| "strcmp: expected string/char scalar inputs".to_string())?;
    let sb = extract_scalar_string(&b).ok_or_else(|| "strcmp: expected string/char scalar inputs".to_string())?;
    Ok(sa == sb)
}

#[runtime_builtin(name = "strcmpi")]
fn strcmpi_builtin(a: Value, b: Value) -> Result<bool, String> {
    let sa = extract_scalar_string(&a).ok_or_else(|| "strcmpi: expected string/char scalar inputs".to_string())?;
    let sb = extract_scalar_string(&b).ok_or_else(|| "strcmpi: expected string/char scalar inputs".to_string())?;
    Ok(sa.eq_ignore_ascii_case(&sb))
}

fn set_from_strings(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::StringArray(sa) => Ok(sa.data.clone()),
        Value::Cell(ca) => {
            let mut out = Vec::new();
            for p in &ca.data {
                let elem = &**p;
                if let Some(s) = extract_scalar_string(elem) { out.push(s); }
            }
            Ok(out)
        }
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) => Ok(vec![ca.data.iter().collect()]),
        _ => Err("ismember: expected set to be string array or cell of strings".to_string()),
    }
}

#[runtime_builtin(name = "ismember")]
fn ismember_strings(a: Value, set: Value) -> Result<Value, String> {
    let set_vec = set_from_strings(&set)?;
    match a {
        Value::String(s) => Ok(Value::Bool(set_vec.iter().any(|x| x == &s))),
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::Bool(set_vec.iter().any(|x| x == &s)))
        }
        Value::StringArray(sa) => {
            let mask: Vec<f64> = sa.data.iter().map(|s| if set_vec.iter().any(|x| x == s) { 1.0 } else { 0.0 }).collect();
            // Return a row vector 1xN
            let out = Tensor::new_2d(mask, 1, sa.data.len())?;
            Ok(Value::Tensor(out))
        }
        _ => Err("ismember: expected string or string array as first argument".to_string()),
    }
}

// ---------------------------
// Struct utilities
// ---------------------------

#[runtime_builtin(name = "fieldnames")]
fn fieldnames_builtin(s: Value) -> Result<Value, String> {
    match s {
        Value::Struct(st) => {
            let mut names: Vec<Value> = Vec::with_capacity(st.fields.len());
            for k in st.fields.keys() {
                names.push(Value::String(k.clone()));
            }
            let cell = runmat_builtins::CellArray::new(names.clone(), 1, names.len())
                .map_err(|e| format!("fieldnames: {e}"))?;
            Ok(Value::Cell(cell))
        }
        _ => Err("fieldnames: input must be a struct".to_string()),
    }
}

#[runtime_builtin(name = "isfield")]
fn isfield_builtin(s: Value, name: Value) -> Result<bool, String> {
    let key = extract_scalar_string(&name).ok_or_else(|| "isfield: field name must be string/char".to_string())?;
    match s {
        Value::Struct(st) => Ok(st.fields.contains_key(&key)),
        _ => Err("isfield: first argument must be a struct".to_string()),
    }
}


