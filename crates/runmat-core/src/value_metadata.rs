use runmat_builtins::{LogicalArray, NumericDType, Value};

/// MATLAB-style class name for a runtime value.
pub fn matlab_class_name(value: &Value) -> String {
    match value {
        Value::Num(_) | Value::Tensor(_) | Value::ComplexTensor(_) | Value::Complex(_, _) => {
            "double".to_string()
        }
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::String(_) | Value::StringArray(_) => "string".to_string(),
        Value::CharArray(_) => "char".to_string(),
        Value::Cell(_) => "cell".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
        Value::FunctionHandle(_) | Value::Closure(_) => "function_handle".to_string(),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                "handle".to_string()
            } else {
                handle.class_name.clone()
            }
        }
        Value::Listener(_) => "event.listener".to_string(),
        Value::Object(obj) => obj.class_name.clone(),
        Value::ClassRef(_) => "meta.class".to_string(),
        Value::MException(_) => "MException".to_string(),
    }
}

/// Returns the MATLAB-style shape for the provided value when applicable.
pub fn value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(vec![1, 1]),
        Value::LogicalArray(arr) => Some(arr.shape.clone()),
        Value::StringArray(sa) => Some(sa.shape.clone()),
        Value::String(s) => Some(vec![1, s.chars().count()]),
        Value::CharArray(ca) => Some(vec![ca.rows, ca.cols]),
        Value::Tensor(t) => Some(t.shape.clone()),
        Value::ComplexTensor(t) => Some(t.shape.clone()),
        Value::Cell(ca) => Some(ca.shape.clone()),
        Value::GpuTensor(handle) => Some(handle.shape.clone()),
        _ => None,
    }
}

/// Returns a MATLAB dtype label for numeric values when available.
pub fn numeric_dtype_label(value: &Value) -> Option<&'static str> {
    match value {
        Value::Num(_) | Value::Complex(_, _) => Some("double"),
        Value::Tensor(t) => Some(match t.dtype {
            NumericDType::F32 => "single",
            NumericDType::F64 => "double",
        }),
        Value::LogicalArray(_) => Some("logical"),
        Value::Int(iv) => Some(iv.class_name()),
        _ => None,
    }
}

/// Rough estimate of the in-memory footprint for the provided value, in bytes.
pub fn approximate_size_bytes(value: &Value) -> Option<u64> {
    Some(match value {
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) => 8,
        Value::Bool(_) => 1,
        Value::LogicalArray(arr) => arr.data.len() as u64,
        Value::Tensor(t) => (t.data.len() * 8) as u64,
        Value::ComplexTensor(t) => (t.data.len() * 16) as u64,
        Value::String(s) => s.len() as u64,
        Value::StringArray(sa) => sa.data.iter().map(|s| s.len() as u64).sum(),
        Value::CharArray(ca) => (ca.rows * ca.cols) as u64,
        _ => return None,
    })
}

/// Produce a numeric preview (up to `limit` elements) for scalars and dense arrays.
pub fn preview_numeric_values(value: &Value, limit: usize) -> Option<(Vec<f64>, bool)> {
    match value {
        Value::Num(n) => Some((vec![*n], false)),
        Value::Int(iv) => Some((vec![iv.to_f64()], false)),
        Value::Bool(flag) => Some((vec![if *flag { 1.0 } else { 0.0 }], false)),
        Value::Tensor(t) => Some(preview_f64_slice(&t.data, limit)),
        Value::LogicalArray(arr) => Some(preview_logical_slice(arr, limit)),
        Value::StringArray(_) | Value::String(_) | Value::CharArray(_) => None,
        Value::ComplexTensor(_) | Value::Complex(_, _) => None,
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::GpuTensor(_) => None,
    }
}

fn preview_f64_slice(data: &[f64], limit: usize) -> (Vec<f64>, bool) {
    if data.len() > limit {
        (data[..limit].to_vec(), true)
    } else {
        (data.to_vec(), false)
    }
}

fn preview_logical_slice(arr: &LogicalArray, limit: usize) -> (Vec<f64>, bool) {
    let truncated = arr.data.len() > limit;
    let mut preview = Vec::with_capacity(arr.data.len().min(limit));
    for value in arr.data.iter().take(limit) {
        preview.push(if *value == 0 { 0.0 } else { 1.0 });
    }
    (preview, truncated)
}
