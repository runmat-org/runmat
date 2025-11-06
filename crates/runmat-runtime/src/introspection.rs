#[cfg(test)]
use runmat_builtins::Value;

#[cfg(test)]
#[allow(dead_code)]
fn numel_builtin(a: Value) -> Result<f64, String> {
    Ok(crate::builtins::common::shape::value_numel(&a) as f64)
}

// ---------------------------
// String predicates and ops
// ---------------------------

#[cfg(test)]
#[allow(dead_code)]
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

#[cfg(test)]
#[allow(dead_code)]
fn strcmp_builtin(a: Value, b: Value) -> Result<bool, String> {
    let sa = extract_scalar_string(&a).ok_or("strcmp: expected scalar string")?;
    let sb = extract_scalar_string(&b).ok_or("strcmp: expected scalar string")?;
    Ok(sa == sb)
}

#[cfg(test)]
#[allow(dead_code)]
fn strcmpi_builtin(a: Value, b: Value) -> Result<bool, String> {
    let sa = extract_scalar_string(&a).ok_or("strcmpi: expected scalar string")?;
    let sb = extract_scalar_string(&b).ok_or("strcmpi: expected scalar string")?;
    Ok(sa.to_lowercase() == sb.to_lowercase())
}
