use runmat_builtins::Value;

use super::state::FigureHandle;
use super::style::value_as_string;

pub(super) fn handles_from_value(value: &Value, ctx: &str) -> Result<Vec<FigureHandle>, String> {
    match value {
        Value::Num(v) => Ok(vec![handle_from_scalar(*v, ctx)?]),
        Value::Int(i) => Ok(vec![handle_from_scalar(i.to_f64(), ctx)?]),
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                return Err(format!("{ctx}: handle array cannot be empty"));
            }
            let mut handles = Vec::with_capacity(tensor.data.len());
            for val in &tensor.data {
                handles.push(handle_from_scalar(*val, ctx)?);
            }
            Ok(handles)
        }
        Value::CharArray(_) | Value::String(_) => {
            let text = parse_string(value).unwrap_or_default();
            if text.is_empty() {
                return Err(format!("{ctx}: handle string cannot be empty"));
            }
            Ok(vec![handle_from_string(&text, ctx)?])
        }
        _ => Err(format!("{ctx}: unsupported argument type")),
    }
}

pub(super) fn parse_string(value: &Value) -> Option<String> {
    value_as_string(value).map(|s| s.trim().to_string())
}

pub(super) fn handle_from_string(text: &str, ctx: &str) -> Result<FigureHandle, String> {
    if text.trim().is_empty() {
        return Err(format!("{ctx}: handle text cannot be empty"));
    }
    if let Ok(id) = text.trim().parse::<u32>() {
        if id == 0 {
            return Err(format!("{ctx}: figure handle must be positive"));
        }
        return Ok(FigureHandle::from(id));
    }
    Err(format!(
        "{ctx}: expected numeric figure handle text, got `{text}`"
    ))
}

pub(super) fn handle_from_scalar(value: f64, ctx: &str) -> Result<FigureHandle, String> {
    if !value.is_finite() {
        return Err(format!("{ctx}: figure handle must be finite"));
    }
    let rounded = value.round() as i64;
    if rounded <= 0 {
        return Err(format!("{ctx}: figure handle must be positive"));
    }
    Ok(FigureHandle::from(rounded as u32))
}
