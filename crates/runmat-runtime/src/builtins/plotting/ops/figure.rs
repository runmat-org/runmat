//! MATLAB-compatible `figure` builtin for selecting/creating plotting windows.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::state::{new_figure_handle, select_figure, FigureHandle};

fn parse_handle(value: &Value) -> Result<Option<FigureHandle>, String> {
    match value {
        Value::Num(v) => {
            if !v.is_finite() {
                return Err("figure: handle must be finite".to_string());
            }
            let id = (*v).round() as i64;
            if id <= 0 {
                return Err("figure: handle must be positive".to_string());
            }
            Ok(Some(FigureHandle::from(id as u32)))
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err("figure: handle tensor must contain a single element".to_string());
            }
            let val = tensor.data[0];
            if !val.is_finite() {
                return Err("figure: handle must be finite".to_string());
            }
            let id = val.round() as i64;
            if id <= 0 {
                return Err("figure: handle must be positive".to_string());
            }
            Ok(Some(FigureHandle::from(id as u32)))
        }
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            parse_string_handle(text.trim())
        }
        Value::String(s) => parse_string_handle(s.trim()),
        _ => Err("figure: unsupported handle type".to_string()),
    }
}

fn parse_string_handle(text: &str) -> Result<Option<FigureHandle>, String> {
    if text.eq_ignore_ascii_case("next") {
        Ok(None)
    } else {
        text.parse::<u32>()
            .map(FigureHandle::from)
            .map(Some)
            .map_err(|_| "figure: invalid handle string".to_string())
    }
}

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create or select a plotting figure.",
    keywords = "figure,plotting",
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(rest: Vec<Value>) -> Result<f64, String> {
    let handle = if rest.is_empty() {
        new_figure_handle()
    } else {
        match parse_handle(&rest[0])? {
            Some(handle) => {
                select_figure(handle);
                handle
            }
            None => new_figure_handle(),
        }
    };
    Ok(handle.as_u32() as f64)
}
