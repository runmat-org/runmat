//! MATLAB-compatible `figure` builtin for selecting/creating plotting windows.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::state::{new_figure_handle, select_figure, FigureHandle};
use crate::build_runtime_error;

fn parse_handle(value: &Value) -> crate::BuiltinResult<Option<FigureHandle>> {
    match value {
        Value::Num(v) => {
            if !v.is_finite() {
                return Err(plotting_error("figure", "figure: handle must be finite"));
            }
            let id = (*v).round() as i64;
            if id <= 0 {
                return Err(plotting_error("figure", "figure: handle must be positive"));
            }
            Ok(Some(FigureHandle::from(id as u32)))
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(plotting_error(
                    "figure",
                    "figure: handle tensor must contain a single element",
                ));
            }
            let val = tensor.data[0];
            if !val.is_finite() {
                return Err(plotting_error("figure", "figure: handle must be finite"));
            }
            let id = val.round() as i64;
            if id <= 0 {
                return Err(plotting_error("figure", "figure: handle must be positive"));
            }
            Ok(Some(FigureHandle::from(id as u32)))
        }
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            parse_string_handle(text.trim())
        }
        Value::String(s) => parse_string_handle(s.trim()),
        _ => Err(plotting_error("figure", "figure: unsupported handle type")),
    }
}

fn parse_string_handle(text: &str) -> crate::BuiltinResult<Option<FigureHandle>> {
    if text.eq_ignore_ascii_case("next") {
        Ok(None)
    } else {
        let id = text.parse::<u32>().map_err(|err| {
            build_runtime_error("figure: invalid handle string")
                .with_builtin("figure")
                .with_source(err)
                .build()
        })?;
        Ok(Some(FigureHandle::from(id)))
    }
}

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create or select a plotting figure.",
    keywords = "figure,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
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
