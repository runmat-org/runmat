use runmat_builtins::Value;

use super::state::FigureHandle;
use super::style::value_as_string;
use super::plotting_error;

use crate::{build_runtime_error, BuiltinResult};

pub(super) fn handles_from_value(value: &Value, ctx: &str) -> BuiltinResult<Vec<FigureHandle>> {
    match value {
        Value::Num(v) => Ok(vec![handle_from_scalar(*v, ctx)?]),
        Value::Int(i) => Ok(vec![handle_from_scalar(i.to_f64(), ctx)?]),
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                return Err(plotting_error(ctx, format!("{ctx}: handle array cannot be empty")));
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
                return Err(plotting_error(ctx, format!("{ctx}: handle string cannot be empty")));
            }
            Ok(vec![handle_from_string(&text, ctx)?])
        }
        _ => Err(plotting_error(ctx, format!("{ctx}: unsupported argument type"))),
    }
}

pub(super) fn parse_string(value: &Value) -> Option<String> {
    value_as_string(value).map(|s| s.trim().to_string())
}

pub(super) fn handle_from_string(text: &str, ctx: &str) -> BuiltinResult<FigureHandle> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: handle text cannot be empty"),
        ));
    }
    let id = trimmed.parse::<u32>().map_err(|err| {
        build_runtime_error(format!("{ctx}: expected numeric figure handle text, got `{text}`"))
            .with_builtin(ctx)
            .with_source(err)
            .build()
            .into()
    })?;
    if id == 0 {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle must be positive"),
        ));
    }
    Ok(FigureHandle::from(id))
}

pub(super) fn handle_from_scalar(value: f64, ctx: &str) -> BuiltinResult<FigureHandle> {
    if !value.is_finite() {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle must be finite"),
        ));
    }
    let rounded = value.round() as i64;
    if rounded <= 0 {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle must be positive"),
        ));
    }
    Ok(FigureHandle::from(rounded as u32))
}
