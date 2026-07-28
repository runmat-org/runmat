use runmat_builtins::Value;

use crate::builtins::common::tensor;
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::state::FigureHandle;
use crate::builtins::plotting::style::value_as_string;
use crate::{build_runtime_error, BuiltinResult};

pub fn handles_from_value(value: &Value, ctx: &str) -> BuiltinResult<Vec<FigureHandle>> {
    match value {
        Value::Num(v) => Ok(vec![handle_from_scalar(*v, ctx)?]),
        Value::Int(i) => Ok(vec![handle_from_scalar(i.to_f64(), ctx)?]),
        Value::Tensor(tensor) => {
            let data = tensor::tensor_values_f64(tensor);
            if data.is_empty() {
                return Err(plotting_error(
                    ctx,
                    format!("{ctx}: handle array cannot be empty"),
                ));
            }
            data.iter().map(|v| handle_from_scalar(*v, ctx)).collect()
        }
        Value::CharArray(_) | Value::String(_) => {
            let text = parse_string(value).unwrap_or_default();
            if text.is_empty() {
                return Err(plotting_error(
                    ctx,
                    format!("{ctx}: handle string cannot be empty"),
                ));
            }
            Ok(vec![handle_from_string(&text, ctx)?])
        }
        _ => Err(plotting_error(
            ctx,
            format!("{ctx}: unsupported argument type"),
        )),
    }
}

pub fn parse_string(value: &Value) -> Option<String> {
    value_as_string(value).map(|s| s.trim().to_string())
}

pub fn handle_from_string(text: &str, ctx: &str) -> BuiltinResult<FigureHandle> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: handle text cannot be empty"),
        ));
    }
    let id = trimmed.parse::<u32>().map_err(|err| {
        build_runtime_error(format!(
            "{ctx}: expected numeric figure handle text, got `{text}`"
        ))
        .with_builtin(ctx)
        .with_source(err)
        .build()
    })?;
    if id == 0 {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle must be positive"),
        ));
    }
    Ok(FigureHandle::from(id))
}

pub fn handle_from_scalar(value: f64, ctx: &str) -> BuiltinResult<FigureHandle> {
    if !value.is_finite() {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle must be finite"),
        ));
    }
    let rounded = value.round();
    if rounded <= 0.0 {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle must be positive"),
        ));
    }
    if (rounded - value).abs() > f64::EPSILON || rounded > u32::MAX as f64 {
        return Err(plotting_error(
            ctx,
            format!("{ctx}: figure handle is too large"),
        ));
    }
    Ok(FigureHandle::from(rounded as u32))
}

pub fn parse_optional_figure_handle(
    value: &Value,
    ctx: &str,
) -> BuiltinResult<Option<FigureHandle>> {
    match value {
        Value::CharArray(_) | Value::String(_) => {
            let text = parse_string(value).unwrap_or_default();
            parse_string_handle_or_next(&text, ctx)
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let value = tensor::tensor_value_f64(tensor, 0);
            Ok(Some(handle_from_scalar(value, ctx)?))
        }
        Value::Num(v) => Ok(Some(handle_from_scalar(*v, ctx)?)),
        Value::Int(i) => Ok(Some(handle_from_scalar(i.to_f64(), ctx)?)),
        _ => Err(plotting_error(
            ctx,
            format!("{ctx}: unsupported handle type"),
        )),
    }
}

pub fn parse_string_handle_or_next(text: &str, ctx: &str) -> BuiltinResult<Option<FigureHandle>> {
    if text.eq_ignore_ascii_case("next") {
        Ok(None)
    } else {
        Ok(Some(handle_from_string(text, ctx)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn handles_read_typed_integer_storage() {
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U32(vec![1, 2]), vec![1, 2]).expect("handles"),
        );

        let handles = handles_from_value(&value, "figure").expect("handles");

        assert_eq!(handles, vec![FigureHandle::from(1), FigureHandle::from(2)]);
    }

    #[test]
    fn handle_parser_rejects_unrepresentable_float_before_cast() {
        assert!(handle_from_scalar((u32::MAX as f64) + 1.0, "figure").is_err());
        assert!(handle_from_scalar(1.5, "figure").is_err());
    }
}
