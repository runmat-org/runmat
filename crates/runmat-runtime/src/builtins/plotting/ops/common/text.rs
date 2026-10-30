use runmat_builtins::BuiltinErrorDescriptor;
use runmat_plot::plots::{LegendStyle, TextStyle};
use runmat_value::NumericScalar;
use runmat_value::Value;

use crate::builtins::plotting::properties::{parse_text_style_pairs, split_legend_style_pairs};
use crate::builtins::plotting::state::{
    axes_handle_exists, current_axes_state, decode_axes_handle, FigureError, FigureHandle,
};
use crate::builtins::plotting::style::value_as_string;
use crate::builtins::plotting::{plotting_error, plotting_error_with_source};
use crate::BuiltinResult;

use super::handles::numeric_handle_scalar;

#[derive(Clone, Debug)]
pub struct TextCommand {
    target: Option<(FigureHandle, usize)>,
    pub text: String,
    pub style: TextStyle,
}

impl TextCommand {
    /// The implicit current axes is selected only after text and style
    /// validation has completed without an error.
    pub fn resolve_target(&self) -> (FigureHandle, usize) {
        self.target.unwrap_or_else(current_axes_target)
    }
}

#[derive(Clone, Debug)]
pub struct LegendCommand {
    pub target: (FigureHandle, usize),
    pub enabled: bool,
    pub labels: Option<Vec<String>>,
    pub style: LegendStyle,
}

pub fn value_as_text_lines(
    value: &Value,
) -> Result<Option<Vec<String>>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Some(vec![text.as_str().to_owned()])
        }
        Value::CharArray(chars) => Some(vec![runmat_value::decode_utf16_code_units(chars.data())?]),
        Value::StringArray(arr) => arr
            .data()
            .iter()
            .map(|text| text.as_str().map(str::to_owned))
            .collect(),
        Value::Cell(cell) => {
            let mut lines = Vec::new();
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    let value = &cell.data()[row * cell.cols + col];
                    let Some(line) = value_as_string(value)? else {
                        return Ok(None);
                    };
                    lines.push(line);
                }
            }
            Some(lines)
        }
        _ => None,
    })
}

pub fn value_as_text_string(
    value: &Value,
) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(value_as_text_lines(value)?.map(|lines| lines.join("\n")))
}

pub fn current_axes_target() -> (FigureHandle, usize) {
    let state = current_axes_state();
    (state.handle, state.active_index)
}

pub fn map_figure_error(builtin: &'static str, err: FigureError) -> crate::RuntimeError {
    let message = format!("{builtin}: {err}");
    plotting_error_with_source(builtin, message, err)
}

pub fn annotation_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    source: crate::RuntimeError,
) -> crate::RuntimeError {
    if source.identifier().is_some() {
        return source;
    }
    let mut builder = crate::build_runtime_error(format!("{builtin}: {}", source.message))
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub fn parse_text_command(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<TextCommand> {
    if args.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected text input"),
        ));
    }
    let DeferredAxesArguments { target, rest } = split_axes_target_deferred(builtin, args)?;
    if rest.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected text input"),
        ));
    }

    let text = value_as_text_string(&rest[0]).map_err(|source| crate::builtins::plotting::plotting_error_with_source(builtin,source.to_string(),source))?.ok_or_else(|| {
        plotting_error(
            builtin,
            format!(
                "{builtin}: expected text as char array, string, string array, or cell array of strings"
            ),
        )
    })?;
    let style = parse_text_style_pairs(builtin, rest.range(1..)?)?;
    Ok(TextCommand {
        target,
        text,
        style,
    })
}

pub fn parse_numeric_text_command(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<TextCommand> {
    if args.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected text input"),
        ));
    }
    let numeric_text_form = args.len() % 2 == 1
        && args
            .first()
            .is_some_and(|value| format_numeric_text(value).is_some());
    let DeferredAxesArguments { target, rest } = if args.len() == 1 || numeric_text_form {
        DeferredAxesArguments {
            target: None,
            rest: args,
        }
    } else {
        split_axes_target_deferred(builtin, args)?
    };
    if rest.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected text input"),
        ));
    }
    let text = value_as_text_string(&rest[0]).map_err(|source| crate::builtins::plotting::plotting_error_with_source(builtin,source.to_string(),source))?
        .or_else(|| format_numeric_text(&rest[0]))
        .ok_or_else(|| {
            plotting_error(
                builtin,
                format!(
                    "{builtin}: expected text as char array, string, string array, cell array of strings, or numeric scalar"
                ),
            )
        })?;
    let style = parse_text_style_pairs(builtin, rest.range(1..)?)?;
    Ok(TextCommand {
        target,
        text,
        style,
    })
}

/// Format a numeric annotation scalar without routing native integers through binary64.
pub fn format_numeric_text(value: &Value) -> Option<String> {
    match value {
        Value::Int(value) => Some(value.decimal_string()),
        Value::Num(value) if value.is_finite() => Some(format!("{value}")),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            format_numeric_scalar(tensor.numeric_value_at(0)?)
        }
        _ => None,
    }
}

fn format_numeric_scalar(value: NumericScalar) -> Option<String> {
    match value {
        NumericScalar::F64(value) if value.is_finite() => Some(format!("{value}")),
        NumericScalar::F32(value) if value.is_finite() => Some(format!("{value}")),
        NumericScalar::I8(value) => Some(value.to_string()),
        NumericScalar::I16(value) => Some(value.to_string()),
        NumericScalar::I32(value) => Some(value.to_string()),
        NumericScalar::I64(value) => Some(value.to_string()),
        NumericScalar::U8(value) => Some(value.to_string()),
        NumericScalar::U16(value) => Some(value.to_string()),
        NumericScalar::U32(value) => Some(value.to_string()),
        NumericScalar::U64(value) => Some(value.to_string()),
        _ => None,
    }
}

pub fn parse_legend_command(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<LegendCommand> {
    let (target, rest) = split_axes_target(builtin, args)?;
    let (rest, style) = split_legend_style_pairs(builtin, rest)?;

    if rest.is_empty() {
        return Ok(LegendCommand {
            target,
            enabled: true,
            labels: None,
            style,
        });
    }

    if rest.len() == 1 {
        if let Some(mode) = value_as_string(&rest[0]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                builtin,
                source.to_string(),
                source,
            )
        })? {
            match mode.trim().to_ascii_lowercase().as_str() {
                "on" | "show" => {
                    return Ok(LegendCommand {
                        target,
                        enabled: true,
                        labels: None,
                        style,
                    });
                }
                "off" | "hide" => {
                    return Ok(LegendCommand {
                        target,
                        enabled: false,
                        labels: None,
                        style,
                    });
                }
                "boxon" => {
                    let mut style = style;
                    style.box_visible = Some(true);
                    return Ok(LegendCommand {
                        target,
                        enabled: true,
                        labels: None,
                        style,
                    });
                }
                "boxoff" => {
                    let mut style = style;
                    style.box_visible = Some(false);
                    return Ok(LegendCommand {
                        target,
                        enabled: true,
                        labels: None,
                        style,
                    });
                }
                _ => {}
            }
        }
    }

    let labels = collect_label_strings(builtin, rest)?;
    if labels.is_empty() {
        return Err(plotting_error(builtin, "legend: labels cannot be empty"));
    }
    Ok(LegendCommand {
        target,
        enabled: true,
        labels: Some(labels),
        style,
    })
}

pub(crate) fn split_axes_target<'a>(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<((FigureHandle, usize), crate::call::FunctionArgumentView<'a>)> {
    let DeferredAxesArguments { target, rest } = split_axes_target_deferred(builtin, args)?;
    Ok((target.unwrap_or_else(current_axes_target), rest))
}

/// Resolve an explicit axes handle without creating the implicit current axes.
/// Callers with potentially pending preparation can publish the implicit scene
/// only after their input and cancellation checks have completed.
pub(crate) struct DeferredAxesArguments<'a> {
    pub(crate) target: Option<(FigureHandle, usize)>,
    pub(crate) rest: crate::call::FunctionArgumentView<'a>,
}

pub(crate) fn split_axes_target_deferred<'a>(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<DeferredAxesArguments<'a>> {
    if let Some(first) = args.first() {
        if let Some(target) = try_parse_axes_target(first) {
            if axes_handle_exists(target.0, target.1) {
                return Ok(DeferredAxesArguments {
                    target: Some(target),
                    rest: args.range(1..)?,
                });
            }
            return Err(plotting_error(
                builtin,
                format!("{builtin}: invalid axes handle"),
            ));
        }
    }
    Ok(DeferredAxesArguments {
        target: None,
        rest: args,
    })
}

fn try_parse_axes_target(value: &Value) -> Option<(FigureHandle, usize)> {
    decode_axes_handle(numeric_handle_scalar(value)?).ok()
}

fn collect_label_strings(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<String>> {
    let mut labels = Vec::new();
    for arg in args {
        match arg {
            Value::StringArray(arr) => labels.extend(
                arr.data()
                    .iter()
                    .map(|text| {
                        text.as_str().map(str::to_owned).ok_or_else(|| {
                            plotting_error(builtin, "legend: labels must be strings or char arrays")
                        })
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?,
            ),
            Value::Cell(cell) => {
                for row in 0..cell.rows {
                    for col in 0..cell.cols {
                        let value = &cell.data()[row * cell.cols + col];
                        labels.push(
                            value_as_text_string(value)
                                .map_err(|source| {
                                    crate::builtins::plotting::plotting_error_with_source(
                                        builtin,
                                        source.to_string(),
                                        source,
                                    )
                                })?
                                .ok_or_else(|| {
                                    plotting_error(
                                        builtin,
                                        "legend: labels must be strings or char arrays",
                                    )
                                })?,
                        );
                    }
                }
            }
            _ => labels.push(
                value_as_text_string(arg)
                    .map_err(|source| {
                        crate::builtins::plotting::plotting_error_with_source(
                            builtin,
                            source.to_string(),
                            source,
                        )
                    })?
                    .ok_or_else(|| {
                        plotting_error(builtin, "legend: labels must be strings or char arrays")
                    })?,
            ),
        }
    }
    Ok(labels)
}

#[cfg(test)]
pub fn vec4_eq(a: Option<glam::Vec4>, b: glam::Vec4) -> bool {
    a.map(|v| (v - b).abs().max_element() < 1e-6)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn axes_target_parser_reads_typed_integer_storage_exactly() {
        let encoded =
            crate::builtins::plotting::state::encode_axes_handle(FigureHandle::from(7), 2);
        let tensor =
            Tensor::new_integer(IntegerStorage::U32(vec![encoded as u32]), vec![1, 1]).unwrap();

        assert_eq!(
            try_parse_axes_target(&Value::Tensor(tensor)),
            Some((FigureHandle::from(7), 2))
        );
    }
}
