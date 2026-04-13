use runmat_builtins::Value;
use runmat_plot::plots::{LegendStyle, TextStyle};

use crate::builtins::plotting::properties::{parse_text_style_pairs, split_legend_style_pairs};
use crate::builtins::plotting::state::{
    axes_handle_exists, current_axes_state, decode_axes_handle, FigureError, FigureHandle,
};
use crate::builtins::plotting::style::value_as_string;
use crate::builtins::plotting::{plotting_error, plotting_error_with_source};
use crate::BuiltinResult;

#[derive(Clone, Debug)]
pub struct TextCommand {
    pub target: (FigureHandle, usize),
    pub text: String,
    pub style: TextStyle,
}

#[derive(Clone, Debug)]
pub struct LegendCommand {
    pub target: (FigureHandle, usize),
    pub enabled: bool,
    pub labels: Option<Vec<String>>,
    pub style: LegendStyle,
}

pub fn value_as_text_lines(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::String(text) => Some(vec![text.clone()]),
        Value::CharArray(chars) => Some(vec![chars.data.iter().collect()]),
        Value::StringArray(arr) => Some(arr.data.clone()),
        Value::Cell(cell) => {
            let mut lines = Vec::new();
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    let value = cell.get(row, col).ok()?;
                    lines.push(value_as_string(&value)?);
                }
            }
            Some(lines)
        }
        _ => None,
    }
}

pub fn value_as_text_string(value: &Value) -> Option<String> {
    value_as_text_lines(value).map(|lines| lines.join("\n"))
}

pub fn current_axes_target() -> (FigureHandle, usize) {
    let state = current_axes_state();
    (state.handle, state.active_index)
}

pub fn map_figure_error(builtin: &'static str, err: FigureError) -> crate::RuntimeError {
    let message = format!("{builtin}: {err}");
    plotting_error_with_source(builtin, message, err)
}

pub fn parse_text_command(builtin: &'static str, args: &[Value]) -> BuiltinResult<TextCommand> {
    let (target, rest) = split_axes_target(builtin, args)?;
    if rest.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected text input"),
        ));
    }

    let text = value_as_text_string(&rest[0]).ok_or_else(|| {
        plotting_error(
            builtin,
            format!(
                "{builtin}: expected text as char array, string, string array, or cell array of strings"
            ),
        )
    })?;
    let style = parse_text_style_pairs(builtin, &rest[1..])?;
    Ok(TextCommand {
        target,
        text,
        style,
    })
}

pub fn parse_legend_command(builtin: &'static str, args: &[Value]) -> BuiltinResult<LegendCommand> {
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
        if let Some(mode) = value_as_string(&rest[0]) {
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
    args: &'a [Value],
) -> BuiltinResult<((FigureHandle, usize), &'a [Value])> {
    if let Some(first) = args.first() {
        if let Some(target) = try_parse_axes_target(first) {
            if axes_handle_exists(target.0, target.1) {
                return Ok((target, &args[1..]));
            }
            return Err(plotting_error(
                builtin,
                format!("{builtin}: invalid axes handle"),
            ));
        }
    }
    Ok((current_axes_target(), args))
}

fn try_parse_axes_target(value: &Value) -> Option<(FigureHandle, usize)> {
    match value {
        Value::Num(v) => decode_axes_handle(*v).ok(),
        Value::Int(i) => decode_axes_handle(i.to_f64()).ok(),
        Value::Tensor(tensor) if tensor.data.len() == 1 => decode_axes_handle(tensor.data[0]).ok(),
        _ => None,
    }
}

fn collect_label_strings(builtin: &'static str, args: &[Value]) -> BuiltinResult<Vec<String>> {
    let mut labels = Vec::new();
    for arg in args {
        match arg {
            Value::StringArray(arr) => labels.extend(arr.data.iter().cloned()),
            Value::Cell(cell) => {
                for row in 0..cell.rows {
                    for col in 0..cell.cols {
                        let value = cell.get(row, col).map_err(|err| {
                            plotting_error(builtin, format!("legend: invalid label cell: {err}"))
                        })?;
                        labels.push(value_as_text_string(&value).ok_or_else(|| {
                            plotting_error(builtin, "legend: labels must be strings or char arrays")
                        })?);
                    }
                }
            }
            _ => labels.push(value_as_text_string(arg).ok_or_else(|| {
                plotting_error(builtin, "legend: labels must be strings or char arrays")
            })?),
        }
    }
    Ok(labels)
}

#[cfg(test)]
pub fn vec4_eq(a: Option<glam::Vec4>, b: glam::Vec4) -> bool {
    a.map(|v| (v - b).abs().max_element() < 1e-6)
        .unwrap_or(false)
}
