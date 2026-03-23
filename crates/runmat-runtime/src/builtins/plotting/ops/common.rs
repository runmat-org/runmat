#[cfg(test)]
use glam::Vec4;
use runmat_builtins::Value;
use runmat_plot::plots::{LegendStyle, TextStyle};

use super::state::{
    axes_handle_exists, current_axes_state, decode_axes_handle, FigureError, FigureHandle,
};
use super::style::{
    parse_color_value, value_as_bool, value_as_f64, value_as_string, LineStyleParseOptions,
};
use super::{plotting_error, plotting_error_with_source};
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

    let text = value_as_string(&rest[0]).ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: expected text as char array or string"),
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
                    })
                }
                "off" | "hide" => {
                    return Ok(LegendCommand {
                        target,
                        enabled: false,
                        labels: None,
                        style,
                    })
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

fn split_axes_target<'a>(
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

fn parse_text_style_pairs(builtin: &'static str, args: &[Value]) -> BuiltinResult<TextStyle> {
    if args.is_empty() {
        return Ok(TextStyle::default());
    }
    if args.len() % 2 != 0 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    let opts = LineStyleParseOptions::generic(builtin);
    let mut style = TextStyle::default();
    for pair in args.chunks_exact(2) {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: property names must be strings"),
                )
            })?
            .trim()
            .to_ascii_lowercase();
        match key.as_str() {
            "color" => style.color = Some(parse_color_value(&opts, &pair[1])?),
            "fontsize" => {
                style.font_size = Some(value_as_f64(&pair[1]).ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: FontSize must be numeric"))
                })? as f32)
            }
            "interpreter" => {
                style.interpreter = Some(value_as_string(&pair[1]).ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: Interpreter must be a string"))
                })?)
            }
            "visible" => {
                style.visible = value_as_bool(&pair[1]).ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: Visible must be logical"))
                })?
            }
            other => {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported property `{other}`"),
                ))
            }
        }
    }
    Ok(style)
}

fn split_legend_style_pairs<'a>(
    builtin: &'static str,
    args: &'a [Value],
) -> BuiltinResult<(&'a [Value], LegendStyle)> {
    let opts = LineStyleParseOptions::generic(builtin);
    let mut style = LegendStyle::default();
    let mut split = args.len();
    while split >= 2 {
        let key_idx = split - 2;
        let Some(key) = value_as_string(&args[key_idx]) else {
            break;
        };
        let key = key.trim().to_ascii_lowercase();
        match key.as_str() {
            "location" => {
                style.location =
                    Some(value_as_string(&args[key_idx + 1]).ok_or_else(|| {
                        plotting_error(builtin, "legend: Location must be a string")
                    })?);
                split -= 2;
            }
            "fontsize" => {
                style.font_size =
                    Some(value_as_f64(&args[key_idx + 1]).ok_or_else(|| {
                        plotting_error(builtin, "legend: FontSize must be numeric")
                    })? as f32);
                split -= 2;
            }
            "textcolor" | "color" => {
                style.text_color = Some(parse_color_value(&opts, &args[key_idx + 1])?);
                split -= 2;
            }
            "visible" => {
                style.visible = value_as_bool(&args[key_idx + 1])
                    .ok_or_else(|| plotting_error(builtin, "legend: Visible must be logical"))?;
                split -= 2;
            }
            _ => break,
        }
    }
    Ok((&args[..split], style))
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
                        labels.push(value_as_string(&value).ok_or_else(|| {
                            plotting_error(builtin, "legend: labels must be strings or char arrays")
                        })?);
                    }
                }
            }
            _ => labels.push(value_as_string(arg).ok_or_else(|| {
                plotting_error(builtin, "legend: labels must be strings or char arrays")
            })?),
        }
    }
    Ok(labels)
}

#[cfg(test)]
pub fn vec4_eq(a: Option<Vec4>, b: Vec4) -> bool {
    a.map(|v| (v - b).abs().max_element() < 1e-6)
        .unwrap_or(false)
}
