//! Compatibility stubs for MATLAB command-style plotting/layout verbs.
//! These are intentionally side-effect-light to keep fusion/analysis tolerant.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plotting_error;

fn as_lower_str(val: &Value) -> Option<String> {
    match val {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::CharArray(c) => Some(c.data.iter().collect::<String>().to_ascii_lowercase()),
        _ => None,
    }
}

fn accepts_arg(arg: Option<&Value>, allowed: &[&str]) -> bool {
    match arg.and_then(as_lower_str) {
        Some(s) => allowed.is_empty() || allowed.iter().any(|a| *a == s),
        None => allowed.is_empty() || arg.is_none(),
    }
}

#[runtime_builtin(
    name = "grid",
    category = "plotting",
    summary = "Toggle grid lines on current axes.",
    keywords = "grid,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn grid_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    if !accepts_arg(args.first(), &["on", "off"]) {
        return Err(plotting_error("grid", "grid: expected 'on' or 'off'"));
    }
    Ok("grid toggled".into())
}

#[runtime_builtin(
    name = "box",
    category = "plotting",
    summary = "Toggle box outline on current axes.",
    keywords = "box,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn box_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    if !accepts_arg(args.first(), &["on", "off"]) {
        return Err(plotting_error("box", "box: expected 'on' or 'off'"));
    }
    Ok("box toggled".into())
}

#[runtime_builtin(
    name = "axis",
    category = "plotting",
    summary = "Adjust axis limits/aspect.",
    keywords = "axis,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn axis_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    // Accept common command-style args; ignore for now.
    let ok = args.first().is_none_or(|v| {
        matches!(
            as_lower_str(v).as_deref(),
            Some("auto" | "manual" | "tight" | "equal" | "ij" | "xy")
        )
    });
    if !ok {
        return Err(plotting_error("axis", "axis: unsupported argument"));
    }
    Ok("axis updated".into())
}

#[runtime_builtin(
    name = "shading",
    category = "plotting",
    summary = "Set shading mode for surface/mesh plots.",
    keywords = "shading,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn shading_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    if !accepts_arg(args.first(), &["flat", "interp", "faceted"]) {
        return Err(plotting_error(
            "shading",
            "shading: expected 'flat', 'interp', or 'faceted'",
        ));
    }
    Ok("shading set".into())
}

#[runtime_builtin(
    name = "colormap",
    category = "plotting",
    summary = "Set colormap.",
    keywords = "colormap,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn colormap_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("colormap set".into())
}

#[runtime_builtin(
    name = "colorbar",
    category = "plotting",
    summary = "Show or hide colorbar.",
    keywords = "colorbar,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn colorbar_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("colorbar toggled".into())
}

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create/select figure.",
    keywords = "figure,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn figure_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("figure selected".into())
}

#[runtime_builtin(
    name = "subplot",
    category = "plotting",
    summary = "Select subplot layout/axes.",
    keywords = "subplot,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn subplot_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("subplot selected".into())
}

#[runtime_builtin(
    name = "clf",
    category = "plotting",
    summary = "Clear current figure.",
    keywords = "clf,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn clf_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("figure cleared".into())
}

#[runtime_builtin(
    name = "cla",
    category = "plotting",
    summary = "Clear current axes.",
    keywords = "cla,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn cla_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("axes cleared".into())
}

#[runtime_builtin(
    name = "close",
    category = "plotting",
    summary = "Close figure.",
    keywords = "close,plotting",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::compat_cmds"
)]
pub fn close_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    Ok("figure closed".into())
}
