//! MATLAB command-style plotting/layout verbs.
//!
//! These operate on the active figure/axes state (grid/axis/cla/colormap/shading/colorbar).

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::state::{
    clear_current_axes, set_axis_equal, set_axis_limits, set_box_enabled, set_colorbar_enabled,
    set_colormap, set_grid_enabled, set_surface_shading, toggle_box, toggle_colorbar, toggle_grid,
};
use crate::builtins::plotting::type_resolvers::string_type;

fn as_lower_str(val: &Value) -> Option<String> {
    match val {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::CharArray(c) => Some(c.data.iter().collect::<String>().to_ascii_lowercase()),
        _ => None,
    }
}

fn parse_on_off(
    builtin: &'static str,
    arg: Option<&Value>,
) -> Result<Option<bool>, crate::RuntimeError> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let Some(s) = as_lower_str(arg) else {
        return Err(plotting_error(builtin, "expected string argument"));
    };
    match s.trim() {
        "on" => Ok(Some(true)),
        "off" => Ok(Some(false)),
        other => Err(plotting_error(
            builtin,
            format!("expected 'on' or 'off' (got '{other}')"),
        )),
    }
}

#[runtime_builtin(
    name = "grid",
    category = "plotting",
    summary = "Toggle grid lines on current axes.",
    keywords = "grid,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn grid_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    match parse_on_off("grid", args.first())? {
        Some(enabled) => {
            set_grid_enabled(enabled);
            Ok(if enabled { "grid on" } else { "grid off" }.to_string())
        }
        None => {
            let enabled = toggle_grid();
            Ok(if enabled { "grid on" } else { "grid off" }.to_string())
        }
    }
}

#[runtime_builtin(
    name = "box",
    category = "plotting",
    summary = "Toggle axes box outline.",
    keywords = "box,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn box_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    match parse_on_off("box", args.first())? {
        Some(enabled) => {
            set_box_enabled(enabled);
            Ok(if enabled { "box on" } else { "box off" }.to_string())
        }
        None => {
            let enabled = toggle_box();
            Ok(if enabled { "box on" } else { "box off" }.to_string())
        }
    }
}

#[runtime_builtin(
    name = "axis",
    category = "plotting",
    summary = "Set axis limits/aspect.",
    keywords = "axis,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn axis_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    if args.is_empty() {
        return Ok("axis".to_string());
    }

    // Numeric form: axis([xmin xmax ymin ymax])
    if let Value::Tensor(t) = &args[0] {
        if t.data.len() == 4 {
            let xmin = t.data[0];
            let xmax = t.data[1];
            let ymin = t.data[2];
            let ymax = t.data[3];
            if !(xmin.is_finite() && xmax.is_finite() && ymin.is_finite() && ymax.is_finite()) {
                return Err(plotting_error("axis", "axis: limits must be finite"));
            }
            set_axis_limits(Some((xmin, xmax)), Some((ymin, ymax)));
            return Ok("axis limits set".to_string());
        }
    }

    let Some(mode) = as_lower_str(&args[0]) else {
        return Err(plotting_error(
            "axis",
            "axis: expected a string mode or a 4-element vector",
        ));
    };
    match mode.trim() {
        "equal" => {
            set_axis_equal(true);
            Ok("axis equal".to_string())
        }
        "auto" => {
            set_axis_equal(false);
            set_axis_limits(None, None);
            Ok("axis auto".to_string())
        }
        "tight" => {
            // Treat as auto; camera fit uses data bounds.
            set_axis_limits(None, None);
            Ok("axis tight".to_string())
        }
        other => Err(plotting_error(
            "axis",
            format!("axis: unsupported argument '{other}'"),
        )),
    }
}

#[runtime_builtin(
    name = "cla",
    category = "plotting",
    summary = "Clear current axes.",
    keywords = "cla,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn cla_builtin(_args: Vec<Value>) -> crate::BuiltinResult<String> {
    clear_current_axes();
    Ok("axes cleared".to_string())
}

#[runtime_builtin(
    name = "colormap",
    category = "plotting",
    summary = "Set the active colormap.",
    keywords = "colormap,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colormap_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let Some(arg) = args.first() else {
        return Err(plotting_error("colormap", "colormap: expected a name"));
    };
    let Some(name) = as_lower_str(arg) else {
        return Err(plotting_error(
            "colormap",
            "colormap: expected a string name",
        ));
    };
    let cmap = match name.trim() {
        "parula" => runmat_plot::plots::surface::ColorMap::Parula,
        "viridis" => runmat_plot::plots::surface::ColorMap::Viridis,
        "plasma" => runmat_plot::plots::surface::ColorMap::Plasma,
        "inferno" => runmat_plot::plots::surface::ColorMap::Inferno,
        "magma" => runmat_plot::plots::surface::ColorMap::Magma,
        "turbo" => runmat_plot::plots::surface::ColorMap::Turbo,
        "jet" => runmat_plot::plots::surface::ColorMap::Jet,
        "hot" => runmat_plot::plots::surface::ColorMap::Hot,
        "cool" => runmat_plot::plots::surface::ColorMap::Cool,
        "spring" => runmat_plot::plots::surface::ColorMap::Spring,
        "summer" => runmat_plot::plots::surface::ColorMap::Summer,
        "autumn" => runmat_plot::plots::surface::ColorMap::Autumn,
        "winter" => runmat_plot::plots::surface::ColorMap::Winter,
        "gray" | "grey" => runmat_plot::plots::surface::ColorMap::Gray,
        "bone" => runmat_plot::plots::surface::ColorMap::Bone,
        "copper" => runmat_plot::plots::surface::ColorMap::Copper,
        "pink" => runmat_plot::plots::surface::ColorMap::Pink,
        "lines" => runmat_plot::plots::surface::ColorMap::Lines,
        other => {
            return Err(plotting_error(
                "colormap",
                format!("colormap: unknown colormap '{other}'"),
            ))
        }
    };
    set_colormap(cmap);
    Ok(format!("colormap {name}"))
}

#[runtime_builtin(
    name = "shading",
    category = "plotting",
    summary = "Set shading mode for surface plots.",
    keywords = "shading,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn shading_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let Some(arg) = args.first() else {
        return Err(plotting_error(
            "shading",
            "shading: expected 'flat', 'interp', or 'faceted'",
        ));
    };
    let Some(mode) = as_lower_str(arg) else {
        return Err(plotting_error("shading", "shading: expected a string"));
    };
    let shading = match mode.trim() {
        "flat" => runmat_plot::plots::surface::ShadingMode::Flat,
        "interp" => runmat_plot::plots::surface::ShadingMode::Smooth,
        "faceted" => runmat_plot::plots::surface::ShadingMode::Faceted,
        other => {
            return Err(plotting_error(
                "shading",
                format!("shading: unknown mode '{other}'"),
            ))
        }
    };
    set_surface_shading(shading);
    Ok(format!("shading {mode}"))
}

#[runtime_builtin(
    name = "colorbar",
    category = "plotting",
    summary = "Toggle colorbar visibility.",
    keywords = "colorbar,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colorbar_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    match parse_on_off("colorbar", args.first())? {
        Some(enabled) => {
            set_colorbar_enabled(enabled);
            Ok(if enabled {
                "colorbar on"
            } else {
                "colorbar off"
            }
            .to_string())
        }
        None => {
            let enabled = toggle_colorbar();
            Ok(if enabled {
                "colorbar on"
            } else {
                "colorbar off"
            }
            .to_string())
        }
    }
}
