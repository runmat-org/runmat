//! MATLAB command-style plotting/layout verbs.
//!
//! These operate on the active figure/axes state (grid/axis/cla/colormap/shading/colorbar).

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::{as_lower_str, parse_on_off};
use super::plotting_error;
use super::state::{
    clear_current_axes, set_axis_equal, set_axis_limits, set_box_enabled, set_colorbar_enabled,
    set_colormap, set_grid_enabled, set_surface_shading, set_z_limits, toggle_box, toggle_colorbar,
    toggle_grid,
};
use crate::builtins::plotting::type_resolvers::bool_type;

#[runtime_builtin(
    name = "grid",
    category = "plotting",
    summary = "Toggle grid lines on current axes.",
    keywords = "grid,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn grid_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_on_off("grid", args.first())? {
        Some(enabled) => {
            set_grid_enabled(enabled);
            Ok(enabled)
        }
        None => {
            let enabled = toggle_grid();
            Ok(enabled)
        }
    }
}

#[runtime_builtin(
    name = "box",
    category = "plotting",
    summary = "Toggle axes box outline.",
    keywords = "box,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn box_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_on_off("box", args.first())? {
        Some(enabled) => {
            set_box_enabled(enabled);
            Ok(enabled)
        }
        None => {
            let enabled = toggle_box();
            Ok(enabled)
        }
    }
}

#[runtime_builtin(
    name = "axis",
    category = "plotting",
    summary = "Set axis limits/aspect.",
    keywords = "axis,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn axis_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    if args.is_empty() {
        return Ok(true);
    }

    // Numeric form: axis([xmin xmax ymin ymax]) or axis([xmin xmax ymin ymax zmin zmax])
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
            return Ok(true);
        }
        if t.data.len() == 6 {
            let xmin = t.data[0];
            let xmax = t.data[1];
            let ymin = t.data[2];
            let ymax = t.data[3];
            let zmin = t.data[4];
            let zmax = t.data[5];
            if !(xmin.is_finite()
                && xmax.is_finite()
                && ymin.is_finite()
                && ymax.is_finite()
                && zmin.is_finite()
                && zmax.is_finite())
            {
                return Err(plotting_error("axis", "axis: limits must be finite"));
            }
            if xmax < xmin || ymax < ymin || zmax < zmin {
                return Err(plotting_error("axis", "axis: limits must be increasing"));
            }
            set_axis_limits(Some((xmin, xmax)), Some((ymin, ymax)));
            set_z_limits(Some((zmin, zmax)));
            return Ok(true);
        }
    }

    let Some(mode) = as_lower_str(&args[0]) else {
        return Err(plotting_error(
            "axis",
            "axis: expected a string mode or a 4-element or 6-element vector",
        ));
    };
    match mode.trim() {
        "equal" => {
            set_axis_equal(true);
            Ok(true)
        }
        "auto" => {
            set_axis_equal(false);
            set_axis_limits(None, None);
            Ok(true)
        }
        "tight" => {
            // Treat as auto; camera fit uses data bounds.
            set_axis_limits(None, None);
            Ok(true)
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
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn cla_builtin(_args: Vec<Value>) -> crate::BuiltinResult<bool> {
    clear_current_axes();
    Ok(true)
}

#[runtime_builtin(
    name = "colormap",
    category = "plotting",
    summary = "Set the active colormap.",
    keywords = "colormap,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colormap_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
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
    Ok(true)
}

#[runtime_builtin(
    name = "shading",
    category = "plotting",
    summary = "Set shading mode for surface plots.",
    keywords = "shading,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn shading_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
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
    Ok(true)
}

#[runtime_builtin(
    name = "colorbar",
    category = "plotting",
    summary = "Toggle colorbar visibility.",
    keywords = "colorbar,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colorbar_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_on_off("colorbar", args.first())? {
        Some(enabled) => {
            set_colorbar_enabled(enabled);
            Ok(enabled)
        }
        None => {
            let enabled = toggle_colorbar();
            Ok(enabled)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_builtins::{NumericDType, Tensor};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn axis_accepts_six_element_3d_limits() {
        let _guard = setup();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .unwrap();

        axis_builtin(vec![Value::Tensor(Tensor {
            rows: 1,
            cols: 6,
            shape: vec![1, 6],
            data: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            dtype: NumericDType::F64,
        })])
        .unwrap();
        let zlim = get_builtin(vec![Value::Num(ax), Value::String("ZLim".into())]).unwrap();
        let zlim = Tensor::try_from(&zlim).unwrap();
        assert_eq!(zlim.data, vec![4.0, 5.0]);
    }
}
