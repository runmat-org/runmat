//! MATLAB command-style plotting/layout verbs.
//!
//! These operate on the active figure/axes state (grid/axis/cla/colormap/shading/colorbar).

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::{as_lower_str, parse_on_off};
use super::state::{
    clear_current_axes, set_axis_equal, set_axis_limits, set_box_enabled, set_colorbar_enabled,
    set_colormap, set_grid_enabled, set_minor_grid_enabled, set_surface_shading, set_z_limits,
    toggle_box, toggle_colorbar, toggle_grid, toggle_minor_grid,
};
use crate::builtins::plotting::type_resolvers::bool_type;
use crate::{build_runtime_error, RuntimeError};

const GRID_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grid enabled state after command execution.",
}];
const GRID_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const GRID_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Grid mode token ('on'|'off'|'minor').",
}];
const GRID_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = grid()",
        inputs: &GRID_INPUTS_NONE,
        outputs: &GRID_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = grid(mode)",
        inputs: &GRID_INPUTS_MODE,
        outputs: &GRID_OUTPUT_ENABLED,
    },
];
const GRID_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRID.INVALID_ARGUMENT",
    identifier: Some("RunMat:grid:InvalidArgument"),
    when: "Grid mode argument is unsupported.",
    message: "grid: invalid argument",
};
const GRID_ERRORS: [BuiltinErrorDescriptor; 1] = [GRID_ERROR_INVALID_ARGUMENT];
pub const GRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRID_ERRORS,
};

enum GridMode {
    ToggleMajor,
    Major(bool),
    ToggleMinor,
}

const BOX_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Box outline enabled state after command execution.",
}];
const BOX_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const BOX_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Box mode token ('on'|'off').",
}];
const BOX_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = box()",
        inputs: &BOX_INPUTS_NONE,
        outputs: &BOX_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = box(mode)",
        inputs: &BOX_INPUTS_MODE,
        outputs: &BOX_OUTPUT_ENABLED,
    },
];
const BOX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOX.INVALID_ARGUMENT",
    identifier: Some("RunMat:box:InvalidArgument"),
    when: "Box mode argument is unsupported.",
    message: "box: invalid argument",
};
const BOX_ERRORS: [BuiltinErrorDescriptor; 1] = [BOX_ERROR_INVALID_ARGUMENT];
pub const BOX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BOX_ERRORS,
};

const AXIS_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True on success.",
}];
const AXIS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const AXIS_INPUTS_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axis limits vector [xmin xmax ymin ymax] or [xmin xmax ymin ymax zmin zmax].",
}];
const AXIS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode token: 'equal'|'auto'|'tight'|'manual'|'ij'|'xy'|'on'|'off'.",
}];
const AXIS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ok = axis()",
        inputs: &AXIS_INPUTS_NONE,
        outputs: &AXIS_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = axis([xmin xmax ymin ymax | ... zmin zmax])",
        inputs: &AXIS_INPUTS_LIMITS,
        outputs: &AXIS_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = axis(mode)",
        inputs: &AXIS_INPUTS_MODE,
        outputs: &AXIS_OUTPUT_OK,
    },
];
const AXIS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXIS.INVALID_ARGUMENT",
    identifier: Some("RunMat:axis:InvalidArgument"),
    when: "Axis argument is unsupported, malformed, non-finite, or has invalid bounds ordering.",
    message: "axis: invalid argument",
};
const AXIS_ERRORS: [BuiltinErrorDescriptor; 1] = [AXIS_ERROR_INVALID_ARGUMENT];
pub const AXIS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AXIS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AXIS_ERRORS,
};

const CLA_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when current axes are cleared.",
}];
const CLA_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const CLA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ok = cla()",
    inputs: &CLA_INPUTS_NONE,
    outputs: &CLA_OUTPUT_OK,
}];
const CLA_ERRORS: [BuiltinErrorDescriptor; 0] = [];
pub const CLA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLA_ERRORS,
};

const COLORMAP_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True on successful colormap update.",
}];
const COLORMAP_INPUTS_NAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Colormap name.",
}];
const COLORMAP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ok = colormap(name)",
    inputs: &COLORMAP_INPUTS_NAME,
    outputs: &COLORMAP_OUTPUT_OK,
}];
const COLORMAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:colormap:InvalidArgument"),
    when: "Colormap name is missing, non-string, or unknown.",
    message: "colormap: invalid argument",
};
const COLORMAP_ERRORS: [BuiltinErrorDescriptor; 1] = [COLORMAP_ERROR_INVALID_ARGUMENT];
pub const COLORMAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORMAP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORMAP_ERRORS,
};

const SHADING_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True on successful shading mode update.",
}];
const SHADING_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shading mode token: 'flat'|'interp'|'faceted'.",
}];
const SHADING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ok = shading(mode)",
    inputs: &SHADING_INPUTS_MODE,
    outputs: &SHADING_OUTPUT_OK,
}];
const SHADING_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHADING.INVALID_ARGUMENT",
    identifier: Some("RunMat:shading:InvalidArgument"),
    when: "Shading mode is missing, non-string, or unsupported.",
    message: "shading: invalid argument",
};
const SHADING_ERRORS: [BuiltinErrorDescriptor; 1] = [SHADING_ERROR_INVALID_ARGUMENT];
pub const SHADING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SHADING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SHADING_ERRORS,
};

const COLORBAR_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Colorbar enabled state after command execution.",
}];
const COLORBAR_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const COLORBAR_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Colorbar mode token ('on'|'off').",
}];
const COLORBAR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = colorbar()",
        inputs: &COLORBAR_INPUTS_NONE,
        outputs: &COLORBAR_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = colorbar(mode)",
        inputs: &COLORBAR_INPUTS_MODE,
        outputs: &COLORBAR_OUTPUT_ENABLED,
    },
];
const COLORBAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORBAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:colorbar:InvalidArgument"),
    when: "Colorbar mode argument is unsupported.",
    message: "colorbar: invalid argument",
};
const COLORBAR_ERRORS: [BuiltinErrorDescriptor; 1] = [COLORBAR_ERROR_INVALID_ARGUMENT];
pub const COLORBAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORBAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORBAR_ERRORS,
};

fn cmd_error_with_message(
    builtin: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "grid",
    category = "plotting",
    summary = "Toggle axes grid lines.",
    keywords = "grid,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::GRID_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn grid_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_grid_mode(&args)? {
        GridMode::ToggleMajor => {
            let enabled = toggle_grid();
            Ok(enabled)
        }
        GridMode::Major(enabled) => {
            set_grid_enabled(enabled);
            if !enabled {
                set_minor_grid_enabled(false);
            }
            Ok(enabled)
        }
        GridMode::ToggleMinor => {
            let enabled = toggle_minor_grid();
            Ok(enabled)
        }
    }
}

fn parse_grid_mode(args: &[Value]) -> crate::BuiltinResult<GridMode> {
    if args.len() > 1 {
        return Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected at most one mode argument",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        ));
    }

    let Some(arg) = args.first() else {
        return Ok(GridMode::ToggleMajor);
    };
    let Some(mode) = as_lower_str(arg) else {
        return Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected string argument",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        ));
    };
    match mode.trim() {
        "on" => Ok(GridMode::Major(true)),
        "off" => Ok(GridMode::Major(false)),
        "minor" => Ok(GridMode::ToggleMinor),
        other => Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected 'on', 'off', or 'minor' (got '{other}')",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        )),
    }
}

#[runtime_builtin(
    name = "box",
    category = "plotting",
    summary = "Toggle axes box outlines.",
    keywords = "box,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::BOX_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn box_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_on_off("box", args.first()).map_err(|err| {
        cmd_error_with_message(
            "box",
            format!("{}: {}", BOX_ERROR_INVALID_ARGUMENT.message, err.message()),
            &BOX_ERROR_INVALID_ARGUMENT,
        )
    })? {
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
    summary = "Set axis limits and aspect behavior.",
    keywords = "axis,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::AXIS_DESCRIPTOR),
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
                return Err(cmd_error_with_message(
                    "axis",
                    AXIS_ERROR_INVALID_ARGUMENT.message,
                    &AXIS_ERROR_INVALID_ARGUMENT,
                ));
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
                return Err(cmd_error_with_message(
                    "axis",
                    AXIS_ERROR_INVALID_ARGUMENT.message,
                    &AXIS_ERROR_INVALID_ARGUMENT,
                ));
            }
            if xmax < xmin || ymax < ymin || zmax < zmin {
                return Err(cmd_error_with_message(
                    "axis",
                    AXIS_ERROR_INVALID_ARGUMENT.message,
                    &AXIS_ERROR_INVALID_ARGUMENT,
                ));
            }
            set_axis_limits(Some((xmin, xmax)), Some((ymin, ymax)));
            set_z_limits(Some((zmin, zmax)));
            return Ok(true);
        }
    }

    let Some(mode) = as_lower_str(&args[0]) else {
        return Err(cmd_error_with_message(
            "axis",
            AXIS_ERROR_INVALID_ARGUMENT.message,
            &AXIS_ERROR_INVALID_ARGUMENT,
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
        "manual" | "ij" | "xy" | "on" | "off" => {
            // These MATLAB axis modes are accepted as command tokens for compatibility.
            // The current plot scene model does not yet track axis visibility, direction,
            // or manual limit-lock state separately from concrete limits.
            Ok(true)
        }
        other => Err(cmd_error_with_message(
            "axis",
            format!(
                "{}: unsupported argument '{other}'",
                AXIS_ERROR_INVALID_ARGUMENT.message
            ),
            &AXIS_ERROR_INVALID_ARGUMENT,
        )),
    }
}

#[runtime_builtin(
    name = "cla",
    category = "plotting",
    summary = "Clear the current axes.",
    keywords = "cla,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::CLA_DESCRIPTOR),
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
    descriptor(crate::builtins::plotting::cmds::COLORMAP_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colormap_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    let Some(arg) = args.first() else {
        return Err(cmd_error_with_message(
            "colormap",
            COLORMAP_ERROR_INVALID_ARGUMENT.message,
            &COLORMAP_ERROR_INVALID_ARGUMENT,
        ));
    };
    let Some(name) = as_lower_str(arg) else {
        return Err(cmd_error_with_message(
            "colormap",
            COLORMAP_ERROR_INVALID_ARGUMENT.message,
            &COLORMAP_ERROR_INVALID_ARGUMENT,
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
            return Err(cmd_error_with_message(
                "colormap",
                format!(
                    "{}: unknown colormap '{other}'",
                    COLORMAP_ERROR_INVALID_ARGUMENT.message
                ),
                &COLORMAP_ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    set_colormap(cmap);
    Ok(true)
}

#[runtime_builtin(
    name = "shading",
    category = "plotting",
    summary = "Set surface shading mode (flat, interp, or faceted).",
    keywords = "shading,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::SHADING_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn shading_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    let Some(arg) = args.first() else {
        return Err(cmd_error_with_message(
            "shading",
            SHADING_ERROR_INVALID_ARGUMENT.message,
            &SHADING_ERROR_INVALID_ARGUMENT,
        ));
    };
    let Some(mode) = as_lower_str(arg) else {
        return Err(cmd_error_with_message(
            "shading",
            SHADING_ERROR_INVALID_ARGUMENT.message,
            &SHADING_ERROR_INVALID_ARGUMENT,
        ));
    };
    let shading = match mode.trim() {
        "flat" => runmat_plot::plots::surface::ShadingMode::Flat,
        "interp" => runmat_plot::plots::surface::ShadingMode::Smooth,
        "faceted" => runmat_plot::plots::surface::ShadingMode::Faceted,
        other => {
            return Err(cmd_error_with_message(
                "shading",
                format!(
                    "{}: unknown mode '{other}'",
                    SHADING_ERROR_INVALID_ARGUMENT.message
                ),
                &SHADING_ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    set_surface_shading(shading);
    Ok(true)
}

#[runtime_builtin(
    name = "colorbar",
    category = "plotting",
    summary = "Show, hide, or toggle colorbars.",
    keywords = "colorbar,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::COLORBAR_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colorbar_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_on_off("colorbar", args.first()).map_err(|err| {
        cmd_error_with_message(
            "colorbar",
            format!(
                "{}: {}",
                COLORBAR_ERROR_INVALID_ARGUMENT.message,
                err.message()
            ),
            &COLORBAR_ERROR_INVALID_ARGUMENT,
        )
    })? {
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

    #[test]
    fn axis_accepts_common_command_modes() {
        let _guard = setup();
        for mode in ["equal", "auto", "tight", "manual", "ij", "xy", "on", "off"] {
            axis_builtin(vec![Value::String(mode.into())])
                .unwrap_or_else(|err| panic!("axis {mode} should be accepted: {err:?}"));
        }
    }

    #[test]
    fn grid_minor_toggles_minor_grid_without_changing_major_grid() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

        let enabled = grid_builtin(vec![Value::String("minor".into())]).unwrap();
        assert!(enabled);
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("Grid".into())]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("MinorGrid".into())]).unwrap(),
            Value::Bool(true)
        );

        let enabled = grid_builtin(vec![Value::String("minor".into())]).unwrap();
        assert!(!enabled);
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("Grid".into())]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("MinorGrid".into())]).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn grid_off_disables_major_and_minor_grid() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

        grid_builtin(vec![Value::String("minor".into())]).unwrap();
        let enabled = grid_builtin(vec![Value::String("off".into())]).unwrap();
        assert!(!enabled);
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("Grid".into())]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("MinorGrid".into())]).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn grid_rejects_extra_arguments() {
        let _guard = setup();
        let err = grid_builtin(vec![
            Value::String("minor".into()),
            Value::String("on".into()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:grid:InvalidArgument"));
    }

    #[test]
    fn command_descriptors_cover_core_forms() {
        let grid_labels: Vec<&str> = GRID_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(grid_labels.contains(&"enabled = grid()"));
        assert!(grid_labels.contains(&"enabled = grid(mode)"));

        let box_labels: Vec<&str> = BOX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(box_labels.contains(&"enabled = box()"));
        assert!(box_labels.contains(&"enabled = box(mode)"));

        let axis_labels: Vec<&str> = AXIS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(axis_labels.contains(&"ok = axis()"));
        assert!(axis_labels.contains(&"ok = axis([xmin xmax ymin ymax | ... zmin zmax])"));
        assert!(axis_labels.contains(&"ok = axis(mode)"));

        let cla_labels: Vec<&str> = CLA_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(cla_labels.contains(&"ok = cla()"));

        let colormap_labels: Vec<&str> = COLORMAP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(colormap_labels.contains(&"ok = colormap(name)"));

        let shading_labels: Vec<&str> = SHADING_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(shading_labels.contains(&"ok = shading(mode)"));

        let colorbar_labels: Vec<&str> = COLORBAR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(colorbar_labels.contains(&"enabled = colorbar()"));
        assert!(colorbar_labels.contains(&"enabled = colorbar(mode)"));
    }
}
