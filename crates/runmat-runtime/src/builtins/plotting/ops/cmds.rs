//! MATLAB command-style plotting/layout verbs.
//!
//! These operate on the active figure/axes state (grid/axis/cla/colormap/shading/colorbar).

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::colormap_arrays::parse_rgb_colormap_tensor;
use super::op_common::cmd_parsing::{as_lower_str, parse_on_off};
use super::state::{
    axes_metadata_snapshot, axis_display_bounds_snapshot_for_axes, clear_current_axes,
    color_limits_snapshot, current_axes_state, current_figure_handle, set_axes_style_for_axes,
    set_axis_equal, set_axis_equal_and_limits, set_axis_limits, set_box_enabled,
    set_color_limits_runtime, set_colorbar_enabled, set_colormap, set_colormap_with_length,
    set_grid_and_minor_grid_enabled, set_hidden_line_removal_for_axes, set_surface_shading,
    set_z_limits, toggle_box, toggle_colorbar, toggle_grid, toggle_minor_grid, z_limits_snapshot,
    FigureHandle,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::type_resolvers::{axis_type, bool_type};
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

const AXIS_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "lim",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current x/y limits, with z limits included for a configured 3-D view.",
}];
const AXIS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const AXIS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const AXIS_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle whose limits are queried.",
}];
const AXIS_INPUTS_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Four-, six-, or eight-element x/y, x/y/z, or x/y/z/color limits vector.",
}];
const AXIS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode token: 'equal'|'image'|'auto'|'tight'|'manual'|'ij'|'xy'|'on'|'off'.",
}];
const AXIS_INPUTS_VISIBILITY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "visibility",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes visibility as on/off, logical true/false, or numeric 1/0.",
}];
const AXIS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "lim = axis()",
        inputs: &AXIS_INPUTS_NONE,
        outputs: &AXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "lim = axis(ax)",
        inputs: &AXIS_INPUTS_TARGET,
        outputs: &AXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "axis([xmin xmax ymin ymax | ... cmin cmax])",
        inputs: &AXIS_INPUTS_LIMITS,
        outputs: &AXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "axis(mode)",
        inputs: &AXIS_INPUTS_MODE,
        outputs: &AXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "axis(visibility)",
        inputs: &AXIS_INPUTS_VISIBILITY,
        outputs: &AXIS_OUTPUTS_NONE,
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

const AXIS_LIMIT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "limits",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Four-, six-, and eight-element limit vectors accept every built-in integer class.",
    }];
const AXIS_VISIBILITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "visibility",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Current numeric visibility syntax accepts scalar zero or one; logical false/true is the noninteger counterpart.",
    }];
pub const AXIS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "axis(integer_limits)",
        inputs: &AXIS_LIMIT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Authoritative integer storage is shape- and order-validated before one conversion into host f64 graphics limits; the separate query form returns double limits.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "axis(integer_visibility)",
        inputs: &AXIS_VISIBILITY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact scalar zero and one are accepted, without provider dispatch or integer output.",
    },
];

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
const COLORMAP_INPUTS_RGB: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "map",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 RGB colormap array with values in [0, 1].",
}];
const COLORMAP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "ok = colormap(name)",
        inputs: &COLORMAP_INPUTS_NAME,
        outputs: &COLORMAP_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = colormap(map)",
        inputs: &COLORMAP_INPUTS_RGB,
        outputs: &COLORMAP_OUTPUT_OK,
    },
];
const COLORMAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:colormap:InvalidArgument"),
    when: "Colormap input is missing, unknown, or not a valid m-by-3 RGB array.",
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

const HIDDEN_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hidden-line-removal state after command execution.",
}];
const HIDDEN_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const HIDDEN_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Hidden-line-removal mode token ('on'|'off').",
}];
const HIDDEN_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];
const HIDDEN_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"toggle\""),
        description: "Hidden-line-removal mode token ('on'|'off').",
    },
];
const HIDDEN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "enabled = hidden()",
        inputs: &HIDDEN_INPUTS_NONE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(mode)",
        inputs: &HIDDEN_INPUTS_MODE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(ax)",
        inputs: &HIDDEN_INPUTS_AX,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(ax, mode)",
        inputs: &HIDDEN_INPUTS_AX_MODE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
];
const HIDDEN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIDDEN.INVALID_ARGUMENT",
    identifier: Some("RunMat:hidden:InvalidArgument"),
    when: "Hidden-line-removal mode or axes target is unsupported.",
    message: "hidden: invalid argument",
};
const HIDDEN_ERRORS: [BuiltinErrorDescriptor; 1] = [HIDDEN_ERROR_INVALID_ARGUMENT];
pub const HIDDEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HIDDEN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HIDDEN_ERRORS,
};
type HiddenAxesTarget = Option<(super::state::FigureHandle, usize)>;

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
            set_grid_and_minor_grid_enabled(enabled, if enabled { None } else { Some(false) });
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
    summary = "Query or set axis limits, visibility, and aspect behavior.",
    keywords = "axis,plotting",
    suppress_auto_output = true,
    type_resolver(axis_type),
    descriptor(crate::builtins::plotting::cmds::AXIS_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::cmds::AXIS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn axis_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Ok(axis_query_value(None)?);
    }
    if args.len() != 1 {
        return Err(axis_invalid_argument());
    }

    if let Value::Tensor(t) = &args[0] {
        if matches!(tensor_utils::tensor_element_len(t), 4 | 6 | 8) {
            let values = tensor_utils::tensor_values_f64(t);
            let state = current_axes_state();
            let bounds = axis_display_bounds_snapshot_for_axes(state.handle, state.active_index)
                .ok()
                .flatten()
                .unwrap_or((0.0, 1.0, 0.0, 1.0));
            let x = axis_limit_pair(values[0], values[1], bounds.0, bounds.1)?;
            let y = axis_limit_pair(values[2], values[3], bounds.2, bounds.3)?;
            set_axis_limits(Some(x), Some(y));
            if values.len() >= 6 {
                let z_auto = z_limits_snapshot().unwrap_or((0.0, 1.0));
                set_z_limits(Some(axis_limit_pair(
                    values[4], values[5], z_auto.0, z_auto.1,
                )?));
            }
            if values.len() == 8 {
                let c_auto = color_limits_snapshot().unwrap_or((0.0, 1.0));
                set_color_limits_runtime(Some(axis_limit_pair(
                    values[6], values[7], c_auto.0, c_auto.1,
                )?));
            }
            return Ok(Value::Bool(true));
        }
    }

    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&args[0], "axis") {
        return axis_query_value(Some((handle, axes_index)));
    }

    if let Some(visible) = axis_visibility_value(&args[0])? {
        set_current_axis_visibility(visible)?;
        return Ok(Value::Bool(true));
    }

    let Some(mode) = as_lower_str(&args[0]) else {
        return Err(axis_invalid_argument());
    };
    match mode.trim() {
        "equal" => {
            set_axis_equal(true);
            Ok(Value::Bool(true))
        }
        "auto" => {
            set_axis_equal_and_limits(false, None, None);
            set_z_limits(None);
            Ok(Value::Bool(true))
        }
        "tight" => {
            // Treat as auto; camera fit uses data bounds.
            set_axis_limits(None, None);
            set_z_limits(None);
            Ok(Value::Bool(true))
        }
        "image" => {
            set_axis_equal_and_limits(true, None, None);
            Ok(Value::Bool(true))
        }
        "on" => {
            set_current_axis_visibility(true)?;
            Ok(Value::Bool(true))
        }
        "off" => {
            set_current_axis_visibility(false)?;
            Ok(Value::Bool(true))
        }
        "manual" | "ij" | "xy" => {
            // These MATLAB axis modes are accepted as command tokens for compatibility.
            // The current plot scene model does not yet track direction or manual
            // limit-lock state separately from concrete limits.
            Ok(Value::Bool(true))
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

fn axis_invalid_argument() -> RuntimeError {
    cmd_error_with_message(
        "axis",
        AXIS_ERROR_INVALID_ARGUMENT.message,
        &AXIS_ERROR_INVALID_ARGUMENT,
    )
}

fn axis_limit_pair(
    lower: f64,
    upper: f64,
    automatic_lower: f64,
    automatic_upper: f64,
) -> crate::BuiltinResult<(f64, f64)> {
    let lower = if lower == f64::NEG_INFINITY {
        automatic_lower
    } else if lower.is_finite() {
        lower
    } else {
        return Err(axis_invalid_argument());
    };
    let upper = if upper == f64::INFINITY {
        automatic_upper
    } else if upper.is_finite() {
        upper
    } else {
        return Err(axis_invalid_argument());
    };
    if upper <= lower {
        return Err(axis_invalid_argument());
    }
    Ok((lower, upper))
}

fn axis_visibility_value(value: &Value) -> crate::BuiltinResult<Option<bool>> {
    let scalar = match value {
        Value::Bool(value) => return Ok(Some(*value)),
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor_utils::tensor_element_len(tensor) == 1 => {
            Some(tensor_utils::tensor_values_f64(tensor)[0])
        }
        _ => None,
    };
    match scalar {
        Some(0.0) => Ok(Some(false)),
        Some(1.0) => Ok(Some(true)),
        Some(_) => Err(axis_invalid_argument()),
        None => Ok(None),
    }
}

fn set_current_axis_visibility(visible: bool) -> crate::BuiltinResult<()> {
    let state = current_axes_state();
    let mut style = axes_metadata_snapshot(state.handle, state.active_index)
        .map_err(|_| axis_invalid_argument())?
        .axes_style;
    style.visible = visible;
    set_axes_style_for_axes(state.handle, state.active_index, style)
        .map_err(|_| axis_invalid_argument())
}

fn axis_query_value(target: Option<(FigureHandle, usize)>) -> crate::BuiltinResult<Value> {
    let state = current_axes_state();
    let (handle, axes_index) = target.unwrap_or((state.handle, state.active_index));
    let meta = axes_metadata_snapshot(handle, axes_index).map_err(|_| axis_invalid_argument())?;
    let bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
        .map_err(|_| axis_invalid_argument())?
        .unwrap_or((0.0, 1.0, 0.0, 1.0));
    let x = meta.x_limits.unwrap_or((bounds.0, bounds.1));
    let y = meta.y_limits.unwrap_or((bounds.2, bounds.3));
    let mut values = vec![x.0, x.1, y.0, y.1];
    if let Some(z) = meta.z_limits {
        values.extend_from_slice(&[z.0, z.1]);
    }
    let len = values.len();
    Ok(Value::Tensor(
        Tensor::new(values, vec![1, len]).expect("axis query row"),
    ))
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
    let [arg] = args.as_slice() else {
        return Err(cmd_error_with_message(
            "colormap",
            COLORMAP_ERROR_INVALID_ARGUMENT.message,
            &COLORMAP_ERROR_INVALID_ARGUMENT,
        ));
    };

    if let Some(name) = as_lower_str(arg) {
        let Some(cmap) = runmat_plot::plots::surface::ColorMap::from_name(&name) else {
            let other = name.trim();
            return Err(cmd_error_with_message(
                "colormap",
                format!(
                    "{}: unknown colormap '{other}'",
                    COLORMAP_ERROR_INVALID_ARGUMENT.message
                ),
                &COLORMAP_ERROR_INVALID_ARGUMENT,
            ));
        };
        set_colormap(cmap);
        return Ok(true);
    }

    if let Value::Tensor(tensor) = arg {
        let (cmap, len) = parse_rgb_colormap_tensor(tensor, "colormap")?;
        set_colormap_with_length(cmap, len);
        return Ok(true);
    };

    Err(cmd_error_with_message(
        "colormap",
        COLORMAP_ERROR_INVALID_ARGUMENT.message,
        &COLORMAP_ERROR_INVALID_ARGUMENT,
    ))
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
    name = "hidden",
    category = "plotting",
    summary = "Set axes hidden-line-removal state.",
    keywords = "hidden,hiddenline,plotting,surface,mesh",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::HIDDEN_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn hidden_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    let (target, rest) = parse_hidden_target(args)?;
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let axes = current_axes_state();
        (current_figure_handle(), axes.active_index)
    });
    let requested = parse_on_off("hidden", rest.first()).map_err(|err| {
        cmd_error_with_message(
            "hidden",
            format!(
                "{}: {}",
                HIDDEN_ERROR_INVALID_ARGUMENT.message,
                err.message()
            ),
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        )
    })?;
    if rest.len() > 1 {
        return Err(cmd_error_with_message(
            "hidden",
            "hidden: expected at most one mode argument after optional axes handle",
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        ));
    }
    let current = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| {
            cmd_error_with_message(
                "hidden",
                format!("{}: {err}", HIDDEN_ERROR_INVALID_ARGUMENT.message),
                &HIDDEN_ERROR_INVALID_ARGUMENT,
            )
        })?
        .hidden_line_removal;
    let enabled = requested.unwrap_or(!current);
    set_hidden_line_removal_for_axes(handle, axes_index, enabled).map_err(|err| {
        cmd_error_with_message(
            "hidden",
            format!("{}: {err}", HIDDEN_ERROR_INVALID_ARGUMENT.message),
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok(enabled)
}

fn parse_hidden_target(args: Vec<Value>) -> crate::BuiltinResult<(HiddenAxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, "hidden") {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
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
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::{IntegerStorage, Tensor};

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

        axis_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 6])
                .expect("six-element axis limits"),
        )])
        .unwrap();
        let zlim = get_builtin(vec![Value::Num(ax), Value::String("ZLim".into())]).unwrap();
        let zlim = Tensor::try_from(&zlim).unwrap();
        assert_eq!(zlim.materialize_f64(), vec![4.0, 5.0]);
    }

    #[test]
    fn axis_limits_read_typed_integer_storage_exactly() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();
        let limits = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![0, 10, -2, 2, 4, 5]),
            vec![1, 6],
        )
        .expect("typed limits");

        axis_builtin(vec![Value::Tensor(limits)]).unwrap();

        let xlim =
            Tensor::try_from(&get_builtin(vec![ax.clone(), Value::String("XLim".into())]).unwrap())
                .unwrap();
        let ylim =
            Tensor::try_from(&get_builtin(vec![ax.clone(), Value::String("YLim".into())]).unwrap())
                .unwrap();
        let zlim = Tensor::try_from(&get_builtin(vec![ax, Value::String("ZLim".into())]).unwrap())
            .unwrap();
        assert_eq!(xlim.materialize_f64(), vec![0.0, 10.0]);
        assert_eq!(ylim.materialize_f64(), vec![-2.0, 2.0]);
        assert_eq!(zlim.materialize_f64(), vec![4.0, 5.0]);
    }

    #[test]
    fn axis_limits_accept_all_integer_classes_and_query_double_graphics_state() {
        let _guard = setup();
        let storages = [
            IntegerStorage::I8(vec![0, 10, 1, 2]),
            IntegerStorage::I16(vec![0, 10, 1, 2]),
            IntegerStorage::I32(vec![0, 10, 1, 2]),
            IntegerStorage::I64(vec![0, 10, 1, 2]),
            IntegerStorage::U8(vec![0, 10, 1, 2]),
            IntegerStorage::U16(vec![0, 10, 1, 2]),
            IntegerStorage::U32(vec![0, 10, 1, 2]),
            IntegerStorage::U64(vec![0, 10, 1, 2]),
        ];
        for storage in storages {
            let limits = Tensor::new_integer(storage, vec![1, 4]).expect("limits");
            assert_eq!(
                axis_builtin(vec![Value::Tensor(limits)]).expect("set limits"),
                Value::Bool(true)
            );
            let Value::Tensor(actual) = axis_builtin(Vec::new()).expect("query limits") else {
                panic!("expected limit tensor");
            };
            assert_eq!(actual.materialize_f64(), vec![0.0, 10.0, 1.0, 2.0]);
            assert_eq!(actual.numeric_dtype(), runmat_builtins::NumericDType::F64);
        }
    }

    #[test]
    fn axis_supports_eight_limits_semiautomatic_endpoints_and_strict_ordering() {
        let _guard = setup();
        axis_builtin(vec![Value::Tensor(
            Tensor::new(
                vec![f64::NEG_INFINITY, 10.0, 0.0, f64::INFINITY],
                vec![1, 4],
            )
            .expect("semiautomatic limits"),
        )])
        .expect("semiautomatic axis");
        axis_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 2.0, 8.0], vec![1, 8])
                .expect("eight limits"),
        )])
        .expect("eight-element axis");
        assert_eq!(color_limits_snapshot(), Some((2.0, 8.0)));

        let equal = Tensor::new(vec![0.0, 0.0, 0.0, 1.0], vec![1, 4]).expect("equal");
        assert!(axis_builtin(vec![Value::Tensor(equal)]).is_err());
    }

    #[test]
    fn axis_visibility_accepts_logical_and_all_integer_scalar_classes() {
        let _guard = setup();
        let state = current_axes_state();
        let zeros = [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ];
        for storage in zeros {
            let scalar = Tensor::new_integer(storage, vec![1, 1]).expect("visibility");
            axis_builtin(vec![Value::Tensor(scalar)]).expect("axis off");
            assert!(
                !axes_metadata_snapshot(state.handle, state.active_index)
                    .expect("metadata")
                    .axes_style
                    .visible
            );
        }
        axis_builtin(vec![Value::Bool(true)]).expect("axis on");
        assert!(
            axes_metadata_snapshot(state.handle, state.active_index)
                .expect("metadata")
                .axes_style
                .visible
        );
    }

    #[test]
    fn axis_rejects_resident_limits_before_provider_access() {
        let _guard = setup();
        let resident = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 4],
            device_id: 0,
            buffer_id: 9_361_001,
        };
        let error = axis_builtin(vec![Value::GpuTensor(resident)])
            .expect_err("resident graphics limits must reject");
        assert_eq!(error.identifier(), Some("RunMat:axis:InvalidArgument"));
    }

    #[test]
    fn axis_accepts_common_command_modes() {
        let _guard = setup();
        for mode in [
            "equal", "auto", "tight", "image", "manual", "ij", "xy", "on", "off",
        ] {
            axis_builtin(vec![Value::String(mode.into())])
                .unwrap_or_else(|err| panic!("axis {mode} should be accepted: {err:?}"));
        }
    }

    #[test]
    fn axis_image_enables_equal_aspect_and_data_fitted_limits() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

        axis_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.0, 10.0, -2.0, 2.0], vec![1, 4]).expect("four-element axis limits"),
        )])
        .unwrap();
        axis_builtin(vec![Value::String("image".into())]).unwrap();

        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("AxisEqual".into())]).unwrap(),
            Value::Bool(true)
        );
        let xlim = get_builtin(vec![ax.clone(), Value::String("XLim".into())]).unwrap();
        let ylim = get_builtin(vec![ax, Value::String("YLim".into())]).unwrap();
        let xlim = Tensor::try_from(&xlim).unwrap();
        let ylim = Tensor::try_from(&ylim).unwrap();
        assert!(xlim.materialize_f64().iter().all(|value| value.is_nan()));
        assert!(ylim.materialize_f64().iter().all(|value| value.is_nan()));
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
    fn hidden_toggles_and_sets_current_axes_hidden_line_removal() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("HiddenLineRemoval".into())]).unwrap(),
            Value::String("on".into())
        );
        assert!(!hidden_builtin(vec![]).unwrap());
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("HiddenLineRemoval".into())]).unwrap(),
            Value::String("off".into())
        );
        assert!(hidden_builtin(vec![Value::String("on".into())]).unwrap());
        assert_eq!(
            get_builtin(vec![ax, Value::String("HiddenLineRemoval".into())]).unwrap(),
            Value::String("on".into())
        );
    }

    #[test]
    fn hidden_accepts_axes_target_without_selecting_current_axes() {
        let _guard = setup();
        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        hidden_builtin(vec![Value::Num(ax1), Value::String("off".into())]).unwrap();

        assert_eq!(
            get_builtin(vec![
                Value::Num(ax1),
                Value::String("HiddenLineRemoval".into())
            ])
            .unwrap(),
            Value::String("off".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(ax2),
                Value::String("HiddenLineRemoval".into())
            ])
            .unwrap(),
            Value::String("on".into())
        );
    }

    #[test]
    fn hidden_line_removal_property_set_get_and_rejects_invalid_values() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

        set_builtin(vec![
            ax.clone(),
            Value::String("HiddenLineRemoval".into()),
            Value::String("off".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("HiddenLineRemoval".into())]).unwrap(),
            Value::String("off".into())
        );

        let err = set_builtin(vec![
            ax,
            Value::String("HiddenLineRemoval".into()),
            Value::String("maybe".into()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:set:InvalidArgument"));
    }

    #[test]
    fn hidden_rejects_extra_arguments() {
        let _guard = setup();
        let err = hidden_builtin(vec![
            Value::String("on".into()),
            Value::String("off".into()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:hidden:InvalidArgument"));
    }

    #[test]
    fn colormap_accepts_rgb_matrix_lookup_tables() {
        let _guard = setup();
        colormap_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.2, 0.8, 0.4, 0.1, 0.6, 0.0], vec![2, 3]).expect("RGB colormap"),
        )])
        .unwrap();

        let figure = clone_figure(current_figure_handle()).expect("current figure");
        let meta = figure
            .axes_metadata(figure.active_axes_index)
            .expect("axes");
        let runmat_plot::plots::surface::ColorMap::Listed(colors) = &meta.colormap else {
            panic!("expected listed colormap");
        };
        assert_eq!(colors.as_ref(), &[[0.2, 0.4, 0.6], [0.8, 0.1, 0.0]]);
    }

    #[test]
    fn colormap_preserves_generated_parula_rows_as_listed_matrix() {
        let _guard = setup();
        let generated = crate::builtins::plotting::colormap_arrays::colormap_tensor(
            runmat_plot::plots::surface::ColorMap::Parula,
            8,
        );

        colormap_builtin(vec![Value::Tensor(generated)]).unwrap();

        let figure = clone_figure(current_figure_handle()).expect("current figure");
        let meta = figure
            .axes_metadata(figure.active_axes_index)
            .expect("axes");
        let runmat_plot::plots::surface::ColorMap::Listed(colors) = &meta.colormap else {
            panic!("expected listed colormap");
        };
        assert_eq!(colors.len(), 8);
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
        assert!(axis_labels.contains(&"lim = axis()"));
        assert!(axis_labels.contains(&"lim = axis(ax)"));
        assert!(axis_labels.contains(&"axis([xmin xmax ymin ymax | ... cmin cmax])"));
        assert!(axis_labels.contains(&"axis(mode)"));
        assert!(axis_labels.contains(&"axis(visibility)"));

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
        assert!(colormap_labels.contains(&"ok = colormap(map)"));

        let shading_labels: Vec<&str> = SHADING_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(shading_labels.contains(&"ok = shading(mode)"));

        let hidden_labels: Vec<&str> = HIDDEN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(hidden_labels.contains(&"enabled = hidden()"));
        assert!(hidden_labels.contains(&"enabled = hidden(mode)"));
        assert!(hidden_labels.contains(&"enabled = hidden(ax)"));
        assert!(hidden_labels.contains(&"enabled = hidden(ax, mode)"));

        let colorbar_labels: Vec<&str> = COLORBAR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(colorbar_labels.contains(&"enabled = colorbar()"));
        assert!(colorbar_labels.contains(&"enabled = colorbar(mode)"));
    }
}
