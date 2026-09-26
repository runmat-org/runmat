use super::error::cmd_error_with_message;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, axis_display_bounds_snapshot_for_axes, current_axes_state,
    current_figure_handle_if_exists, set_axes_style_for_axes, set_axis_equal,
    set_axis_equal_and_limits, set_axis_limits, set_color_limits_runtime, set_z_limits,
    FigureHandle,
};
use crate::builtins::plotting::type_resolvers::axis_type;
use crate::RuntimeError;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
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
        return axis_query_value(None);
    }
    if args.len() != 1 {
        return Err(axis_invalid_argument());
    }

    if let Value::Tensor(t) = &args[0] {
        if matches!(tensor_utils::tensor_element_len(t), 4 | 6 | 8) {
            let values = tensor_utils::tensor_values_f64(t);
            let state = current_figure_handle_if_exists().map(|_| current_axes_state());
            let bounds = state
                .as_ref()
                .and_then(|state| {
                    axis_display_bounds_snapshot_for_axes(state.handle, state.active_index)
                        .ok()
                        .flatten()
                })
                .unwrap_or((0.0, 1.0, 0.0, 1.0));
            let meta = state
                .as_ref()
                .and_then(|state| axes_metadata_snapshot(state.handle, state.active_index).ok());
            let x = axis_limit_pair(values[0], values[1], bounds.0, bounds.1)?;
            let y = axis_limit_pair(values[2], values[3], bounds.2, bounds.3)?;
            let z = if values.len() >= 6 {
                let z_auto = meta
                    .as_ref()
                    .and_then(|meta| meta.z_limits)
                    .unwrap_or((0.0, 1.0));
                Some(axis_limit_pair(values[4], values[5], z_auto.0, z_auto.1)?)
            } else {
                None
            };
            let c = if values.len() == 8 {
                let c_auto = meta
                    .as_ref()
                    .and_then(|meta| meta.color_limits)
                    .unwrap_or((0.0, 1.0));
                Some(axis_limit_pair(values[6], values[7], c_auto.0, c_auto.1)?)
            } else {
                None
            };
            // Admit every requested pair before publishing any limit change.
            set_axis_limits(Some(x), Some(y));
            if let Some(z) = z {
                set_z_limits(Some(z));
            }
            if let Some(c) = c {
                set_color_limits_runtime(Some(c));
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
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let state = current_axes_state();
        (state.handle, state.active_index)
    });
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
#[cfg(test)]
#[path = "axis/tests.rs"]
mod tests;
