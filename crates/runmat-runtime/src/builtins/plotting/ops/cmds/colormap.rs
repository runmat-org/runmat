use super::error::cmd_error_with_message;
use crate::builtins::plotting::colormap_arrays::{colormap_tensor, parse_rgb_colormap_tensor};
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, clone_figure, colormap_length_for_axes, current_axes_state,
    current_colormap_length, set_colormap_for_axes_with_length, set_colormap_with_length,
    FigureHandle,
};
use crate::builtins::plotting::type_resolvers::get_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerClass, BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};
const COLORMAP_OUTPUT_MAP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cmap",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current colormap as an m-by-3 normalized double RGB matrix.",
}];
const COLORMAP_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const COLORMAP_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
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
    description: "m-by-3 single/double RGB map in [0,1], or uint8 RGB map in [0,255].",
}];
const COLORMAP_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure or axes handle.",
}];
const COLORMAP_INPUTS_TARGET_MAP: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure or axes handle.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Colormap name or m-by-3 numeric RGB map.",
    },
];
const COLORMAP_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "cmap = colormap()",
        inputs: &COLORMAP_INPUTS_NONE,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
    BuiltinSignatureDescriptor {
        label: "cmap = colormap(target)",
        inputs: &COLORMAP_INPUTS_TARGET,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
    BuiltinSignatureDescriptor {
        label: "colormap(name)",
        inputs: &COLORMAP_INPUTS_NAME,
        outputs: &COLORMAP_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "colormap(map)",
        inputs: &COLORMAP_INPUTS_RGB,
        outputs: &COLORMAP_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "colormap(target, map)",
        inputs: &COLORMAP_INPUTS_TARGET_MAP,
        outputs: &COLORMAP_OUTPUTS_NONE,
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

pub(crate) const COLORMAP_NON_UINT8_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colormap-non-uint8-integer-map",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colormap with an integer RGB map other than uint8 is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColormapNonUint8IntegerMapExtension"),
    };

pub const COLORMAP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [COLORMAP_NON_UINT8_INTEGER_EXTENSION];

const COLORMAP_UINT8_CLASSES: [BuiltinIntegerClass; 1] = [BuiltinIntegerClass::Uint8];
const COLORMAP_NON_UINT8_CLASSES: [BuiltinIntegerClass; 7] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const COLORMAP_UINT8_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "map",
    classes: &COLORMAP_UINT8_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Public uint8 RGB maps use the full [0,255] channel range and are normalized to [0,1].",
}];
const COLORMAP_NON_UINT8_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "map",
        classes: &COLORMAP_NON_UINT8_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode admits the other seven integer classes only when every RGB component is exactly 0 or 1.",
    }];

pub const COLORMAP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "colormap(uint8_map)",
        inputs: &COLORMAP_UINT8_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8 channels are divided by 255 at the explicit host graphics-state boundary; the applied map is stored as normalized double and resident inputs remain unsupported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "colormap(non_uint8_integer_map)",
        inputs: &COLORMAP_NON_UINT8_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "This useful RunMat-only form validates exact native 0/1 components before conversion to normalized host graphics metadata; resident inputs reject without provider access.",
    },
];

#[runtime_builtin(
    name = "colormap",
    category = "plotting",
    summary = "Set the active colormap.",
    keywords = "colormap,plotting",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::cmds::COLORMAP_DESCRIPTOR),
    extensions(crate::builtins::plotting::cmds::COLORMAP_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::cmds::COLORMAP_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colormap_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        let state = current_axes_state();
        return colormap_query(state.handle, state.active_index);
    }
    if args.len() == 1 {
        let typed_integer_map = matches!(&args[0], Value::Int(_))
            || matches!(&args[0], Value::Tensor(tensor) if tensor.integer_storage().is_some());
        if !typed_integer_map {
            if let Ok(target) = resolve_plot_handle(&args[0], "colormap") {
                return match target {
                    PlotHandle::Axes(handle, axes) => colormap_query(handle, axes),
                    PlotHandle::Figure(handle) => {
                        let axes = clone_figure(handle)
                            .map(|figure| figure.active_axes_index)
                            .unwrap_or(0);
                        colormap_query(handle, axes)
                    }
                    _ => Err(cmd_error_with_message(
                        "colormap",
                        "target must be a figure or axes handle",
                        &COLORMAP_ERROR_INVALID_ARGUMENT,
                    )),
                };
            }
        }
        let (cmap, len) = colormap_parse_map(&args[0])?;
        let output = Value::Tensor(colormap_tensor(cmap.clone(), len));
        set_colormap_with_length(cmap, len);
        return Ok(output);
    }
    if args.len() == 2 {
        if matches!(&args[0], Value::Int(_))
            || matches!(&args[0], Value::Tensor(tensor) if tensor.integer_storage().is_some())
        {
            return Err(cmd_error_with_message(
                "colormap",
                "target must be a figure or axes graphics handle, not a typed integer",
                &COLORMAP_ERROR_INVALID_ARGUMENT,
            ));
        }
        let target = resolve_plot_handle(&args[0], "colormap")?;
        let (cmap, len) = colormap_parse_map(&args[1])?;
        let output = Value::Tensor(colormap_tensor(cmap.clone(), len));
        match target {
            PlotHandle::Axes(handle, axes) => {
                set_colormap_for_axes_with_length(handle, axes, cmap, len).map_err(|err| {
                    cmd_error_with_message(
                        "colormap",
                        format!("{}: {err}", COLORMAP_ERROR_INVALID_ARGUMENT.message),
                        &COLORMAP_ERROR_INVALID_ARGUMENT,
                    )
                })?;
            }
            PlotHandle::Figure(handle) => {
                let count = clone_figure(handle)
                    .map(|figure| figure.axes_count())
                    .unwrap_or(1)
                    .max(1);
                for axes in 0..count {
                    set_colormap_for_axes_with_length(handle, axes, cmap.clone(), len).map_err(
                        |err| {
                            cmd_error_with_message(
                                "colormap",
                                format!("{}: {err}", COLORMAP_ERROR_INVALID_ARGUMENT.message),
                                &COLORMAP_ERROR_INVALID_ARGUMENT,
                            )
                        },
                    )?;
                }
            }
            _ => {
                return Err(cmd_error_with_message(
                    "colormap",
                    "target must be a figure or axes handle",
                    &COLORMAP_ERROR_INVALID_ARGUMENT,
                ))
            }
        }
        return Ok(output);
    }
    Err(cmd_error_with_message(
        "colormap",
        "expected colormap(), colormap(map), colormap(target), or colormap(target, map)",
        &COLORMAP_ERROR_INVALID_ARGUMENT,
    ))
}

fn colormap_query(handle: FigureHandle, axes: usize) -> crate::BuiltinResult<Value> {
    let metadata = axes_metadata_snapshot(handle, axes).map_err(|err| {
        cmd_error_with_message(
            "colormap",
            format!("{}: {err}", COLORMAP_ERROR_INVALID_ARGUMENT.message),
            &COLORMAP_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok(Value::Tensor(colormap_tensor(
        metadata.colormap,
        colormap_length_for_axes(handle, axes),
    )))
}

fn colormap_parse_map(
    arg: &Value,
) -> crate::BuiltinResult<(runmat_plot::plots::surface::ColorMap, usize)> {
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
        let len = current_colormap_length();
        return Ok((cmap, len));
    }

    if let Value::Tensor(tensor) = arg {
        if tensor.integer_storage().is_some() && tensor.numeric_dtype() != NumericDType::U8 {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COLORMAP_NON_UINT8_INTEGER_EXTENSION,
                "colormap",
            )?;
        }
        return parse_rgb_colormap_tensor(tensor, "colormap");
    };

    Err(cmd_error_with_message(
        "colormap",
        COLORMAP_ERROR_INVALID_ARGUMENT.message,
        &COLORMAP_ERROR_INVALID_ARGUMENT,
    ))
}
#[cfg(test)]
#[path = "colormap/tests.rs"]
mod tests;
