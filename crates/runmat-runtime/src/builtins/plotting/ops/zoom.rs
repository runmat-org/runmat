use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    HandleRef, ObjectInstance, StructValue, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    apply_zoom_factor_for_axes, apply_zoom_factor_for_figure, current_figure_handle,
    figure_handle_exists, reset_zoom_baseline_for_axes, reset_zoom_baseline_for_figure,
    restore_zoom_baseline_for_axes, restore_zoom_baseline_for_figure, select_figure,
    set_zoom_direction_for_figure, set_zoom_enabled_for_figure, set_zoom_legacy_for_figure,
    set_zoom_mode_for_axes, set_zoom_mode_for_figure, set_zoom_motion_for_figure,
    set_zoom_right_click_action_for_figure, zoom_state_snapshot, FigureHandle, ZoomDirection,
    ZoomModeCommand, ZoomMotion, ZoomRightClickAction, ZoomStateSnapshot,
};
use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "zoom";
const ZOOM_CLASS_NAME: &str = "matlab.graphics.interaction.internal.zoom";

const ZOOM_INTEGER_TARGET_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "zoom-integer-target-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded figure or axes handle",
    error_identifier: Some("RunMat:compatibility:ZoomIntegerTargetHandleExtension"),
};
pub const ZOOM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ZOOM_INTEGER_TARGET_EXTENSION];
const ZOOM_INTEGER_FACTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "factor",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive numeric zoom factor admits positive integer scalars.",
    }];
const ZOOM_INTEGER_TARGET_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "target",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for RunMat's encoded graphics handles are separately gated.",
    }];
pub const ZOOM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "zoom(integer_factor)",
        inputs: &ZOOM_INTEGER_FACTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positive finite factor crosses the graphics scaling boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "zoom(integer_target, option_or_factor)",
        inputs: &ZOOM_INTEGER_TARGET_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict mode rejects the encoded-handle alias before graphics state access.",
    },
];

const ZOOM_OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zoom mode object for the current or specified figure.",
}];

const ZOOM_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns \"ok\" when the zoom command succeeds.",
}];

const ZOOM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const ZOOM_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zoom option: on, off, toggle, xon, yon, reset, or out.",
}];

const ZOOM_INPUTS_FACTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "factor",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive numeric zoom factor.",
}];

const ZOOM_INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure handle.",
}];

const ZOOM_INPUTS_TARGET_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Figure or axes handle.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zoom option or factor.",
    },
];

const ZOOM_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "z = zoom()",
        inputs: &ZOOM_INPUTS_NONE,
        outputs: &ZOOM_OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "zoom()",
        inputs: &ZOOM_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = zoom(option)",
        inputs: &ZOOM_INPUTS_OPTION,
        outputs: &ZOOM_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = zoom(factor)",
        inputs: &ZOOM_INPUTS_FACTOR,
        outputs: &ZOOM_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = zoom(target, option)",
        inputs: &ZOOM_INPUTS_TARGET_OPTION,
        outputs: &ZOOM_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "z = zoom(fig)",
        inputs: &ZOOM_INPUTS_FIGURE,
        outputs: &ZOOM_OUTPUT_OBJECT,
    },
];

const ZOOM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZOOM.INVALID_ARGUMENT",
    identifier: Some("RunMat:zoom:InvalidArgument"),
    when: "Argument count, target handle, factor, option, or zoom property is invalid.",
    message: "zoom: invalid argument",
};

const ZOOM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZOOM.INTERNAL",
    identifier: Some("RunMat:zoom:Internal"),
    when: "Internal zoom state or handle-object update fails.",
    message: "zoom: internal operation failed",
};

const ZOOM_ERRORS: [BuiltinErrorDescriptor; 2] = [ZOOM_ERROR_INVALID_ARGUMENT, ZOOM_ERROR_INTERNAL];

pub const ZOOM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZOOM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZOOM_ERRORS,
};

fn zoom_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    zoom_error(&ZOOM_ERROR_INVALID_ARGUMENT, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    zoom_error(&ZOOM_ERROR_INTERNAL, detail)
}

fn requested_outputs() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn status() -> Value {
    Value::String("ok".into())
}

fn ensure_current_zoom_figure() -> FigureHandle {
    let handle = current_figure_handle();
    if !figure_handle_exists(handle) {
        select_figure(handle);
    }
    handle
}

#[runtime_builtin(
    name = "zoom",
    category = "plotting",
    summary = "Control interactive and programmatic axes zooming.",
    keywords = "zoom,plotting,axes,figure,pan,interaction",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::zoom::ZOOM_DESCRIPTOR),
    extensions(crate::builtins::plotting::zoom::ZOOM_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::zoom::ZOOM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::zoom"
)]
pub fn zoom_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ZOOM_INTEGER_TARGET_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match args.as_slice() {
        [] => {
            let handle = ensure_current_zoom_figure();
            if requested_outputs() == 0 {
                set_zoom_mode_for_figure(handle, ZoomModeCommand::Toggle)
                    .map_err(|err| invalid(err.to_string()))?;
                Ok(status())
            } else {
                let snapshot =
                    zoom_state_snapshot(handle, None).map_err(|err| invalid(err.to_string()))?;
                zoom_object_from_snapshot(snapshot)
            }
        }
        [single] => {
            if let Some(option) = super::style::value_as_string(single) {
                apply_option_to_figure(ensure_current_zoom_figure(), &option)?;
                return Ok(status());
            }
            let factor = numeric_scalar(single)
                .ok_or_else(|| invalid("expected an option string or positive zoom factor"))?;
            if requested_outputs() > 0
                && !crate::builtins::common::validation::value_has_native_integer_class(single)
            {
                let figure = FigureHandle::from(rounded_handle_id(factor)?);
                if let Ok(snapshot) = zoom_state_snapshot(figure, None) {
                    return zoom_object_from_snapshot(snapshot);
                }
            }
            validate_factor(factor)?;
            apply_zoom_factor_for_figure(ensure_current_zoom_figure(), factor)
                .map_err(|err| invalid(err.to_string()))?;
            Ok(status())
        }
        [target, command] => {
            let target = resolve_zoom_target(target)?;
            if let Some(option) = super::style::value_as_string(command) {
                apply_option_to_target(target, &option)?;
                return Ok(status());
            }
            let factor = numeric_scalar(command)
                .ok_or_else(|| invalid("expected an option string or positive zoom factor"))?;
            validate_factor(factor)?;
            match target {
                ZoomTarget::Figure(handle) => apply_zoom_factor_for_figure(handle, factor)
                    .map_err(|err| invalid(err.to_string()))?,
                ZoomTarget::Axes(handle, axes) => apply_zoom_factor_for_axes(handle, axes, factor)
                    .map_err(|err| invalid(err.to_string()))?,
            }
            Ok(status())
        }
        _ => Err(invalid(
            "expected zoom(), zoom(option), zoom(factor), or zoom(target, option)",
        )),
    }
}

#[derive(Clone, Copy, Debug)]
enum ZoomTarget {
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
}

fn resolve_zoom_target(value: &Value) -> BuiltinResult<ZoomTarget> {
    match resolve_plot_handle(value, BUILTIN_NAME).map_err(|err| invalid(err.message))? {
        PlotHandle::Figure(handle) => Ok(ZoomTarget::Figure(handle)),
        PlotHandle::Axes(handle, axes) => Ok(ZoomTarget::Axes(handle, axes)),
        _ => Err(invalid("target must be a figure or axes handle")),
    }
}

fn numeric_scalar(value: &Value) -> Option<f64> {
    super::style::value_as_f64(value)
}

fn rounded_handle_id(value: f64) -> BuiltinResult<u32> {
    if !value.is_finite() || value < 1.0 || value > u32::MAX as f64 {
        return Err(invalid("figure handle must be a positive finite scalar"));
    }
    Ok(value.round() as u32)
}

fn validate_factor(value: f64) -> BuiltinResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid("factor must be a positive finite scalar"))
    }
}

fn apply_option_to_figure(handle: FigureHandle, option: &str) -> BuiltinResult<()> {
    match parse_option(option)? {
        ZoomOption::Mode(command) => {
            set_zoom_mode_for_figure(handle, command).map_err(|err| invalid(err.to_string()))
        }
        ZoomOption::Reset => {
            reset_zoom_baseline_for_figure(handle).map_err(|err| invalid(err.to_string()))
        }
        ZoomOption::Out => {
            restore_zoom_baseline_for_figure(handle).map_err(|err| invalid(err.to_string()))
        }
    }
}

fn apply_option_to_target(target: ZoomTarget, option: &str) -> BuiltinResult<()> {
    match (target, parse_option(option)?) {
        (ZoomTarget::Figure(handle), ZoomOption::Mode(command)) => {
            set_zoom_mode_for_figure(handle, command).map_err(|err| invalid(err.to_string()))
        }
        (ZoomTarget::Axes(handle, axes), ZoomOption::Mode(command)) => {
            set_zoom_mode_for_axes(handle, axes, command).map_err(|err| invalid(err.to_string()))
        }
        (ZoomTarget::Figure(handle), ZoomOption::Reset) => {
            reset_zoom_baseline_for_figure(handle).map_err(|err| invalid(err.to_string()))
        }
        (ZoomTarget::Axes(handle, axes), ZoomOption::Reset) => {
            reset_zoom_baseline_for_axes(handle, axes).map_err(|err| invalid(err.to_string()))
        }
        (ZoomTarget::Figure(handle), ZoomOption::Out) => {
            restore_zoom_baseline_for_figure(handle).map_err(|err| invalid(err.to_string()))
        }
        (ZoomTarget::Axes(handle, axes), ZoomOption::Out) => {
            restore_zoom_baseline_for_axes(handle, axes).map_err(|err| invalid(err.to_string()))
        }
    }
}

enum ZoomOption {
    Mode(ZoomModeCommand),
    Reset,
    Out,
}

fn parse_option(option: &str) -> BuiltinResult<ZoomOption> {
    match option.trim().to_ascii_lowercase().as_str() {
        "on" => Ok(ZoomOption::Mode(ZoomModeCommand::On)),
        "off" => Ok(ZoomOption::Mode(ZoomModeCommand::Off)),
        "toggle" => Ok(ZoomOption::Mode(ZoomModeCommand::Toggle)),
        "xon" => Ok(ZoomOption::Mode(ZoomModeCommand::XOn)),
        "yon" => Ok(ZoomOption::Mode(ZoomModeCommand::YOn)),
        "reset" => Ok(ZoomOption::Reset),
        "out" => Ok(ZoomOption::Out),
        other => Err(invalid(format!("unsupported option '{other}'"))),
    }
}

fn zoom_object_from_snapshot(snapshot: ZoomStateSnapshot) -> BuiltinResult<Value> {
    let mut object = ObjectInstance::new(ZOOM_CLASS_NAME.to_string());
    update_object_properties(&mut object, snapshot);
    let target =
        runmat_gc::gc_allocate(Value::Object(object)).map_err(|err| internal(err.to_string()))?;
    Ok(Value::HandleObject(HandleRef {
        class_name: ZOOM_CLASS_NAME.to_string(),
        target,
        valid: true,
    }))
}

fn update_object_properties(object: &mut ObjectInstance, snapshot: ZoomStateSnapshot) {
    object.properties.insert(
        "Enable".to_string(),
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    object.properties.insert(
        "Motion".to_string(),
        Value::String(snapshot.mode.motion.as_str().into()),
    );
    object.properties.insert(
        "Direction".to_string(),
        Value::String(snapshot.mode.direction.as_str().into()),
    );
    object.properties.insert(
        "RightClickAction".to_string(),
        Value::String(snapshot.mode.right_click_action.as_str().into()),
    );
    object
        .properties
        .entry("ContextMenu".to_string())
        .or_insert(Value::Num(0.0));
    object
        .properties
        .entry("UIContextMenu".to_string())
        .or_insert(Value::Num(0.0));
    object
        .properties
        .entry("ButtonDownFilter".to_string())
        .or_insert(Value::String(String::new()));
    object
        .properties
        .entry("ActionPreCallback".to_string())
        .or_insert(Value::String(String::new()));
    object
        .properties
        .entry("ActionPostCallback".to_string())
        .or_insert(Value::String(String::new()));
    object.properties.insert(
        "FigureHandle".to_string(),
        Value::Num(snapshot.figure.as_u32() as f64),
    );
    object.properties.insert(
        "AxesIndex".to_string(),
        snapshot
            .axes_index
            .map(|index| Value::Num((index + 1) as f64))
            .unwrap_or(Value::Num(0.0)),
    );
    object.properties.insert(
        "UseLegacyExplorationModes".to_string(),
        Value::String(
            if snapshot.mode.use_legacy_exploration_modes {
                "on"
            } else {
                "off"
            }
            .into(),
        ),
    );
}

fn read_zoom_handle(value: &Value) -> Option<&HandleRef> {
    match value {
        Value::HandleObject(handle)
            if handle.class_name == ZOOM_CLASS_NAME
                && handle.valid
                && crate::is_handle_valid(handle) =>
        {
            Some(handle)
        }
        _ => None,
    }
}

fn snapshot_from_object(handle: &HandleRef) -> BuiltinResult<ZoomStateSnapshot> {
    let (figure, axes_index) = runmat_gc::gc_with_value(&handle.target, |value| {
        let Value::Object(object) = value else {
            return None;
        };
        if object.class_name != ZOOM_CLASS_NAME {
            return None;
        }
        let figure = object
            .properties
            .get("FigureHandle")
            .and_then(numeric_scalar)?;
        let axes_index = object
            .properties
            .get("AxesIndex")
            .and_then(numeric_scalar)
            .and_then(|value| {
                if value >= 1.0 {
                    Some(value.round() as usize - 1)
                } else {
                    None
                }
            });
        Some((FigureHandle::from(figure.round() as u32), axes_index))
    })
    .map_err(|err| internal(err.to_string()))?
    .ok_or_else(|| invalid("invalid zoom object"))?;
    zoom_state_snapshot(figure, axes_index).map_err(|err| invalid(err.to_string()))
}

fn zoom_struct(snapshot: ZoomStateSnapshot) -> Value {
    let mut st = StructValue::new();
    st.insert(
        "Enable",
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    st.insert(
        "Motion",
        Value::String(snapshot.mode.motion.as_str().into()),
    );
    st.insert(
        "Direction",
        Value::String(snapshot.mode.direction.as_str().into()),
    );
    st.insert(
        "RightClickAction",
        Value::String(snapshot.mode.right_click_action.as_str().into()),
    );
    st.insert("ContextMenu", Value::Num(0.0));
    st.insert("UIContextMenu", Value::Num(0.0));
    st.insert("ButtonDownFilter", Value::String(String::new()));
    st.insert("ActionPreCallback", Value::String(String::new()));
    st.insert("ActionPostCallback", Value::String(String::new()));
    st.insert("FigureHandle", Value::Num(snapshot.figure.as_u32() as f64));
    st.insert(
        "UseLegacyExplorationModes",
        Value::String(
            if snapshot.mode.use_legacy_exploration_modes {
                "on"
            } else {
                "off"
            }
            .into(),
        ),
    );
    Value::Struct(st)
}

pub fn get_zoom_object_property(
    value: &Value,
    property: Option<&str>,
    _builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let Some(handle) = read_zoom_handle(value) else {
        return Ok(None);
    };
    let snapshot = snapshot_from_object(handle)?;
    let value = match property {
        None => {
            let mut value = zoom_struct(snapshot);
            if let Value::Struct(st) = &mut value {
                for (name, default) in [
                    ("ContextMenu", Value::Num(0.0)),
                    ("UIContextMenu", Value::Num(0.0)),
                    ("ButtonDownFilter", Value::String(String::new())),
                    ("ActionPreCallback", Value::String(String::new())),
                    ("ActionPostCallback", Value::String(String::new())),
                ] {
                    st.insert(name, object_property_or_default(handle, name, default)?);
                }
            }
            value
        }
        Some(name) => match canonical_property(name).as_deref() {
            Some("Enable") => {
                Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into())
            }
            Some("Motion") => Value::String(snapshot.mode.motion.as_str().into()),
            Some("Direction") => Value::String(snapshot.mode.direction.as_str().into()),
            Some("RightClickAction") => {
                Value::String(snapshot.mode.right_click_action.as_str().into())
            }
            Some("ContextMenu") | Some("UIContextMenu") => {
                object_property_or_default(handle, name, Value::Num(0.0))?
            }
            Some("ButtonDownFilter") | Some("ActionPreCallback") | Some("ActionPostCallback") => {
                object_property_or_default(handle, name, Value::String(String::new()))?
            }
            Some("FigureHandle") => Value::Num(snapshot.figure.as_u32() as f64),
            Some("UseLegacyExplorationModes") => Value::String(
                if snapshot.mode.use_legacy_exploration_modes {
                    "on"
                } else {
                    "off"
                }
                .into(),
            ),
            _ => return Err(invalid(format!("unsupported zoom property '{name}'"))),
        },
    };
    Ok(Some(value))
}

fn object_property_or_default(
    handle: &HandleRef,
    name: &str,
    default: Value,
) -> BuiltinResult<Value> {
    let key = canonical_property(name).unwrap_or_else(|| name.to_string());
    runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) => object.properties.get(&key).cloned().unwrap_or(default),
        _ => default,
    })
    .map_err(|err| internal(err.to_string()))
}

pub fn set_zoom_object_properties(
    value: &Value,
    args: &[Value],
    _builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(handle) = read_zoom_handle(value).cloned() else {
        return Ok(None);
    };
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(invalid("expected zoom property/value pairs"));
    }
    let snapshot = snapshot_from_object(&handle)?;
    let mut idx = 0;
    while idx < args.len() {
        let name = super::style::value_as_string(&args[idx])
            .ok_or_else(|| invalid("zoom property names must be text"))?;
        let value = &args[idx + 1];
        apply_zoom_property(snapshot.figure, &name, value)?;
        write_object_property(&handle, &name, value.clone())?;
        idx += 2;
    }
    let refreshed = snapshot_from_object(&handle)?;
    runmat_gc::gc_with_value_mut(&handle.target, |target| {
        if let Value::Object(object) = target {
            update_object_properties(object, refreshed);
        }
    })
    .map_err(|err| internal(err.to_string()))?;
    Ok(Some(()))
}

fn write_object_property(handle: &HandleRef, name: &str, value: Value) -> BuiltinResult<()> {
    let key = canonical_property(name).unwrap_or_else(|| name.to_string());
    runmat_gc::gc_with_value_mut(&handle.target, |target| {
        if let Value::Object(object) = target {
            object.properties.insert(key, value);
        }
    })
    .map_err(|err| internal(err.to_string()))
}

fn apply_zoom_property(figure: FigureHandle, name: &str, value: &Value) -> BuiltinResult<()> {
    match canonical_property(name).as_deref() {
        Some("Enable") => {
            let enabled = parse_on_off(value, "Enable")?;
            set_zoom_enabled_for_figure(figure, enabled).map_err(|err| invalid(err.to_string()))
        }
        Some("Motion") => set_zoom_motion_for_figure(figure, parse_motion(value)?)
            .map_err(|err| invalid(err.to_string())),
        Some("Direction") => set_zoom_direction_for_figure(figure, parse_direction(value)?)
            .map_err(|err| invalid(err.to_string())),
        Some("RightClickAction") => {
            set_zoom_right_click_action_for_figure(figure, parse_right_click_action(value)?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some("UseLegacyExplorationModes") => {
            set_zoom_legacy_for_figure(figure, parse_on_off(value, "UseLegacyExplorationModes")?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some("FigureHandle") | Some("AxesIndex") => Err(invalid(format!("{name} is read-only"))),
        Some("ContextMenu")
        | Some("UIContextMenu")
        | Some("ButtonDownFilter")
        | Some("ActionPreCallback")
        | Some("ActionPostCallback") => Ok(()),
        _ => Err(invalid(format!("unsupported zoom property '{name}'"))),
    }
}

fn parse_on_off(value: &Value, property: &str) -> BuiltinResult<bool> {
    super::style::value_as_bool(value)
        .ok_or_else(|| invalid(format!("{property} must be 'on' or 'off'")))
}

fn parse_motion(value: &Value) -> BuiltinResult<ZoomMotion> {
    let text = super::style::value_as_string(value)
        .ok_or_else(|| invalid("Motion must be 'both', 'horizontal', or 'vertical'"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "both" => Ok(ZoomMotion::Both),
        "horizontal" => Ok(ZoomMotion::Horizontal),
        "vertical" => Ok(ZoomMotion::Vertical),
        _ => Err(invalid(
            "Motion must be 'both', 'horizontal', or 'vertical'",
        )),
    }
}

fn parse_direction(value: &Value) -> BuiltinResult<ZoomDirection> {
    let text = super::style::value_as_string(value)
        .ok_or_else(|| invalid("Direction must be 'in' or 'out'"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "in" => Ok(ZoomDirection::In),
        "out" => Ok(ZoomDirection::Out),
        _ => Err(invalid("Direction must be 'in' or 'out'")),
    }
}

fn parse_right_click_action(value: &Value) -> BuiltinResult<ZoomRightClickAction> {
    let text = super::style::value_as_string(value)
        .ok_or_else(|| invalid("RightClickAction must be 'PostContextMenu' or 'InverseZoom'"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "postcontextmenu" => Ok(ZoomRightClickAction::PostContextMenu),
        "inversezoom" => Ok(ZoomRightClickAction::InverseZoom),
        _ => Err(invalid(
            "RightClickAction must be 'PostContextMenu' or 'InverseZoom'",
        )),
    }
}

fn canonical_property(name: &str) -> Option<String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some("Enable".into()),
        "motion" => Some("Motion".into()),
        "direction" => Some("Direction".into()),
        "rightclickaction" => Some("RightClickAction".into()),
        "contextmenu" => Some("ContextMenu".into()),
        "uicontextmenu" => Some("UIContextMenu".into()),
        "buttondownfilter" => Some("ButtonDownFilter".into()),
        "actionprecallback" => Some("ActionPreCallback".into()),
        "actionpostcallback" => Some("ActionPostCallback".into()),
        "figurehandle" => Some("FigureHandle".into()),
        "axesindex" => Some("AxesIndex".into()),
        "uselegacyexplorationmodes" => Some("UseLegacyExplorationModes".into()),
        _ => None,
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
    use runmat_builtins::IntValue;

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn zoom_option_toggles_mode_and_returns_live_object_state() {
        let _guard = setup_plot_tests();
        zoom_builtin(vec![Value::String("on".into())]).unwrap();

        let z = zoom_builtin(Vec::new()).unwrap();
        assert_eq!(
            get_builtin(vec![z.clone(), Value::String("Enable".into())]).unwrap(),
            Value::String("on".into())
        );

        set_builtin(vec![
            z.clone(),
            Value::String("Motion".into()),
            Value::String("horizontal".into()),
            Value::String("Direction".into()),
            Value::String("out".into()),
            Value::String("RightClickAction".into()),
            Value::String("InverseZoom".into()),
            Value::String("ActionPreCallback".into()),
            Value::String("pre".into()),
        ])
        .unwrap();

        assert_eq!(
            get_builtin(vec![z.clone(), Value::String("Motion".into())]).unwrap(),
            Value::String("horizontal".into())
        );
        assert_eq!(
            get_builtin(vec![z, Value::String("RightClickAction".into())]).unwrap(),
            Value::String("InverseZoom".into())
        );
        let z = zoom_builtin(Vec::new()).unwrap();
        set_builtin(vec![
            z.clone(),
            Value::String("ActionPreCallback".into()),
            Value::String("pre".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![z, Value::String("ActionPreCallback".into())]).unwrap(),
            Value::String("pre".into())
        );

        let z = zoom_builtin(Vec::new()).unwrap();
        set_builtin(vec![
            z.clone(),
            Value::String("Motion".into()),
            Value::String("horizontal".into()),
            Value::String("Enable".into()),
            Value::String("on".into()),
            Value::String("ContextMenu".into()),
            Value::Num(123.0),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![z.clone(), Value::String("Motion".into())]).unwrap(),
            Value::String("horizontal".into())
        );
        assert_eq!(
            get_builtin(vec![z, Value::String("ContextMenu".into())]).unwrap(),
            Value::Num(123.0)
        );
    }

    #[test]
    fn zoom_factor_updates_current_axes_limits_directly() {
        let _guard = setup_plot_tests();
        crate::builtins::plotting::state::set_axis_limits(Some((0.0, 10.0)), Some((0.0, 20.0)));

        let _outputs = crate::output_count::push_output_count(Some(0));
        zoom_builtin(vec![Value::Num(2.0)]).unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_limits, Some((2.5, 7.5)));
        assert_eq!(meta.y_limits, Some((5.0, 15.0)));
    }

    #[test]
    fn zoom_reset_and_out_restore_recorded_baseline() {
        let _guard = setup_plot_tests();
        crate::builtins::plotting::state::set_axis_limits(Some((0.0, 10.0)), Some((0.0, 20.0)));
        zoom_builtin(vec![Value::String("reset".into())]).unwrap();

        crate::builtins::plotting::state::set_axis_limits(Some((1.0, 2.0)), Some((3.0, 4.0)));
        zoom_builtin(vec![Value::String("out".into())]).unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_limits, Some((0.0, 10.0)));
        assert_eq!(meta.y_limits, Some((0.0, 20.0)));
    }

    #[test]
    fn zoom_explicit_axes_does_not_change_current_axes() {
        let _guard = setup_plot_tests();
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
        let (figure, axes_index) =
            crate::builtins::plotting::state::decode_axes_handle(ax1).expect("axes handle");
        crate::builtins::plotting::state::set_axis_limits_for_axes(
            figure,
            axes_index,
            Some((0.0, 10.0)),
            Some((0.0, 20.0)),
        )
        .unwrap();
        crate::builtins::plotting::state::select_axes_for_figure(figure, 1).unwrap();

        let _outputs = crate::output_count::push_output_count(Some(0));
        zoom_builtin(vec![Value::Num(ax1), Value::Num(2.0)]).unwrap();

        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );
        let fig = clone_figure(figure).unwrap();
        let first = fig.axes_metadata(0).unwrap();
        let second = fig.axes_metadata(1).unwrap();
        assert_eq!(first.x_limits, Some((2.5, 7.5)));
        assert_eq!(first.y_limits, Some((5.0, 15.0)));
        assert_eq!(second.x_limits, None);
        assert_eq!(second.y_limits, None);
    }

    #[test]
    fn zoom_descriptor_covers_object_and_statement_forms() {
        let labels: Vec<&str> = ZOOM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"z = zoom()"));
        assert!(labels.contains(&"zoom()"));
        assert!(labels.contains(&"status = zoom(factor)"));
        assert!(labels.contains(&"status = zoom(target, option)"));
    }

    #[test]
    fn typed_integer_single_argument_is_a_zoom_factor() {
        let _guard = setup_plot_tests();
        crate::builtins::plotting::state::set_axis_limits(Some((0.0, 10.0)), Some((0.0, 20.0)));
        let _outputs = crate::output_count::push_output_count(Some(0));
        zoom_builtin(vec![Value::Int(IntValue::U8(2))]).expect("integer factor");
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_limits, Some((2.5, 7.5)));
        assert_eq!(meta.y_limits, Some((5.0, 15.0)));
    }
}
