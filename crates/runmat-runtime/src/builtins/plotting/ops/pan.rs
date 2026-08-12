//! MATLAB-compatible `pan` interaction-mode builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, ObjectInstance, StructValue, Value};

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    current_figure_handle, figure_handle_exists, pan_state_snapshot, select_figure,
    set_pan_enabled_for_figure, set_pan_mode_for_axes, set_pan_mode_for_figure,
    set_pan_motion_for_figure, FigureHandle, PanModeCommand, PanStateSnapshot, ZoomMotion,
};
use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pan";
const PAN_CLASS_NAME: &str = "matlab.graphics.interaction.internal.pan";

const PAN_OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pan mode object for the current or specified figure.",
}];

const PAN_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns \"ok\" when the pan command succeeds.",
}];

const PAN_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const PAN_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pan option: on, off, toggle, xon, or yon.",
}];

const PAN_INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure handle.",
}];

const PAN_INPUTS_TARGET_OPTION: [BuiltinParamDescriptor; 2] = [
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
        description: "Pan option.",
    },
];

const PAN_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "p = pan()",
        inputs: &PAN_INPUTS_NONE,
        outputs: &PAN_OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "pan()",
        inputs: &PAN_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = pan(option)",
        inputs: &PAN_INPUTS_OPTION,
        outputs: &PAN_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = pan(target, option)",
        inputs: &PAN_INPUTS_TARGET_OPTION,
        outputs: &PAN_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "p = pan(fig)",
        inputs: &PAN_INPUTS_FIGURE,
        outputs: &PAN_OUTPUT_OBJECT,
    },
];

const PAN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAN.INVALID_ARGUMENT",
    identifier: Some("RunMat:pan:InvalidArgument"),
    when: "Argument count, target handle, option, or pan property is invalid.",
    message: "pan: invalid argument",
};

const PAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAN.INTERNAL",
    identifier: Some("RunMat:pan:Internal"),
    when: "Internal pan state or handle-object update fails.",
    message: "pan: internal operation failed",
};

const PAN_ERRORS: [BuiltinErrorDescriptor; 2] = [PAN_ERROR_INVALID_ARGUMENT, PAN_ERROR_INTERNAL];

pub const PAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAN_ERRORS,
};

fn pan_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    pan_error(&PAN_ERROR_INVALID_ARGUMENT, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    pan_error(&PAN_ERROR_INTERNAL, detail)
}

fn requested_outputs() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn status() -> Value {
    Value::String("ok".into())
}

fn ensure_current_pan_figure() -> FigureHandle {
    let handle = current_figure_handle();
    if !figure_handle_exists(handle) {
        select_figure(handle);
    }
    handle
}

#[runtime_builtin(
    name = "pan",
    category = "plotting",
    summary = "Control interactive axes panning.",
    keywords = "pan,plotting,axes,figure,interaction",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::pan::PAN_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::pan"
)]
pub fn pan_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => {
            let handle = ensure_current_pan_figure();
            if requested_outputs() == 0 {
                set_pan_mode_for_figure(handle, PanModeCommand::Toggle)
                    .map_err(|err| invalid(err.to_string()))?;
                Ok(status())
            } else {
                pan_object_from_snapshot(
                    pan_state_snapshot(handle, None).map_err(|err| invalid(err.to_string()))?,
                )
            }
        }
        [single] => {
            if let Some(option) = super::style::value_as_string(single) {
                apply_option_to_figure(ensure_current_pan_figure(), &option)?;
                return Ok(status());
            }
            let figure = existing_figure_handle(single)?;
            let snapshot =
                pan_state_snapshot(figure, None).map_err(|err| invalid(err.to_string()))?;
            pan_object_from_snapshot(snapshot)
        }
        [target, command] => {
            let target = resolve_pan_target(target)?;
            let option = super::style::value_as_string(command)
                .ok_or_else(|| invalid("expected a pan option string"))?;
            apply_option_to_target(target, &option)?;
            Ok(status())
        }
        _ => Err(invalid(
            "expected pan(), pan(option), or pan(target, option)",
        )),
    }
}

#[derive(Clone, Copy, Debug)]
enum PanTarget {
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
}

fn resolve_pan_target(value: &Value) -> BuiltinResult<PanTarget> {
    match resolve_plot_handle(value, BUILTIN_NAME).map_err(|err| invalid(err.message))? {
        PlotHandle::Figure(handle) => Ok(PanTarget::Figure(handle)),
        PlotHandle::Axes(handle, axes) => Ok(PanTarget::Axes(handle, axes)),
        _ => Err(invalid("target must be a figure or axes handle")),
    }
}

fn rounded_handle_id(value: &Value) -> BuiltinResult<u32> {
    let value = numeric_scalar(value).ok_or_else(|| invalid("figure handle must be numeric"))?;
    if !value.is_finite() || value < 1.0 || value > u32::MAX as f64 {
        return Err(invalid("figure handle must be a positive finite scalar"));
    }
    Ok(value.round() as u32)
}

fn existing_figure_handle(value: &Value) -> BuiltinResult<FigureHandle> {
    let handle = FigureHandle::from(rounded_handle_id(value)?);
    if figure_handle_exists(handle) {
        Ok(handle)
    } else {
        Err(invalid("figure handle does not exist"))
    }
}

fn numeric_scalar(value: &Value) -> Option<f64> {
    super::style::value_as_f64(value)
}

fn apply_option_to_figure(handle: FigureHandle, option: &str) -> BuiltinResult<()> {
    set_pan_mode_for_figure(handle, parse_option(option)?).map_err(|err| invalid(err.to_string()))
}

fn apply_option_to_target(target: PanTarget, option: &str) -> BuiltinResult<()> {
    match target {
        PanTarget::Figure(handle) => set_pan_mode_for_figure(handle, parse_option(option)?)
            .map_err(|err| invalid(err.to_string())),
        PanTarget::Axes(handle, axes) => set_pan_mode_for_axes(handle, axes, parse_option(option)?)
            .map_err(|err| invalid(err.to_string())),
    }
}

fn parse_option(option: &str) -> BuiltinResult<PanModeCommand> {
    match option.trim().to_ascii_lowercase().as_str() {
        "on" => Ok(PanModeCommand::On),
        "off" => Ok(PanModeCommand::Off),
        "toggle" => Ok(PanModeCommand::Toggle),
        "xon" => Ok(PanModeCommand::XOn),
        "yon" => Ok(PanModeCommand::YOn),
        other => Err(invalid(format!("unsupported option '{other}'"))),
    }
}

fn pan_object_from_snapshot(snapshot: PanStateSnapshot) -> BuiltinResult<Value> {
    let mut object = ObjectInstance::new(PAN_CLASS_NAME.to_string());
    update_object_properties(&mut object, snapshot);
    let target =
        runmat_gc::gc_allocate(Value::Object(object)).map_err(|err| internal(err.to_string()))?;
    Ok(Value::HandleObject(HandleRef {
        class_name: PAN_CLASS_NAME.to_string(),
        target,
        valid: true,
    }))
}

fn update_object_properties(object: &mut ObjectInstance, snapshot: PanStateSnapshot) {
    object.properties.insert(
        "Enable".to_string(),
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    object.properties.insert(
        "Motion".to_string(),
        Value::String(snapshot.mode.motion.as_str().into()),
    );
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
}

fn read_pan_handle(value: &Value) -> Option<&HandleRef> {
    match value {
        Value::HandleObject(handle)
            if handle.class_name == PAN_CLASS_NAME
                && handle.valid
                && crate::is_handle_valid(handle) =>
        {
            Some(handle)
        }
        _ => None,
    }
}

fn snapshot_from_object(handle: &HandleRef) -> BuiltinResult<PanStateSnapshot> {
    let (figure, axes_index) = runmat_gc::gc_with_value(&handle.target, |value| {
        let Value::Object(object) = value else {
            return None;
        };
        if object.class_name != PAN_CLASS_NAME {
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
    .ok_or_else(|| invalid("invalid pan object"))?;
    pan_state_snapshot(figure, axes_index).map_err(|err| invalid(err.to_string()))
}

fn pan_struct(snapshot: PanStateSnapshot) -> Value {
    let mut st = StructValue::new();
    st.insert(
        "Enable",
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    st.insert(
        "Motion",
        Value::String(snapshot.mode.motion.as_str().into()),
    );
    st.insert("UIContextMenu", Value::Num(0.0));
    st.insert("ButtonDownFilter", Value::String(String::new()));
    st.insert("ActionPreCallback", Value::String(String::new()));
    st.insert("ActionPostCallback", Value::String(String::new()));
    st.insert("FigureHandle", Value::Num(snapshot.figure.as_u32() as f64));
    Value::Struct(st)
}

pub fn get_pan_object_property(
    value: &Value,
    property: Option<&str>,
    _builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let Some(handle) = read_pan_handle(value) else {
        return Ok(None);
    };
    let snapshot = snapshot_from_object(handle)?;
    let value = match property {
        None => {
            let mut value = pan_struct(snapshot);
            if let Value::Struct(st) = &mut value {
                for (name, default) in [
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
            Some("UIContextMenu") => object_property_or_default(handle, name, Value::Num(0.0))?,
            Some("ButtonDownFilter") | Some("ActionPreCallback") | Some("ActionPostCallback") => {
                object_property_or_default(handle, name, Value::String(String::new()))?
            }
            Some("FigureHandle") => Value::Num(snapshot.figure.as_u32() as f64),
            _ => return Err(invalid(format!("unsupported pan property '{name}'"))),
        },
    };
    Ok(Some(value))
}

pub fn set_pan_object_properties(
    value: &Value,
    args: &[Value],
    _builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(handle) = read_pan_handle(value).cloned() else {
        return Ok(None);
    };
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(invalid("expected pan property/value pairs"));
    }
    let snapshot = snapshot_from_object(&handle)?;
    for pair in args.chunks_exact(2) {
        let name = super::style::value_as_string(&pair[0])
            .ok_or_else(|| invalid("pan property names must be text"))?;
        apply_pan_property(snapshot.figure, &name, &pair[1])?;
        write_object_property(&handle, &name, pair[1].clone())?;
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

fn write_object_property(handle: &HandleRef, name: &str, value: Value) -> BuiltinResult<()> {
    let key = canonical_property(name).unwrap_or_else(|| name.to_string());
    runmat_gc::gc_with_value_mut(&handle.target, |target| {
        if let Value::Object(object) = target {
            object.properties.insert(key, value);
        }
    })
    .map_err(|err| internal(err.to_string()))
}

fn apply_pan_property(figure: FigureHandle, name: &str, value: &Value) -> BuiltinResult<()> {
    match canonical_property(name).as_deref() {
        Some("Enable") => set_pan_enabled_for_figure(figure, parse_on_off(value, "Enable")?)
            .map_err(|err| invalid(err.to_string())),
        Some("Motion") => set_pan_motion_for_figure(figure, parse_motion(value)?)
            .map_err(|err| invalid(err.to_string())),
        Some("FigureHandle") | Some("AxesIndex") => Err(invalid(format!("{name} is read-only"))),
        Some("UIContextMenu")
        | Some("ButtonDownFilter")
        | Some("ActionPreCallback")
        | Some("ActionPostCallback") => Ok(()),
        _ => Err(invalid(format!("unsupported pan property '{name}'"))),
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

fn canonical_property(name: &str) -> Option<String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some("Enable".into()),
        "motion" => Some("Motion".into()),
        "uicontextmenu" => Some("UIContextMenu".into()),
        "buttondownfilter" => Some("ButtonDownFilter".into()),
        "actionprecallback" => Some("ActionPreCallback".into()),
        "actionpostcallback" => Some("ActionPostCallback".into()),
        "figurehandle" => Some("FigureHandle".into()),
        "axesindex" => Some("AxesIndex".into()),
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
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn pan_option_and_object_track_live_state() {
        let _guard = setup_plot_tests();
        pan_builtin(vec![Value::String("xon".into())]).unwrap();
        let p = pan_builtin(Vec::new()).unwrap();
        assert_eq!(
            get_builtin(vec![p.clone(), Value::String("Enable".into())]).unwrap(),
            Value::String("on".into())
        );
        assert_eq!(
            get_builtin(vec![p.clone(), Value::String("Motion".into())]).unwrap(),
            Value::String("horizontal".into())
        );

        set_builtin(vec![
            p.clone(),
            Value::String("Motion".into()),
            Value::String("vertical".into()),
            Value::String("Enable".into()),
            Value::String("off".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![p, Value::String("Enable".into())]).unwrap(),
            Value::String("off".into())
        );
    }

    #[test]
    fn pan_figure_handle_returns_object() {
        let _guard = setup_plot_tests();
        pan_builtin(vec![Value::String("off".into())]).unwrap();
        let figure = current_figure_handle().as_u32() as f64;
        let p = pan_builtin(vec![Value::Num(figure)]).unwrap();
        assert!(matches!(p, Value::HandleObject(_)));
    }

    #[test]
    fn pan_rejects_missing_explicit_figure_handle() {
        let _guard = setup_plot_tests();
        let err = pan_builtin(vec![Value::Num(999.0)]).unwrap_err();
        assert!(err.to_string().contains("figure handle does not exist"));
    }
}
