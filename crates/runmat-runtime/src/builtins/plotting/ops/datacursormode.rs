//! MATLAB-compatible `datacursormode` interaction-state builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    HandleRef, ObjectInstance, StructValue, Value,
};
use runmat_macros::runtime_builtin;

use super::state::{
    current_figure_handle, data_cursor_state_snapshot, figure_handle_exists, select_figure,
    set_data_cursor_display_style_for_figure, set_data_cursor_enabled_for_figure,
    set_data_cursor_snap_to_data_vertex_for_figure, DataCursorStateSnapshot, FigureHandle,
};
use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "datacursormode";
const DATA_CURSOR_CLASS_NAME: &str = "matlab.graphics.shape.internal.DataCursorManager";

const OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dcm",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Data cursor mode object for the current or specified figure.",
}];

const OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns \"ok\" when the command succeeds.",
}];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode option: on, off, or toggle.",
}];

const INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure handle.",
}];

const INPUTS_FIGURE_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure handle.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mode option: on, off, or toggle.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "dcm = datacursormode()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "datacursormode()",
        inputs: &INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = datacursormode(option)",
        inputs: &INPUTS_OPTION,
        outputs: &OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "dcm = datacursormode(fig)",
        inputs: &INPUTS_FIGURE,
        outputs: &OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "status = datacursormode(fig, option)",
        inputs: &INPUTS_FIGURE_OPTION,
        outputs: &OUTPUT_STATUS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATACURSORMODE.INVALID_ARGUMENT",
    identifier: Some("RunMat:datacursormode:InvalidArgument"),
    when: "Argument count, figure handle, option, or property is invalid.",
    message: "datacursormode: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATACURSORMODE.INTERNAL",
    identifier: Some("RunMat:datacursormode:Internal"),
    when: "Internal data cursor state or handle-object update fails.",
    message: "datacursormode: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const DATACURSORMODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn err(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    err(&ERROR_INVALID_ARGUMENT, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    err(&ERROR_INTERNAL, detail)
}

fn requested_outputs() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn ensure_current_figure() -> FigureHandle {
    let handle = current_figure_handle();
    if !figure_handle_exists(handle) {
        select_figure(handle);
    }
    handle
}

#[runtime_builtin(
    name = "datacursormode",
    category = "plotting",
    summary = "Control data cursor interaction mode.",
    keywords = "datacursormode,data cursor,plotting,interaction",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::datacursormode::DATACURSORMODE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::datacursormode"
)]
pub fn datacursormode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => {
            let figure = ensure_current_figure();
            if requested_outputs() == 0 {
                toggle_figure(figure)?;
                Ok(Value::String("ok".into()))
            } else {
                object_from_snapshot(
                    data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?,
                )
            }
        }
        [single] => {
            if let Some(option) = super::style::value_as_string(single) {
                apply_option(ensure_current_figure(), &option)?;
                return Ok(Value::String("ok".into()));
            }
            let figure = existing_figure_handle(single)?;
            object_from_snapshot(
                data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?,
            )
        }
        [figure, option] => {
            let figure = existing_figure_handle(figure)?;
            let option = super::style::value_as_string(option)
                .ok_or_else(|| invalid("expected option string"))?;
            apply_option(figure, &option)?;
            Ok(Value::String("ok".into()))
        }
        _ => Err(invalid(
            "expected datacursormode(), datacursormode(option), or datacursormode(fig, option)",
        )),
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

fn apply_option(figure: FigureHandle, option: &str) -> BuiltinResult<()> {
    match option.trim().to_ascii_lowercase().as_str() {
        "on" => set_data_cursor_enabled_for_figure(figure, true),
        "off" => set_data_cursor_enabled_for_figure(figure, false),
        "toggle" => {
            let snapshot =
                data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?;
            set_data_cursor_enabled_for_figure(figure, !snapshot.mode.enabled)
        }
        other => return Err(invalid(format!("unsupported option '{other}'"))),
    }
    .map_err(|err| invalid(err.to_string()))
}

fn toggle_figure(figure: FigureHandle) -> BuiltinResult<()> {
    let snapshot = data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?;
    set_data_cursor_enabled_for_figure(figure, !snapshot.mode.enabled)
        .map_err(|err| invalid(err.to_string()))
}

fn object_from_snapshot(snapshot: DataCursorStateSnapshot) -> BuiltinResult<Value> {
    let mut object = ObjectInstance::new(DATA_CURSOR_CLASS_NAME.to_string());
    update_object_properties(&mut object, snapshot);
    let target =
        runmat_gc::gc_allocate(Value::Object(object)).map_err(|err| internal(err.to_string()))?;
    Ok(Value::HandleObject(HandleRef {
        class_name: DATA_CURSOR_CLASS_NAME.to_string(),
        target,
        valid: true,
    }))
}

fn update_object_properties(object: &mut ObjectInstance, snapshot: DataCursorStateSnapshot) {
    object.properties.insert(
        "Enable".into(),
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    object.properties.insert(
        "SnapToDataVertex".into(),
        Value::String(
            if snapshot.mode.snap_to_data_vertex {
                "on"
            } else {
                "off"
            }
            .into(),
        ),
    );
    object.properties.insert(
        "DisplayStyle".into(),
        Value::String(snapshot.mode.display_style),
    );
    object
        .properties
        .entry("UpdateFcn".into())
        .or_insert(Value::String(String::new()));
    object.properties.insert(
        "FigureHandle".into(),
        Value::Num(snapshot.figure.as_u32() as f64),
    );
}

fn read_data_cursor_handle(value: &Value) -> Option<&HandleRef> {
    match value {
        Value::HandleObject(handle)
            if handle.class_name == DATA_CURSOR_CLASS_NAME
                && handle.valid
                && crate::is_handle_valid(handle) =>
        {
            Some(handle)
        }
        _ => None,
    }
}

fn snapshot_from_object(handle: &HandleRef) -> BuiltinResult<DataCursorStateSnapshot> {
    let figure = runmat_gc::gc_with_value(&handle.target, |value| {
        let Value::Object(object) = value else {
            return None;
        };
        if object.class_name != DATA_CURSOR_CLASS_NAME {
            return None;
        }
        object
            .properties
            .get("FigureHandle")
            .and_then(numeric_scalar)
            .map(|figure| FigureHandle::from(figure.round() as u32))
    })
    .map_err(|err| internal(err.to_string()))?
    .ok_or_else(|| invalid("invalid datacursormode object"))?;
    data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))
}

fn cursor_struct(snapshot: DataCursorStateSnapshot) -> Value {
    let mut st = StructValue::new();
    st.insert(
        "Enable",
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    st.insert(
        "SnapToDataVertex",
        Value::String(
            if snapshot.mode.snap_to_data_vertex {
                "on"
            } else {
                "off"
            }
            .into(),
        ),
    );
    st.insert("DisplayStyle", Value::String(snapshot.mode.display_style));
    st.insert("UpdateFcn", Value::String(String::new()));
    st.insert("FigureHandle", Value::Num(snapshot.figure.as_u32() as f64));
    Value::Struct(st)
}

pub fn get_data_cursor_object_property(
    value: &Value,
    property: Option<&str>,
    _builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let Some(handle) = read_data_cursor_handle(value) else {
        return Ok(None);
    };
    let snapshot = snapshot_from_object(handle)?;
    let value = match property {
        None => {
            let mut value = cursor_struct(snapshot);
            if let Value::Struct(st) = &mut value {
                st.insert(
                    "UpdateFcn",
                    object_property_or_default(handle, "UpdateFcn", Value::String(String::new()))?,
                );
            }
            value
        }
        Some(name) => match canonical_property(name).as_deref() {
            Some("Enable") => {
                Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into())
            }
            Some("SnapToDataVertex") => Value::String(
                if snapshot.mode.snap_to_data_vertex {
                    "on"
                } else {
                    "off"
                }
                .into(),
            ),
            Some("DisplayStyle") => Value::String(snapshot.mode.display_style),
            Some("UpdateFcn") => {
                object_property_or_default(handle, "UpdateFcn", Value::String(String::new()))?
            }
            Some("FigureHandle") => Value::Num(snapshot.figure.as_u32() as f64),
            _ => {
                return Err(invalid(format!(
                    "unsupported datacursormode property '{name}'"
                )))
            }
        },
    };
    Ok(Some(value))
}

pub fn set_data_cursor_object_properties(
    value: &Value,
    args: &[Value],
    _builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(handle) = read_data_cursor_handle(value).cloned() else {
        return Ok(None);
    };
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(invalid("expected datacursormode property/value pairs"));
    }
    let snapshot = snapshot_from_object(&handle)?;
    for pair in args.chunks_exact(2) {
        let name = super::style::value_as_string(&pair[0])
            .ok_or_else(|| invalid("datacursormode property names must be text"))?;
        apply_property(snapshot.figure, &name, &pair[1])?;
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

fn apply_property(figure: FigureHandle, name: &str, value: &Value) -> BuiltinResult<()> {
    match canonical_property(name).as_deref() {
        Some("Enable") => {
            set_data_cursor_enabled_for_figure(figure, parse_on_off(value, "Enable")?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some("SnapToDataVertex") => set_data_cursor_snap_to_data_vertex_for_figure(
            figure,
            parse_on_off(value, "SnapToDataVertex")?,
        )
        .map_err(|err| invalid(err.to_string())),
        Some("DisplayStyle") => {
            let style = super::style::value_as_string(value)
                .ok_or_else(|| invalid("DisplayStyle must be 'datatip' or 'window'"))?;
            let normalized = match style.trim().to_ascii_lowercase().as_str() {
                "datatip" => "datatip",
                "window" => "window",
                _ => return Err(invalid("DisplayStyle must be 'datatip' or 'window'")),
            };
            set_data_cursor_display_style_for_figure(figure, normalized.to_string())
                .map_err(|err| invalid(err.to_string()))
        }
        Some("UpdateFcn") => Ok(()),
        Some("FigureHandle") => Err(invalid("FigureHandle is read-only")),
        _ => Err(invalid(format!(
            "unsupported datacursormode property '{name}'"
        ))),
    }
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

fn parse_on_off(value: &Value, property: &str) -> BuiltinResult<bool> {
    super::style::value_as_bool(value)
        .ok_or_else(|| invalid(format!("{property} must be 'on' or 'off'")))
}

fn canonical_property(name: &str) -> Option<String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some("Enable".into()),
        "snaptodatavertex" => Some("SnapToDataVertex".into()),
        "displaystyle" => Some("DisplayStyle".into()),
        "updatefcn" => Some("UpdateFcn".into()),
        "figurehandle" => Some("FigureHandle".into()),
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
    fn datacursormode_option_and_object_track_live_state() {
        let _guard = setup_plot_tests();
        datacursormode_builtin(vec![Value::String("on".into())]).unwrap();
        let dcm = datacursormode_builtin(Vec::new()).unwrap();
        assert_eq!(
            get_builtin(vec![dcm.clone(), Value::String("Enable".into())]).unwrap(),
            Value::String("on".into())
        );

        set_builtin(vec![
            dcm.clone(),
            Value::String("DisplayStyle".into()),
            Value::String("window".into()),
            Value::String("SnapToDataVertex".into()),
            Value::String("off".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![dcm.clone(), Value::String("DisplayStyle".into())]).unwrap(),
            Value::String("window".into())
        );
        assert_eq!(
            get_builtin(vec![dcm, Value::String("SnapToDataVertex".into())]).unwrap(),
            Value::String("off".into())
        );
    }

    #[test]
    fn datacursormode_figure_handle_returns_object() {
        let _guard = setup_plot_tests();
        datacursormode_builtin(vec![Value::String("off".into())]).unwrap();
        let figure = current_figure_handle().as_u32() as f64;
        let dcm = datacursormode_builtin(vec![Value::Num(figure)]).unwrap();
        assert!(matches!(dcm, Value::HandleObject(_)));
    }

    #[test]
    fn datacursormode_rejects_missing_explicit_figure_handle() {
        let _guard = setup_plot_tests();
        let err = datacursormode_builtin(vec![Value::Num(999.0), Value::String("on".into())])
            .unwrap_err();
        assert!(err.to_string().contains("figure handle does not exist"));
    }
}
