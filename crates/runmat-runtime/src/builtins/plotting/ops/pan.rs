//! MATLAB-compatible `pan` interaction-mode builtin.

use runmat_builtins::BuiltinErrorDescriptor;

use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, ObjectInstance, Value};

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    current_figure_handle, figure_handle_exists, pan_state_snapshot, select_figure,
    set_pan_enabled_for_figure, set_pan_mode_for_axes, set_pan_mode_for_figure,
    set_pan_motion_for_figure, FigureHandle, PanModeCommand, PanStateSnapshot, ZoomMotion,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pan";
const PAN_CLASS_NAME: runmat_types::StaticClassIdentity = runmat_types::standard::PAN_CONTROL;

pub use runmat_builtins::PAN_INTEGER_TARGET_EXTENSION as INTEGER_TARGET_EXTENSION;

pub use runmat_builtins::PAN_ERROR_INVALID_ARGUMENT;

pub use runmat_builtins::PAN_ERROR_INTERNAL;

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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::pan"
)]
pub fn pan_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    match (args.len(), args.first(), args.get(1)) {
        (0, _, _) => {
            let handle = ensure_current_pan_figure();
            if requested_outputs() == 0 {
                set_pan_mode_for_figure(handle, PanModeCommand::Toggle)
                    .map_err(|err| invalid(err.to_string()))?;
                Ok(status())
            } else {
                pan_object_from_snapshot(
                    runtime.as_ref(),
                    pan_state_snapshot(handle, None).map_err(|err| invalid(err.to_string()))?,
                )
            }
        }
        (1, Some(single), _) => {
            if let Some(option) = super::style::value_as_string(single).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "pan",
                    source.to_string(),
                    source,
                )
            })? {
                apply_option_to_figure(ensure_current_pan_figure(), &option)?;
                return Ok(status());
            }
            gate_integer_target(single)?;
            let figure = existing_figure_handle(single)?;
            let snapshot =
                pan_state_snapshot(figure, None).map_err(|err| invalid(err.to_string()))?;
            pan_object_from_snapshot(runtime.as_ref(), snapshot)
        }
        (2, Some(target), Some(command)) => {
            gate_integer_target(target)?;
            let target = resolve_pan_target(target)?;
            let option = super::style::value_as_string(command)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "pan",
                        source.to_string(),
                        source,
                    )
                })?
                .ok_or_else(|| invalid("expected a pan option string"))?;
            apply_option_to_target(target, &option)?;
            Ok(status())
        }
        _ => Err(invalid(
            "expected pan(), pan(option), or pan(target, option)",
        )),
    }
}

fn gate_integer_target(value: &Value) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_TARGET_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
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

fn pan_object_from_snapshot(
    runtime: Option<&crate::context::RuntimeContext>,
    snapshot: PanStateSnapshot,
) -> BuiltinResult<Value> {
    use super::op_common::mode_object::{Field, FieldValue};
    let mut fields = [
        Field {
            name: "Enable",
            value: FieldValue::Text(if snapshot.mode.enabled { "on" } else { "off" }),
        },
        Field {
            name: "Motion",
            value: FieldValue::Text(snapshot.mode.motion.as_str()),
        },
        Field {
            name: "UIContextMenu",
            value: FieldValue::Value(Some(Value::Num(0.0))),
        },
        Field {
            name: "ButtonDownFilter",
            value: FieldValue::Text(""),
        },
        Field {
            name: "ActionPreCallback",
            value: FieldValue::Text(""),
        },
        Field {
            name: "ActionPostCallback",
            value: FieldValue::Text(""),
        },
        Field {
            name: "FigureHandle",
            value: FieldValue::Value(Some(Value::Num(snapshot.figure.as_u32() as f64))),
        },
        Field {
            name: "AxesIndex",
            value: FieldValue::Value(Some(
                snapshot
                    .axes_index
                    .map(|index| Value::Num((index + 1) as f64))
                    .unwrap_or(Value::Num(0.0)),
            )),
        },
    ];
    let object = super::op_common::mode_object::construct(
        runtime,
        PAN_CLASS_NAME,
        super::state::mode_properties::receiver(snapshot.figure, BUILTIN_NAME)?,
        &mut fields,
    )?;
    let (object, _roots) = object.into_value_and_lease();
    let Value::Object(object) = object else {
        unreachable!("pan snapshot publishes an object")
    };
    let class_name = object.class_name_owner().clone();
    let target =
        runmat_gc::gc_allocate(Value::Object(object)).map_err(|err| internal(err.to_string()))?;
    Ok(Value::HandleObject(HandleRef {
        class_name,
        target,
        valid: true,
    }))
}

fn update_object_properties(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    snapshot: PanStateSnapshot,
) -> BuiltinResult<()> {
    use super::op_common::mode_object::{Field, FieldValue};
    let mut fields = [
        Field {
            name: "Enable",
            value: FieldValue::Text(if snapshot.mode.enabled { "on" } else { "off" }),
        },
        Field {
            name: "Motion",
            value: FieldValue::Text(snapshot.mode.motion.as_str()),
        },
        Field {
            name: "FigureHandle",
            value: FieldValue::Value(Some(Value::Num(snapshot.figure.as_u32() as f64))),
        },
        Field {
            name: "AxesIndex",
            value: FieldValue::Value(Some(
                snapshot
                    .axes_index
                    .map(|index| Value::Num((index + 1) as f64))
                    .unwrap_or(Value::Num(0.0)),
            )),
        },
    ];
    super::op_common::mode_object::prepare_fields(runtime, &mut fields)?;
    crate::execution::object::replacement::replace_properties(
        runtime,
        object,
        fields.iter().map(|field| (field.name, field.value())),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:pan:HostAllocation",
                message,
                source,
            )
        },
        invalid,
    )
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
        if *object.class_name() != PAN_CLASS_NAME {
            return None;
        }
        let receiver = object.graphics_receiver()?.clone();
        let axes_index = object
            .properties()
            .get("AxesIndex")
            .and_then(numeric_scalar)
            .and_then(|value| {
                if value >= 1.0 {
                    Some(value.round() as usize - 1)
                } else {
                    None
                }
            });
        Some((receiver, axes_index))
    })
    .map_err(|err| internal(err.to_string()))?
    .ok_or_else(|| invalid("invalid pan object"))?;
    pan_state_snapshot(figure, axes_index).map_err(|err| invalid(err.to_string()))
}

fn pan_struct(snapshot: PanStateSnapshot) -> Value {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert(
        "Enable",
        Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into()),
    );
    st.insert(
        "Motion",
        Value::String(snapshot.mode.motion.as_str().into()),
    );
    st.insert("UIContextMenu", Value::Num(0.0));
    st.insert("ButtonDownFilter", Value::string(String::new()));
    st.insert("ActionPreCallback", Value::string(String::new()));
    st.insert("ActionPostCallback", Value::string(String::new()));
    st.insert("FigureHandle", Value::Num(snapshot.figure.as_u32() as f64));
    Value::Struct(st.finish())
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
            let Value::Struct(st) = pan_struct(snapshot) else {
                unreachable!("pan snapshot is a structure")
            };
            let mut st = runmat_value::UnboundedStructValueBuilder::from_structure(st);
            for (name, default) in [
                ("UIContextMenu", Value::Num(0.0)),
                ("ButtonDownFilter", Value::string(String::new())),
                ("ActionPreCallback", Value::string(String::new())),
                ("ActionPostCallback", Value::string(String::new())),
            ] {
                st.insert(name, object_property_or_default(handle, name, default)?);
            }
            Value::Struct(st.finish())
        }
        Some(name) => match canonical_property(name) {
            Some(PanProperty::Enable) => {
                Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into())
            }
            Some(PanProperty::Motion) => Value::String(snapshot.mode.motion.as_str().into()),
            Some(PanProperty::UIContextMenu) => {
                object_property_or_default(handle, name, Value::Num(0.0))?
            }
            Some(PanProperty::ButtonDownFilter)
            | Some(PanProperty::ActionPreCallback)
            | Some(PanProperty::ActionPostCallback) => {
                object_property_or_default(handle, name, Value::string(String::new()))?
            }
            Some(PanProperty::FigureHandle) => Value::Num(snapshot.figure.as_u32() as f64),
            _ => return Err(invalid(format!("unsupported pan property '{name}'"))),
        },
    };
    Ok(Some(value))
}

pub(crate) fn set_pan_object_properties_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    args: crate::call::FunctionArgumentView<'_>,
    _builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(handle) = read_pan_handle(value).cloned() else {
        return Ok(None);
    };
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(invalid("expected pan property/value pairs"));
    }
    snapshot_from_object(&handle)?;
    for pair in args.chunks_exact(2) {
        let name = super::properties::property_name_text(&pair[0], "set")
            .map_err(|_| invalid("pan property names must be scalar text"))?;
        apply_pan_property(mode_receiver(&handle)?, &name, &pair[1])?;
        write_object_property(runtime, &handle, &name, pair[1].clone())?;
    }
    let refreshed = snapshot_from_object(&handle)?;
    let updated = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        if let Value::Object(object) = target {
            update_object_properties(runtime, object, refreshed)
        } else {
            Err(invalid("pan handle target is not an object"))
        }
    })
    .map_err(|err| internal(err.to_string()))?;
    updated?;
    Ok(Some(()))
}

fn object_property_or_default(
    handle: &HandleRef,
    name: &str,
    default: Value,
) -> BuiltinResult<Value> {
    if let Some(key) = canonical_property(name).and_then(PanProperty::persistent) {
        return super::state::mode_properties::get(
            &mode_receiver(handle)?,
            key,
            default,
            BUILTIN_NAME,
        );
    }
    let key = canonical_property(name)
        .map(PanProperty::name)
        .unwrap_or(name)
        .to_string();
    runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) => object.properties().get(&key).cloned().unwrap_or(default),
        _ => default,
    })
    .map_err(|err| internal(err.to_string()))
}

fn write_object_property(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &HandleRef,
    name: &str,
    value: Value,
) -> BuiltinResult<()> {
    if let Some(key) = canonical_property(name).and_then(PanProperty::persistent) {
        return super::state::mode_properties::set_in_context(
            runtime,
            &mode_receiver(handle)?,
            key,
            value,
            BUILTIN_NAME,
        );
    }
    let key = canonical_property(name)
        .map(PanProperty::name)
        .unwrap_or(name)
        .to_string();
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        if let Value::Object(object) = target {
            crate::execution::object::replacement::replace_property(
                runtime,
                object,
                &key,
                value,
                |message, source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        runtime,
                        "RunMat:pan:HostAllocation",
                        message,
                        source,
                    )
                },
                internal,
            )?;
        }
        Ok(())
    })
    .map_err(|err| internal(err.to_string()))?
}

fn apply_pan_property(
    figure: impl Into<super::state::FigureTarget>,
    name: &str,
    value: &Value,
) -> BuiltinResult<()> {
    match canonical_property(name) {
        Some(PanProperty::Enable) => {
            set_pan_enabled_for_figure(figure, parse_on_off(value, "Enable")?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some(PanProperty::Motion) => set_pan_motion_for_figure(figure, parse_motion(value)?)
            .map_err(|err| invalid(err.to_string())),
        Some(PanProperty::FigureHandle) | Some(PanProperty::AxesIndex) => {
            Err(invalid(format!("{name} is read-only")))
        }
        Some(PanProperty::UIContextMenu)
        | Some(PanProperty::ButtonDownFilter)
        | Some(PanProperty::ActionPreCallback)
        | Some(PanProperty::ActionPostCallback) => Ok(()),
        _ => Err(invalid(format!("unsupported pan property '{name}'"))),
    }
}

fn parse_on_off(value: &Value, property: &str) -> BuiltinResult<bool> {
    super::style::value_as_bool(value)
        .ok_or_else(|| invalid(format!("{property} must be 'on' or 'off'")))
}

fn parse_motion(value: &Value) -> BuiltinResult<ZoomMotion> {
    let text = super::style::value_as_string(value)
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source("pan", source.to_string(), source)
        })?
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

#[derive(Clone, Copy)]
enum PanProperty {
    Enable,
    Motion,
    UIContextMenu,
    ButtonDownFilter,
    ActionPreCallback,
    ActionPostCallback,
    FigureHandle,
    AxesIndex,
}
impl PanProperty {
    fn name(self) -> &'static str {
        match self {
            Self::Enable => "Enable",
            Self::Motion => "Motion",
            Self::UIContextMenu => "UIContextMenu",
            Self::ButtonDownFilter => "ButtonDownFilter",
            Self::ActionPreCallback => "ActionPreCallback",
            Self::ActionPostCallback => "ActionPostCallback",
            Self::FigureHandle => "FigureHandle",
            Self::AxesIndex => "AxesIndex",
        }
    }
    fn persistent(self) -> Option<super::state::mode_properties::ModeProperty> {
        use super::state::mode_properties::ModeProperty as Property;
        use crate::context::GraphicsCallbackKind as Kind;
        match self {
            Self::UIContextMenu => Some(Property::PanContextMenu),
            Self::ButtonDownFilter => Some(Property::Callback(Kind::PanButtonDownFilter)),
            Self::ActionPreCallback => Some(Property::Callback(Kind::PanActionPre)),
            Self::ActionPostCallback => Some(Property::Callback(Kind::PanActionPost)),
            _ => None,
        }
    }
}
fn canonical_property(name: &str) -> Option<PanProperty> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some(PanProperty::Enable),
        "motion" => Some(PanProperty::Motion),
        "uicontextmenu" => Some(PanProperty::UIContextMenu),
        "buttondownfilter" => Some(PanProperty::ButtonDownFilter),
        "actionprecallback" => Some(PanProperty::ActionPreCallback),
        "actionpostcallback" => Some(PanProperty::ActionPostCallback),
        "figurehandle" => Some(PanProperty::FigureHandle),
        "axesindex" => Some(PanProperty::AxesIndex),
        _ => None,
    }
}

fn mode_receiver(handle: &HandleRef) -> BuiltinResult<runmat_value::GraphicsReceiverRef> {
    runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) if *object.class_name() == PAN_CLASS_NAME => {
            object.graphics_receiver().cloned()
        }
        _ => None,
    })
    .map_err(|cause| internal(cause.to_string()))?
    .ok_or_else(|| invalid("pan object has no exact Figure receiver"))
}

#[cfg(test)]
mod tests {
    fn pan_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        arguments: Vec<Value>,
    ) -> BuiltinResult<Value> {
        let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
        super::pan_builtin(runtime, arguments)
    }

    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn make_pan(args: Vec<Value>) -> BuiltinResult<Value> {
        pan_builtin(None, args)
    }

    fn make_zoom(args: Vec<Value>) -> BuiltinResult<Value> {
        crate::builtins::plotting::zoom::zoom_builtin(
            None,
            crate::call::BuiltinArguments::from_borrowed(&args, None)?,
        )
    }

    fn make_data_cursor(args: Vec<Value>) -> BuiltinResult<Value> {
        crate::builtins::plotting::datacursormode::datacursormode_builtin(
            None,
            crate::call::BuiltinArguments::from_borrowed(&args, None)?,
        )
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn mode_properties_are_figure_owned_and_old_mode_objects_cannot_retarget_reused_figures() {
        type Constructor = fn(Vec<Value>) -> BuiltinResult<Value>;
        let cases: [(Constructor, &str, &str, Value); 3] = [
            (
                make_pan,
                "ActionPreCallback",
                "UIContextMenu",
                Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            ),
            (
                make_zoom,
                "ActionPostCallback",
                "ContextMenu",
                Value::Int(runmat_value::IntValue::I64(i64::MIN)),
            ),
            (
                make_data_cursor,
                "UpdateFcn",
                "Interpreter",
                Value::String("latex".into()),
            ),
        ];
        for (make, callback_property, metadata_property, metadata) in cases {
            let _guard = lock_plot_registry();
            ensure_plot_test_env();
            super::super::state::reset_plot_state();
            runmat_gc::gc_test_context(|| {
                let first = crate::execution::RootedValue::new(
                    make(Vec::new()).expect("first mode handle"),
                )
                .unwrap();
                let callback = Value::FunctionHandle("retained_original_callback".into());
                set_builtin(vec![
                    first.value().clone(),
                    Value::String(callback_property.into()),
                    callback.clone(),
                    Value::String(metadata_property.into()),
                    metadata.clone(),
                ])
                .expect("store figure behavior");
                let second = crate::execution::RootedValue::new(
                    make(Vec::new()).expect("independent mode handle"),
                )
                .unwrap();
                assert_eq!(
                    get_builtin(vec![
                        second.value().clone(),
                        Value::String(callback_property.into())
                    ])
                    .unwrap(),
                    callback
                );
                assert_eq!(
                    get_builtin(vec![
                        second.value().clone(),
                        Value::String(metadata_property.into())
                    ])
                    .unwrap(),
                    metadata
                );
                let replacement = Value::FunctionHandle("replacement_callback".into());
                set_builtin(vec![
                    second.value().clone(),
                    Value::String(callback_property.into()),
                    replacement.clone(),
                ])
                .unwrap();
                assert_eq!(
                    get_builtin(vec![
                        first.value().clone(),
                        Value::String(callback_property.into())
                    ])
                    .unwrap(),
                    replacement
                );
                let empty =
                    Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap());
                set_builtin(vec![
                    first.value().clone(),
                    Value::String(callback_property.into()),
                    empty.clone(),
                ])
                .unwrap();
                assert_eq!(
                    get_builtin(vec![
                        second.value().clone(),
                        Value::String(callback_property.into())
                    ])
                    .unwrap(),
                    empty
                );
                let old = current_figure_handle();
                super::super::state::close_figure(Some(old)).unwrap();
                let fresh = crate::execution::RootedValue::new(
                    make(Vec::new()).expect("replacement figure and mode handle"),
                )
                .unwrap();
                assert_eq!(
                    current_figure_handle(),
                    old,
                    "exercise ordinary reused default coordinate"
                );
                assert!(get_builtin(vec![
                    first.value().clone(),
                    Value::String(callback_property.into())
                ])
                .is_err());
                assert!(
                    get_builtin(vec![second.value().clone(), Value::String("Enable".into())])
                        .is_err()
                );
                assert!(
                    get_builtin(vec![fresh.value().clone(), Value::String("Enable".into())])
                        .is_ok()
                );
            });
        }
    }

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
        pan_builtin(None, vec![Value::String("xon".into())]).unwrap();
        let p = pan_builtin(None, Vec::new()).unwrap();
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
        pan_builtin(None, vec![Value::String("off".into())]).unwrap();
        let figure = current_figure_handle().as_u32() as f64;
        let p = pan_builtin(None, vec![Value::Num(figure)]).unwrap();
        assert!(matches!(p, Value::HandleObject(_)));
    }

    #[test]
    fn pan_rejects_missing_explicit_figure_handle() {
        let _guard = setup_plot_tests();
        let err = pan_builtin(None, vec![Value::Num(999.0)]).unwrap_err();
        assert!(err.to_string().contains("figure handle does not exist"));
    }

    #[test]
    fn pan_enable_accepts_zero_and_one_from_every_integer_class() {
        let _guard = setup_plot_tests();
        let p = pan_builtin(None, Vec::new()).expect("pan object");
        for (enabled, disabled) in [
            (runmat_value::IntValue::I8(1), runmat_value::IntValue::I8(0)),
            (
                runmat_value::IntValue::I16(1),
                runmat_value::IntValue::I16(0),
            ),
            (
                runmat_value::IntValue::I32(1),
                runmat_value::IntValue::I32(0),
            ),
            (
                runmat_value::IntValue::I64(1),
                runmat_value::IntValue::I64(0),
            ),
            (runmat_value::IntValue::U8(1), runmat_value::IntValue::U8(0)),
            (
                runmat_value::IntValue::U16(1),
                runmat_value::IntValue::U16(0),
            ),
            (
                runmat_value::IntValue::U32(1),
                runmat_value::IntValue::U32(0),
            ),
            (
                runmat_value::IntValue::U64(1),
                runmat_value::IntValue::U64(0),
            ),
        ] {
            set_builtin(vec![
                p.clone(),
                Value::String("Enable".into()),
                Value::Int(enabled),
            ])
            .expect("enable pan");
            assert_eq!(
                get_builtin(vec![p.clone(), Value::String("Enable".into())]).unwrap(),
                Value::String("on".into())
            );
            set_builtin(vec![
                p.clone(),
                Value::String("Enable".into()),
                Value::Int(disabled),
            ])
            .expect("disable pan");
            assert_eq!(
                get_builtin(vec![p.clone(), Value::String("Enable".into())]).unwrap(),
                Value::String("off".into())
            );
        }
    }

    #[test]
    fn pan_typed_target_is_gated_before_handle_resolution() {
        let _guard = setup_plot_tests();
        let figure = current_figure_handle().as_u32();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = pan_builtin(None, vec![Value::Int(runmat_value::IntValue::U32(figure))])
            .expect_err("typed target must be gated");

        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PanIntegerGraphicsTargetExtension")
        );
    }
}
