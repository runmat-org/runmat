use runmat_builtins::BuiltinErrorDescriptor;

use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, ObjectInstance, Value};

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
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "zoom";
const ZOOM_CLASS_NAME: runmat_types::StaticClassIdentity = runmat_types::standard::ZOOM_CONTROL;

pub use runmat_builtins::ZOOM_INTEGER_TARGET_EXTENSION;

pub use runmat_builtins::ZOOM_ERROR_INVALID_ARGUMENT;

pub use runmat_builtins::ZOOM_ERROR_INTERNAL;

#[cfg(test)]
use runmat_builtins::ZOOM_DESCRIPTOR;

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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::zoom"
)]
pub fn zoom_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    match (args.len(), args.first(), args.get(1)) {
        (0, _, _) => {
            let handle = ensure_current_zoom_figure();
            if requested_outputs() == 0 {
                set_zoom_mode_for_figure(handle, ZoomModeCommand::Toggle)
                    .map_err(|err| invalid(err.to_string()))?;
                Ok(status())
            } else {
                let snapshot =
                    zoom_state_snapshot(handle, None).map_err(|err| invalid(err.to_string()))?;
                zoom_object_from_snapshot(runtime.as_ref(), snapshot)
            }
        }
        (1, Some(single), _) => {
            if let Some(option) = super::style::value_as_string(single).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "zoom",
                    source.to_string(),
                    source,
                )
            })? {
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
                    return zoom_object_from_snapshot(runtime.as_ref(), snapshot);
                }
            }
            validate_factor(factor)?;
            apply_zoom_factor_for_figure(ensure_current_zoom_figure(), factor)
                .map_err(|err| invalid(err.to_string()))?;
            Ok(status())
        }
        (2, Some(target), Some(command)) => {
            let target = resolve_zoom_target(target)?;
            if let Some(option) = super::style::value_as_string(command).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "zoom",
                    source.to_string(),
                    source,
                )
            })? {
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

fn zoom_object_from_snapshot(
    runtime: Option<&crate::context::RuntimeContext>,
    snapshot: ZoomStateSnapshot,
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
            name: "Direction",
            value: FieldValue::Text(snapshot.mode.direction.as_str()),
        },
        Field {
            name: "RightClickAction",
            value: FieldValue::Text(snapshot.mode.right_click_action.as_str()),
        },
        Field {
            name: "ContextMenu",
            value: FieldValue::Value(Some(Value::Num(0.0))),
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
        Field {
            name: "UseLegacyExplorationModes",
            value: FieldValue::Text(if snapshot.mode.use_legacy_exploration_modes {
                "on"
            } else {
                "off"
            }),
        },
    ];
    let object = super::op_common::mode_object::construct(
        runtime,
        ZOOM_CLASS_NAME,
        super::state::mode_properties::receiver(snapshot.figure, BUILTIN_NAME)?,
        &mut fields,
    )?;
    let (object, _roots) = object.into_value_and_lease();
    let Value::Object(object) = object else {
        unreachable!("zoom snapshot publishes an object")
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
    snapshot: ZoomStateSnapshot,
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
            name: "Direction",
            value: FieldValue::Text(snapshot.mode.direction.as_str()),
        },
        Field {
            name: "RightClickAction",
            value: FieldValue::Text(snapshot.mode.right_click_action.as_str()),
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
        Field {
            name: "UseLegacyExplorationModes",
            value: FieldValue::Text(if snapshot.mode.use_legacy_exploration_modes {
                "on"
            } else {
                "off"
            }),
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
                "RunMat:zoom:HostAllocation",
                message,
                source,
            )
        },
        invalid,
    )
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
        if *object.class_name() != ZOOM_CLASS_NAME {
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
    .ok_or_else(|| invalid("invalid zoom object"))?;
    zoom_state_snapshot(figure, axes_index).map_err(|err| invalid(err.to_string()))
}

fn zoom_struct(snapshot: ZoomStateSnapshot) -> Value {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
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
    st.insert("ButtonDownFilter", Value::string(String::new()));
    st.insert("ActionPreCallback", Value::string(String::new()));
    st.insert("ActionPostCallback", Value::string(String::new()));
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
    Value::Struct(st.finish())
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
            let Value::Struct(st) = zoom_struct(snapshot) else {
                unreachable!("zoom snapshot is a structure")
            };
            let mut st = runmat_value::UnboundedStructValueBuilder::from_structure(st);
            for (name, default) in [
                ("ContextMenu", Value::Num(0.0)),
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
            Some(ZoomProperty::Enable) => {
                Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into())
            }
            Some(ZoomProperty::Motion) => Value::String(snapshot.mode.motion.as_str().into()),
            Some(ZoomProperty::Direction) => Value::String(snapshot.mode.direction.as_str().into()),
            Some(ZoomProperty::RightClickAction) => {
                Value::String(snapshot.mode.right_click_action.as_str().into())
            }
            Some(ZoomProperty::ContextMenu) | Some(ZoomProperty::UIContextMenu) => {
                object_property_or_default(handle, name, Value::Num(0.0))?
            }
            Some(ZoomProperty::ButtonDownFilter)
            | Some(ZoomProperty::ActionPreCallback)
            | Some(ZoomProperty::ActionPostCallback) => {
                object_property_or_default(handle, name, Value::string(String::new()))?
            }
            Some(ZoomProperty::FigureHandle) => Value::Num(snapshot.figure.as_u32() as f64),
            Some(ZoomProperty::UseLegacyExplorationModes) => Value::String(
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
    if let Some(key) = canonical_property(name).and_then(ZoomProperty::persistent) {
        return super::state::mode_properties::get(
            &mode_receiver(handle)?,
            key,
            default,
            BUILTIN_NAME,
        );
    }
    let key = canonical_property(name)
        .map(ZoomProperty::name)
        .unwrap_or(name)
        .to_string();
    runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) => object.properties().get(&key).cloned().unwrap_or(default),
        _ => default,
    })
    .map_err(|err| internal(err.to_string()))
}

pub(crate) fn set_zoom_object_properties_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    args: crate::call::FunctionArgumentView<'_>,
    _builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(handle) = read_zoom_handle(value).cloned() else {
        return Ok(None);
    };
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(invalid("expected zoom property/value pairs"));
    }
    snapshot_from_object(&handle)?;
    let mut idx = 0;
    while idx < args.len() {
        let name = super::properties::property_name_text(&args[idx], "set")
            .map_err(|_| invalid("zoom property names must be scalar text"))?;
        let value = &args[idx + 1];
        apply_zoom_property(mode_receiver(&handle)?, &name, value)?;
        write_object_property(runtime, &handle, &name, value.clone())?;
        idx += 2;
    }
    let refreshed = snapshot_from_object(&handle)?;
    let updated = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        if let Value::Object(object) = target {
            update_object_properties(runtime, object, refreshed)
        } else {
            Err(invalid("zoom handle target is not an object"))
        }
    })
    .map_err(|err| internal(err.to_string()))?;
    updated?;
    Ok(Some(()))
}

fn write_object_property(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &HandleRef,
    name: &str,
    value: Value,
) -> BuiltinResult<()> {
    if let Some(key) = canonical_property(name).and_then(ZoomProperty::persistent) {
        return super::state::mode_properties::set_in_context(
            runtime,
            &mode_receiver(handle)?,
            key,
            value,
            BUILTIN_NAME,
        );
    }
    let key = canonical_property(name)
        .map(ZoomProperty::name)
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
                        "RunMat:zoom:HostAllocation",
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

fn apply_zoom_property(
    figure: impl Into<super::state::FigureTarget>,
    name: &str,
    value: &Value,
) -> BuiltinResult<()> {
    match canonical_property(name) {
        Some(ZoomProperty::Enable) => {
            let enabled = parse_on_off(value, "Enable")?;
            set_zoom_enabled_for_figure(figure, enabled).map_err(|err| invalid(err.to_string()))
        }
        Some(ZoomProperty::Motion) => set_zoom_motion_for_figure(figure, parse_motion(value)?)
            .map_err(|err| invalid(err.to_string())),
        Some(ZoomProperty::Direction) => {
            set_zoom_direction_for_figure(figure, parse_direction(value)?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some(ZoomProperty::RightClickAction) => {
            set_zoom_right_click_action_for_figure(figure, parse_right_click_action(value)?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some(ZoomProperty::UseLegacyExplorationModes) => {
            set_zoom_legacy_for_figure(figure, parse_on_off(value, "UseLegacyExplorationModes")?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some(ZoomProperty::FigureHandle) | Some(ZoomProperty::AxesIndex) => {
            Err(invalid(format!("{name} is read-only")))
        }
        Some(ZoomProperty::ContextMenu)
        | Some(ZoomProperty::UIContextMenu)
        | Some(ZoomProperty::ButtonDownFilter)
        | Some(ZoomProperty::ActionPreCallback)
        | Some(ZoomProperty::ActionPostCallback) => Ok(()),
        _ => Err(invalid(format!("unsupported zoom property '{name}'"))),
    }
}

fn parse_on_off(value: &Value, property: &str) -> BuiltinResult<bool> {
    super::style::value_as_bool(value)
        .ok_or_else(|| invalid(format!("{property} must be 'on' or 'off'")))
}

fn parse_motion(value: &Value) -> BuiltinResult<ZoomMotion> {
    let text = super::style::value_as_string(value)
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "zoom",
                source.to_string(),
                source,
            )
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

fn parse_direction(value: &Value) -> BuiltinResult<ZoomDirection> {
    let text = super::style::value_as_string(value)
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "zoom",
                source.to_string(),
                source,
            )
        })?
        .ok_or_else(|| invalid("Direction must be 'in' or 'out'"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "in" => Ok(ZoomDirection::In),
        "out" => Ok(ZoomDirection::Out),
        _ => Err(invalid("Direction must be 'in' or 'out'")),
    }
}

fn parse_right_click_action(value: &Value) -> BuiltinResult<ZoomRightClickAction> {
    let text = super::style::value_as_string(value)
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "zoom",
                source.to_string(),
                source,
            )
        })?
        .ok_or_else(|| invalid("RightClickAction must be 'PostContextMenu' or 'InverseZoom'"))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "postcontextmenu" => Ok(ZoomRightClickAction::PostContextMenu),
        "inversezoom" => Ok(ZoomRightClickAction::InverseZoom),
        _ => Err(invalid(
            "RightClickAction must be 'PostContextMenu' or 'InverseZoom'",
        )),
    }
}

#[derive(Clone, Copy)]
enum ZoomProperty {
    Enable,
    Motion,
    Direction,
    RightClickAction,
    ContextMenu,
    UIContextMenu,
    ButtonDownFilter,
    ActionPreCallback,
    ActionPostCallback,
    FigureHandle,
    AxesIndex,
    UseLegacyExplorationModes,
}
impl ZoomProperty {
    fn name(self) -> &'static str {
        match self {
            Self::Enable => "Enable",
            Self::Motion => "Motion",
            Self::Direction => "Direction",
            Self::RightClickAction => "RightClickAction",
            Self::ContextMenu => "ContextMenu",
            Self::UIContextMenu => "UIContextMenu",
            Self::ButtonDownFilter => "ButtonDownFilter",
            Self::ActionPreCallback => "ActionPreCallback",
            Self::ActionPostCallback => "ActionPostCallback",
            Self::FigureHandle => "FigureHandle",
            Self::AxesIndex => "AxesIndex",
            Self::UseLegacyExplorationModes => "UseLegacyExplorationModes",
        }
    }
    fn persistent(self) -> Option<super::state::mode_properties::ModeProperty> {
        use super::state::mode_properties::ModeProperty as Property;
        use crate::context::GraphicsCallbackKind as Kind;
        match self {
            Self::ContextMenu => Some(Property::ZoomContextMenu),
            Self::UIContextMenu => Some(Property::ZoomLegacyContextMenu),
            Self::ButtonDownFilter => Some(Property::Callback(Kind::ZoomButtonDownFilter)),
            Self::ActionPreCallback => Some(Property::Callback(Kind::ZoomActionPre)),
            Self::ActionPostCallback => Some(Property::Callback(Kind::ZoomActionPost)),
            _ => None,
        }
    }
}
fn canonical_property(name: &str) -> Option<ZoomProperty> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some(ZoomProperty::Enable),
        "motion" => Some(ZoomProperty::Motion),
        "direction" => Some(ZoomProperty::Direction),
        "rightclickaction" => Some(ZoomProperty::RightClickAction),
        "contextmenu" => Some(ZoomProperty::ContextMenu),
        "uicontextmenu" => Some(ZoomProperty::UIContextMenu),
        "buttondownfilter" => Some(ZoomProperty::ButtonDownFilter),
        "actionprecallback" => Some(ZoomProperty::ActionPreCallback),
        "actionpostcallback" => Some(ZoomProperty::ActionPostCallback),
        "figurehandle" => Some(ZoomProperty::FigureHandle),
        "axesindex" => Some(ZoomProperty::AxesIndex),
        "uselegacyexplorationmodes" => Some(ZoomProperty::UseLegacyExplorationModes),
        _ => None,
    }
}

fn mode_receiver(handle: &HandleRef) -> BuiltinResult<runmat_value::GraphicsReceiverRef> {
    runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) if *object.class_name() == ZOOM_CLASS_NAME => {
            object.graphics_receiver().cloned()
        }
        _ => None,
    })
    .map_err(|cause| internal(cause.to_string()))?
    .ok_or_else(|| invalid("mode object has no exact Figure receiver"))
}

#[cfg(test)]
mod tests {
    fn zoom_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        arguments: Vec<Value>,
    ) -> BuiltinResult<Value> {
        let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
        super::zoom_builtin(runtime, arguments)
    }

    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_value::IntValue;

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
        zoom_builtin(None, vec![Value::String("on".into())]).unwrap();

        let z = zoom_builtin(None, Vec::new()).unwrap();
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
        let z = zoom_builtin(None, Vec::new()).unwrap();
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

        let z = zoom_builtin(None, Vec::new()).unwrap();
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

        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        zoom_builtin(None, vec![Value::Num(2.0)]).unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_limits, Some((2.5, 7.5)));
        assert_eq!(meta.y_limits, Some((5.0, 15.0)));
    }

    #[test]
    fn zoom_reset_and_out_restore_recorded_baseline() {
        let _guard = setup_plot_tests();
        crate::builtins::plotting::state::set_axis_limits(Some((0.0, 10.0)), Some((0.0, 20.0)));
        zoom_builtin(None, vec![Value::String("reset".into())]).unwrap();

        crate::builtins::plotting::state::set_axis_limits(Some((1.0, 2.0)), Some((3.0, 4.0)));
        zoom_builtin(None, vec![Value::String("out".into())]).unwrap();

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
        crate::builtins::plotting::state::set_axis_manual_limits_for_axes(
            figure,
            axes_index,
            Some((0.0, 10.0)),
            Some((0.0, 20.0)),
        )
        .unwrap();
        crate::builtins::plotting::state::select_axes_for_figure(figure, 1).unwrap();

        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        zoom_builtin(None, vec![Value::Num(ax1), Value::Num(2.0)]).unwrap();

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
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        zoom_builtin(None, vec![Value::Int(IntValue::U8(2))]).expect("integer factor");
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_limits, Some((2.5, 7.5)));
        assert_eq!(meta.y_limits, Some((5.0, 15.0)));
    }
}
