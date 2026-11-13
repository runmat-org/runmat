//! MATLAB-compatible `datacursormode` interaction-state builtin.

use runmat_builtins::BuiltinErrorDescriptor;

use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, IntValue, ObjectInstance, Tensor, Value};

use super::state::{
    current_figure_handle, data_cursor_state_snapshot, figure_handle_exists, select_figure,
    set_data_cursor_display_style_for_figure, set_data_cursor_enabled_for_figure,
    set_data_cursor_snap_to_data_vertex_for_figure, DataCursorStateSnapshot, FigureHandle,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "datacursormode";
const DATA_CURSOR_CLASS_NAME: runmat_types::StaticClassIdentity =
    runmat_types::standard::DATA_CURSOR_CONTROL;

pub use runmat_builtins::DATACURSORMODE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT;

pub use runmat_builtins::DATACURSORMODE_ERROR_INTERNAL as ERROR_INTERNAL;

#[cfg(test)]
use runmat_builtins::DATACURSORMODE_DESCRIPTOR;

pub use runmat_builtins::DATACURSORMODE_DATACURSOR_NUMERIC_HANDLE_EXTENSION as DATACURSOR_NUMERIC_HANDLE_EXTENSION;
use runmat_builtins::DATACURSORMODE_DATACURSOR_ONOFF_ALIASES_EXTENSION as DATACURSOR_ONOFF_ALIASES_EXTENSION;
pub use runmat_builtins::DATACURSORMODE_DATACURSOR_STATUS_OUTPUT_EXTENSION as DATACURSOR_STATUS_OUTPUT_EXTENSION;

#[cfg(test)]
use runmat_builtins::DATACURSORMODE_INTEGER_CAPABILITIES;

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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::datacursormode"
)]
pub fn datacursormode_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    match (args.len(), args.first(), args.get(1)) {
        (0, _, _) => {
            let figure = ensure_current_figure();
            if requested_outputs() == 0 {
                ensure_command_status_allowed()?;
                toggle_figure(figure)?;
                command_status()
            } else {
                object_from_snapshot(
                    runtime.as_ref(),
                    data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?,
                )
            }
        }
        (1, Some(single), _) => {
            if let Some(option) = super::style::value_as_string(single).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "datacursormode",
                    source.to_string(),
                    source,
                )
            })? {
                ensure_command_status_allowed()?;
                apply_option(ensure_current_figure(), &option)?;
                return command_status();
            }
            let figure = existing_figure_handle(single)?;
            object_from_snapshot(
                runtime.as_ref(),
                data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?,
            )
        }
        (2, Some(figure), Some(option)) => {
            let figure = existing_figure_handle(figure)?;
            let option = super::style::value_as_string(option)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "datacursormode",
                        source.to_string(),
                        source,
                    )
                })?
                .ok_or_else(|| invalid("expected option string"))?;
            ensure_command_status_allowed()?;
            apply_option(figure, &option)?;
            command_status()
        }
        _ => Err(invalid(
            "expected datacursormode(), datacursormode(option), or datacursormode(fig, option)",
        )),
    }
}

fn command_status() -> BuiltinResult<Value> {
    Ok(Value::String("ok".into()))
}

fn ensure_command_status_allowed() -> BuiltinResult<()> {
    if requested_outputs() > 0 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATACURSOR_STATUS_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn rounded_handle_id(value: &Value) -> BuiltinResult<u32> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &DATACURSOR_NUMERIC_HANDLE_EXTENSION,
        BUILTIN_NAME,
    )?;
    let raw = exact_handle_u64(value)?;
    u32::try_from(raw).map_err(|_| invalid("figure handle is too large"))
}

fn exact_handle_u64(value: &Value) -> BuiltinResult<u64> {
    match value {
        Value::Int(value) => value
            .try_to_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid("figure handle must be a positive integer scalar")),
        Value::Num(value) => exact_f64_handle(*value),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return exact_handle_u64(&Value::Int(
                    storage.value_at(0).expect("scalar integer storage"),
                ));
            }
            exact_f64_handle(crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
        }
        _ => Err(invalid("figure handle must be a numeric scalar")),
    }
}

fn exact_f64_handle(value: f64) -> BuiltinResult<u64> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > u64::MAX as f64 {
        return Err(invalid("figure handle must be a positive integer scalar"));
    }
    Ok(value as u64)
}

fn existing_figure_handle(value: &Value) -> BuiltinResult<FigureHandle> {
    let handle = FigureHandle::from(rounded_handle_id(value)?);
    if figure_handle_exists(handle) {
        Ok(handle)
    } else {
        Err(invalid("figure handle does not exist"))
    }
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

fn object_from_snapshot(
    runtime: Option<&crate::context::RuntimeContext>,
    snapshot: DataCursorStateSnapshot,
) -> BuiltinResult<Value> {
    use super::op_common::mode_object::{Field, FieldValue};
    let mut fields = [
        Field {
            name: "Enable",
            value: FieldValue::Text(if snapshot.mode.enabled { "on" } else { "off" }),
        },
        Field {
            name: "SnapToDataVertex",
            value: FieldValue::Text(if snapshot.mode.snap_to_data_vertex {
                "on"
            } else {
                "off"
            }),
        },
        Field {
            name: "DisplayStyle",
            value: FieldValue::Text(&snapshot.mode.display_style),
        },
        Field {
            name: "UpdateFcn",
            value: FieldValue::EmptyF64Tensor(&[0, 0]),
        },
        Field {
            name: "Interpreter",
            value: FieldValue::Text("tex"),
        },
        Field {
            name: "Figure",
            value: FieldValue::Value(Some(Value::Num(snapshot.figure.as_u32() as f64))),
        },
    ];
    let object = super::op_common::mode_object::construct(
        runtime,
        DATA_CURSOR_CLASS_NAME,
        super::state::mode_properties::receiver(snapshot.figure, BUILTIN_NAME)?,
        &mut fields,
    )?;
    let (object, _roots) = object.into_value_and_lease();
    let Value::Object(object) = object else {
        unreachable!("data cursor snapshot publishes an object")
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
    snapshot: DataCursorStateSnapshot,
) -> BuiltinResult<()> {
    use super::op_common::mode_object::{Field, FieldValue};
    let mut fields = [
        Field {
            name: "Enable",
            value: FieldValue::Text(if snapshot.mode.enabled { "on" } else { "off" }),
        },
        Field {
            name: "SnapToDataVertex",
            value: FieldValue::Text(if snapshot.mode.snap_to_data_vertex {
                "on"
            } else {
                "off"
            }),
        },
        Field {
            name: "DisplayStyle",
            value: FieldValue::Text(&snapshot.mode.display_style),
        },
        Field {
            name: "Figure",
            value: FieldValue::Value(Some(Value::Num(snapshot.figure.as_u32() as f64))),
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
                "RunMat:datacursormode:HostAllocation",
                message,
                source,
            )
        },
        invalid,
    )
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
        if *object.class_name() != DATA_CURSOR_CLASS_NAME {
            return None;
        }
        object.graphics_receiver().cloned()
    })
    .map_err(|err| internal(err.to_string()))?
    .ok_or_else(|| invalid("invalid datacursormode object"))?;
    data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))
}

fn cursor_struct(snapshot: DataCursorStateSnapshot) -> Value {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
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
    st.insert("DisplayStyle", Value::string(snapshot.mode.display_style));
    st.insert("UpdateFcn", empty_update_fcn());
    st.insert("Interpreter", Value::String("tex".into()));
    st.insert("Figure", Value::Num(snapshot.figure.as_u32() as f64));
    Value::Struct(st.finish())
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
            let Value::Struct(st) = cursor_struct(snapshot) else {
                unreachable!("cursor snapshot is a structure")
            };
            let mut st = runmat_value::UnboundedStructValueBuilder::from_structure(st);
            st.insert(
                "UpdateFcn",
                object_property_or_default(handle, "UpdateFcn", empty_update_fcn())?,
            );
            Value::Struct(st.finish())
        }
        Some(name) => match canonical_property(name) {
            Some(DataCursorProperty::Enable) => {
                Value::String(if snapshot.mode.enabled { "on" } else { "off" }.into())
            }
            Some(DataCursorProperty::SnapToDataVertex) => Value::String(
                if snapshot.mode.snap_to_data_vertex {
                    "on"
                } else {
                    "off"
                }
                .into(),
            ),
            Some(DataCursorProperty::DisplayStyle) => Value::string(snapshot.mode.display_style),
            Some(DataCursorProperty::UpdateFcn) => {
                object_property_or_default(handle, "UpdateFcn", empty_update_fcn())?
            }
            Some(DataCursorProperty::Interpreter) => {
                object_property_or_default(handle, "Interpreter", Value::String("tex".into()))?
            }
            Some(DataCursorProperty::Figure) => Value::Num(snapshot.figure.as_u32() as f64),
            _ => {
                return Err(invalid(format!(
                    "unsupported datacursormode property '{name}'"
                )))
            }
        },
    };
    Ok(Some(value))
}

#[cfg(test)]
pub fn set_data_cursor_object_properties(
    value: &Value,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    set_data_cursor_object_properties_in_context(
        None,
        value,
        crate::call::FunctionArgumentView::from_slice(args),
        builtin,
    )
}

pub(crate) fn set_data_cursor_object_properties_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    args: crate::call::FunctionArgumentView<'_>,
    _builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(handle) = read_data_cursor_handle(value).cloned() else {
        return Ok(None);
    };
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(invalid("expected datacursormode property/value pairs"));
    }
    snapshot_from_object(&handle)?;
    for pair in args.chunks_exact(2) {
        let name = super::properties::property_name_text(&pair[0], "set")
            .map_err(|_| invalid("datacursormode property names must be scalar text"))?;
        apply_property(mode_receiver(&handle)?, &name, &pair[1])?;
        write_object_property(runtime, &handle, &name, pair[1].clone())?;
    }
    let refreshed = snapshot_from_object(&handle)?;
    let updated = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        if let Value::Object(object) = target {
            update_object_properties(runtime, object, refreshed)
        } else {
            Err(invalid("datacursormode handle target is not an object"))
        }
    })
    .map_err(|err| internal(err.to_string()))?;
    updated?;
    Ok(Some(()))
}

fn apply_property(
    figure: impl Into<super::state::FigureTarget>,
    name: &str,
    value: &Value,
) -> BuiltinResult<()> {
    match canonical_property(name) {
        Some(DataCursorProperty::Enable) => {
            set_data_cursor_enabled_for_figure(figure, parse_enable(value)?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some(DataCursorProperty::SnapToDataVertex) => {
            set_data_cursor_snap_to_data_vertex_for_figure(figure, parse_snap(value)?)
                .map_err(|err| invalid(err.to_string()))
        }
        Some(DataCursorProperty::DisplayStyle) => {
            let style = super::style::value_as_string(value)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "datacursormode",
                        source.to_string(),
                        source,
                    )
                })?
                .ok_or_else(|| invalid("DisplayStyle must be 'datatip' or 'window'"))?;
            let normalized = match style.trim().to_ascii_lowercase().as_str() {
                "datatip" => "datatip",
                "window" => "window",
                _ => return Err(invalid("DisplayStyle must be 'datatip' or 'window'")),
            };
            set_data_cursor_display_style_for_figure(figure, normalized.to_string())
                .map_err(|err| invalid(err.to_string()))
        }
        Some(DataCursorProperty::UpdateFcn) => validate_update_fcn(value),
        Some(DataCursorProperty::Interpreter) => {
            let interpreter = super::style::value_as_string(value)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "datacursormode",
                        source.to_string(),
                        source,
                    )
                })?
                .ok_or_else(|| invalid("Interpreter must be 'tex', 'latex', or 'none'"))?;
            match interpreter.trim().to_ascii_lowercase().as_str() {
                "tex" | "latex" | "none" => Ok(()),
                _ => Err(invalid("Interpreter must be 'tex', 'latex', or 'none'")),
            }
        }
        Some(DataCursorProperty::Figure) => Err(invalid("Figure is read-only")),
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
    if let Some(key) = canonical_property(name).and_then(DataCursorProperty::persistent) {
        return super::state::mode_properties::get(
            &mode_receiver(handle)?,
            key,
            default,
            BUILTIN_NAME,
        );
    }
    let key = canonical_property(name)
        .map(DataCursorProperty::name)
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
    if let Some(key) = canonical_property(name).and_then(DataCursorProperty::persistent) {
        return super::state::mode_properties::set_in_context(
            runtime,
            &mode_receiver(handle)?,
            key,
            value,
            BUILTIN_NAME,
        );
    }
    let key = canonical_property(name)
        .map(DataCursorProperty::name)
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
                        "RunMat:datacursormode:HostAllocation",
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

fn empty_update_fcn() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty UpdateFcn value"))
}

fn validate_update_fcn(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) if tensor.is_empty() => Ok(()),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_) => Ok(()),
        _ => Err(invalid("UpdateFcn must be empty or a function handle")),
    }
}

fn parse_enable(value: &Value) -> BuiltinResult<bool> {
    if let Some(value) = strict_numeric_on_off(value) {
        return Ok(value);
    }
    if let Some(text) = super::style::value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source(
            "datacursormode",
            source.to_string(),
            source,
        )
    })? {
        match text.trim().to_ascii_lowercase().as_str() {
            "on" => return Ok(true),
            "off" => return Ok(false),
            _ => {}
        }
    }
    broad_on_off_alias(value, "Enable")
}

fn parse_snap(value: &Value) -> BuiltinResult<bool> {
    if let Some(text) = super::style::value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source(
            "datacursormode",
            source.to_string(),
            source,
        )
    })? {
        match text.trim().to_ascii_lowercase().as_str() {
            "on" => return Ok(true),
            "off" => return Ok(false),
            _ => {}
        }
    }
    broad_on_off_alias(value, "SnapToDataVertex")
}

fn broad_on_off_alias(value: &Value, property: &str) -> BuiltinResult<bool> {
    let parsed = super::style::value_as_bool(value)
        .ok_or_else(|| invalid(format!("{property} must be 'on' or 'off'")))?;
    crate::compatibility::ensure_builtin_extension_enabled(
        &DATACURSOR_ONOFF_ALIASES_EXTENSION,
        BUILTIN_NAME,
    )?;
    Ok(parsed)
}

fn strict_numeric_on_off(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::Int(value) => match value {
            IntValue::I8(0)
            | IntValue::I16(0)
            | IntValue::I32(0)
            | IntValue::I64(0)
            | IntValue::U8(0)
            | IntValue::U16(0)
            | IntValue::U32(0)
            | IntValue::U64(0) => Some(false),
            IntValue::I8(1)
            | IntValue::I16(1)
            | IntValue::I32(1)
            | IntValue::I64(1)
            | IntValue::U8(1)
            | IntValue::U16(1)
            | IntValue::U32(1)
            | IntValue::U64(1) => Some(true),
            _ => None,
        },
        Value::Num(0.0) => Some(false),
        Value::Num(1.0) => Some(true),
        Value::LogicalArray(array) if array.data.len() == 1 => Some(array.data[0] != 0),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                strict_numeric_on_off(&Value::Int(storage.value_at(0)?))
            } else {
                strict_numeric_on_off(&Value::Num(
                    crate::builtins::common::tensor::tensor_value_f64(tensor, 0),
                ))
            }
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum DataCursorProperty {
    Enable,
    SnapToDataVertex,
    DisplayStyle,
    UpdateFcn,
    Interpreter,
    Figure,
}
impl DataCursorProperty {
    fn name(self) -> &'static str {
        match self {
            Self::Enable => "Enable",
            Self::SnapToDataVertex => "SnapToDataVertex",
            Self::DisplayStyle => "DisplayStyle",
            Self::UpdateFcn => "UpdateFcn",
            Self::Interpreter => "Interpreter",
            Self::Figure => "Figure",
        }
    }
    fn persistent(self) -> Option<super::state::mode_properties::ModeProperty> {
        use super::state::mode_properties::ModeProperty as Property;
        use crate::context::GraphicsCallbackKind as Kind;
        match self {
            Self::UpdateFcn => Some(Property::Callback(Kind::DataCursorUpdate)),
            Self::Interpreter => Some(Property::DataCursorInterpreter),
            _ => None,
        }
    }
}
fn canonical_property(name: &str) -> Option<DataCursorProperty> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some(DataCursorProperty::Enable),
        "snaptodatavertex" => Some(DataCursorProperty::SnapToDataVertex),
        "displaystyle" => Some(DataCursorProperty::DisplayStyle),
        "updatefcn" => Some(DataCursorProperty::UpdateFcn),
        "interpreter" => Some(DataCursorProperty::Interpreter),
        "figure" => Some(DataCursorProperty::Figure),
        _ => None,
    }
}

fn mode_receiver(handle: &HandleRef) -> BuiltinResult<runmat_value::GraphicsReceiverRef> {
    runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) if *object.class_name() == DATA_CURSOR_CLASS_NAME => {
            object.graphics_receiver().cloned()
        }
        _ => None,
    })
    .map_err(|cause| internal(cause.to_string()))?
    .ok_or_else(|| invalid("mode object has no exact Figure receiver"))
}

#[cfg(test)]
mod tests {
    fn datacursormode_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        arguments: Vec<Value>,
    ) -> BuiltinResult<Value> {
        let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
        super::datacursormode_builtin(runtime, arguments)
    }

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
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        datacursormode_builtin(None, vec![Value::String("on".into())]).unwrap();
        let dcm = datacursormode_builtin(None, Vec::new()).unwrap();
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
    fn datacursormode_public_descriptor_excludes_extension_only_outputs_and_aliases() {
        let labels = DATACURSORMODE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "dcm = datacursormode()",
                "datacursormode()",
                "datacursormode(option)",
            ]
        );
        assert!(DATACURSORMODE_DESCRIPTOR
            .signatures
            .iter()
            .all(
                |signature| !signature.label.contains("status") && !signature.label.contains("fig")
            ));
    }

    #[test]
    fn datacursormode_figure_handle_returns_object() {
        let _guard = setup_plot_tests();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        datacursormode_builtin(None, vec![Value::String("off".into())]).unwrap();
        let figure = current_figure_handle().as_u32() as f64;
        let dcm = datacursormode_builtin(None, vec![Value::Num(figure)]).unwrap();
        assert!(matches!(dcm, Value::HandleObject(_)));
    }

    #[test]
    fn datacursormode_rejects_missing_explicit_figure_handle() {
        let _guard = setup_plot_tests();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = datacursormode_builtin(None, vec![Value::Num(999.0), Value::String("on".into())])
            .unwrap_err();
        assert!(err.to_string().contains("figure handle does not exist"));
    }

    #[test]
    fn datacursormode_extensions_are_independently_gated() {
        let _guard = setup_plot_tests();
        let figure = current_figure_handle().as_u32();
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let _output = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let error = datacursormode_builtin(None, vec![Value::String("on".into())])
            .expect_err("status output gate");
        assert_eq!(
            error.identifier(),
            DATACURSOR_STATUS_OUTPUT_EXTENSION.error_identifier
        );
        drop(_output);

        let error = datacursormode_builtin(None, vec![Value::Int(IntValue::U32(figure))])
            .expect_err("numeric handle gate");
        assert_eq!(
            error.identifier(),
            DATACURSOR_NUMERIC_HANDLE_EXTENSION.error_identifier
        );

        let dcm = datacursormode_builtin(None, Vec::new()).unwrap();
        let error = set_data_cursor_object_properties(
            &dcm,
            &[Value::String("SnapToDataVertex".into()), Value::Num(1.0)],
            "set",
        )
        .expect_err("broad on/off alias gate");
        assert_eq!(
            error.identifier(),
            DATACURSOR_ONOFF_ALIASES_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn datacursormode_exact_handle_and_official_property_contract() {
        let _guard = setup_plot_tests();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let figure = ensure_current_figure().as_u32() as f64;
        assert!(datacursormode_builtin(None, vec![Value::Num(figure + 0.5)]).is_err());
        assert!(datacursormode_builtin(
            None,
            vec![Value::Tensor(
                runmat_value::Tensor::new(vec![figure, figure], vec![1, 2]).unwrap(),
            )]
        )
        .is_err());

        let dcm = datacursormode_builtin(None, vec![Value::Num(figure)]).unwrap();
        assert_eq!(
            get_data_cursor_object_property(&dcm, Some("Interpreter"), "get").unwrap(),
            Some(Value::String("tex".into()))
        );
        assert_eq!(
            get_data_cursor_object_property(&dcm, Some("Figure"), "get").unwrap(),
            Some(Value::Num(figure))
        );
        assert!(set_data_cursor_object_properties(
            &dcm,
            &[Value::String("Figure".into()), Value::Num(figure)],
            "set",
        )
        .is_err());
        assert!(set_data_cursor_object_properties(
            &dcm,
            &[
                Value::String("Interpreter".into()),
                Value::String("latex".into()),
            ],
            "set",
        )
        .is_ok());
    }

    #[test]
    fn datacursormode_enable_accepts_scalar_logical_and_updatefcn_is_constrained() {
        let _guard = setup_plot_tests();
        let dcm = datacursormode_builtin(None, Vec::new()).unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for flag in [0, 1] {
            let logical = Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![flag], vec![1, 1]).unwrap(),
            );
            assert!(set_data_cursor_object_properties(
                &dcm,
                &[Value::String("Enable".into()), logical],
                "set",
            )
            .is_ok());
        }

        assert_eq!(
            get_data_cursor_object_property(&dcm, Some("UpdateFcn"), "get").unwrap(),
            Some(empty_update_fcn())
        );
        assert!(set_data_cursor_object_properties(
            &dcm,
            &[Value::String("UpdateFcn".into()), Value::Num(1.0)],
            "set",
        )
        .is_err());
        let callback = Value::FunctionHandle("cursor_label".into());
        assert!(set_data_cursor_object_properties(
            &dcm,
            &[Value::String("UpdateFcn".into()), callback.clone()],
            "set",
        )
        .is_ok());
        assert_eq!(
            get_data_cursor_object_property(&dcm, Some("UpdateFcn"), "get").unwrap(),
            Some(callback)
        );
        assert!(set_data_cursor_object_properties(
            &dcm,
            &[Value::String("UpdateFcn".into()), empty_update_fcn()],
            "set",
        )
        .is_ok());
    }

    #[test]
    fn datacursormode_parses_all_eight_integer_figure_aliases_exactly() {
        let _guard = setup_plot_tests();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let figure = ensure_current_figure().as_u32();
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            assert_eq!(exact_handle_u64(&Value::Int(value)).unwrap(), 1);
        }
        assert!(matches!(
            datacursormode_builtin(None, vec![Value::Int(IntValue::U32(figure))]).unwrap(),
            Value::HandleObject(_)
        ));
        let error =
            datacursormode_builtin(None, vec![Value::Int(IntValue::U64(9_007_199_254_740_993))])
                .expect_err("large integer must not round into a handle");
        assert!(error.message().contains("too large"));
    }

    #[test]
    fn datacursormode_rejects_registered_resident_handle_without_provider_access() {
        let _guard = setup_plot_tests();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap();
            provider.reset_telemetry();
            let error = datacursormode_builtin(None, vec![Value::GpuTensor(handle.clone())])
                .expect_err("resident figure alias must reject");
            assert!(error.message().contains("numeric scalar"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let _ = handle.release();
        });
    }

    #[test]
    fn datacursormode_integer_capabilities_cover_all_classes_and_resident_rejection() {
        assert_eq!(DATACURSORMODE_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            DATACURSORMODE_INTEGER_CAPABILITIES[0].inputs[0]
                .classes
                .len(),
            8
        );
        assert_eq!(
            DATACURSORMODE_INTEGER_CAPABILITIES[1].inputs[0]
                .classes
                .len(),
            8
        );
        assert!(DATACURSORMODE_INTEGER_CAPABILITIES[2].inputs[0]
            .classes
            .is_empty());

        let _guard = setup_plot_tests();
        let dcm = datacursormode_builtin(None, Vec::new()).unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            assert!(set_data_cursor_object_properties(
                &dcm,
                &[Value::String("Enable".into()), Value::Int(value)],
                "set",
            )
            .is_ok());
        }
    }
}
