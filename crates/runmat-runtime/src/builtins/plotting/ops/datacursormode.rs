//! MATLAB-compatible `datacursormode` interaction-state builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, IntValue, ObjectInstance, StructValue, Tensor, Value};

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

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode option: on, off, or toggle.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
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
        label: "datacursormode(option)",
        inputs: &INPUTS_OPTION,
        outputs: &[],
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

pub const DATACURSOR_NUMERIC_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datacursormode-numeric-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "datacursormode with RunMat numeric graphics-handle aliases is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatacursormodeNumericHandleExtension"),
    };
pub const DATACURSOR_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datacursormode-status-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "status output from a datacursormode command form is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatacursormodeStatusOutputExtension"),
    };
pub const DATACURSOR_ONOFF_ALIASES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datacursormode-onoff-aliases",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "nonstandard datacursormode on/off numeric and text aliases are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatacursormodeOnOffAliasesExtension"),
    };
pub const DATACURSORMODE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    DATACURSOR_NUMERIC_HANDLE_EXTENSION,
    DATACURSOR_STATUS_OUTPUT_EXTENSION,
    DATACURSOR_ONOFF_ALIASES_EXTENSION,
];

const DATACURSOR_INTEGER_HANDLE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integers are exact scalar aliases for RunMat's numeric figure registry and are separately compatibility gated; this extension-only form is not advertised as a public signature.",
    }];
const DATACURSOR_INTEGER_ENABLE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Enable",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric on/off property accepts only exact scalar 0 or 1; other numeric truthiness is a separately gated alias.",
    }];
const DATACURSOR_RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident fig or property",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident numeric arrays reject without provider access.",
    }];
pub const DATACURSORMODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "dcm = datacursormode(integer_fig)", inputs: &DATACURSOR_INTEGER_HANDLE, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "RunMat-only exact numeric handle alias for the current numeric graphics registry; native Figure-object admission is not implemented." },
    BuiltinIntegerCapabilityDescriptor { form: "set(dcm, 'Enable', integer_state)", inputs: &DATACURSOR_INTEGER_ENABLE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight classes are admitted only for authoritative scalar zero or one." },
    BuiltinIntegerCapabilityDescriptor { form: "datacursormode(resident_integer, ...)", inputs: &DATACURSOR_RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Interaction metadata never gathers a resident array to reinterpret it as a handle or switch." },
];

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
    extensions(crate::builtins::plotting::datacursormode::DATACURSORMODE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::plotting::datacursormode::DATACURSORMODE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::plotting::datacursormode"
)]
pub fn datacursormode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => {
            let figure = ensure_current_figure();
            if requested_outputs() == 0 {
                ensure_command_status_allowed()?;
                toggle_figure(figure)?;
                command_status()
            } else {
                object_from_snapshot(
                    data_cursor_state_snapshot(figure).map_err(|err| invalid(err.to_string()))?,
                )
            }
        }
        [single] => {
            if let Some(option) = super::style::value_as_string(single) {
                ensure_command_status_allowed()?;
                apply_option(ensure_current_figure(), &option)?;
                return command_status();
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
        .or_insert_with(empty_update_fcn);
    object
        .properties
        .entry("Interpreter".into())
        .or_insert(Value::String("tex".into()));
    object
        .properties
        .insert("Figure".into(), Value::Num(snapshot.figure.as_u32() as f64));
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
            .get("Figure")
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
    st.insert("UpdateFcn", empty_update_fcn());
    st.insert("Interpreter", Value::String("tex".into()));
    st.insert("Figure", Value::Num(snapshot.figure.as_u32() as f64));
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
                    object_property_or_default(handle, "UpdateFcn", empty_update_fcn())?,
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
                object_property_or_default(handle, "UpdateFcn", empty_update_fcn())?
            }
            Some("Interpreter") => {
                object_property_or_default(handle, "Interpreter", Value::String("tex".into()))?
            }
            Some("Figure") => Value::Num(snapshot.figure.as_u32() as f64),
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
        Some("Enable") => set_data_cursor_enabled_for_figure(figure, parse_enable(value)?)
            .map_err(|err| invalid(err.to_string())),
        Some("SnapToDataVertex") => {
            set_data_cursor_snap_to_data_vertex_for_figure(figure, parse_snap(value)?)
                .map_err(|err| invalid(err.to_string()))
        }
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
        Some("UpdateFcn") => validate_update_fcn(value),
        Some("Interpreter") => {
            let interpreter = super::style::value_as_string(value)
                .ok_or_else(|| invalid("Interpreter must be 'tex', 'latex', or 'none'"))?;
            match interpreter.trim().to_ascii_lowercase().as_str() {
                "tex" | "latex" | "none" => Ok(()),
                _ => Err(invalid("Interpreter must be 'tex', 'latex', or 'none'")),
            }
        }
        Some("Figure") => Err(invalid("Figure is read-only")),
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

fn empty_update_fcn() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty UpdateFcn value"))
}

fn validate_update_fcn(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) if tensor.is_empty() => Ok(()),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. } => Ok(()),
        _ => Err(invalid("UpdateFcn must be empty or a function handle")),
    }
}

fn parse_enable(value: &Value) -> BuiltinResult<bool> {
    if let Some(value) = strict_numeric_on_off(value) {
        return Ok(value);
    }
    if let Some(text) = super::style::value_as_string(value) {
        match text.trim().to_ascii_lowercase().as_str() {
            "on" => return Ok(true),
            "off" => return Ok(false),
            _ => {}
        }
    }
    broad_on_off_alias(value, "Enable")
}

fn parse_snap(value: &Value) -> BuiltinResult<bool> {
    if let Some(text) = super::style::value_as_string(value) {
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

fn canonical_property(name: &str) -> Option<String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "enable" => Some("Enable".into()),
        "snaptodatavertex" => Some("SnapToDataVertex".into()),
        "displaystyle" => Some("DisplayStyle".into()),
        "updatefcn" => Some("UpdateFcn".into()),
        "interpreter" => Some("Interpreter".into()),
        "figure" => Some("Figure".into()),
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
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
        datacursormode_builtin(vec![Value::String("off".into())]).unwrap();
        let figure = current_figure_handle().as_u32() as f64;
        let dcm = datacursormode_builtin(vec![Value::Num(figure)]).unwrap();
        assert!(matches!(dcm, Value::HandleObject(_)));
    }

    #[test]
    fn datacursormode_rejects_missing_explicit_figure_handle() {
        let _guard = setup_plot_tests();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = datacursormode_builtin(vec![Value::Num(999.0), Value::String("on".into())])
            .unwrap_err();
        assert!(err.to_string().contains("figure handle does not exist"));
    }

    #[test]
    fn datacursormode_extensions_are_independently_gated() {
        let _guard = setup_plot_tests();
        let figure = current_figure_handle().as_u32();
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let _output = crate::output_count::push_output_count(Some(1));
        let error = datacursormode_builtin(vec![Value::String("on".into())])
            .expect_err("status output gate");
        assert_eq!(
            error.identifier(),
            DATACURSOR_STATUS_OUTPUT_EXTENSION.error_identifier
        );
        drop(_output);

        let error = datacursormode_builtin(vec![Value::Int(IntValue::U32(figure))])
            .expect_err("numeric handle gate");
        assert_eq!(
            error.identifier(),
            DATACURSOR_NUMERIC_HANDLE_EXTENSION.error_identifier
        );

        let dcm = datacursormode_builtin(Vec::new()).unwrap();
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
        assert!(datacursormode_builtin(vec![Value::Num(figure + 0.5)]).is_err());
        assert!(datacursormode_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![figure, figure], vec![1, 2]).unwrap(),
        )])
        .is_err());

        let dcm = datacursormode_builtin(vec![Value::Num(figure)]).unwrap();
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
        let dcm = datacursormode_builtin(Vec::new()).unwrap();
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
            datacursormode_builtin(vec![Value::Int(IntValue::U32(figure))]).unwrap(),
            Value::HandleObject(_)
        ));
        let error = datacursormode_builtin(vec![Value::Int(IntValue::U64(9_007_199_254_740_993))])
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
            let error = datacursormode_builtin(vec![Value::GpuTensor(handle.clone())])
                .expect_err("resident figure alias must reject");
            assert!(error.message().contains("numeric scalar"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let _ = provider.free(&handle);
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
        let dcm = datacursormode_builtin(Vec::new()).unwrap();
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
