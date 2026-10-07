//! Compatibility filesystem, path, process, preference, and environment helpers.

use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::process::Command;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, FileAttributeControl, FileAttributeControlError, RehashMode,
    FILEATTRIB_ERROR_TOO_MANY_OUTPUTS, FILEATTRIB_ERROR_UNSUPPORTED_PERMISSION, MEMMAPFILE_CLASS,
    MEMMAPFILE_DESCRIPTOR, MEMMAPFILE_ERROR_INVALID_CALL, MEMMAPFILE_ERROR_INVALID_CONTROL,
    MEMMAPFILE_ERROR_INVALID_FILENAME, MEMMAPFILE_ERROR_READ_FAILED,
    MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS, MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
    MEMMAPFILE_INTEGER_CONTROLS_EXTENSION,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
#[cfg(any(test, windows))]
use runmat_value::CellArray;
use runmat_value::{CharArray, ObjectInstance, StructValue, Value};

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::output_count;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const INPUTS_THREE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "input1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input argument.",
    },
    BuiltinParamDescriptor {
        name: "input2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input argument.",
    },
    BuiltinParamDescriptor {
        name: "input3",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional third input argument.",
    },
];
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
macro_rules! simple_descriptor {
    ($sig:ident, $desc:ident, $label:expr, $inputs:expr, $outputs:expr, $mode:expr) => {
        const $sig: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: $outputs,
        }];
        pub const $desc: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$sig,
            output_mode: $mode,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
    };
}

simple_descriptor!(
    WINQUERYREG_SIGNATURES,
    WINQUERYREG_DESCRIPTOR,
    "value = winqueryreg(root, key, valuename)",
    &INPUTS_THREE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
pub const WINQUERYREG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = winqueryreg(rootkey, subkey, valname) for REG_DWORD",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Text-only registry lookup inputs can return a documented exact int32 scalar for a 32-bit REG_DWORD value. The Windows command output is decoded as a 32-bit bit pattern without a floating conversion.",
    }];

pub(super) fn compat_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn map_control_flow(name: &str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{name}: {}", err.message()))
        .with_builtin(name)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) async fn gather_args(name: &str, args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(|err| map_control_flow(name, err))?,
        );
    }
    Ok(out)
}

pub(super) fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match crate::builtins::common::validation::value_text_scalar_string(value) {
        Some(mut text) => {
            if matches!(value, Value::CharArray(_)) {
                while text.ends_with(' ') {
                    text.pop();
                }
            }
            Ok(text)
        }
        None => Err(compat_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

fn char_row_to_string(array: &CharArray) -> String {
    let mut text: String = array.data.iter().take(array.cols).collect();
    while text.ends_with(' ') {
        text.pop();
    }
    text
}

pub(super) fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn expand_path_for_builtin(text: &str, name: &str) -> BuiltinResult<PathBuf> {
    let expanded = expand_user_path(text.trim(), name).map_err(|err| compat_error(name, err))?;
    Ok(PathBuf::from(expanded))
}

pub(super) fn value_to_path(value: &Value, name: &str, arg: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, arg)?;
    expand_path_for_builtin(&text, name)
}

fn output_list_for_count(default: Vec<Value>) -> Value {
    if let Some(count) = output_count::current_output_count() {
        return output_count::output_list_with_padding(count, default);
    }
    if default.len() == 1 {
        default.into_iter().next().unwrap()
    } else {
        Value::OutputList(default)
    }
}

fn value_contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(value) => value.data.iter().any(value_contains_resident),
        Value::Struct(value) => value.fields.values().any(value_contains_resident),
        Value::Object(value) => value.properties.values().any(value_contains_resident),
        Value::Closure(value) => value.captures.iter().any(value_contains_resident),
        Value::OutputList(values) => values.iter().any(value_contains_resident),
        _ => false,
    }
}

#[runtime_builtin(
    name = "system",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn system_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let requested = output_count::current_output_count();
    if requested.is_some_and(|count| count > 2) {
        return Err(compat_error(
            "system",
            "system: at most two outputs are supported",
        ));
    }
    if args.is_empty() || args.len() > 2 {
        return Err(compat_error(
            "system",
            "system: expected command and optional echo flag",
        ));
    }
    let command = crate::builtins::common::validation::value_text_scalar_string(&args[0])
        .ok_or_else(|| {
            compat_error(
                "system",
                "system: command must be a string scalar or character vector",
            )
        })?;
    let echo = if let Some(option) = args.get(1) {
        let option = crate::builtins::common::validation::value_text_scalar_string(option)
            .ok_or_else(|| {
                compat_error(
                    "system",
                    "system: option must be a string scalar or character vector",
                )
            })?;
        if !option.eq_ignore_ascii_case("-echo") {
            return Err(compat_error(
                "system",
                "system: optional second argument must be '-echo'",
            ));
        }
        true
    } else {
        false
    };
    let result = system::run(runtime.as_ref(), &command).await?;
    if (echo || requested == Some(0)) && !result.1.is_empty() {
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, &result.1);
    }
    let outputs = vec![Value::Num(result.0 as f64), char_value(&result.1)];
    if let Some(count) = requested {
        return Ok(output_count::output_list_with_padding(count, outputs));
    }
    Ok(Value::Num(result.0 as f64))
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Num(v) => *v != 0.0,
        Value::Int(v) => !v.is_zero(),
        Value::String(s) => !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false"),
        Value::CharArray(ca) => {
            let s = char_row_to_string(ca);
            !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false")
        }
        _ => true,
    }
}

#[cfg(windows)]
fn cellstr(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    Ok(Value::Cell(
        CellArray::new(values.into_iter().map(|s| char_value(&s)).collect(), len, 1)
            .map_err(|err| compat_error("cellstr", err))?,
    ))
}

#[runtime_builtin(
    name = "fileattrib",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn fileattrib_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if output_count::current_output_count().is_some_and(|count| count > 2) {
        let mut failure = build_runtime_error(FILEATTRIB_ERROR_TOO_MANY_OUTPUTS.message)
            .with_builtin("fileattrib");
        if let Some(identifier) = FILEATTRIB_ERROR_TOO_MANY_OUTPUTS.identifier {
            failure = failure.with_identifier(identifier);
        }
        return Err(failure.build());
    }
    if args.is_empty() || args.len() > 3 {
        return Err(compat_error(
            "fileattrib",
            "fileattrib: expected path and optional attribute flag",
        ));
    }
    let path = value_to_path(&args[0], "fileattrib", "path")?;
    let readonly = parse_fileattrib_flags(&args[1..])?;
    let selected = CapturedFsPath::capture(path, "fileattrib")?;
    if let Some(readonly) = readonly {
        selected.set_readonly(readonly).await.map_err(|source| {
            build_runtime_error(format!("fileattrib: {source}"))
                .with_builtin("fileattrib")
                .with_source(source)
                .build()
        })?;
    }
    match selected.metadata().await {
        Ok(meta) => Ok(output_list_for_count(vec![
            Value::Num(1.0),
            Value::Struct(fileattrib_struct(&selected.path, &meta)),
        ])),
        Err(err) => {
            selected.check_current().map_err(|source| {
                build_runtime_error(format!("fileattrib: {source}"))
                    .with_builtin("fileattrib")
                    .with_source(source)
                    .build()
            })?;
            Ok(output_list_for_count(vec![
                Value::Num(0.0),
                Value::Struct(error_struct(&err.to_string())),
            ]))
        }
    }
}

fn parse_fileattrib_flags(flags: &[Value]) -> BuiltinResult<Option<bool>> {
    // Decode the complete option list before the first filesystem mutation.
    // The last writable flag determines the single provider-owned update.
    let mut readonly = None;
    for flag_value in flags {
        let flag = scalar_text(flag_value, "fileattrib", "attribute")?;
        match FileAttributeControl::parse(&flag) {
            Ok(FileAttributeControl::SetReadOnly(value)) => readonly = Some(value),
            Err(FileAttributeControlError::ReadPermissionUnavailable) => {
                let mut failure =
                    build_runtime_error(FILEATTRIB_ERROR_UNSUPPORTED_PERMISSION.message)
                        .with_builtin("fileattrib");
                if let Some(identifier) = FILEATTRIB_ERROR_UNSUPPORTED_PERMISSION.identifier {
                    failure = failure.with_identifier(identifier);
                }
                return Err(failure.build());
            }
            Err(FileAttributeControlError::UnsupportedFlag(other)) => {
                return Err(compat_error(
                    "fileattrib",
                    format!("fileattrib: unsupported attribute flag '{other}'"),
                ));
            }
        }
    }
    Ok(readonly)
}

fn fileattrib_struct(path: &Path, meta: &vfs::FsMetadata) -> StructValue {
    let mut st = StructValue::new();
    st.insert("Name", char_value(&path_to_string(path)));
    st.insert("archive", Value::Num(f64::NAN));
    st.insert("system", Value::Num(f64::NAN));
    st.insert("hidden", Value::Num(f64::NAN));
    st.insert("directory", Value::Bool(meta.is_dir()));
    st.insert("UserRead", Value::Num(f64::NAN));
    st.insert("UserWrite", Value::Bool(!meta.is_readonly()));
    st.insert("UserExecute", Value::Num(f64::NAN));
    st.insert("GroupRead", Value::Num(f64::NAN));
    st.insert("GroupWrite", Value::Num(f64::NAN));
    st.insert("GroupExecute", Value::Num(f64::NAN));
    st.insert("OtherRead", Value::Num(f64::NAN));
    st.insert("OtherWrite", Value::Num(f64::NAN));
    st.insert("OtherExecute", Value::Num(f64::NAN));
    st
}

fn error_struct(message: &str) -> StructValue {
    let mut st = StructValue::new();
    st.insert("message", char_value(message));
    st
}

#[runtime_builtin(
    name = "rehash",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn rehash_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    use crate::user_functions::DynamicFunctionRefreshRequest;

    if output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(build_runtime_error("rehash: too many output arguments")
            .with_builtin("rehash")
            .with_identifier("RunMat:rehash:TooManyOutputs")
            .build());
    }
    let request = match args.as_slice() {
        [] => DynamicFunctionRefreshRequest::Default,
        [argument] => {
            let mode = scalar_text(argument, "rehash", "mode").map_err(|source| {
                build_runtime_error(source.message().to_owned())
                    .with_builtin("rehash")
                    .with_identifier("RunMat:rehash:InvalidMode")
                    .with_source(source)
                    .build()
            })?;
            match RehashMode::parse(&mode) {
                Some(RehashMode::Path) => DynamicFunctionRefreshRequest::Path,
                Some(RehashMode::Toolbox) => DynamicFunctionRefreshRequest::Toolbox,
                Some(RehashMode::ToolboxCache) => DynamicFunctionRefreshRequest::ToolboxCache,
                _ => {
                    return Err(build_runtime_error(
                        "rehash: expected path, toolbox, or toolboxcache",
                    )
                    .with_builtin("rehash")
                    .with_identifier("RunMat:rehash:InvalidMode")
                    .build());
                }
            }
        }
        _ => {
            return Err(build_runtime_error("rehash: too many input arguments")
                .with_builtin("rehash")
                .with_identifier("RunMat:rehash:InvalidCall")
                .build());
        }
    };
    crate::user_functions::try_refresh_dynamic_functions(request)
        .await
        .ok_or_else(|| {
            build_runtime_error("rehash: no source-cache refresh service is installed")
                .with_builtin("rehash")
                .with_identifier("RunMat:rehash:Unavailable")
                .build()
        })??;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "userpath",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn userpath_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    user_path::run(runtime, args).await
}

#[runtime_builtin(
    name = "restoredefaultpath",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn restoredefaultpath_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    user_path::restore(runtime, args)
}

#[runtime_builtin(
    name = "memmapfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn memmapfile_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    output_count::validate_fixed_outputs(
        &MEMMAPFILE_DESCRIPTOR,
        &MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS,
        "memmapfile",
    )?;
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
            "memmapfile",
        )?;
    }
    if args.is_empty() {
        return Err(memmapfile::typed_error(
            &MEMMAPFILE_ERROR_INVALID_CALL,
            "memmapfile: filename is required",
        ));
    }
    let filename = value_to_path(&args[0], "memmapfile", "filename").map_err(|source| {
        build_runtime_error(source.message().to_owned())
            .with_builtin("memmapfile")
            .with_identifier(
                MEMMAPFILE_ERROR_INVALID_FILENAME
                    .identifier
                    .expect("typed filename error"),
            )
            .with_source(source)
            .build()
    })?;
    let option_args = &args[1..];
    if !option_args.len().is_multiple_of(2) {
        return Err(memmapfile::typed_error(
            &MEMMAPFILE_ERROR_INVALID_CALL,
            "memmapfile: name-value options must be paired",
        ));
    }
    let mut options = Vec::with_capacity(option_args.len() / 2);
    for pair in option_args.chunks_exact(2) {
        let option = MemmapOption::from_value(&pair[0])?;
        if crate::builtins::common::validation::value_contains_native_integer_class(&pair[1]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &MEMMAPFILE_INTEGER_CONTROLS_EXTENSION,
                "memmapfile",
            )?;
        }
        if !value_contains_resident(&pair[1]) {
            option.validate_host_value(&pair[1])?;
        }
        options.push((option, &pair[1]));
    }
    let input = CapturedFsPath::capture(filename, "memmapfile").map_err(|source| {
        build_runtime_error(source.message().to_owned())
            .with_builtin("memmapfile")
            .with_identifier(
                MEMMAPFILE_ERROR_READ_FAILED
                    .identifier
                    .expect("typed source access error"),
            )
            .with_source(source)
            .build()
    })?;
    let mut writable = false;
    let mut offset = 0usize;
    let mut format = MemmapFormat::default();
    let mut repeat: Option<usize> = None;
    let mut cooperation = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    if let Some(cooperation) = &mut cooperation {
        cooperation
            .checkpoint(0)
            .await
            .map_err(|source| source.into_runtime_error())?;
    }
    for (option, value) in options {
        let gathered = if let Some(cooperation) = &cooperation {
            cooperation
                .await_cancellable(gather_if_needed_async(value))
                .await
                .map_err(|source| source.into_runtime_error())?
        } else {
            gather_if_needed_async(value).await
        };
        let value = gathered.map_err(|source| map_control_flow("memmapfile", source))?;
        match option {
            MemmapOption::Writable => writable = truthy(&value),
            MemmapOption::Offset => offset = numeric_usize(&value, "Offset")?,
            MemmapOption::Format => format = MemmapFormat::from_value(&value)?,
            MemmapOption::Repeat => {
                repeat = memmapfile::repeat_count(&value)?;
            }
        }
    }
    format.validate_layout(repeat)?;
    let opened = if let Some(cooperation) = &cooperation {
        cooperation
            .await_cancellable(input.open_mapped(writable))
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        input.open_mapped(writable).await
    };
    let mapped = opened.map_err(|source| {
        build_runtime_error(format!("memmapfile: {source}"))
            .with_builtin("memmapfile")
            .with_identifier(
                MEMMAPFILE_ERROR_READ_FAILED
                    .identifier
                    .expect("typed mapped-source error"),
            )
            .with_source(source)
            .build()
    })?;
    if let Some(backing) = mapped {
        let extent = usize::try_from(backing.opened_extent()).map_err(|source| {
            build_runtime_error("memmapfile: mapped source exceeds addressable size")
                .with_builtin("memmapfile")
                .with_identifier(
                    MEMMAPFILE_ERROR_INVALID_CONTROL
                        .identifier
                        .expect("typed mapped-size error"),
                )
                .with_source(source)
                .build()
        })?;
        if offset > extent {
            return Err(memmapfile::typed_error(
                &MEMMAPFILE_ERROR_INVALID_CONTROL,
                "memmapfile: Offset exceeds file size",
            ));
        }
        let authority = memmapfile::issue_authority(runtime.as_ref(), backing)?;
        let receiver = runmat_value::MemmapReceiverState {
            authority: runmat_value::MemmapReceiverAuthority::Opened(authority),
            offset,
            repeat,
            writable,
            fields: format.receiver_fields(),
            format_presentation: format.receiver_presentation(),
        };
        let mut object =
            ObjectInstance::new(MEMMAPFILE_CLASS.owned()).with_memmap_receiver(receiver);
        object
            .properties
            .insert("Filename".into(), char_value(&path_to_string(&input.path)));
        if let Some(cooperation) = &mut cooperation {
            cooperation
                .checkpoint(0)
                .await
                .map_err(|source| source.into_runtime_error())?;
        }
        input.check_current().map_err(|source| {
            build_runtime_error("memmapfile: mapped source authority changed before publication")
                .with_builtin("memmapfile")
                .with_identifier(
                    MEMMAPFILE_ERROR_READ_FAILED
                        .identifier
                        .expect("typed source error"),
                )
                .with_source(source)
                .build()
        })?;
        memmapfile::ensure_class_registered();
        return Ok(Value::Object(object));
    }
    if writable {
        return Err(memmapfile::typed_error(
            &MEMMAPFILE_ERROR_INVALID_CONTROL,
            "memmapfile: selected filesystem provider cannot supply writable mapped ranges",
        ));
    }
    let read = if let Some(cooperation) = &cooperation {
        cooperation
            .await_cancellable(input.read())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        input.read().await
    };
    let bytes = read.map_err(|source| {
        build_runtime_error(format!("memmapfile: {source}"))
            .with_builtin("memmapfile")
            .with_identifier(
                MEMMAPFILE_ERROR_READ_FAILED
                    .identifier
                    .expect("typed read error"),
            )
            .with_source(source)
            .build()
    })?;
    if offset > bytes.len() {
        return Err(memmapfile::typed_error(
            &MEMMAPFILE_ERROR_INVALID_CONTROL,
            "memmapfile: Offset exceeds file size",
        ));
    }
    let data_value = format.decode(&bytes[offset..], repeat)?;
    let receiver = runmat_value::MemmapReceiverState {
        authority: runmat_value::MemmapReceiverAuthority::Detached(Box::new(data_value)),
        offset,
        repeat,
        writable: false,
        fields: format.receiver_fields(),
        format_presentation: format.receiver_presentation(),
    };
    let mut object = ObjectInstance::new(MEMMAPFILE_CLASS.owned()).with_memmap_receiver(receiver);
    object.properties.insert(
        "Filename".to_string(),
        char_value(&path_to_string(&input.path)),
    );
    if let Some(cooperation) = &mut cooperation {
        cooperation
            .checkpoint(0)
            .await
            .map_err(|source| source.into_runtime_error())?;
    }
    input.check_current().map_err(|source| {
        build_runtime_error("memmapfile: source authority changed before publication")
            .with_builtin("memmapfile")
            .with_identifier(
                MEMMAPFILE_ERROR_READ_FAILED
                    .identifier
                    .expect("typed source error"),
            )
            .with_source(source)
            .build()
    })?;
    memmapfile::ensure_class_registered();
    Ok(Value::Object(object))
}

mod captured_fs_path;
#[cfg(test)]
#[path = "compat/tests/bindings.rs"]
mod binding_tests;
#[path = "compat/user_path.rs"]
mod user_path;
#[path = "compat/system.rs"]
mod system;
use captured_fs_path::CapturedFsPath;
pub(crate) mod memmapfile;
#[cfg(test)]
use memmapfile::MemmapFormatPresentation;
use memmapfile::{numeric_usize, MemmapFormat, MemmapOption};

#[runtime_builtin(
    name = "winqueryreg",
    category = "io/repl_fs",
    summary = "Query values from the Windows registry.",
    keywords = "winqueryreg,windows,registry",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::data_unknown_type),
    descriptor(crate::builtins::io::repl_fs::compat::WINQUERYREG_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::repl_fs::compat::WINQUERYREG_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn winqueryreg_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_contains_native_integer_class(value)
            || value_contains_resident(value)
    }) {
        return Err(compat_error(
            "winqueryreg",
            "winqueryreg: registry selectors must be host string scalars or character vectors",
        ));
    }
    let args = gather_args("winqueryreg", &args).await?;
    if args.len() < 2 || args.len() > 3 {
        return Err(compat_error(
            "winqueryreg",
            "winqueryreg: expected root, key, and optional value name",
        ));
    }
    let root = scalar_text(&args[0], "winqueryreg", "root")?;
    if root == "name" && args.len() == 3 {
        let rootkey = scalar_text(&args[1], "winqueryreg", "root")?;
        let subkey = scalar_text(&args[2], "winqueryreg", "key")?;
        return query_windows_registry_names(&rootkey, &subkey);
    }
    let key = scalar_text(&args[1], "winqueryreg", "key")?;
    let value = args
        .get(2)
        .map(|value| scalar_text(value, "winqueryreg", "value"))
        .transpose()?;
    query_windows_registry(&root, &key, value.as_deref())
}

#[cfg(windows)]
fn query_windows_registry(root: &str, key: &str, value: Option<&str>) -> BuiltinResult<Value> {
    let mut full = root.to_string();
    if !key.is_empty() {
        full.push('\\');
        full.push_str(key);
    }
    let mut cmd = Command::new("reg");
    cmd.args(["query", &full]);
    if let Some(value) = value {
        cmd.args(["/v", value]);
    }
    let output = cmd
        .output()
        .map_err(|err| compat_error("winqueryreg", format!("winqueryreg: {err}")))?;
    if !output.status.success() {
        return Err(compat_error(
            "winqueryreg",
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    parse_registry_query_value(&String::from_utf8_lossy(&output.stdout))
}

#[cfg_attr(not(windows), allow(dead_code))]
fn parse_registry_query_value(output: &str) -> BuiltinResult<Value> {
    for line in output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if let Some(index) = fields.iter().position(|field| *field == "REG_DWORD") {
            let raw = fields.get(index + 1).ok_or_else(|| {
                compat_error("winqueryreg", "winqueryreg: malformed REG_DWORD output")
            })?;
            let bits = raw
                .strip_prefix("0x")
                .map_or_else(|| raw.parse::<u32>(), |hex| u32::from_str_radix(hex, 16))
                .map_err(|_| {
                    compat_error("winqueryreg", "winqueryreg: malformed REG_DWORD value")
                })?;
            return Ok(Value::Int(runmat_value::IntValue::I32(bits as i32)));
        }
        for kind in ["REG_SZ", "REG_EXPAND_SZ"] {
            if let Some(index) = fields.iter().position(|field| *field == kind) {
                return Ok(char_value(&fields[index + 1..].join(" ")));
            }
        }
    }
    Err(compat_error(
        "winqueryreg",
        "winqueryreg: registry query returned no supported value",
    ))
}

#[cfg(windows)]
fn query_windows_registry_names(root: &str, key: &str) -> BuiltinResult<Value> {
    let mut full = root.to_string();
    if !key.is_empty() {
        full.push('\\');
        full.push_str(key);
    }
    let output = Command::new("reg")
        .args(["query", &full])
        .output()
        .map_err(|err| compat_error("winqueryreg", format!("winqueryreg: {err}")))?;
    if !output.status.success() {
        return Err(compat_error(
            "winqueryreg",
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    let names = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            fields
                .iter()
                .position(|field| field.starts_with("REG_"))
                .map(|index| fields[..index].join(" "))
                .filter(|name| !name.is_empty())
        })
        .collect();
    cellstr(names)
}

#[cfg(not(windows))]
fn query_windows_registry_names(_root: &str, _key: &str) -> BuiltinResult<Value> {
    Err(compat_error(
        "winqueryreg",
        "winqueryreg: Windows registry is unavailable on this platform",
    ))
}

#[cfg(not(windows))]
fn query_windows_registry(_root: &str, _key: &str, _value: Option<&str>) -> BuiltinResult<Value> {
    Err(compat_error(
        "winqueryreg",
        "winqueryreg: Windows registry is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::memmapfile::{shape_from_value, MemmapDataType};
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use runmat_builtins::{
        BuiltinIntegerAuditKind, FILEATTRIB_INTEGER_AUDIT, MEMMAPFILE_INTEGER_CAPABILITIES,
        USERPATH_INTEGER_AUDIT,
    };
    use runmat_filesystem::FsProvider;
    use runmat_value::{IntegerStorage, Tensor};
    use std::sync::Arc;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[path = "preferences.rs"]
    mod preference_tests;

    fn mapped_property(receiver: Value, name: &str) -> Value {
        let path = crate::object::indexing::ObjectSubscriptPath::single(
            crate::object::indexing::ObjectSubscript::member(name),
        )
        .to_standard_substruct_value()
        .expect("property path");
        run(memmapfile::live::subsref(None, receiver, path)).expect("mapped property")
    }

    #[test]
    fn winqueryreg_decodes_dword_as_exact_int32() {
        let positive = parse_registry_query_value(
            "HKEY_CURRENT_USER\\Software\\RunMat\n    Count    REG_DWORD    0x7fffffff\n",
        )
        .expect("positive DWORD");
        assert_eq!(positive, Value::Int(runmat_value::IntValue::I32(i32::MAX)));

        let bit_pattern = parse_registry_query_value(
            "HKEY_CURRENT_USER\\Software\\RunMat\n    Count    REG_DWORD    0xffffffff\n",
        )
        .expect("full DWORD bit pattern");
        assert_eq!(bit_pattern, Value::Int(runmat_value::IntValue::I32(-1)));
    }

    fn unregistered_resident_value() -> Value {
        Value::GpuTensor(
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
                .handle,
        )
    }

    #[test]
    fn textual_filesystem_apis_are_integer_inapplicable() {
        assert_eq!(
            FILEATTRIB_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert_eq!(
            USERPATH_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
    }

    #[test]
    fn userpath_rejects_integer_and_resident_values_before_provider_access() {
        for invalid in [
            Value::Int(runmat_value::IntValue::I64(i64::MIN)),
            unregistered_resident_value(),
        ] {
            let error = run(userpath_builtin(None, vec![invalid])).expect_err("invalid path text");
            assert!(error.message().contains("must be host text"));
            assert!(!error.message().to_ascii_lowercase().contains("provider"));
        }
    }

    #[test]
    fn memmapfile_decodes_named_typed_format() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let _provider_guard = vfs::provider_override_lock();
        let path = std::env::temp_dir().join("runmat_memmapfile_test.bin");
        let provider = Arc::new(vfs::MemoryFsProvider::with_current_dir(std::env::temp_dir()));
        provider.write_project_path(&path, &[1u8, 0, 2, 0]).unwrap();
        let _selected_provider = vfs::replace_provider(provider);
        let shape = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
        let fmt = Value::Cell(
            CellArray::new(
                vec![char_value("uint16"), shape, char_value("samples")],
                1,
                3,
            )
            .unwrap(),
        );
        let value = run(memmapfile_builtin(
            None,
            vec![
                Value::String(path_to_string(&path)),
                char_value("Format"),
                fmt,
            ],
        ))
        .unwrap();
        let Value::Struct(data) = mapped_property(value, "Data") else {
            panic!("expected Data struct");
        };
        let Some(Value::Tensor(samples)) = data.fields.get("samples") else {
            panic!("expected samples tensor");
        };
        assert_eq!(samples.shape(), &[2, 1]);
        assert_eq!(
            samples.integer_storage(),
            Some(&IntegerStorage::U16(vec![1, 2]))
        );
    }

    #[test]
    fn memmapfile_integer_controls_are_gated_before_file_access() {
        assert_eq!(MEMMAPFILE_INTEGER_CAPABILITIES.len(), 3);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(memmapfile_builtin(
            None,
            vec![
                Value::String("definitely-missing.bin".into()),
                Value::String("Offset".into()),
                Value::Int(runmat_value::IntValue::U64(0)),
            ],
        ))
        .expect_err("typed integer control must be gated");
        assert_eq!(
            error.identifier(),
            MEMMAPFILE_INTEGER_CONTROLS_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn memmapfile_rejects_excess_requested_outputs_before_file_access() {
        let _requested = output_count::push_output_count(Some(2));
        let error = run(memmapfile_builtin(
            None,
            vec![Value::String("definitely-missing-output-count.bin".into())],
        ))
        .expect_err("fixed single result");
        assert_eq!(
            error.identifier(),
            MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[test]
    fn memmapfile_typed_shape_offset_and_repeat_parsers_are_exact() {
        use runmat_value::{IntValue, IntegerStorage};

        assert_eq!(
            shape_from_value(&Value::Int(IntValue::U16(7))).unwrap(),
            vec![7, 1]
        );
        let shape = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2])
            .expect("typed shape vector");
        assert_eq!(shape_from_value(&Value::Tensor(shape)).unwrap(), vec![2, 3]);
        let single_shape = Tensor::from_f32(vec![4.0, 5.0], vec![1, 2]).expect("single shape");
        assert_eq!(
            shape_from_value(&Value::Tensor(single_shape)).unwrap(),
            vec![4, 5]
        );
        assert!(shape_from_value(&Value::Int(IntValue::I8(-1))).is_err());
        assert!(shape_from_value(&Value::Int(IntValue::U8(0))).is_err());
        assert!(shape_from_value(&Value::Num(2.5)).is_err());
        assert!(shape_from_value(&Value::Num(usize::MAX as f64 + 1.0)).is_err());

        assert_eq!(
            numeric_usize(&Value::Int(IntValue::U16(64)), "Offset").unwrap(),
            64
        );
        assert!(numeric_usize(&Value::Int(IntValue::I8(-1)), "Offset").is_err());
        assert!(numeric_usize(&Value::Num(2.5), "Offset").is_err());
        assert!(numeric_usize(&Value::Num(usize::MAX as f64 + 1.0), "Offset").is_err());

        let maximum = numeric_usize(&Value::Int(IntValue::U64(u64::MAX)), "Repeat");
        if usize::BITS == 64 {
            assert_eq!(maximum.unwrap(), usize::MAX);
        } else {
            assert!(maximum.is_err());
        }
    }

    #[test]
    fn memmapfile_record_and_repeat_lengths_reject_overflow_without_allocation() {
        let impossible_record = MemmapFormat {
            dtype: MemmapDataType::parse("uint8").unwrap(),
            shape: vec![usize::MAX, 2],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        };
        assert!(impossible_record.decode(&[], None).is_err());

        let impossible_repeat = MemmapFormat {
            dtype: MemmapDataType::parse("uint8").unwrap(),
            shape: vec![2, 1],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        };
        let error = impossible_repeat
            .decode(&[0], Some(usize::MAX))
            .expect_err("requested records overflow before payload allocation");
        assert!(error.message().contains("Repeat exceeds addressable size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn memmapfile_heterogeneous_format_table_preserves_record_fields_and_integer_bits() {
        let shape = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap());
        let format = Value::Cell(
            CellArray::new(
                vec![
                    char_value("int16"),
                    shape.clone(),
                    char_value("signed"),
                    char_value("uint64"),
                    shape,
                    char_value("wide"),
                ],
                2,
                3,
            )
            .unwrap(),
        );
        let parsed = MemmapFormat::from_value(&format).unwrap();
        let Value::Cell(storage) = &format else {
            unreachable!("constructed cell format")
        };
        let visible_order = storage.to_column_major();
        let normalized = Value::Cell(
            CellArray::from_column_major(visible_order, vec![2, 3])
                .expect("MATLAB-visible column-major format table"),
        );
        assert_eq!(
            MemmapFormat::from_value(&normalized)
                .unwrap()
                .to_value()
                .unwrap(),
            format
        );
        parsed.validate_layout(Some(2)).unwrap();
        assert_eq!(parsed.to_value().unwrap(), format);
        let bytes = [
            (-7_i16).to_le_bytes().to_vec(),
            u64::MAX.to_le_bytes().to_vec(),
            9_i16.to_le_bytes().to_vec(),
            (u64::MAX - 1).to_le_bytes().to_vec(),
        ]
        .concat();
        let Value::StructArray(records) = parsed.decode(&bytes, Some(2)).unwrap() else {
            panic!("two records must be a structure array");
        };
        let expected = Value::StructArray(records.clone());
        let records = records.into_elements();
        assert_eq!(records.len(), 2);
        for (record, signed, wide) in [(&records[0], -7, u64::MAX), (&records[1], 9, u64::MAX - 1)]
        {
            let Some(Value::Tensor(signed_value)) = record.fields.get("signed") else {
                panic!("signed field");
            };
            let Some(Value::Tensor(wide_value)) = record.fields.get("wide") else {
                panic!("wide field");
            };
            assert_eq!(
                signed_value.integer_storage(),
                Some(&IntegerStorage::I16(vec![signed]))
            );
            assert_eq!(
                wide_value.integer_storage(),
                Some(&IntegerStorage::U64(vec![wide]))
            );
        }

        let _lock = vfs::provider_override_lock();
        #[cfg(not(target_arch = "wasm32"))]
        let root = std::env::temp_dir().join("runmat-memmap-table");
        #[cfg(target_arch = "wasm32")]
        let root = PathBuf::from("/runmat-memmap-table");
        let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
            root.clone(),
        ));
        provider
            .write_project_path(root.join("records.bin"), &bytes)
            .expect("table input");
        let _provider = vfs::replace_provider(provider);
        vfs::set_current_dir(&root).expect("table cwd");
        let result = futures::executor::block_on(memmapfile_builtin(
            None,
            vec![
                Value::String("records.bin".into()),
                char_value("Format"),
                format,
                char_value("Repeat"),
                Value::Num(2.0),
            ],
        ))
        .expect("mapped table");
        let Value::Object(mapped) = result else {
            panic!("memmapfile result");
        };
        assert_eq!(mapped_property(Value::Object(mapped), "Data"), expected);
    }

    #[test]
    fn memmapfile_table_admission_rejects_duplicate_and_overflow_before_read() {
        let shape = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap());
        let duplicated = Value::Cell(
            CellArray::new(
                vec![
                    char_value("uint8"),
                    shape.clone(),
                    char_value("same"),
                    char_value("uint8"),
                    shape,
                    char_value("same"),
                ],
                2,
                3,
            )
            .unwrap(),
        );
        let error = futures::executor::block_on(memmapfile_builtin(
            None,
            vec![
                Value::String("definitely-missing-format-table.bin".into()),
                char_value("Format"),
                duplicated,
            ],
        ))
        .expect_err("duplicate fields fail before file access");
        assert!(error.message().contains("unique"));
        let overflow = MemmapFormat {
            dtype: MemmapDataType::parse("uint64").unwrap(),
            shape: vec![usize::MAX, 1],
            field: Some("wide".into()),
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::Cell { columns: 3 },
        };
        assert!(overflow.validate_layout(None).is_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn memmapfile_named_shape_presentation_does_not_round_wide_dimensions() {
        let format = MemmapFormat {
            dtype: MemmapDataType::parse("uint8").unwrap(),
            shape: vec![(1usize << 53) + 1, 1],
            field: Some("wide".into()),
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::Cell { columns: 3 },
        };
        let presented = format.to_value().expect("exact shape presentation");
        let reparsed = MemmapFormat::from_value(&presented).expect("exact shape parsing");
        assert_eq!(reparsed.shape, format.shape);
    }

    #[test]
    fn memmapfile_single_payload_preserves_negative_zero_and_nan_bits() {
        let bits = [0x8000_0000_u32, 0x7fc1_2345_u32];
        let bytes = bits
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let format = MemmapFormat {
            dtype: MemmapDataType::parse("single").unwrap(),
            shape: vec![1, 2],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        };
        let Value::Tensor(result) = format.decode(&bytes, None).unwrap() else {
            panic!("single tensor");
        };
        assert_eq!(
            result
                .as_f32_slice()
                .expect("native Float32 payload")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
    }

    #[test]
    fn memmapfile_known_invalid_controls_reject_before_earlier_download_or_read() {
        let filename = Value::String("definitely-missing-preflight.bin".into());
        let earlier_resident = unregistered_resident_value();
        let unknown = run(memmapfile_builtin(
            None,
            vec![
                filename.clone(),
                Value::String("Offset".into()),
                earlier_resident.clone(),
                Value::String("Unknown".into()),
                Value::Num(1.0),
            ],
        ))
        .expect_err("unknown option is a preflight error");
        assert!(unknown.message().contains("unsupported option"));
        assert!(!unknown.message().to_ascii_lowercase().contains("provider"));

        let invalid = run(memmapfile_builtin(
            None,
            vec![
                filename,
                Value::String("Offset".into()),
                earlier_resident,
                Value::String("Format".into()),
                Value::String("not-a-supported-dtype".into()),
            ],
        ))
        .expect_err("invalid known format is a preflight error");
        assert!(invalid.message().contains("unsupported Format type"));
        assert!(!invalid.message().to_ascii_lowercase().contains("provider"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn memmapfile_captured_input_retains_provider_and_cwd_after_ambient_swap() {
        use runmat_filesystem::MemoryFsProvider;
        use std::sync::Arc;

        let _lock = vfs::provider_override_lock();
        #[cfg(not(target_arch = "wasm32"))]
        let selected_dir = std::env::temp_dir().join("runmat-memmap-selected");
        #[cfg(not(target_arch = "wasm32"))]
        let other_dir = std::env::temp_dir().join("runmat-memmap-other");
        #[cfg(target_arch = "wasm32")]
        let selected_dir = PathBuf::from("/runmat-memmap-selected");
        #[cfg(target_arch = "wasm32")]
        let other_dir = PathBuf::from("/runmat-memmap-other");
        let selected = Arc::new(MemoryFsProvider::with_current_dir(selected_dir.clone()));
        let other = Arc::new(MemoryFsProvider::with_current_dir(other_dir.clone()));
        selected
            .write_project_path(selected_dir.join("source.bin"), b"selected")
            .expect("selected source");
        other
            .write_project_path(selected_dir.join("source.bin"), b"wrong owner")
            .expect("same path on foreign owner");
        let _selected = vfs::replace_provider(selected);
        vfs::set_current_dir(&selected_dir).expect("selected cwd");
        let captured = CapturedFsPath::capture(PathBuf::from("source.bin"), "memmapfile")
            .expect("capture selected input");
        let _other = vfs::replace_provider(other);
        vfs::set_current_dir(&other_dir).expect("other cwd");
        assert_eq!(captured.path, selected_dir.join("source.bin"));
        assert_eq!(
            futures::executor::block_on(captured.read()).expect("pinned read"),
            b"selected"
        );
    }

    #[test]
    fn memmapfile_decodes_all_integer_formats_without_float_coercion() {
        let cases = [
            (
                "int8",
                vec![i8::MIN as u8, i8::MAX as u8],
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            ),
            (
                "uint8",
                vec![0, u8::MAX],
                IntegerStorage::U8(vec![0, u8::MAX]),
            ),
            (
                "int16",
                [i16::MIN.to_le_bytes(), i16::MAX.to_le_bytes()].concat(),
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            ),
            (
                "uint16",
                [0_u16.to_le_bytes(), u16::MAX.to_le_bytes()].concat(),
                IntegerStorage::U16(vec![0, u16::MAX]),
            ),
            (
                "int32",
                [i32::MIN.to_le_bytes(), i32::MAX.to_le_bytes()].concat(),
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            ),
            (
                "uint32",
                [0_u32.to_le_bytes(), u32::MAX.to_le_bytes()].concat(),
                IntegerStorage::U32(vec![0, u32::MAX]),
            ),
            (
                "int64",
                [i64::MIN.to_le_bytes(), i64::MAX.to_le_bytes()].concat(),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            ),
            (
                "uint64",
                [0_u64.to_le_bytes(), u64::MAX.to_le_bytes()].concat(),
                IntegerStorage::U64(vec![0, u64::MAX]),
            ),
        ];

        for (dtype, bytes, expected) in cases {
            let mapped = MemmapFormat {
                dtype: MemmapDataType::parse(dtype).unwrap(),
                shape: vec![1, 2],
                field: None,
                additional_fields: Vec::new(),
                presentation: MemmapFormatPresentation::TypeName,
            }
            .decode(&bytes, None)
            .unwrap();
            let Value::Tensor(tensor) = mapped else {
                panic!("expected tensor");
            };
            assert_eq!(tensor.shape(), &[1, 2], "{dtype}");
            assert_eq!(tensor.integer_storage(), Some(&expected), "{dtype}");
        }
    }

    #[test]
    fn memmapfile_preserves_integer_storage_for_repeated_records() {
        let bytes = [
            1_u32.to_le_bytes(),
            u32::MAX.to_le_bytes(),
            3_u32.to_le_bytes(),
            4_u32.to_le_bytes(),
        ]
        .concat();
        let mapped = MemmapFormat {
            dtype: MemmapDataType::parse("uint32").unwrap(),
            shape: vec![2, 1],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        }
        .decode(&bytes, Some(2))
        .unwrap();
        let Value::Tensor(tensor) = mapped else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape(), &[2, 1, 2]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U32(vec![1, u32::MAX, 3, 4]))
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn system_two_outputs_returns_status_and_text() {
        use std::rc::Rc;
        struct Process;
        impl crate::context::RuntimeProcessService for Process {
            fn output_complete(
                &self,
                command: runmat_process_host::HostCommand,
            ) -> crate::context::RuntimeServiceFuture<
                runmat_process_host::ProcessHostResult<runmat_process_host::CompleteProcessOutput>,
            > {
                Box::pin(command.output_complete())
            }
            fn output_complete_with_input(
                &self,
                command: runmat_process_host::HostCommand,
                input: Vec<u8>,
            ) -> crate::context::RuntimeServiceFuture<
                runmat_process_host::ProcessHostResult<runmat_process_host::CompleteProcessOutput>,
            > {
                Box::pin(command.output_complete_with_input(input))
            }
        }
        let _serial = vfs::provider_override_lock();
        let _native = vfs::replace_provider(Arc::new(vfs::NativeFsProvider));
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default().with_process(Rc::new(Process)),
        );
        #[cfg(windows)]
        let command = "echo hello";
        #[cfg(not(windows))]
        let command = "printf hello";
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = system_builtin(Some(runtime), vec![Value::String(command.to_string())])
            .await
            .unwrap();
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        assert_eq!(values[0], Value::Num(0.0));
        #[cfg(windows)]
        let expected_output = "hello\r\n";
        #[cfg(not(windows))]
        let expected_output = "hello";
        assert_eq!(values[1], char_value(expected_output));
    }

    #[test]
    fn system_rejects_integer_commands_before_process_access() {
        let error = run(system_builtin(
            None,
            vec![Value::Int(runmat_value::IntValue::U64(u64::MAX))],
        ))
        .expect_err("integer command must reject");
        assert!(error.message().contains("command must be"));
    }

    #[test]
    fn system_rejects_integer_echo_options_before_process_access() {
        let error = run(system_builtin(
            None,
            vec![
                Value::String("printf should-not-run".to_string()),
                Value::Int(runmat_value::IntValue::U8(1)),
            ],
        ))
        .expect_err("integer option must reject");
        assert!(error.message().contains("option must be"));
    }

    #[test]
    fn system_rejects_excess_outputs_before_process_access() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let error = run(system_builtin(
            None,
            vec![Value::String("printf should-not-run".to_string())],
        ))
        .expect_err("unsupported output count must reject");
        assert!(error.message().contains("at most two outputs"));
    }

    #[test]
    fn fileattrib_can_toggle_user_write_flag() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_fileattrib_test.txt");
        std::fs::write(&path, "data").unwrap();
        let rejected = run(fileattrib_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("-w"),
            char_value("unsupported"),
        ]))
        .expect_err("a later invalid flag cannot leave an earlier mutation");
        assert!(rejected.message().contains("unsupported attribute flag"));
        assert!(!std::fs::metadata(&path).unwrap().permissions().readonly());
        run(fileattrib_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("-w"),
        ]))
        .unwrap();
        let readonly = std::fs::metadata(&path).unwrap().permissions().readonly();
        assert!(readonly);
        run(fileattrib_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("+w"),
        ]))
        .unwrap();
        let readonly = std::fs::metadata(&path).unwrap().permissions().readonly();
        assert!(!readonly);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn fileattrib_read_flag_rejects_before_an_earlier_write_flag_mutates() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let _provider_lock = vfs::provider_override_lock();
        #[cfg(not(target_arch = "wasm32"))]
        let root = std::env::temp_dir().join("runmat-fileattrib-preflight");
        #[cfg(target_arch = "wasm32")]
        let root = PathBuf::from("/runmat-fileattrib-preflight");
        let provider = Arc::new(vfs::MemoryFsProvider::with_current_dir(root.clone()));
        let path = root.join("input.bin");
        provider.write_project_path(&path, b"value").unwrap();
        let _installed = vfs::replace_provider(provider.clone());
        let error = run(fileattrib_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("-w"),
            char_value("+r"),
        ]))
        .expect_err("unsupported read permission must reject before mutation");
        assert_eq!(
            error.identifier(),
            Some("RunMat:fileattrib:UnsupportedPermission")
        );
        assert!(!futures::executor::block_on(provider.metadata(&path))
            .unwrap()
            .is_readonly());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileattrib_output_count_admits_zero_one_two_and_rejects_three_before_mutation() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let _provider_lock = vfs::provider_override_lock();
        #[cfg(windows)]
        let root = PathBuf::from("C:\\runmat-fileattrib-output-count");
        #[cfg(not(windows))]
        let root = PathBuf::from("/runmat-fileattrib-output-count");
        let provider = Arc::new(vfs::MemoryFsProvider::with_current_dir(&root));
        let path = root.join("input.bin");
        provider.write_project_path(&path, b"value").unwrap();
        let _installed = vfs::replace_provider(provider.clone());
        vfs::set_current_dir(&root).unwrap();

        for count in 0..=2 {
            let _outputs = output_count::push_output_count(Some(count));
            let Value::OutputList(outputs) = run(fileattrib_builtin(vec![
                Value::String("input.bin".into()),
                char_value("+w"),
            ]))
            .expect("supported requested output count") else {
                panic!("explicit count produces output list");
            };
            assert_eq!(outputs.len(), count);
            if count > 0 {
                assert_eq!(outputs[0], Value::Num(1.0));
            }
            if count == 2 {
                assert!(matches!(outputs[1], Value::Struct(_)));
            }
        }

        let _outputs = output_count::push_output_count(Some(3));
        let error = run(fileattrib_builtin(vec![
            Value::String("input.bin".into()),
            char_value("-w"),
        ]))
        .expect_err("third output rejects before read-only mutation");
        assert_eq!(
            error.identifier(),
            FILEATTRIB_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        assert!(!futures::executor::block_on(provider.metadata(&path))
            .unwrap()
            .is_readonly());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_text_controls_reject_character_pages_without_filesystem_effects() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let _provider_lock = vfs::provider_override_lock();
        #[cfg(windows)]
        let root = PathBuf::from("C:\\runmat-fileattrib-paged-text");
        #[cfg(not(windows))]
        let root = PathBuf::from("/runmat-fileattrib-paged-text");
        let provider = Arc::new(vfs::MemoryFsProvider::with_current_dir(&root));
        let path = root.join("input.bin");
        provider.write_project_path(&path, b"value").unwrap();
        let _installed = vfs::replace_provider(provider.clone());
        vfs::set_current_dir(&root).unwrap();

        let page = Value::CharArray(
            CharArray::new_with_shape(vec!['-', 'w'], vec![1, 1, 2]).expect("two pages"),
        );
        assert!(scalar_text(&page, "fileattrib", "attribute").is_err());
        let accepted = Value::CharArray(
            CharArray::new_with_shape(vec!['-', 'w', ' '], vec![1, 3, 1])
                .expect("trailing singleton page"),
        );
        assert_eq!(
            scalar_text(&accepted, "fileattrib", "attribute").unwrap(),
            "-w"
        );
        assert!(run(fileattrib_builtin(vec![page.clone(), char_value("-w"),])).is_err());
        assert!(run(fileattrib_builtin(vec![
            Value::String("input.bin".into()),
            page.clone(),
        ]))
        .is_err());
        assert!(run(rehash_builtin(vec![page])).is_err());
        assert!(!futures::executor::block_on(provider.metadata(&path))
            .unwrap()
            .is_readonly());
    }

    #[test]
    fn fileattrib_captured_path_retains_provider_for_mutation_and_metadata() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let _provider_lock = vfs::provider_override_lock();
        #[cfg(not(target_arch = "wasm32"))]
        let root = std::env::temp_dir().join("runmat-fileattrib-selected");
        #[cfg(target_arch = "wasm32")]
        let root = PathBuf::from("/runmat-fileattrib-selected");
        let selected = Arc::new(vfs::MemoryFsProvider::with_current_dir(root.clone()));
        let other = Arc::new(vfs::MemoryFsProvider::with_current_dir(root.clone()));
        let path = root.join("input.bin");
        selected.write_project_path(&path, b"selected").unwrap();
        other.write_project_path(&path, b"foreign").unwrap();
        let _selected = vfs::replace_provider(selected.clone());
        vfs::set_current_dir(&root).unwrap();
        let captured = CapturedFsPath::capture(PathBuf::from("input.bin"), "fileattrib").unwrap();
        let _other = vfs::replace_provider(other.clone());
        futures::executor::block_on(captured.set_readonly(true)).unwrap();
        assert!(futures::executor::block_on(captured.metadata())
            .unwrap()
            .is_readonly());
        assert!(futures::executor::block_on(selected.metadata(&path))
            .unwrap()
            .is_readonly());
        assert!(!futures::executor::block_on(other.metadata(&path))
            .unwrap()
            .is_readonly());
    }

    #[test]
    fn fileattrib_reports_unavailable_permission_facts_as_nan() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let _provider_lock = vfs::provider_override_lock();
        #[cfg(not(target_arch = "wasm32"))]
        let root = std::env::temp_dir().join("runmat-fileattrib-unknown-facts");
        #[cfg(target_arch = "wasm32")]
        let root = PathBuf::from("/runmat-fileattrib-unknown-facts");
        let provider = Arc::new(vfs::MemoryFsProvider::with_current_dir(root.clone()));
        provider
            .write_project_path(root.join("sample.bin"), b"value")
            .unwrap();
        let _installed = vfs::replace_provider(provider);
        vfs::set_current_dir(&root).unwrap();
        let _outputs = output_count::push_output_count(Some(2));
        let Value::OutputList(outputs) =
            run(fileattrib_builtin(vec![Value::String("sample.bin".into())])).unwrap()
        else {
            panic!("two requested fileattrib outputs");
        };
        assert_eq!(outputs[0], Value::Num(1.0));
        let Value::Struct(attributes) = &outputs[1] else {
            panic!("attribute struct");
        };
        for name in [
            "archive",
            "system",
            "hidden",
            "UserRead",
            "UserExecute",
            "GroupRead",
            "GroupWrite",
            "GroupExecute",
            "OtherRead",
            "OtherWrite",
            "OtherExecute",
        ] {
            assert!(
                matches!(attributes.fields.get(name), Some(Value::Num(value)) if value.is_nan()),
                "{name}"
            );
        }
        assert_eq!(attributes.fields.get("UserWrite"), Some(&Value::Bool(true)));
        assert_eq!(
            attributes.fields.get("directory"),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn fileattrib_rejects_numeric_and_resident_inputs_before_external_access() {
        let numeric =
            run(fileattrib_builtin(vec![Value::Num(1.0)])).expect_err("numeric path must reject");
        assert!(numeric.message().contains("path must be"));

        let numeric_flag = run(fileattrib_builtin(vec![
            Value::String("missing-fileattrib-path".to_string()),
            Value::Num(1.0),
        ]))
        .expect_err("numeric attribute must reject");
        assert!(numeric_flag.message().contains("attribute must be"));

        let resident_path = run(fileattrib_builtin(vec![unregistered_resident_value()]))
            .expect_err("resident path must reject");
        assert!(resident_path.message().contains("path must be"));
        assert!(!resident_path
            .message()
            .to_ascii_lowercase()
            .contains("provider"));

        let resident_flag = run(fileattrib_builtin(vec![
            Value::String("missing-fileattrib-path".to_string()),
            unregistered_resident_value(),
        ]))
        .expect_err("resident attribute must reject");
        assert!(resident_flag.message().contains("attribute must be"));
        assert!(!resident_flag
            .message()
            .to_ascii_lowercase()
            .contains("provider"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rehash_routes_typed_modes_to_source_refresh_without_clearing_values() {
        use crate::user_functions::DynamicFunctionRefreshRequest;
        use std::cell::RefCell;
        use std::rc::Rc;

        let observed = Rc::new(RefCell::new(Vec::new()));
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let calls = Rc::clone(&observed);
        context.set_dynamic_function_refresher(Some(Rc::new(move |_runtime, request| {
            let calls = Rc::clone(&calls);
            Box::pin(async move {
                calls.borrow_mut().push(request);
                Ok(())
            })
        })));
        let _active = context.enter();
        for arguments in [
            vec![],
            vec![char_value("path")],
            vec![Value::String("toolbox".into())],
            vec![char_value("toolboxcache")],
        ] {
            assert_eq!(run(rehash_builtin(arguments)).unwrap(), Value::Num(0.0));
        }
        assert_eq!(
            observed.borrow().as_slice(),
            &[
                DynamicFunctionRefreshRequest::Default,
                DynamicFunctionRefreshRequest::Path,
                DynamicFunctionRefreshRequest::Toolbox,
                DynamicFunctionRefreshRequest::ToolboxCache,
            ]
        );
        let error = run(rehash_builtin(vec![char_value("not-a-mode")]))
            .expect_err("invalid mode must fail before cache callback");
        assert!(error.message().contains("expected path"));
        assert_eq!(error.identifier(), Some("RunMat:rehash:InvalidMode"));
        assert_eq!(observed.borrow().len(), 4);
        let error = run(rehash_builtin(vec![Value::Num(1.0)]))
            .expect_err("numeric mode must fail before cache callback");
        assert!(error.message().contains("mode must be"));
        assert_eq!(error.identifier(), Some("RunMat:rehash:InvalidMode"));
        assert_eq!(observed.borrow().len(), 4);
        let error = run(rehash_builtin(vec![
            char_value("path"),
            char_value("toolbox"),
        ]))
        .expect_err("extra mode must fail before cache callback");
        assert_eq!(error.identifier(), Some("RunMat:rehash:InvalidCall"));
        assert_eq!(observed.borrow().len(), 4);
        let _outputs = output_count::push_output_count(Some(2));
        let error = run(rehash_builtin(Vec::new()))
            .expect_err("unsupported output count must fail before cache callback");
        assert_eq!(error.identifier(), Some("RunMat:rehash:TooManyOutputs"));
        assert_eq!(observed.borrow().len(), 4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rehash_without_an_installed_source_refresh_service_is_not_a_noop_success() {
        use std::rc::Rc;
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _active = context.enter();
        let error = run(rehash_builtin(Vec::new())).expect_err("no cache owner exists");
        assert_eq!(error.identifier(), Some("RunMat:rehash:Unavailable"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cancelled_rehash_rejects_before_refresh_callback() {
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::cell::Cell;
        use std::rc::Rc;

        let cancellation = CancellationSource::new();
        cancellation.cancel(CancellationReason::User);
        let calls = Rc::new(Cell::new(0));
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_cancellation_token(cancellation.token());
        let counted = Rc::clone(&calls);
        context.set_dynamic_function_refresher(Some(Rc::new(move |_, _| {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        })));
        let _active = context.enter();
        let error = run(rehash_builtin(Vec::new())).expect_err("cancelled refresh cannot mutate");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(matches!(
            error.source.as_ref().and_then(|source| {
                source.downcast_ref::<crate::execution::ExecutionServiceError>()
            }),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ));
        assert_eq!(calls.get(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pending_rehash_cancellation_keeps_builtin_identifier_and_retires_callback() {
        use runmat_execution::{cancellation::CancellationSource, CancellationReason};
        use std::cell::Cell;
        use std::future::Future;
        use std::pin::Pin;
        use std::rc::Rc;
        use std::task::{Context, Poll};

        struct PendingRefresh(Rc<Cell<bool>>);
        impl Future for PendingRefresh {
            type Output = Result<(), RuntimeError>;
            fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Pending
            }
        }
        impl Drop for PendingRefresh {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let cancellation = CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_cancellation_token(cancellation.token());
        let selected = Rc::clone(&dropped);
        context.set_dynamic_function_refresher(Some(Rc::new(move |_, _| {
            Box::pin(PendingRefresh(Rc::clone(&selected)))
        })));
        let _active = context.enter();
        let mut call = Box::pin(rehash_builtin(Vec::new()));
        let waker = futures::task::noop_waker();
        let mut poll = Context::from_waker(&waker);
        assert!(call.as_mut().poll(&mut poll).is_pending());
        assert!(cancellation.cancel(CancellationReason::User));
        let Poll::Ready(Err(error)) = call.as_mut().poll(&mut poll) else {
            panic!("pending refresh must finish with typed cancellation")
        };
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(matches!(
            error.source.as_ref().and_then(|source| {
                source.downcast_ref::<crate::execution::ExecutionServiceError>()
            }),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ));
        assert!(dropped.get());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dropped_pending_rehash_drops_selected_refresh_without_publication() {
        use std::cell::Cell;
        use std::future::Future;
        use std::pin::Pin;
        use std::rc::Rc;
        use std::task::{Context, Poll};

        struct PendingRefresh(Rc<Cell<bool>>);
        impl Future for PendingRefresh {
            type Output = Result<(), RuntimeError>;
            fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Pending
            }
        }
        impl Drop for PendingRefresh {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let dropped = Rc::new(Cell::new(false));
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let selected = Rc::clone(&dropped);
        context.set_dynamic_function_refresher(Some(Rc::new(move |_, _| {
            Box::pin(PendingRefresh(Rc::clone(&selected)))
        })));
        let _active = context.enter();
        let mut call = Box::pin(rehash_builtin(Vec::new()));
        let waker = futures::task::noop_waker();
        assert!(call
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        drop(call);
        assert!(dropped.get());
    }
}
