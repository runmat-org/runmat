//! Compatibility filesystem, path, process, preference, and environment helpers.

use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor, MEMMAPFILE_CLASS, MEMMAPFILE_DESCRIPTOR,
    MEMMAPFILE_ERROR_TOO_MANY_OUTPUTS, MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
    MEMMAPFILE_INTEGER_CONTROLS_EXTENSION,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, ObjectInstance, StructValue, Value};

use crate::builtins::common::fs::{expand_user_path, home_directory, path_to_string};
use crate::builtins::common::path_state::set_path_string;
use crate::output_count;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "input",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input argument.",
}];
const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
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
];
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
const OUTPUT_STATUS_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "0 on success, nonzero on failure.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Command output or diagnostic text.",
    },
];
const OUTPUT_STATUS_ATTRIB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "1 when attributes were read, otherwise 0.",
    },
    BuiltinParamDescriptor {
        name: "attributes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute struct or diagnostic struct.",
    },
];

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

const SYSTEM_INPUTS_COMMAND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "command",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operating-system command to execute.",
}];
const SYSTEM_INPUTS_ECHO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "command",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Operating-system command to execute.",
    },
    BuiltinParamDescriptor {
        name: "echo",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("'-echo'"),
        description: "The literal '-echo', which also writes command output to the console.",
    },
];
const SYSTEM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[status, output] = system(command)",
        inputs: &SYSTEM_INPUTS_COMMAND,
        outputs: &OUTPUT_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status, output] = system(command, '-echo')",
        inputs: &SYSTEM_INPUTS_ECHO,
        outputs: &OUTPUT_STATUS_MESSAGE,
    },
];
pub const SYSTEM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYSTEM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub const SYSTEM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "system accepts a host character vector or string command, not numeric input; integer and resident numeric values reject before provider or process access.",
};
simple_descriptor!(
    FILEATTRIB_SIGNATURES,
    FILEATTRIB_DESCRIPTOR,
    "[status, attributes] = fileattrib(path)",
    &INPUTS_ONE,
    &OUTPUT_STATUS_ATTRIB,
    BuiltinOutputMode::ByRequestedOutputCount
);

pub const FILEATTRIB_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "fileattrib accepts host-text paths and textual controls; numeric and provider-resident values are rejected before any gather, provider access, or filesystem operation, while numeric status and attribute fields are outputs only.",
    };
simple_descriptor!(
    REHASH_SIGNATURES,
    REHASH_DESCRIPTOR,
    "rehash()",
    &INPUTS_NONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    USERPATH_SIGNATURES,
    USERPATH_DESCRIPTOR,
    "path = userpath(option)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
pub const USERPATH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "userpath accepts no input or one host text path/command; integer and resident numeric values reject before provider or search-path access.",
};
simple_descriptor!(
    RESTOREDEFAULTPATH_SIGNATURES,
    RESTOREDEFAULTPATH_DESCRIPTOR,
    "path = restoredefaultpath()",
    &INPUTS_NONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
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
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(char_row_to_string(array)),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(compat_error(
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
    category = "io/repl_fs",
    summary = "Execute an operating-system command.",
    keywords = "system,command,shell,process",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::system_type),
    descriptor(crate::builtins::io::repl_fs::compat::SYSTEM_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::SYSTEM_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn system_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(compat_error(
            "system",
            "system: expected command and optional echo flag",
        ));
    }
    let command = scalar_text(&args[0], "system", "command")?;
    let echo = if let Some(option) = args.get(1) {
        let option = scalar_text(option, "system", "option")?;
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
    let result = run_system_command(&command)?;
    let requested = output_count::current_output_count();
    if (echo || requested == Some(0)) && !result.1.is_empty() {
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, &result.1);
    }
    let outputs = vec![Value::Num(result.0 as f64), char_value(&result.1)];
    if let Some(count) = requested {
        return Ok(output_count::output_list_with_padding(count, outputs));
    }
    Ok(Value::Num(result.0 as f64))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_system_command(command: &str) -> BuiltinResult<(i32, String)> {
    #[cfg(windows)]
    let output = Command::new("cmd").args(["/C", command]).output();
    #[cfg(not(windows))]
    let output = Command::new("sh").args(["-c", command]).output();
    let output = output.map_err(|err| compat_error("system", format!("system: {err}")))?;
    let status = output.status.code().unwrap_or(1);
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((status, text))
}

#[cfg(target_arch = "wasm32")]
fn run_system_command(_command: &str) -> BuiltinResult<(i32, String)> {
    Ok((
        1,
        "system: process execution is unavailable in WebAssembly".to_string(),
    ))
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
    category = "io/repl_fs",
    summary = "Return file attribute metadata.",
    keywords = "fileattrib,file,folder,attributes,metadata",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::fileattrib_type),
    descriptor(crate::builtins::io::repl_fs::compat::FILEATTRIB_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::FILEATTRIB_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn fileattrib_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
        match flag.to_ascii_lowercase().as_str() {
            "+w" | "w" => readonly = Some(false),
            "-w" => readonly = Some(true),
            "+r" | "r" | "-r" => {
                return Err(build_runtime_error(
                    "fileattrib: read permission changes require provider permission support",
                )
                .with_builtin("fileattrib")
                .with_identifier("RunMat:fileattrib:UnsupportedPermission")
                .build());
            }
            other => {
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
    st.insert("hidden", Value::Bool(is_hidden_path(path)));
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

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

#[runtime_builtin(
    name = "rehash",
    category = "io/repl_fs",
    summary = "Refresh function/path caches.",
    keywords = "rehash,path,cache,refresh",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::repl_fs::compat::REHASH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn rehash_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "userpath",
    category = "io/repl_fs",
    summary = "Query or set the user path.",
    keywords = "userpath,path,user,home",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::USERPATH_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::USERPATH_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn userpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_has_native_integer_class(value)
            || matches!(value, Value::GpuTensor(_))
    }) {
        return Err(compat_error(
            "userpath",
            "userpath: path or command must be host text",
        ));
    }
    let args = gather_args("userpath", &args).await?;
    match args.len() {
        0 => Ok(char_value(&default_userpath())),
        1 => {
            let command = scalar_text(&args[0], "userpath", "command")?;
            match command.to_ascii_lowercase().as_str() {
                "reset" => Ok(char_value(&default_userpath())),
                "clear" => Ok(char_value("")),
                path => Ok(char_value(path)),
            }
        }
        _ => Err(compat_error(
            "userpath",
            "userpath: too many input arguments",
        )),
    }
}

fn default_userpath() -> String {
    home_directory()
        .map(|home| home.join("Documents").join("MATLAB"))
        .map(|path| path_to_string(&path))
        .unwrap_or_default()
}

#[runtime_builtin(
    name = "restoredefaultpath",
    category = "io/repl_fs",
    summary = "Restore the RunMat search path to its default value.",
    keywords = "restoredefaultpath,path,default",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::RESTOREDEFAULTPATH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn restoredefaultpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(compat_error(
            "restoredefaultpath",
            "restoredefaultpath: too many input arguments",
        ));
    }
    let root = path_to_string(&super::installation_path::root());
    set_path_string(&root);
    Ok(char_value(&root))
}

#[runtime_builtin(
    name = "memmapfile",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn memmapfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
        return Err(compat_error(
            "memmapfile",
            "memmapfile: filename is required",
        ));
    }
    let filename = value_to_path(&args[0], "memmapfile", "filename")?;
    let option_args = &args[1..];
    if option_args.len() % 2 != 0 {
        return Err(compat_error(
            "memmapfile",
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
    let input = CapturedFsPath::capture(filename, "memmapfile")?;
    let mut writable = false;
    let mut offset = 0usize;
    let mut format = MemmapFormat::default();
    let mut repeat: Option<usize> = None;
    for (option, value) in options {
        let value = gather_if_needed_async(value)
            .await
            .map_err(|source| map_control_flow("memmapfile", source))?;
        match option {
            MemmapOption::Writable => writable = truthy(&value),
            MemmapOption::Offset => offset = numeric_usize(&value, "memmapfile", "Offset")?,
            MemmapOption::Format => format = MemmapFormat::from_value(&value)?,
            MemmapOption::Repeat => {
                repeat = Some(numeric_usize(&value, "memmapfile", "Repeat")?);
            }
        }
    }
    format.validate_layout(repeat)?;
    let bytes = input.read().await.map_err(|source| {
        build_runtime_error(format!("memmapfile: {source}"))
            .with_builtin("memmapfile")
            .with_source(source)
            .build()
    })?;
    if offset > bytes.len() {
        return Err(compat_error(
            "memmapfile",
            "memmapfile: Offset exceeds file size",
        ));
    }
    let data_value = format.decode(&bytes[offset..], repeat)?;
    let mut object = ObjectInstance::new(MEMMAPFILE_CLASS.owned());
    object.properties.insert(
        "Filename".to_string(),
        char_value(&path_to_string(&input.path)),
    );
    object
        .properties
        .insert("Writable".to_string(), Value::Bool(writable));
    object
        .properties
        .insert("Offset".to_string(), Value::Num(offset as f64));
    object
        .properties
        .insert("Format".to_string(), format.to_value()?);
    object.properties.insert(
        "Repeat".to_string(),
        repeat.map_or_else(|| char_value("Inf"), |value| Value::Num(value as f64)),
    );
    object.properties.insert("Data".to_string(), data_value);
    Ok(Value::Object(object))
}

mod captured_fs_path;
use captured_fs_path::CapturedFsPath;
mod memmapfile;
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
    use crate::builtins::io::repl_fs::preferences::{
        getpref_builtin, ispref_builtin, setpref_builtin,
    };
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use runmat_builtins::{
        GETPREF_DESCRIPTOR, GETPREF_GROUP_QUERY_EXTENSION, GETPREF_INTEGER_CAPABILITIES,
        MEMMAPFILE_INTEGER_CAPABILITIES,
    };
    use runmat_filesystem::FsProvider;
    use runmat_value::{IntegerStorage, Tensor};
    use std::sync::Arc;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
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
            let error = run(userpath_builtin(vec![invalid])).expect_err("invalid path text");
            assert!(error.message().contains("must be host text"));
            assert!(!error.message().to_ascii_lowercase().contains("provider"));
        }
    }

    #[test]
    fn ispref_rejects_integer_and_resident_controls_without_gather() {
        for integer in [
            runmat_value::IntValue::I8(-1),
            runmat_value::IntValue::I16(-2),
            runmat_value::IntValue::I32(-3),
            runmat_value::IntValue::I64(i64::MIN),
            runmat_value::IntValue::U8(1),
            runmat_value::IntValue::U16(2),
            runmat_value::IntValue::U32(3),
            runmat_value::IntValue::U64(u64::MAX),
        ] {
            assert!(run(ispref_builtin(vec![Value::Int(integer)])).is_err());
        }
        assert!(run(ispref_builtin(vec![unregistered_resident_value()])).is_err());
    }

    #[test]
    fn preferences_round_trip() {
        run(setpref_builtin(vec![
            Value::String("runmatTest".to_string()),
            Value::String("answer".to_string()),
            Value::Num(42.0),
        ]))
        .unwrap();
        assert_eq!(
            run(ispref_builtin(vec![
                Value::String("runmatTest".to_string()),
                Value::String("answer".to_string())
            ]))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run(getpref_builtin(vec![
                Value::String("runmatTest".to_string()),
                Value::String("answer".to_string())
            ]))
            .unwrap(),
            Value::Num(42.0)
        );
    }

    #[test]
    fn multi_name_setpref_and_ispref_keep_shape_and_missing_positions() {
        let group = Value::String("runmatMultiPreferenceForms".into());
        let names = Value::StringArray(
            runmat_value::StringArray::new(vec!["first".into(), "second".into()], vec![2, 1])
                .expect("names shape"),
        );
        let values = Value::Cell(
            CellArray::new_with_shape(
                vec![
                    Value::Int(runmat_value::IntValue::U64(u64::MAX)),
                    Value::Num(2.0),
                ],
                vec![2, 1],
            )
            .expect("values shape"),
        );
        assert_eq!(
            run(setpref_builtin(vec![
                group.clone(),
                names.clone(),
                values.clone()
            ]))
            .expect("set both"),
            Value::Num(0.0)
        );
        let queried =
            run(getpref_builtin(vec![group.clone(), names.clone()])).expect("retrieve both");
        assert_eq!(queried, values);
        let flags = run(ispref_builtin(vec![group.clone(), names])).expect("query both");
        let Value::LogicalArray(flags) = flags else {
            panic!("logical array");
        };
        assert_eq!(flags.shape, vec![2, 1]);
        assert_eq!(flags.data.as_slice(), &[1, 1]);

        let mixed = Value::StringArray(
            runmat_value::StringArray::new(vec!["first".into(), "missing".into()], vec![1, 2])
                .expect("mixed shape"),
        );
        let flags = run(ispref_builtin(vec![group, mixed])).expect("mixed query");
        let Value::LogicalArray(flags) = flags else {
            panic!("logical array");
        };
        assert_eq!(flags.shape, vec![1, 2]);
        assert_eq!(flags.data.as_slice(), &[1, 0]);
    }

    #[test]
    fn invalid_multi_value_shape_rejects_before_resident_download_or_store_mutation() {
        let group = Value::String("runmatMultiPreferenceAdmission".into());
        let names = Value::StringArray(
            runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2])
                .expect("names shape"),
        );
        let wrong_shape = Value::Cell(
            CellArray::new_with_shape(
                vec![unregistered_resident_value(), Value::Num(2.0)],
                vec![2, 1],
            )
            .expect("wrong shape"),
        );
        let error = run(setpref_builtin(vec![
            group.clone(),
            names.clone(),
            wrong_shape,
        ]))
        .expect_err("shape must reject before any gather");
        assert!(error.message().contains("matching shape"));
        assert_eq!(
            run(ispref_builtin(vec![group.clone()])).expect("group query"),
            Value::Bool(false)
        );
        let error = run(setpref_builtin(vec![group, names, Value::Num(1.0)]))
            .expect_err("scalar cannot fill multiple settings");
        assert!(error.message().contains("matching shape"));
    }

    #[test]
    fn preference_output_admission_precedes_storage_and_provider_access() {
        let group = Value::String("runmatPreferenceOutputAdmission".into());
        let _count = crate::output_count::push_output_count(Some(2));
        for call in [
            run(getpref_builtin(vec![
                group.clone(),
                Value::String("name".into()),
            ])),
            run(ispref_builtin(vec![group.clone()])),
            run(setpref_builtin(vec![
                group.clone(),
                Value::String("name".into()),
                unregistered_resident_value(),
            ])),
        ] {
            let error = call.expect_err("too many outputs before effects");
            assert!(error.message().contains("too many output"));
        }
        drop(_count);
        assert_eq!(
            run(ispref_builtin(vec![group])).expect("no mutation"),
            Value::Bool(false)
        );
    }

    #[test]
    fn setpref_status_result_is_a_separately_gated_extension() {
        let _count = crate::output_count::push_output_count(Some(1));
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let group = Value::String("runmatSetprefStatusAdmission".into());
        let error = run(setpref_builtin(vec![
            group.clone(),
            Value::String("name".into()),
            unregistered_resident_value(),
        ]))
        .expect_err("status extension must reject before gather");
        assert_eq!(
            error.identifier(),
            runmat_builtins::SETPREF_STATUS_EXTENSION.error_identifier
        );
        assert_eq!(
            run(ispref_builtin(vec![group])).expect("no mutation"),
            Value::Bool(false)
        );
    }

    #[test]
    fn getpref_descriptor_and_integer_capabilities_cover_documented_forms() {
        assert_eq!(GETPREF_DESCRIPTOR.signatures.len(), 4);
        assert_eq!(GETPREF_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(GETPREF_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(
            GETPREF_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(
            GETPREF_INTEGER_CAPABILITIES[0].backend,
            BuiltinIntegerBackendRule::HostOnly
        );
    }

    #[test]
    fn getpref_persists_and_preserves_every_integer_scalar_class() {
        for (index, value) in [
            Value::Int(runmat_value::IntValue::I8(i8::MIN)),
            Value::Int(runmat_value::IntValue::I16(i16::MIN)),
            Value::Int(runmat_value::IntValue::I32(i32::MIN)),
            Value::Int(runmat_value::IntValue::I64(i64::MIN)),
            Value::Int(runmat_value::IntValue::U8(u8::MAX)),
            Value::Int(runmat_value::IntValue::U16(u16::MAX)),
            Value::Int(runmat_value::IntValue::U32(u32::MAX)),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ]
        .into_iter()
        .enumerate()
        {
            let preference = format!("scalar{index}");
            let created = run(getpref_builtin(vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference.clone()),
                value.clone(),
            ]))
            .expect("create integer preference");
            assert_eq!(created, value);
            let stored = run(getpref_builtin(vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference),
            ]))
            .expect("read integer preference");
            assert_eq!(stored, value);
        }
    }

    #[test]
    fn getpref_preserves_every_integer_array_class_and_shape() {
        for (index, storage) in [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ]
        .into_iter()
        .enumerate()
        {
            let value = Value::Tensor(
                Tensor::new_integer(storage, vec![1, 2]).expect("integer preference array"),
            );
            let preference = format!("array{index}");
            let created = run(getpref_builtin(vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference.clone()),
                value.clone(),
            ]))
            .expect("create integer array preference");
            assert_eq!(created, value);
            let stored = run(getpref_builtin(vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference),
            ]))
            .expect("read integer array preference");
            assert_eq!(stored, value);
        }
    }

    #[test]
    fn getpref_existing_value_wins_over_default_without_conversion() {
        let existing = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        run(getpref_builtin(vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            existing.clone(),
        ]))
        .expect("create preference");
        let result = run(getpref_builtin(vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            Value::Num(1.0),
        ]))
        .expect("read existing preference");
        assert_eq!(result, existing);
    }

    #[test]
    fn getpref_preserves_arbitrary_object_values_and_zero_argument_structure() {
        let mut object = ObjectInstance::new("PreferencePayload".to_string());
        object.properties.insert(
            "revision".to_string(),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        );
        let value = Value::Object(object);
        let created = run(getpref_builtin(vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
            value.clone(),
        ]))
        .expect("create object preference");
        assert_eq!(created, value);
        let stored = run(getpref_builtin(vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
        ]))
        .expect("read object preference");
        assert_eq!(stored, value);

        let all = run(getpref_builtin(Vec::new())).expect("all preferences");
        let Value::Struct(all) = all else {
            panic!("expected preference structure");
        };
        assert!(all.fields.contains_key("getprefObjects"));
    }

    #[test]
    fn getpref_multiple_names_preserve_query_shape_and_persist_defaults() {
        let names = Value::StringArray(
            runmat_value::StringArray::new(vec!["signed".into(), "unsigned".into()], vec![2, 1])
                .expect("preference names"),
        );
        let defaults = Value::Cell(
            CellArray::new_with_shape(
                vec![
                    Value::Int(runmat_value::IntValue::I64(i64::MIN)),
                    Value::Int(runmat_value::IntValue::U64(u64::MAX)),
                ],
                vec![2, 1],
            )
            .expect("defaults"),
        );
        let created = run(getpref_builtin(vec![
            Value::String("getprefMultiple".into()),
            names.clone(),
            defaults.clone(),
        ]))
        .expect("create multiple preferences");
        assert_eq!(created, defaults);
        let stored = run(getpref_builtin(vec![
            Value::String("getprefMultiple".into()),
            names,
        ]))
        .expect("read multiple preferences");
        assert_eq!(stored, defaults);

        let cellstr = Value::Cell(
            CellArray::new_with_shape(
                vec![char_value("signed"), char_value("unsigned")],
                vec![1, 2],
            )
            .expect("cellstr names"),
        );
        let stored = run(getpref_builtin(vec![
            Value::String("getprefMultiple".into()),
            cellstr,
        ]))
        .expect("read cellstr preferences");
        let Value::Cell(stored) = stored else {
            panic!("expected cell result");
        };
        assert_eq!(stored.shape, vec![1, 2]);
    }

    #[test]
    fn getpref_group_query_is_a_separately_gated_extension() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(getpref_builtin(vec![Value::String("group".into())]))
            .expect_err("strict group query");
        assert_eq!(
            error.identifier(),
            GETPREF_GROUP_QUERY_EXTENSION.error_identifier
        );
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run(getpref_builtin(vec![Value::String("group".into())]))
            .expect("extension group query");
        assert!(matches!(result, Value::Struct(_)));
    }

    #[test]
    fn getpref_rejects_resident_controls_and_defaults_before_provider_access() {
        let resident = || unregistered_resident_value();
        let error = run(getpref_builtin(vec![
            resident(),
            Value::String("name".into()),
        ]))
        .expect_err("resident group");
        assert!(error.message().contains("group must be"));

        let error = run(getpref_builtin(vec![
            Value::String("getprefResident".into()),
            resident(),
        ]))
        .expect_err("resident preference name");
        assert!(error.message().contains("preference must be"));

        let error = run(getpref_builtin(vec![
            Value::String("getprefResident".into()),
            Value::String("value".into()),
            resident(),
        ]))
        .expect_err("resident default");
        assert!(error.message().contains("provider-resident default"));
    }

    #[test]
    fn getpref_rejects_every_integer_class_in_text_control_positions() {
        for value in [
            Value::Int(runmat_value::IntValue::I8(1)),
            Value::Int(runmat_value::IntValue::I16(1)),
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::Int(runmat_value::IntValue::I64(1)),
            Value::Int(runmat_value::IntValue::U8(1)),
            Value::Int(runmat_value::IntValue::U16(1)),
            Value::Int(runmat_value::IntValue::U32(1)),
            Value::Int(runmat_value::IntValue::U64(1)),
        ] {
            let error = run(getpref_builtin(vec![
                value.clone(),
                Value::String("name".into()),
            ]))
            .expect_err("integer group");
            assert!(error.message().contains("group must be"));
            let error = run(getpref_builtin(vec![
                Value::String("getprefIntegerControls".into()),
                value,
            ]))
            .expect_err("integer preference name");
            assert!(error.message().contains("preference must be"));
        }
    }

    #[test]
    fn memmapfile_decodes_named_typed_format() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_memmapfile_test.bin");
        std::fs::write(&path, [1u8, 0, 2, 0]).unwrap();
        let shape = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
        let fmt = Value::Cell(
            CellArray::new(
                vec![char_value("uint16"), shape, char_value("samples")],
                1,
                3,
            )
            .unwrap(),
        );
        let value = run(memmapfile_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("Format"),
            fmt,
        ]))
        .unwrap();
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        let Some(Value::Struct(data)) = object.properties.get("Data") else {
            panic!("expected Data struct");
        };
        let Some(Value::Tensor(samples)) = data.fields.get("samples") else {
            panic!("expected samples tensor");
        };
        assert_eq!(samples.shape, vec![2, 1]);
        assert_eq!(
            samples.integer_storage(),
            Some(&IntegerStorage::U16(vec![1, 2]))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn memmapfile_integer_controls_are_gated_before_file_access() {
        assert_eq!(MEMMAPFILE_INTEGER_CAPABILITIES.len(), 3);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(memmapfile_builtin(vec![
            Value::String("definitely-missing.bin".into()),
            Value::String("Offset".into()),
            Value::Int(runmat_value::IntValue::U64(0)),
        ]))
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
        let error = run(memmapfile_builtin(vec![Value::String(
            "definitely-missing-output-count.bin".into(),
        )]))
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
            numeric_usize(&Value::Int(IntValue::U16(64)), "memmapfile", "Offset").unwrap(),
            64
        );
        assert!(numeric_usize(&Value::Int(IntValue::I8(-1)), "memmapfile", "Offset").is_err());
        assert!(numeric_usize(&Value::Num(2.5), "memmapfile", "Offset").is_err());
        assert!(
            numeric_usize(&Value::Num(usize::MAX as f64 + 1.0), "memmapfile", "Offset").is_err()
        );

        let maximum = numeric_usize(&Value::Int(IntValue::U64(u64::MAX)), "memmapfile", "Repeat");
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
        };
        assert!(impossible_record.decode(&[], None).is_err());

        let impossible_repeat = MemmapFormat {
            dtype: MemmapDataType::parse("uint8").unwrap(),
            shape: vec![2, 1],
            field: None,
            additional_fields: Vec::new(),
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
        let result = futures::executor::block_on(memmapfile_builtin(vec![
            Value::String("records.bin".into()),
            char_value("Format"),
            format,
            char_value("Repeat"),
            Value::Num(2.0),
        ]))
        .expect("mapped table");
        let Value::Object(mapped) = result else {
            panic!("memmapfile result");
        };
        assert_eq!(mapped.properties.get("Data"), Some(&expected));
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
        let error = futures::executor::block_on(memmapfile_builtin(vec![
            Value::String("definitely-missing-format-table.bin".into()),
            char_value("Format"),
            duplicated,
        ]))
        .expect_err("duplicate fields fail before file access");
        assert!(error.message().contains("unique"));
        let overflow = MemmapFormat {
            dtype: MemmapDataType::parse("uint64").unwrap(),
            shape: vec![usize::MAX, 1],
            field: Some("wide".into()),
            additional_fields: Vec::new(),
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
        let unknown = run(memmapfile_builtin(vec![
            filename.clone(),
            Value::String("Offset".into()),
            earlier_resident.clone(),
            Value::String("Unknown".into()),
            Value::Num(1.0),
        ]))
        .expect_err("unknown option is a preflight error");
        assert!(unknown.message().contains("unsupported option"));
        assert!(!unknown.message().to_ascii_lowercase().contains("provider"));

        let invalid = run(memmapfile_builtin(vec![
            filename,
            Value::String("Offset".into()),
            earlier_resident,
            Value::String("Format".into()),
            Value::String("not-a-supported-dtype".into()),
        ]))
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
            }
            .decode(&bytes, None)
            .unwrap();
            let Value::Tensor(tensor) = mapped else {
                panic!("expected tensor");
            };
            assert_eq!(tensor.shape, vec![1, 2], "{dtype}");
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
        }
        .decode(&bytes, Some(2))
        .unwrap();
        let Value::Tensor(tensor) = mapped else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1, 2]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U32(vec![1, u32::MAX, 3, 4]))
        );
    }

    #[test]
    fn system_two_outputs_returns_status_and_text() {
        #[cfg(windows)]
        let command = "echo hello";
        #[cfg(not(windows))]
        let command = "printf hello";
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = run(system_builtin(vec![Value::String(command.to_string())])).unwrap();
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
        let error = run(system_builtin(vec![Value::Int(
            runmat_value::IntValue::U64(u64::MAX),
        )]))
        .expect_err("integer command must reject");
        assert!(error.message().contains("command must be"));
    }

    #[test]
    fn system_rejects_integer_echo_options_before_process_access() {
        let error = run(system_builtin(vec![
            Value::String("printf should-not-run".to_string()),
            Value::Int(runmat_value::IntValue::U8(1)),
        ]))
        .expect_err("integer option must reject");
        assert!(error.message().contains("option must be"));
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
            "RunMat:fileattrib:UnsupportedPermission"
        );
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
}
