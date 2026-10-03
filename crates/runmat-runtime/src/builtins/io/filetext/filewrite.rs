//! RunMat-native `filewrite` builtin.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntValue, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::OpenOptions;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::filewrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filewrite",
    op_kind: GpuOpKind::Custom("io-file-write"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs synchronous host file I/O; GPU providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::filewrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filewrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Standalone host-side operation; never fused with other kernels.",
};

const BUILTIN_NAME: &str = "filewrite";

const FILEWRITE_NATIVE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filewrite-runmat-native",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "filewrite is a RunMat-native file-writing API with no MATLAB builtin counterpart",
    error_identifier: Some("RunMat:compatibility:FilewriteNativeExtension"),
};

pub const FILEWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FILEWRITE_NATIVE_EXTENSION];

const FILEWRITE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are read from authoritative storage and each value must lie in the inclusive byte range 0..255.",
    }];

pub const FILEWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "count = filewrite(filename, integer_data, ...)",
        inputs: &FILEWRITE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The complete filewrite surface is a gated RunMat-native extension; integer data is validated exactly before encoding and host I/O, resident data gathers to the host, and the byte count is a host double scalar.",
    }];

const FILEWRITE_OUTPUT_COUNT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "count",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written.",
}];
const FILEWRITE_INPUTS_FILENAME_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to target file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or uint8-compatible data to write.",
    },
];
const FILEWRITE_INPUTS_FILENAME_DATA_ENCODING: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to target file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or uint8-compatible data to write.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Positional encoding label.",
    },
];
const FILEWRITE_INPUTS_FILENAME_DATA_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to target file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or uint8-compatible data to write.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option name: 'Encoding' or 'WriteMode'.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value associated with option name.",
    },
];
const FILEWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data, encoding)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA_ENCODING,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data, \"Encoding\", encoding)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA_OPTIONS,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data, ..., \"WriteMode\", mode)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA_OPTIONS,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
];

const FILEWRITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INVALID_INPUT",
    identifier: Some("RunMat:filewrite:InvalidInput"),
    when: "Filename/data argument constraints are violated.",
    message: "filewrite: invalid input arguments",
};
const FILEWRITE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INVALID_OPTION",
    identifier: Some("RunMat:filewrite:InvalidOption"),
    when: "Name/value options are malformed or unsupported.",
    message: "filewrite: invalid option configuration",
};
const FILEWRITE_ERROR_INVALID_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INVALID_DATA",
    identifier: Some("RunMat:filewrite:InvalidData"),
    when: "Input data cannot be converted to writable bytes.",
    message: "filewrite: unsupported data payload",
};
const FILEWRITE_ERROR_ENCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.ENCODE",
    identifier: Some("RunMat:filewrite:EncodeFailed"),
    when: "Encoding selected output bytes/chars fails.",
    message: "filewrite: failed to encode payload",
};
const FILEWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.IO",
    identifier: Some("RunMat:filewrite:IoFailure"),
    when: "Filesystem open/write/flush operation fails.",
    message: "filewrite: file I/O failed",
};
const FILEWRITE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion fails.",
    message: "filewrite: internal error",
};
const FILEWRITE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FILEWRITE_ERROR_INVALID_INPUT,
    FILEWRITE_ERROR_INVALID_OPTION,
    FILEWRITE_ERROR_INVALID_DATA,
    FILEWRITE_ERROR_ENCODE,
    FILEWRITE_ERROR_IO,
    FILEWRITE_ERROR_INTERNAL,
];
pub const FILEWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILEWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILEWRITE_ERRORS,
};

fn filewrite_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filewrite_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn filewrite_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn filewrite_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FILEWRITE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow_with_context(err: RuntimeError, context: &str) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {context}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FILEWRITE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileEncoding {
    Auto,
    Utf8,
    Ascii,
    Latin1,
    Raw,
}

impl FileEncoding {
    fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" | "system" | "native" => Some(FileEncoding::Auto),
            "utf-8" | "utf8" | "utf_8" | "unicode" => Some(FileEncoding::Utf8),
            "ascii" | "us-ascii" | "us_ascii" | "usascii" => Some(FileEncoding::Ascii),
            "latin1" | "latin-1" | "latin_1" | "iso-8859-1" | "iso8859-1" | "iso88591" => {
                Some(FileEncoding::Latin1)
            }
            "raw" | "bytes" | "byte" | "binary" => Some(FileEncoding::Raw),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    Overwrite,
    Append,
}

impl WriteMode {
    fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "overwrite" | "replace" | "truncate" => Some(WriteMode::Overwrite),
            "append" | "add" => Some(WriteMode::Append),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FilewriteOptions {
    encoding: FileEncoding,
    write_mode: WriteMode,
}

impl Default for FilewriteOptions {
    fn default() -> Self {
        Self {
            encoding: FileEncoding::Auto,
            write_mode: WriteMode::Overwrite,
        }
    }
}

#[runtime_builtin(
    name = "filewrite",
    category = "io/filetext",
    summary = "Write text or raw bytes to files.",
    keywords = "filewrite,io,write file,text file,append,encoding",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::filewrite_type),
    descriptor(crate::builtins::io::filetext::filewrite::FILEWRITE_DESCRIPTOR),
    extensions(crate::builtins::io::filetext::filewrite::FILEWRITE_EXTENSIONS),
    integer_capabilities(crate::builtins::io::filetext::filewrite::FILEWRITE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::filetext::filewrite"
)]
async fn filewrite_builtin(
    path: Value,
    data: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &FILEWRITE_NATIVE_EXTENSION,
        BUILTIN_NAME,
    )?;
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let data = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_options(&rest)?;
    let resolved = resolve_path(&path)?;
    let payload = prepare_payload(&data, options.encoding)?;
    let written = write_bytes(&resolved, &payload, options.write_mode).await?;
    Ok(Value::Num(written as f64))
}

async fn gather_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

fn parse_options(args: &[Value]) -> BuiltinResult<FilewriteOptions> {
    if args.is_empty() {
        return Ok(FilewriteOptions::default());
    }

    let mut options = FilewriteOptions::default();
    let mut idx = 0usize;
    let mut encoding_specified = false;
    let mut write_mode_specified = false;

    if !args.is_empty() && !is_keyword(&args[0])? {
        match encoding_from_value(&args[0]) {
            Ok(enc) => {
                options.encoding = enc;
                encoding_specified = true;
                idx = 1;
            }
            Err(err) => {
                if args.len() == 1 {
                    return Err(err);
                }
            }
        }
    }

    if !(args.len() - idx).is_multiple_of(2) {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "expected keyword/value argument pairs",
        ));
    }

    while idx < args.len() {
        let key = keyword_name(&args[idx])?;
        let value = &args[idx + 1];
        if key.eq_ignore_ascii_case("encoding") {
            if encoding_specified {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_OPTION,
                    "duplicate 'Encoding' argument",
                ));
            }
            options.encoding = encoding_from_value(value)?;
            encoding_specified = true;
        } else if key.eq_ignore_ascii_case("writemode") {
            if write_mode_specified {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_OPTION,
                    "duplicate 'WriteMode' argument",
                ));
            }
            options.write_mode = write_mode_from_value(value)?;
            write_mode_specified = true;
        } else {
            return Err(filewrite_error_with_detail(
                &FILEWRITE_ERROR_INVALID_OPTION,
                format!("unrecognised option '{}'", key),
            ));
        }
        idx += 2;
    }

    Ok(options)
}

fn is_keyword(value: &Value) -> BuiltinResult<bool> {
    let text = keyword_name(value)?;
    Ok(text.eq_ignore_ascii_case("encoding") || text.eq_ignore_ascii_case("writemode"))
}

fn keyword_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::CharArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "keyword names must be 1-by-N character vectors",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::StringArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "keyword inputs must be scalar string arrays",
        )),
        other => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            format!("expected keyword as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn encoding_from_value(value: &Value) -> BuiltinResult<FileEncoding> {
    let label = keyword_name(value)?;
    match FileEncoding::from_label(&label) {
        Some(enc) => Ok(enc),
        None if label.trim().is_empty() => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "encoding name must not be empty",
        )),
        None => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            format!("unsupported encoding '{}'", label),
        )),
    }
}

fn write_mode_from_value(value: &Value) -> BuiltinResult<WriteMode> {
    let label = keyword_name(value)?;
    match WriteMode::from_label(&label) {
        Some(mode) => Ok(mode),
        None if label.trim().is_empty() => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            "write mode must not be empty",
        )),
        None => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_OPTION,
            format!(
                "unsupported write mode '{}'; use 'overwrite' or 'append'",
                label
            ),
        )),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let path: String = ca.data.iter().collect();
            normalize_path(&path)
        }
        Value::CharArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            "filename must be a 1-by-N character vector",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        Value::StringArray(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            "string array filename inputs must be scalar",
        )),
        other => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            format!("expected filename as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.is_empty() {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_INPUT,
            "filename must not be empty",
        ));
    }
    Ok(Path::new(raw).to_path_buf())
}

enum Payload {
    Text(Vec<char>),
    Bytes(Vec<u8>),
}

fn prepare_payload(data: &Value, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    let payload = extract_payload(data)?;
    match payload {
        Payload::Text(chars) => encode_text(chars, encoding),
        Payload::Bytes(bytes) => encode_bytes(bytes, encoding),
    }
}

fn extract_payload(data: &Value) -> BuiltinResult<Payload> {
    match data {
        Value::CharArray(ca) => Ok(Payload::Text(char_array_to_text(ca))),
        Value::String(s) => Ok(Payload::Text(s.chars().collect())),
        Value::StringArray(sa) => Ok(Payload::Text(string_array_to_text(sa))),
        Value::Num(n) => {
            let byte = float_to_byte(*n)
                .map_err(|err| map_control_flow_with_context(err, "filewrite: numeric value"))?;
            Ok(Payload::Bytes(vec![byte]))
        }
        Value::Int(i) => {
            let byte = int_value_to_byte(i)
                .map_err(|err| map_control_flow_with_context(err, "filewrite: integer value"))?;
            Ok(Payload::Bytes(vec![byte]))
        }
        Value::Bool(flag) => Ok(Payload::Bytes(vec![if *flag { 1 } else { 0 }])),
        Value::Tensor(t) => Ok(Payload::Bytes(tensor_to_bytes(t)?)),
        Value::LogicalArray(la) => Ok(Payload::Bytes(logical_to_bytes(la))),
        Value::Cell(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            "cell arrays are not supported inputs",
        )),
        Value::GpuTensor(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INTERNAL,
            "GPU tensor should be gathered before filewrite payload extraction",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            "complex data must be converted to text before writing",
        )),
        other => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("unsupported data type {other:?}; expected text or uint8-compatible array"),
        )),
    }
}

fn char_array_to_text(ca: &CharArray) -> Vec<char> {
    if ca.rows <= 1 {
        return ca.data.clone();
    }
    let mut out = Vec::with_capacity(ca.rows * (ca.cols + 1));
    for row in 0..ca.rows {
        for col in 0..ca.cols {
            out.push(ca.data[row * ca.cols + col]);
        }
        if row + 1 < ca.rows {
            out.push('\n');
        }
    }
    out
}

fn string_array_to_text(sa: &StringArray) -> Vec<char> {
    if sa.data.is_empty() {
        return Vec::new();
    }
    let mut combined = String::new();
    for (idx, entry) in sa.data.iter().enumerate() {
        combined.push_str(entry);
        if idx + 1 < sa.data.len() {
            combined.push('\n');
        }
    }
    combined.chars().collect()
}

fn tensor_to_bytes(tensor: &Tensor) -> BuiltinResult<Vec<u8>> {
    let len = tensor_utils::tensor_element_len(tensor);
    let mut out = Vec::with_capacity(len);
    if let Some(storage) = tensor.integer_storage() {
        for idx in 0..len {
            let value = storage
                .value_at(idx)
                .expect("integer storage mirrors tensor shape");
            match int_value_to_byte(&value) {
                Ok(byte) => out.push(byte),
                Err(msg) => {
                    return Err(filewrite_error_with_detail(
                        &FILEWRITE_ERROR_INVALID_DATA,
                        format!("numeric element {} {}", idx, msg),
                    ));
                }
            }
        }
        return Ok(out);
    }
    let values = tensor_utils::tensor_values_f64_cow(tensor);
    for (idx, value) in values.iter().copied().enumerate() {
        match float_to_byte(value) {
            Ok(byte) => out.push(byte),
            Err(msg) => {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_DATA,
                    format!("numeric element {} {}", idx, msg),
                ));
            }
        }
    }
    Ok(out)
}

fn logical_to_bytes(array: &LogicalArray) -> Vec<u8> {
    array
        .data
        .iter()
        .map(|value| if *value != 0 { 1 } else { 0 })
        .collect()
}

fn float_to_byte(value: f64) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not finite; cannot write as raw byte"),
        ));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not an integer in the range 0..255"),
        ));
    }
    let int = rounded as i128;
    if !(0..=255).contains(&int) {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not in the range 0..255"),
        ));
    }
    Ok(int as u8)
}

fn int_value_to_byte(value: &IntValue) -> BuiltinResult<u8> {
    match value {
        IntValue::I8(v) => signed_to_byte(*v as i64),
        IntValue::I16(v) => signed_to_byte(*v as i64),
        IntValue::I32(v) => signed_to_byte(*v as i64),
        IntValue::I64(v) => signed_to_byte(*v),
        IntValue::U8(v) => Ok(*v),
        IntValue::U16(v) => unsigned_to_byte(*v as u64),
        IntValue::U32(v) => unsigned_to_byte(*v as u64),
        IntValue::U64(v) => unsigned_to_byte(*v),
    }
}

fn signed_to_byte(value: i64) -> BuiltinResult<u8> {
    if !(0..=255).contains(&value) {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not in the range 0..255"),
        ));
    }
    Ok(value as u8)
}

fn unsigned_to_byte(value: u64) -> BuiltinResult<u8> {
    if value > 255 {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not in the range 0..255"),
        ));
    }
    Ok(value as u8)
}

fn encode_text(chars: Vec<char>, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    match encoding {
        FileEncoding::Auto | FileEncoding::Utf8 => {
            Ok(chars.iter().collect::<String>().into_bytes())
        }
        FileEncoding::Ascii => encode_ascii_chars(&chars),
        FileEncoding::Latin1 | FileEncoding::Raw => encode_latin_chars(&chars, encoding),
    }
}

fn encode_ascii_chars(chars: &[char]) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(chars.len());
    for &ch in chars {
        if ch as u32 > 0x7F {
            return Err(filewrite_error_with_detail(
                &FILEWRITE_ERROR_ENCODE,
                format!(
                    "character '{}' (U+{:04X}) cannot be encoded as ASCII",
                    ch, ch as u32
                ),
            ));
        }
        out.push(ch as u8);
    }
    Ok(out)
}

fn encode_latin_chars(chars: &[char], encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(chars.len());
    for &ch in chars {
        if ch as u32 > 0xFF {
            return Err(filewrite_error_with_detail(
                &FILEWRITE_ERROR_ENCODE,
                format!(
                    "character '{}' (U+{:04X}) cannot be encoded as {}",
                    ch,
                    ch as u32,
                    match encoding {
                        FileEncoding::Latin1 => "Latin-1",
                        FileEncoding::Raw => "raw bytes",
                        _ => unreachable!(),
                    }
                ),
            ));
        }
        out.push(ch as u8);
    }
    Ok(out)
}

fn encode_bytes(bytes: Vec<u8>, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    if matches!(encoding, FileEncoding::Ascii) {
        for (idx, byte) in bytes.iter().enumerate() {
            if *byte > 0x7F {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_ENCODE,
                    format!("byte 0x{byte:02X} at index {idx} cannot be encoded as ASCII"),
                ));
            }
        }
    }
    Ok(bytes)
}

async fn write_bytes(path: &Path, payload: &[u8], mode: WriteMode) -> BuiltinResult<usize> {
    let mut options = OpenOptions::new();
    options.create(true);
    match mode {
        WriteMode::Overwrite => {
            options.write(true).truncate(true);
        }
        WriteMode::Append => {
            options.write(true).append(true);
        }
    }

    let mut file = options.open_async(path).await.map_err(|err| {
        filewrite_error_with_source(
            format!(
                "{}: unable to open '{}': {}",
                FILEWRITE_ERROR_IO.message,
                path.display(),
                err
            ),
            &FILEWRITE_ERROR_IO,
            err,
        )
    })?;

    file.write_all(payload).map_err(|err| {
        filewrite_error_with_source(
            format!(
                "{}: unable to write to '{}': {}",
                FILEWRITE_ERROR_IO.message,
                path.display(),
                err
            ),
            &FILEWRITE_ERROR_IO,
            err,
        )
    })?;

    file.flush_async().await.map_err(|err| {
        filewrite_error_with_source(
            format!(
                "{}: unable to flush '{}': {}",
                FILEWRITE_ERROR_IO.message,
                path.display(),
                err
            ),
            &FILEWRITE_ERROR_IO,
            err,
        )
    })?;

    Ok(payload.len())
}

#[cfg(test)]
pub(crate) mod tests;
