//! MATLAB-compatible `dlmwrite` builtin for delimiter-separated exports.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_filesystem::{self as vfs, File, OpenOptions};
use runmat_macros::runtime_builtin;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "dlmwrite";
const MAX_NUMERIC_FORMAT_FIELD: usize = 4096;

const DLMWRITE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytesWritten",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written to the output file.",
}];
const DLMWRITE_INPUTS_FILENAME_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
];
const DLMWRITE_INPUTS_FILENAME_DATA_DELIM_ROW_COL: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\",\""),
        description: "Delimiter token.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0"),
        description: "Row offset before writing matrix rows.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0"),
        description: "Column offset before writing matrix columns.",
    },
];
const DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name.",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value.",
    },
];
const DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "args...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional delimiter/offset/append and name-value options.",
    },
];
const DLMWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA,
        outputs: &DLMWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M, delimiter, row, col)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_DELIM_ROW_COL,
        outputs: &DLMWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M, name, optionValue)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE,
        outputs: &DLMWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = dlmwrite(filename, M, args...)",
        inputs: &DLMWRITE_INPUTS_FILENAME_DATA_NAME_VALUE_PAIRS,
        outputs: &DLMWRITE_OUTPUT,
    },
];

const DLMWRITE_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.ARG_CONFIG",
    identifier: None,
    when: "Argument sequence or name-value grammar is malformed.",
    message: "dlmwrite: invalid argument configuration",
};
const DLMWRITE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.FILENAME",
    identifier: None,
    when: "Filename is missing, empty, or not a scalar string/char vector.",
    message: "dlmwrite: invalid filename input",
};
const DLMWRITE_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.OPTION",
    identifier: None,
    when: "Option name or value is invalid.",
    message: "dlmwrite: invalid option value",
};
const DLMWRITE_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.DATA",
    identifier: None,
    when: "Input data cannot be converted to a supported numeric/logical matrix.",
    message: "dlmwrite: invalid input data",
};
const DLMWRITE_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.DATA_SHAPE",
    identifier: None,
    when: "Input matrix is not 2-D.",
    message: "dlmwrite: input must be 2-D",
};
const DLMWRITE_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.FORMAT",
    identifier: None,
    when: "Numeric precision format is invalid or unsupported.",
    message: "dlmwrite: invalid precision format",
};
const DLMWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DLMWRITE.IO",
    identifier: None,
    when: "File inspection/open/write/flush operations fail.",
    message: "dlmwrite: file write failed",
};
const DLMWRITE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    DLMWRITE_ERROR_ARG_CONFIG,
    DLMWRITE_ERROR_FILENAME,
    DLMWRITE_ERROR_OPTION,
    DLMWRITE_ERROR_DATA,
    DLMWRITE_ERROR_DATA_SHAPE,
    DLMWRITE_ERROR_FORMAT,
    DLMWRITE_ERROR_IO,
];
pub const DLMWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLMWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DLMWRITE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::dlmwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dlmwrite",
    op_kind: GpuOpKind::Custom("io-dlmwrite"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; gpuArray inputs are gathered before formatting.",
};

fn dlmwrite_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    dlmwrite_error_with(error, error.message)
}

fn dlmwrite_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn dlmwrite_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::dlmwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dlmwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs synchronous file I/O.",
};

#[runtime_builtin(
    name = "dlmwrite",
    category = "io/tabular",
    summary = "Write numeric matrices to delimiter-separated text files.",
    keywords = "dlmwrite,delimiter,precision,append,roffset,coffset",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::tabular::dlmwrite::DLMWRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::tabular::dlmwrite"
)]
async fn dlmwrite_builtin(
    filename: Value,
    data: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let gathered_path = gather_if_needed_async(&filename)
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&gathered_path)?;

    let mut gathered_args = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_args.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    let options = parse_arguments(&gathered_args)?;

    let gathered_data = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let tensor = tensor::value_into_tensor_for("dlmwrite", gathered_data)
        .map_err(|msg| dlmwrite_error_with(&DLMWRITE_ERROR_DATA, format!("dlmwrite: {msg}")))?;
    ensure_matrix_shape(&tensor)?;

    let bytes = write_dlm(&path, &tensor, &options).await?;
    Ok(Value::Num(bytes as f64))
}

#[derive(Clone, Debug)]
struct DlmWriteOptions {
    delimiter: String,
    newline: LineEnding,
    roffset: usize,
    coffset: usize,
    precision: PrecisionSpec,
    append: bool,
}

impl Default for DlmWriteOptions {
    fn default() -> Self {
        Self {
            delimiter: ",".to_string(),
            newline: LineEnding::platform_default(),
            roffset: 0,
            coffset: 0,
            precision: PrecisionSpec::Significant(5),
            append: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum LineEnding {
    Unix,
    Pc,
    Mac,
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Unix => "\n",
            LineEnding::Pc => "\r\n",
            LineEnding::Mac => "\r",
        }
    }

    fn platform_default() -> Self {
        if cfg!(windows) {
            LineEnding::Pc
        } else {
            LineEnding::Unix
        }
    }
}

#[derive(Clone, Debug)]
enum PrecisionSpec {
    Significant(u32),
    Format(String),
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<DlmWriteOptions> {
    let mut options = DlmWriteOptions::default();
    if args.is_empty() {
        return Ok(options);
    }

    let mut idx = 0usize;
    // Consume any leading '-append' flags.
    while idx < args.len() {
        if is_append_flag(&args[idx]) {
            options.append = true;
            idx += 1;
        } else {
            break;
        }
    }

    // Optional positional delimiter.
    if idx < args.len() && is_positional_delimiter_candidate(&args[idx]) {
        options.delimiter = parse_delimiter_value(&args[idx])?;
        idx += 1;
        // Optional positional row/col offsets if both numeric scalars follow.
        if idx + 1 < args.len()
            && is_numeric_scalar(&args[idx])
            && is_numeric_scalar(&args[idx + 1])
        {
            options.roffset = parse_offset_value(&args[idx], "row offset")?;
            options.coffset = parse_offset_value(&args[idx + 1], "column offset")?;
            idx += 2;
        }
        // Consume any subsequent '-append'.
        while idx < args.len() {
            if is_append_flag(&args[idx]) {
                options.append = true;
                idx += 1;
            } else {
                break;
            }
        }
    }

    // Remaining arguments should be name-value pairs (allowing additional '-append').
    while idx < args.len() {
        if is_append_flag(&args[idx]) {
            options.append = true;
            idx += 1;
            continue;
        }

        let name = value_to_lowercase_string(&args[idx])
            .ok_or_else(|| dlmwrite_error(&DLMWRITE_ERROR_ARG_CONFIG))?;
        idx += 1;
        if idx >= args.len() {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_ARG_CONFIG,
                "dlmwrite: name-value arguments must appear in pairs",
            ));
        }
        let value = &args[idx];
        idx += 1;

        match name.as_str() {
            "delimiter" => {
                options.delimiter = parse_delimiter_value(value)?;
            }
            "precision" => {
                options.precision = parse_precision_value(value)?;
            }
            "newline" => {
                options.newline = parse_newline_value(value)?;
            }
            "roffset" => {
                options.roffset = parse_offset_value(value, "row offset")?;
            }
            "coffset" => {
                options.coffset = parse_offset_value(value, "column offset")?;
            }
            "append" => {
                options.append = parse_append_value(value)?;
            }
            other => {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    format!("dlmwrite: unsupported name-value pair '{other}'"),
                ));
            }
        }
    }

    Ok(options)
}

fn is_append_flag(value: &Value) -> bool {
    match value {
        Value::String(s) => s.trim().eq_ignore_ascii_case("-append"),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            text.trim().eq_ignore_ascii_case("-append")
        }
        Value::StringArray(sa) if sa.data.len() == 1 => {
            sa.data[0].trim().eq_ignore_ascii_case("-append")
        }
        _ => false,
    }
}

fn is_positional_delimiter_candidate(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            if let Some(lower) = value_to_lowercase_string(value) {
                match lower.as_str() {
                    "delimiter" | "precision" | "newline" | "roffset" | "coffset" | "append"
                    | "-append" => return false,
                    _ => {}
                }
            }
            true
        }
        _ => false,
    }
}

fn parse_delimiter_value(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => interpret_delimiter_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            interpret_delimiter_string(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => interpret_delimiter_string(&sa.data[0]),
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: delimiter must be a string scalar or character vector",
        )),
    }
}

fn interpret_delimiter_string(raw: &str) -> BuiltinResult<String> {
    if raw.is_empty() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: delimiter must not be empty",
        ));
    }
    if raw == r"\t" {
        return Ok("\t".to_string());
    }
    if raw == r"\n" {
        return Ok("\n".to_string());
    }
    if raw == r"\r" {
        return Ok("\r".to_string());
    }
    Ok(raw.to_string())
}

fn is_numeric_scalar(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

fn parse_offset_value(value: &Value, context: &str) -> BuiltinResult<usize> {
    let scalar = extract_scalar(value).map_err(|e| {
        dlmwrite_error_with(&DLMWRITE_ERROR_OPTION, format!("dlmwrite: {context} {e}"))
    })?;
    if !scalar.is_finite() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be finite"),
        ));
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > 1e-9 {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be an integer, got {scalar}"),
        ));
    }
    if rounded < 0.0 {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: {context} must be >= 0"),
        ));
    }
    Ok(rounded as usize)
}

fn parse_precision_value(value: &Value) -> BuiltinResult<PrecisionSpec> {
    match value {
        Value::Int(i) => {
            let digits = i.to_i64();
            if digits <= 0 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ));
            }
            Ok(PrecisionSpec::Significant(digits as u32))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision scalar must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1e-9 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision scalar must be an integer",
                ));
            }
            if rounded <= 0.0 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ));
            }
            Ok(PrecisionSpec::Significant(rounded as u32))
        }
        Value::Tensor(t) if t.data.len() == 1 => parse_precision_value(&Value::Num(t.data[0])),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            if logical.data[0] != 0 {
                Ok(PrecisionSpec::Significant(1))
            } else {
                Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ))
            }
        }
        Value::Bool(b) => {
            if *b {
                Ok(PrecisionSpec::Significant(1))
            } else {
                Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_OPTION,
                    "dlmwrite: precision must be a positive integer",
                ))
            }
        }
        Value::String(s) => parse_precision_format(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_precision_format(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => parse_precision_format(&sa.data[0]),
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: precision must be numeric or a format string",
        )),
    }
}

fn parse_precision_format(text: &str) -> BuiltinResult<PrecisionSpec> {
    if text.is_empty() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: precision format string must not be empty",
        ));
    }
    Ok(PrecisionSpec::Format(text.to_string()))
}

fn parse_newline_value(value: &Value) -> BuiltinResult<LineEnding> {
    let text = value_to_lowercase_string(value).ok_or_else(|| {
        dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: newline must be a string scalar or character vector",
        )
    })?;
    match text.as_str() {
        "pc" | "windows" | "crlf" => Ok(LineEnding::Pc),
        "unix" | "lf" => Ok(LineEnding::Unix),
        "mac" | "cr" => Ok(LineEnding::Mac),
        other => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            format!("dlmwrite: unsupported newline setting '{other}' (expected 'pc' or 'unix')"),
        )),
    }
}

fn parse_append_value(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => Ok(*n != 0.0),
        Value::String(s) => parse_bool_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_bool_string(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => parse_bool_string(&sa.data[0]),
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: append value must be logical",
        )),
    }
}

fn parse_bool_string(text: &str) -> BuiltinResult<bool> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "dlmwrite: append value must be logical",
        )),
    }
}

fn extract_scalar(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_OPTION,
            "must be numeric scalar",
        )),
    }
}

fn value_to_lowercase_string(value: &Value) -> Option<String> {
    tensor::value_to_string(value).map(|s| s.trim().to_ascii_lowercase())
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_FILENAME,
                "dlmwrite: filename must be a string scalar or character vector",
            ))
        }
    };
    if raw.trim().is_empty() {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_FILENAME,
            "dlmwrite: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(&raw, BUILTIN_NAME)
        .map_err(|msg| dlmwrite_error_with(&DLMWRITE_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn ensure_matrix_shape(tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape.len() <= 2 {
        return Ok(());
    }
    if tensor.shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(dlmwrite_error_with(
        &DLMWRITE_ERROR_DATA_SHAPE,
        "dlmwrite: input must be 2-D; reshape before writing",
    ))
}

async fn write_dlm(
    path: &Path,
    tensor: &Tensor,
    options: &DlmWriteOptions,
) -> BuiltinResult<usize> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let newline = options.newline.as_str();

    let (existing_nonempty, ends_with_newline) = if options.append {
        match vfs::metadata_async(path).await {
            Ok(meta) if !meta.is_empty() => {
                let ends = file_ends_with_newline(path).await.map_err(|err| {
                    dlmwrite_error_with_source(
                        &DLMWRITE_ERROR_IO,
                        format!(
                            "dlmwrite: failed to inspect existing file \"{}\" ({err})",
                            path.display()
                        ),
                        err,
                    )
                })?;
                (true, ends)
            }
            Ok(_) => (false, false),
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    (false, false)
                } else {
                    return Err(dlmwrite_error_with_source(
                        &DLMWRITE_ERROR_IO,
                        format!("dlmwrite: unable to inspect \"{}\" ({err})", path.display()),
                        err,
                    ));
                }
            }
        }
    } else {
        (false, false)
    };

    let mut open = OpenOptions::new();
    open.create(true);
    if options.append {
        open.append(true);
    } else {
        open.write(true).truncate(true);
    }

    let mut file = open.open_async(path).await.map_err(|err| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            format!(
                "dlmwrite: unable to open \"{}\" for writing ({err})",
                path.display()
            ),
            err,
        )
    })?;

    let mut bytes = 0usize;

    if options.append && existing_nonempty && !ends_with_newline {
        file.write_all(newline.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to insert newline before append ({err})"),
                err,
            )
        })?;
        bytes += newline.len();
    }

    for _ in 0..options.roffset {
        bytes += write_blank_row(
            &mut file,
            cols,
            options.coffset,
            &options.delimiter,
            newline,
        )?;
    }

    if rows == 0 || cols == 0 {
        file.flush_async().await.map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to flush output ({err})"),
                err,
            )
        })?;
        return Ok(bytes);
    }

    for row in 0..rows {
        let mut fields = Vec::with_capacity(options.coffset + cols);
        for _ in 0..options.coffset {
            fields.push(String::new());
        }
        for col in 0..cols {
            let idx = row + col * rows;
            let value = tensor.data[idx];
            fields.push(format_numeric(value, &options.precision)?);
        }
        let line = fields.join(&options.delimiter);
        if !line.is_empty() {
            file.write_all(line.as_bytes()).map_err(|err| {
                dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    format!("dlmwrite: failed to write data ({err})"),
                    err,
                )
            })?;
            bytes += line.len();
        }
        file.write_all(newline.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to write newline ({err})"),
                err,
            )
        })?;
        bytes += newline.len();
    }

    file.flush_async().await.map_err(|err| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            format!("dlmwrite: failed to flush output ({err})"),
            err,
        )
    })?;
    Ok(bytes)
}

fn write_blank_row(
    file: &mut File,
    cols: usize,
    coffset: usize,
    delimiter: &str,
    newline: &str,
) -> BuiltinResult<usize> {
    let mut bytes = 0usize;
    if coffset == 0 && cols == 0 {
        file.write_all(newline.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to write offset newline ({err})"),
                err,
            )
        })?;
        return Ok(newline.len());
    }
    let mut fields = Vec::with_capacity(coffset + cols);
    for _ in 0..coffset {
        fields.push(String::new());
    }
    for _ in 0..cols {
        fields.push(String::new());
    }
    let line = fields.join(delimiter);
    if !line.is_empty() {
        file.write_all(line.as_bytes()).map_err(|err| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                format!("dlmwrite: failed to write offset row ({err})"),
                err,
            )
        })?;
        bytes += line.len();
    }
    file.write_all(newline.as_bytes()).map_err(|err| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            format!("dlmwrite: failed to write offset newline ({err})"),
            err,
        )
    })?;
    bytes += newline.len();
    Ok(bytes)
}

async fn file_ends_with_newline(path: &Path) -> io::Result<bool> {
    let metadata = vfs::metadata_async(path).await?;
    let len = metadata.len();
    if len == 0 {
        return Ok(false);
    }
    let mut file = File::open_async(path).await?;
    let to_read = len.min(2) as usize;
    file.seek(SeekFrom::End(-(to_read as i64)))?;
    let mut buffer = vec![0u8; to_read];
    file.read_exact(&mut buffer)?;
    Ok(buffer.contains(&b'\n') || buffer.contains(&b'\r'))
}

fn format_numeric(value: f64, precision: &PrecisionSpec) -> BuiltinResult<String> {
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        });
    }
    match precision {
        PrecisionSpec::Significant(digits) => {
            if *digits == 0 {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    "dlmwrite: precision must be positive",
                ));
            }
            let fmt = format!("%.{digits}g");
            let mut rendered = c_format(value, &fmt)?;
            if value == 0.0 || rendered == "-0" {
                rendered = "0".to_string();
            }
            Ok(rendered)
        }
        PrecisionSpec::Format(spec) => {
            let rendered = c_format(value, spec)?;
            if value == 0.0 && rendered.starts_with("-") {
                Ok(rendered.trim_start_matches('-').to_string())
            } else {
                Ok(rendered)
            }
        }
    }
}

fn c_format(value: f64, spec: &str) -> BuiltinResult<String> {
    format_float(value, spec)
}

fn format_float(value: f64, spec: &str) -> BuiltinResult<String> {
    let parsed = ParsedFormat::parse(spec)?;
    Ok(parsed.render(value))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatSpecifier {
    Fixed,
    Exponent,
    General,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SignFlag {
    None,
    Plus,
    Space,
}

#[derive(Clone, Copy, Debug)]
struct ParsedFormat {
    specifier: FloatSpecifier,
    uppercase: bool,
    alternate: bool,
    sign: SignFlag,
    left_adjust: bool,
    zero_pad: bool,
    width: Option<usize>,
    precision: Option<usize>,
}

impl ParsedFormat {
    fn parse(input: &str) -> BuiltinResult<Self> {
        use std::iter::Peekable;
        use std::str::Chars;

        fn parse_number(
            chars: &mut Peekable<Chars<'_>>,
            label: &str,
        ) -> BuiltinResult<Option<usize>> {
            let mut value: usize = 0;
            let mut saw_digit = false;
            while let Some(&ch) = chars.peek() {
                if ch.is_ascii_digit() {
                    saw_digit = true;
                    value = value
                        .checked_mul(10)
                        .and_then(|v| v.checked_add((ch as u8 - b'0') as usize))
                        .ok_or_else(|| {
                            dlmwrite_error_with(
                                &DLMWRITE_ERROR_FORMAT,
                                format!("dlmwrite: {label} too large in precision format"),
                            )
                        })?;
                    chars.next();
                } else {
                    break;
                }
            }
            if saw_digit && value > MAX_NUMERIC_FORMAT_FIELD {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    format!("dlmwrite: {label} exceeds maximum supported precision format size"),
                ));
            }
            Ok(if saw_digit { Some(value) } else { None })
        }

        let mut chars = input.chars().peekable();
        match chars.next() {
            Some('%') => {}
            _ => {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    "dlmwrite: precision format must start with '%'",
                ));
            }
        }

        let mut left_adjust = false;
        let mut sign = SignFlag::None;
        let mut zero_pad = false;
        let mut alternate = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '-' => {
                    left_adjust = true;
                    zero_pad = false;
                    chars.next();
                }
                '+' => {
                    sign = SignFlag::Plus;
                    chars.next();
                }
                ' ' => {
                    if sign != SignFlag::Plus {
                        sign = SignFlag::Space;
                    }
                    chars.next();
                }
                '0' => {
                    if !left_adjust {
                        zero_pad = true;
                    }
                    chars.next();
                }
                '#' => {
                    alternate = true;
                    chars.next();
                }
                _ => break,
            }
        }

        let width = parse_number(&mut chars, "field width")?;
        let precision = if matches!(chars.peek(), Some('.')) {
            chars.next();
            parse_number(&mut chars, "precision")?.or(Some(0))
        } else {
            None
        };

        if matches!(chars.peek(), Some('l' | 'L' | 'h')) {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_FORMAT,
                "dlmwrite: length modifiers are not supported in precision formats",
            ));
        }

        let spec_ch = chars.next().ok_or_else(|| {
            dlmwrite_error_with(
                &DLMWRITE_ERROR_FORMAT,
                "dlmwrite: incomplete precision format",
            )
        })?;
        if chars.next().is_some() {
            return Err(dlmwrite_error_with(
                &DLMWRITE_ERROR_FORMAT,
                "dlmwrite: unexpected trailing characters in precision format",
            ));
        }

        let (specifier, uppercase) = match spec_ch {
            'f' => (FloatSpecifier::Fixed, false),
            'F' => (FloatSpecifier::Fixed, true),
            'e' => (FloatSpecifier::Exponent, false),
            'E' => (FloatSpecifier::Exponent, true),
            'g' => (FloatSpecifier::General, false),
            'G' => (FloatSpecifier::General, true),
            other => {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_FORMAT,
                    format!("dlmwrite: unsupported precision format specifier '{other}'"),
                ));
            }
        };

        Ok(Self {
            specifier,
            uppercase,
            alternate,
            sign,
            left_adjust,
            zero_pad: zero_pad && !left_adjust,
            width,
            precision,
        })
    }

    fn render(&self, value: f64) -> String {
        let negative = value.is_sign_negative();
        let magnitude = if negative { -value } else { value };
        let mut body = match self.specifier {
            FloatSpecifier::Fixed => {
                format_fixed_body(magnitude, self.precision.unwrap_or(6), self.alternate)
            }
            FloatSpecifier::Exponent => format_exponential_body(
                magnitude,
                self.precision.unwrap_or(6),
                self.alternate,
                self.uppercase,
            ),
            FloatSpecifier::General => format_general_body(
                magnitude,
                self.precision.unwrap_or(6),
                self.alternate,
                self.uppercase,
            ),
        };
        if self.uppercase {
            body.make_ascii_uppercase();
        }

        let mut prefix = String::new();
        if negative {
            prefix.push('-');
        } else {
            match self.sign {
                SignFlag::Plus => prefix.push('+'),
                SignFlag::Space => prefix.push(' '),
                SignFlag::None => {}
            }
        }

        let total_len = prefix.len() + body.len();
        if let Some(width) = self.width {
            if width > total_len {
                let pad = width - total_len;
                if self.left_adjust {
                    let mut result = prefix;
                    result.push_str(&body);
                    result.extend(std::iter::repeat_n(' ', pad));
                    return result;
                } else if self.zero_pad {
                    let mut result = String::with_capacity(width);
                    result.push_str(&prefix);
                    result.extend(std::iter::repeat_n('0', pad));
                    result.push_str(&body);
                    return result;
                } else {
                    let mut result = String::with_capacity(width);
                    result.extend(std::iter::repeat_n(' ', pad));
                    result.push_str(&prefix);
                    result.push_str(&body);
                    return result;
                }
            }
        }

        let mut result = prefix;
        result.push_str(&body);
        result
    }
}

fn format_fixed_body(value: f64, precision: usize, alternate: bool) -> String {
    let mut s = format!("{:.*}", precision, value);
    if precision == 0 && alternate && !s.contains('.') {
        s.push('.');
    }
    s
}

fn format_exponential_body(
    value: f64,
    precision: usize,
    alternate: bool,
    uppercase: bool,
) -> String {
    let mut s = format!("{:.*e}", precision, value);
    normalize_exponent_notation(&mut s);
    if uppercase {
        s.make_ascii_uppercase();
    }
    if precision == 0 && alternate {
        insert_decimal_point(&mut s);
    }
    s
}

fn format_general_body(value: f64, precision: usize, alternate: bool, uppercase: bool) -> String {
    let effective_precision = precision.max(1);
    let abs_val = value.abs();
    if abs_val == 0.0 {
        return if alternate {
            let mut s = "0.".to_string();
            s.extend(std::iter::repeat_n('0', effective_precision - 1));
            s
        } else {
            "0".to_string()
        };
    }

    let exponent = abs_val.log10().floor() as i32;
    let use_exponent = exponent < -4 || exponent >= effective_precision as i32;
    let mut s = if use_exponent {
        let frac = effective_precision.saturating_sub(1);
        let mut out = format!("{:.*e}", frac, abs_val);
        normalize_exponent_notation(&mut out);
        if uppercase {
            out.make_ascii_uppercase();
        }
        out
    } else {
        let frac = {
            let diff = effective_precision as isize - (exponent + 1) as isize;
            if diff < 0 {
                0
            } else {
                diff as usize
            }
        };
        let mut out = format!("{:.*}", frac, abs_val);
        if uppercase {
            out.make_ascii_uppercase();
        }
        out
    };

    if alternate {
        insert_decimal_point(&mut s);
    } else {
        trim_trailing_zeros(&mut s);
    }
    s
}

fn insert_decimal_point(s: &mut String) {
    if s.contains('.') {
        return;
    }
    if let Some(idx) = find_exponent_index(s) {
        s.insert(idx, '.');
    } else {
        s.push('.');
    }
}

fn trim_trailing_zeros(s: &mut String) {
    if let Some(idx) = find_exponent_index(s) {
        let exponent = s[idx..].to_string();
        let mut mantissa = s[..idx].to_string();
        trim_fraction(&mut mantissa);
        s.clear();
        s.push_str(&mantissa);
        s.push_str(&exponent);
        normalize_exponent_notation(s);
    } else {
        trim_fraction(s);
    }
}

fn trim_fraction(s: &mut String) {
    if let Some(dot_idx) = s.find('.') {
        let mut idx = s.len();
        while idx > dot_idx + 1 && matches!(s.as_bytes().get(idx - 1), Some(b'0')) {
            idx -= 1;
        }
        if idx == dot_idx + 1 {
            idx -= 1;
        }
        s.truncate(idx);
    }
}

fn find_exponent_index(s: &str) -> Option<usize> {
    s.find('e').or_else(|| s.find('E'))
}

fn normalize_exponent_notation(s: &mut String) {
    if let Some(idx) = find_exponent_index(s) {
        let marker = s.as_bytes()[idx] as char;
        let suffix = &s[idx + 1..];
        let (sign, digits) = if let Some(first) = suffix.chars().next() {
            if first == '+' || first == '-' {
                (first, suffix.get(1..).unwrap_or("").to_string())
            } else {
                ('+', suffix.to_string())
            }
        } else {
            ('+', String::from("0"))
        };
        let mut normalized_digits = if digits.is_empty() {
            String::from("0")
        } else {
            digits
        };
        if normalized_digits.is_empty() {
            normalized_digits.push('0');
        }
        if normalized_digits.len() < 2 {
            normalized_digits = format!("{:0>2}", normalized_digits);
        }
        let mut rebuilt = String::with_capacity(idx + 1 + 1 + normalized_digits.len());
        rebuilt.push_str(&s[..idx]);
        rebuilt.push(marker);
        rebuilt.push(sign);
        rebuilt.push_str(&normalized_digits);
        *s = rebuilt;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;

    use crate::builtins::common::fs as fs_helpers;

    fn dlmwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::dlmwrite_builtin(filename, data, rest))
    }

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_dlmwrite_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DLMWRITE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M)"));
        assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M, delimiter, row, col)"));
        assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M, name, optionValue)"));
        assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M, args...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wasm_precision_parser_handles_common_specs() {
        fn fmt(value: f64, spec: &str) -> String {
            super::format_float(value, spec).expect("formatting failed")
        }

        assert_eq!(fmt(12.3456, "%.2f"), "12.35");
        assert_eq!(fmt(-12.3456, "%+08.1f"), "-00012.3");
        assert_eq!(fmt(0.001234, "%.4g"), "0.001234");
        assert_eq!(fmt(12345.0, "%.3g"), "1.23e+04");
        assert_eq!(fmt(1.5, "%#.0f"), "2.");
        assert_eq!(fmt(1.5, "%#.2e"), "1.50e+00");
        assert_eq!(fmt(1.5, "%#.2G"), "1.5");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn precision_parser_rejects_unsafe_native_format_shapes() {
        for spec in [
            "%n",
            "%s",
            "%*f",
            "%.2f %.2f",
            "%.2lf",
            "prefix %.2f",
            "%4097f",
            "%.4097f",
        ] {
            let err = super::format_float(1.0, spec).expect_err("format should be rejected");
            assert!(
                err.message().contains("precision format")
                    || err.message().contains("unsupported precision format"),
                "unexpected error for {spec}: {}",
                err.message()
            );
        }
    }

    fn platform_newline() -> &'static str {
        LineEnding::platform_default().as_str()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_writes_default_comma() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        dlmwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new()).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1,2,3{nl}4,5,6{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_accepts_positional_delimiter_and_offsets() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![
                Value::from(";"),
                Value::Int(IntValue::I32(1)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!(";;;;{nl};;1;2;3{nl};;4;5;6{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_supports_append_and_offsets() {
        let path = temp_path("csv");
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let tensor_b = Tensor::new(vec![10.0, 13.0, 11.0, 14.0, 12.0, 15.0], vec![3, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename.clone()),
            Value::Tensor(tensor_a),
            Vec::new(),
        )
        .unwrap();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor_b),
            vec![
                Value::from("-append"),
                Value::from("roffset"),
                Value::Int(IntValue::I32(1)),
                Value::from("coffset"),
                Value::Int(IntValue::I32(1)),
            ],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(
            contents,
            format!("1,2,3{nl},,{nl},10,14{nl},13,12{nl},11,15{nl}")
        );
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_precision_digits() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![12.34567, std::f64::consts::PI], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("precision"), Value::Int(IntValue::I32(3))],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("12.3,3.14{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_precision_format_string() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![0.25, 0.5, 0.75, 1.5], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![
                Value::from("precision"),
                Value::from("%.4f"),
                Value::from("delimiter"),
                Value::from("\t"),
            ],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("0.2500\t0.7500{nl}0.5000\t1.5000{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_newline_pc() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("newline"), Value::from("pc")],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = LineEnding::Pc.as_str();
        assert_eq!(contents, format!("1{nl}2{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_coffset_inserts_empty_fields() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("coffset"), Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!(",,1,2{nl},,3,4{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_handles_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).unwrap();
            let path = temp_path("csv");
            let filename = path.to_string_lossy().into_owned();
            dlmwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new()).unwrap();
            let contents = fs::read_to_string(&path).unwrap();
            let nl = platform_newline();
            assert_eq!(contents, format!("1,2,3{nl}"));
            let _ = fs::remove_file(path);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn dlmwrite_handles_wgpu_provider_gather() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).unwrap();
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new()).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1,2{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_interprets_control_sequence_delimiters() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("delimiter"), Value::from("\\t")],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1\t2{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_negative_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let message = error_message(
            dlmwrite_builtin(
                Value::from(filename),
                Value::Tensor(tensor),
                vec![Value::from("roffset"), Value::Int(IntValue::I32(-1))],
            )
            .unwrap_err(),
        );
        assert!(message.to_ascii_lowercase().contains("row offset"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_fractional_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let message = error_message(
            dlmwrite_builtin(
                Value::from(filename),
                Value::Tensor(tensor),
                vec![Value::from("coffset"), Value::Num(1.5)],
            )
            .unwrap_err(),
        );
        assert!(message.to_ascii_lowercase().contains("integer"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_empty_delimiter() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let message = error_message(
            dlmwrite_builtin(
                Value::from(filename),
                Value::Tensor(tensor),
                vec![Value::from("")],
            )
            .unwrap_err(),
        );
        assert!(message.to_ascii_lowercase().contains("delimiter"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_precision_zero_error() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let message = error_message(
            dlmwrite_builtin(
                Value::from(filename),
                Value::Tensor(tensor),
                vec![Value::from("precision"), Value::Int(IntValue::I32(0))],
            )
            .unwrap_err(),
        );
        assert!(message.to_ascii_lowercase().contains("precision"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_requires_name_value_pairs() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let message = error_message(
            dlmwrite_builtin(
                Value::from(filename),
                Value::Tensor(tensor),
                vec![Value::from("delimiter")],
            )
            .unwrap_err(),
        );
        assert!(message
            .to_ascii_lowercase()
            .contains("name-value arguments must appear in pairs"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_expands_home_directory() {
        let Some(mut home) = fs_helpers::home_directory() else {
            return;
        };
        let filename = format!(
            "runmat_dlmwrite_home_{}_{}.csv",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        );
        home.push(&filename);

        let tilde_path = format!("~/{}", filename);
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();

        dlmwrite_builtin(Value::from(tilde_path), Value::Tensor(tensor), Vec::new()).unwrap();

        let contents = fs::read_to_string(&home).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("42{nl}"));
        let _ = fs::remove_file(home);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_non_numeric_inputs() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let message = error_message(
            dlmwrite_builtin(
                Value::from(filename),
                Value::String("abc".into()),
                Vec::new(),
            )
            .unwrap_err(),
        );
        assert!(message.contains("dlmwrite"));
    }
}
