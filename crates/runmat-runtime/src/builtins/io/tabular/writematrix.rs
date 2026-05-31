//! MATLAB-compatible `writematrix` builtin for emitting delimited text files.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "writematrix";

const WRITEMATRIX_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytesWritten",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written to the destination file.",
}];
const WRITEMATRIX_INPUTS_DATA_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or string matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
];
const WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or string matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
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
        description: "Value for the preceding option name.",
    },
];
const WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or string matrix to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    },
];
const WRITEMATRIX_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "bytesWritten = writematrix(data, filename)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME,
        outputs: &WRITEMATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = writematrix(data, filename, name, optionValue)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE,
        outputs: &WRITEMATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = writematrix(data, filename, nameValuePairs...)",
        inputs: &WRITEMATRIX_INPUTS_DATA_FILENAME_NAME_VALUE_PAIRS,
        outputs: &WRITEMATRIX_OUTPUT,
    },
];
const WRITEMATRIX_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.ARG_CONFIG",
    identifier: None,
    when: "Filename argument is missing or name-value options are malformed.",
    message: "writematrix: invalid argument configuration",
};
const WRITEMATRIX_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.FILENAME",
    identifier: None,
    when: "Filename is not a valid scalar path string.",
    message: "writematrix: invalid filename input",
};
const WRITEMATRIX_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.OPTION",
    identifier: None,
    when: "A provided option value is invalid.",
    message: "writematrix: invalid option value",
};
const WRITEMATRIX_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.DATA",
    identifier: None,
    when: "Input data cannot be converted into a supported writematrix matrix form.",
    message: "writematrix: invalid input data",
};
const WRITEMATRIX_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.DATA_SHAPE",
    identifier: None,
    when: "Input data has unsupported dimensionality.",
    message: "writematrix: input must be 2-D",
};
const WRITEMATRIX_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEMATRIX.IO",
    identifier: None,
    when: "The destination file cannot be opened or written.",
    message: "writematrix: file write failed",
};
const WRITEMATRIX_ERRORS: [BuiltinErrorDescriptor; 6] = [
    WRITEMATRIX_ERROR_ARG_CONFIG,
    WRITEMATRIX_ERROR_FILENAME,
    WRITEMATRIX_ERROR_OPTION,
    WRITEMATRIX_ERROR_DATA,
    WRITEMATRIX_ERROR_DATA_SHAPE,
    WRITEMATRIX_ERROR_IO,
];
pub const WRITEMATRIX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITEMATRIX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WRITEMATRIX_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::writematrix")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "writematrix",
    op_kind: GpuOpKind::Custom("io-writematrix"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; gpuArray inputs are gathered before serialisation.",
};

fn writematrix_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    writematrix_error_with(error, error.message)
}

fn writematrix_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn writematrix_error_with_source<E>(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::writematrix")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "writematrix",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

#[runtime_builtin(
    name = "writematrix",
    category = "io/tabular",
    summary = "Write numeric or string matrices to delimited text files.",
    keywords = "writematrix,csv,delimited text,write,append,quote strings",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::tabular::writematrix::WRITEMATRIX_DESCRIPTOR),
    builtin_path = "crate::builtins::io::tabular::writematrix"
)]
async fn writematrix_builtin(data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(writematrix_error(&WRITEMATRIX_ERROR_ARG_CONFIG));
    }

    let filename_value = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&filename_value)?;

    let options = parse_options(&rest[1..]).await?;

    let gathered = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let matrix = MatrixData::from_value(gathered)?;

    let bytes_written = write_matrix(&path, &matrix, &options).await?;

    Ok(Value::Num(bytes_written as f64))
}

#[derive(Debug, Clone)]
struct WriteMatrixOptions {
    delimiter: Option<String>,
    write_mode: WriteMode,
    quote_strings: bool,
    line_ending: LineEnding,
    decimal_separator: char,
    file_type: FileType,
}

impl Default for WriteMatrixOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            write_mode: WriteMode::Overwrite,
            quote_strings: true,
            line_ending: LineEnding::Auto,
            decimal_separator: '.',
            file_type: FileType::DelimitedText,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    Overwrite,
    Append,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEnding {
    Auto,
    Unix,
    Windows,
    Mac,
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Auto | LineEnding::Unix => "\n",
            LineEnding::Windows => "\r\n",
            LineEnding::Mac => "\r",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileType {
    DelimitedText,
    Text,
}

async fn parse_options(args: &[Value]) -> BuiltinResult<WriteMatrixOptions> {
    if args.is_empty() {
        return Ok(WriteMatrixOptions::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err(writematrix_error(&WRITEMATRIX_ERROR_ARG_CONFIG));
    }

    let mut options = WriteMatrixOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let name_value = gather_if_needed_async(&args[index])
            .await
            .map_err(map_control_flow)?;
        let name = option_name_from_value(&name_value)?;
        let value = gather_if_needed_async(&args[index + 1])
            .await
            .map_err(map_control_flow)?;
        apply_option(&mut options, &name, &value)?;
        index += 2;
    }
    Ok(options)
}

fn apply_option(options: &mut WriteMatrixOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    if name.eq_ignore_ascii_case("Delimiter") {
        let delimiter = parse_delimiter(value)?;
        options.delimiter = Some(delimiter);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("WriteMode") {
        options.write_mode = parse_write_mode(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("QuoteStrings") {
        options.quote_strings = parse_bool_like(value, "QuoteStrings")?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("DecimalSeparator") {
        options.decimal_separator = parse_decimal_separator(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("LineEnding") {
        options.line_ending = parse_line_ending(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("FileType") {
        options.file_type = parse_file_type(value)?;
        return Ok(());
    }
    // Unsupported or future options are ignored for compatibility with MATLAB's permissive behaviour.
    Ok(())
}

fn option_name_from_value(value: &Value) -> BuiltinResult<String> {
    value_to_string_scalar(value, "option name")
}

fn parse_delimiter(value: &Value) -> BuiltinResult<String> {
    let text = value_to_string_scalar(value, "Delimiter")?;
    if text.is_empty() {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: Delimiter cannot be empty",
        ));
    }
    let trimmed = text.trim();
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "tab" => Ok("\t".to_string()),
        "space" | "whitespace" => Ok(" ".to_string()),
        "comma" => Ok(",".to_string()),
        "semicolon" => Ok(";".to_string()),
        "pipe" => Ok("|".to_string()),
        _ => Ok(trimmed.to_string()),
    }
}

fn parse_write_mode(value: &Value) -> BuiltinResult<WriteMode> {
    let text = value_to_string_scalar(value, "WriteMode")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: WriteMode must be 'overwrite' or 'append'",
        )),
    }
}

fn parse_bool_like(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => {
            let raw = i.to_i64();
            match raw {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_OPTION,
                    format!("writematrix: {context} must be logical (0 or 1)"),
                )),
            }
        }
        Value::Num(n) => {
            if (*n - 0.0).abs() < f64::EPSILON {
                Ok(false)
            } else if (*n - 1.0).abs() < f64::EPSILON {
                Ok(true)
            } else {
                Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_OPTION,
                    format!("writematrix: {context} must be logical (0 or 1)"),
                ))
            }
        }
        _ => {
            let text = value_to_string_scalar(value, context)?;
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "on" | "true" | "yes" | "1" => Ok(true),
                "off" | "false" | "no" | "0" => Ok(false),
                _ => Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_OPTION,
                    format!("writematrix: {context} must be logical (true/on or false/off)"),
                )),
            }
        }
    }
}

fn parse_decimal_separator(value: &Value) -> BuiltinResult<char> {
    let text = value_to_string_scalar(value, "DecimalSeparator")?;
    let mut chars = text.chars();
    let ch = chars.next().ok_or_else(|| {
        writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: DecimalSeparator must be a single character",
        )
    })?;
    if chars.next().is_some() {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: DecimalSeparator must be a single character",
        ));
    }
    if ch == '\n' || ch == '\r' {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: DecimalSeparator cannot be a newline character",
        ));
    }
    Ok(ch)
}

fn parse_line_ending(value: &Value) -> BuiltinResult<LineEnding> {
    let text = value_to_string_scalar(value, "LineEnding")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "auto" => Ok(LineEnding::Auto),
        "unix" => Ok(LineEnding::Unix),
        "pc" | "windows" => Ok(LineEnding::Windows),
        "mac" => Ok(LineEnding::Mac),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: LineEnding must be 'auto', 'unix', 'pc', or 'mac'",
        )),
    }
}

fn parse_file_type(value: &Value) -> BuiltinResult<FileType> {
    let text = value_to_string_scalar(value, "FileType")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "delimitedtext" | "text" => {
            if lowered == "text" {
                Ok(FileType::Text)
            } else {
                Ok(FileType::DelimitedText)
            }
        }
        "spreadsheet" => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: FileType 'spreadsheet' is not supported; export delimited text instead",
        )),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            "writematrix: unsupported FileType",
        )),
    }
}

fn value_to_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_OPTION,
            format!("writematrix: expected {context} as a string scalar or character vector"),
        )),
    }
}

enum MatrixData {
    Numeric(Tensor),
    Text {
        rows: usize,
        cols: usize,
        data: Vec<String>,
    },
}

impl MatrixData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::StringArray(sa) => {
                ensure_matrix_shape(&sa.shape, "string")?;
                let (rows, cols) = if sa.shape.is_empty() {
                    (1, 1)
                } else {
                    (sa.rows, sa.cols)
                };
                Ok(MatrixData::Text {
                    rows,
                    cols,
                    data: sa.data.clone(),
                })
            }
            Value::String(s) => Ok(MatrixData::Text {
                rows: 1,
                cols: 1,
                data: vec![s],
            }),
            Value::CharArray(_) => Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_DATA,
                "writematrix: character arrays are not supported; convert to string arrays or use writecell",
            )),
            Value::Cell(_) => Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_DATA,
                "writematrix: cell arrays are not supported; use writecell for heterogeneous content",
            )),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_DATA,
                "writematrix: complex values are not supported; write real and imaginary parts separately",
            )),
            other => {
                let tensor = tensor::value_into_tensor_for("writematrix", other)
                    .map_err(|msg| {
                        writematrix_error_with(
                            &WRITEMATRIX_ERROR_DATA,
                            format!("writematrix: {msg}"),
                        )
                    })?;
                ensure_matrix_shape(&tensor.shape, "numeric")?;
                Ok(MatrixData::Numeric(tensor))
            }
        }
    }

    fn rows(&self) -> usize {
        match self {
            MatrixData::Numeric(t) => t.rows(),
            MatrixData::Text { rows, .. } => *rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            MatrixData::Numeric(t) => t.cols(),
            MatrixData::Text { cols, .. } => *cols,
        }
    }

    fn format_cell(
        &self,
        row: usize,
        col: usize,
        options: &WriteMatrixOptions,
        delimiter: &str,
    ) -> String {
        match self {
            MatrixData::Numeric(tensor) => {
                let rows = tensor.rows();
                let idx = row + col * rows;
                let value = tensor.data[idx];
                format_numeric(value, options.decimal_separator)
            }
            MatrixData::Text { rows, data, .. } => {
                if *rows == 0 {
                    return String::new();
                }
                let idx = row + col * rows;
                format_string(&data[idx], options.quote_strings, delimiter)
            }
        }
    }
}

fn ensure_matrix_shape(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_DATA_SHAPE,
            format!(
                "writematrix: {context} input must be 2-D; reshape or use writecell for higher dimensions"
            ),
        ))
    }
}

async fn write_matrix(
    path: &Path,
    matrix: &MatrixData,
    options: &WriteMatrixOptions,
) -> BuiltinResult<usize> {
    let delimiter = options.resolve_delimiter(path);
    let line_ending = options.line_ending.as_str();

    let mut open_options = OpenOptions::new();
    open_options.create(true);
    match options.write_mode {
        WriteMode::Overwrite => {
            open_options.write(true).truncate(true);
        }
        WriteMode::Append => {
            open_options.write(true).append(true);
        }
    }

    let mut file = open_options.open_async(path).await.map_err(|err| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            format!(
                "writematrix: unable to open \"{}\" for writing ({err})",
                path.display()
            ),
            err,
        )
    })?;

    let mut bytes_written = 0usize;
    let rows = matrix.rows();
    let cols = matrix.cols();

    if rows == 0 {
        // Nothing else to do; file was truncated or created above.
        file.flush_async().await.map_err(|err| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                format!("writematrix: failed to flush output ({err})"),
                err,
            )
        })?;
        return Ok(0);
    }

    for row in 0..rows {
        for col in 0..cols {
            if col > 0 {
                file.write_all(delimiter.as_bytes()).map_err(|err| {
                    writematrix_error_with_source(
                        &WRITEMATRIX_ERROR_IO,
                        format!("writematrix: failed to write delimiter ({err})"),
                        err,
                    )
                })?;
                bytes_written += delimiter.len();
            }
            let cell = matrix.format_cell(row, col, options, &delimiter);
            if !cell.is_empty() {
                file.write_all(cell.as_bytes()).map_err(|err| {
                    writematrix_error_with_source(
                        &WRITEMATRIX_ERROR_IO,
                        format!("writematrix: failed to write value ({err})"),
                        err,
                    )
                })?;
                bytes_written += cell.len();
            }
        }
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            writematrix_error_with_source(
                &WRITEMATRIX_ERROR_IO,
                format!("writematrix: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }

    file.flush_async().await.map_err(|err| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            format!("writematrix: failed to flush output ({err})"),
            err,
        )
    })?;

    Ok(bytes_written)
}

impl WriteMatrixOptions {
    fn resolve_delimiter(&self, path: &Path) -> String {
        if let Some(custom) = &self.delimiter {
            return custom.clone();
        }
        match self.file_type {
            FileType::Text => " ".to_string(),
            FileType::DelimitedText => default_delimiter_for_path(path),
        }
    }
}

fn default_delimiter_for_path(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("csv") => ",".to_string(),
        Some("tsv") | Some("tab") => "\t".to_string(),
        Some("txt") | Some("dat") | Some("dlm") => " ".to_string(),
        _ => ",".to_string(),
    }
}

fn format_numeric(value: f64, decimal_separator: char) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }

    let abs = value.abs();
    let scientific = abs != 0.0 && !(1e-4..1e15).contains(&abs);
    let raw = if scientific {
        format!("{:.15e}", value)
    } else {
        format!("{:.15}", value)
    };

    let mut formatted = trim_trailing_zeros(raw);
    if formatted == "-0" {
        formatted = "0".to_string();
    }
    if decimal_separator != '.' && formatted.contains('.') {
        formatted = formatted.replace('.', &decimal_separator.to_string());
    }
    formatted
}

fn trim_trailing_zeros(mut value: String) -> String {
    if let Some(exp_pos) = value.find(['e', 'E']) {
        let exponent = value.split_off(exp_pos);
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
        value.push_str(&exponent);
        value
    } else {
        if value.contains('.') {
            while value.ends_with('0') {
                value.pop();
            }
            if value.ends_with('.') {
                value.pop();
            }
        }
        if value.is_empty() {
            "0".to_string()
        } else {
            value
        }
    }
}

fn format_string(value: &str, quote: bool, _delimiter: &str) -> String {
    if !quote {
        return value.to_string();
    }
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        if ch == '"' {
            escaped.push('"');
            escaped.push('"');
        } else {
            escaped.push(ch);
        }
    }
    escaped.push('"');
    escaped
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::CharArray(_) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            "writematrix: expected a 1-by-N character vector for the filename",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        Value::StringArray(_) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            "writematrix: filename string array inputs must be scalar",
        )),
        other => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            format!(
                "writematrix: expected filename as string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_FILENAME,
            "writematrix: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| writematrix_error_with(&WRITEMATRIX_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{StringArray, Tensor};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_writematrix_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = WRITEMATRIX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"bytesWritten = writematrix(data, filename)"));
        assert!(labels.contains(&"bytesWritten = writematrix(data, filename, name, optionValue)"));
        assert!(labels.contains(&"bytesWritten = writematrix(data, filename, nameValuePairs...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_writes_space_delimited_txt() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        block_on(writematrix_builtin(
            Value::Tensor(tensor),
            vec![Value::from(filename)],
        ))
        .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1 2 3\n4 5 6\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_defaults_to_comma_for_csv() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        block_on(writematrix_builtin(
            Value::Tensor(tensor),
            vec![Value::from(filename)],
        ))
        .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1,2,3\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_honours_write_mode_append() {
        let path = temp_path("txt");
        let first = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let second = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        block_on(writematrix_builtin(
            Value::Tensor(first),
            vec![Value::from(filename.clone())],
        ))
        .expect("initial write");

        block_on(writematrix_builtin(
            Value::Tensor(second),
            vec![
                Value::from(filename.clone()),
                Value::from("WriteMode"),
                Value::from("append"),
            ],
        ))
        .expect("append write");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1 2 3\n4 5 6\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_quotes_strings_by_default() {
        let path = temp_path("csv");
        let strings = StringArray::new(vec!["Alice".to_string(), "Bob".to_string()], vec![1, 2])
            .expect("string array");
        let filename = path.to_string_lossy().into_owned();

        block_on(writematrix_builtin(
            Value::StringArray(strings),
            vec![Value::from(filename)],
        ))
        .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "\"Alice\",\"Bob\"\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_accepts_gpu_tensor_inputs() {
        test_support::with_test_provider(|provider| {
            let path = temp_path("csv");
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let filename = path.to_string_lossy().into_owned();

            block_on(writematrix_builtin(
                Value::GpuTensor(handle),
                vec![Value::from(filename)],
            ))
            .expect("writematrix");

            let contents = fs::read_to_string(&path).expect("read contents");
            assert_eq!(contents, "1,2\n");
            let _ = fs::remove_file(path);
        });
    }
}
