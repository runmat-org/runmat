//! MATLAB-compatible `writecell` builtin for heterogeneous cell-array export.

use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, Value,
};
use runmat_filesystem::{File, OpenOptions};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "writecell";
const MAX_EXCEL_ROW_INDEX: usize = 1_048_575;
const MAX_EXCEL_COLUMN_INDEX: usize = 16_383;

const WRITECELL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytesWritten",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written to the destination file.",
}];
const WRITECELL_INPUTS_CELL_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output file path.",
    },
];
const WRITECELL_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array to write.",
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
const WRITECELL_INPUTS_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array to write.",
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
const WRITECELL_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "bytesWritten = writecell(C, filename)",
        inputs: &WRITECELL_INPUTS_CELL_FILENAME,
        outputs: &WRITECELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = writecell(C, filename, name, optionValue)",
        inputs: &WRITECELL_INPUTS_NAME_VALUE,
        outputs: &WRITECELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bytesWritten = writecell(C, filename, nameValuePairs...)",
        inputs: &WRITECELL_INPUTS_NAME_VALUE_PAIRS,
        outputs: &WRITECELL_OUTPUT,
    },
];

const WRITECELL_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.ARG_CONFIG",
    identifier: None,
    when: "Filename argument is missing or name-value options are malformed.",
    message: "writecell: invalid argument configuration",
};
const WRITECELL_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.FILENAME",
    identifier: None,
    when: "Filename is not a valid scalar path string.",
    message: "writecell: invalid filename input",
};
const WRITECELL_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.OPTION",
    identifier: None,
    when: "A provided option value is invalid.",
    message: "writecell: invalid option value",
};
const WRITECELL_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.DATA",
    identifier: None,
    when: "Input data cannot be converted into supported cell export rows.",
    message: "writecell: invalid input data",
};
const WRITECELL_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.DATA_SHAPE",
    identifier: None,
    when: "Input cell array has unsupported dimensionality.",
    message: "writecell: input must be 2-D",
};
const WRITECELL_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITECELL.IO",
    identifier: None,
    when: "The destination file cannot be opened or written.",
    message: "writecell: file write failed",
};
const WRITECELL_ERRORS: [BuiltinErrorDescriptor; 6] = [
    WRITECELL_ERROR_ARG_CONFIG,
    WRITECELL_ERROR_FILENAME,
    WRITECELL_ERROR_OPTION,
    WRITECELL_ERROR_DATA,
    WRITECELL_ERROR_DATA_SHAPE,
    WRITECELL_ERROR_IO,
];

pub const WRITECELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITECELL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WRITECELL_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::writecell")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "writecell",
    op_kind: GpuOpKind::Custom("io-writecell"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Runs entirely on the host; gpuArray values inside cells are gathered before serialisation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::writecell")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "writecell",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

fn writecell_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    writecell_error_with(error, error.message)
}

fn writecell_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn writecell_error_with_source<E>(
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

#[runtime_builtin(
    name = "writecell",
    category = "io/tabular",
    summary = "Write heterogeneous cell arrays to delimited text or spreadsheet files.",
    keywords = "writecell,csv,xlsx,xls,cell array,delimited text,spreadsheet,append,quote strings",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::tabular::writecell::WRITECELL_DESCRIPTOR),
    builtin_path = "crate::builtins::io::tabular::writecell"
)]
async fn writecell_builtin(data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(writecell_error(&WRITECELL_ERROR_ARG_CONFIG));
    }

    let filename_value = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&filename_value)?;
    let options = parse_options(&rest[1..]).await?;

    let gathered = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let table = CellTable::from_value(gathered).await?;

    let bytes_written = match options.resolve_file_type(&path)? {
        OutputFileType::DelimitedText => write_delimited_cells(&path, &table, &options).await?,
        OutputFileType::Spreadsheet => write_spreadsheet_cells(&path, &table, &options).await?,
    };

    Ok(Value::Num(bytes_written as f64))
}

#[derive(Debug, Clone)]
struct WriteCellOptions {
    delimiter: Option<String>,
    write_mode: WriteMode,
    quote_strings: bool,
    line_ending: LineEnding,
    file_type: Option<OutputFileType>,
    sheet: SheetSelector,
    range: Option<RangeStart>,
}

impl Default for WriteCellOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            write_mode: WriteMode::Overwrite,
            quote_strings: true,
            line_ending: LineEnding::Auto,
            file_type: None,
            sheet: SheetSelector::Default,
            range: None,
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
    fn as_str(self) -> &'static str {
        match self {
            LineEnding::Auto | LineEnding::Unix => "\n",
            LineEnding::Windows => "\r\n",
            LineEnding::Mac => "\r",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFileType {
    DelimitedText,
    Spreadsheet,
}

#[derive(Debug, Clone)]
enum SheetSelector {
    Default,
    Name(String),
    Index(usize),
}

#[derive(Debug, Clone, Copy, Default)]
struct RangeStart {
    row: usize,
    col: usize,
}

impl WriteCellOptions {
    fn resolve_file_type(&self, path: &Path) -> BuiltinResult<OutputFileType> {
        if let Some(file_type) = self.file_type {
            if file_type == OutputFileType::Spreadsheet {
                ensure_supported_spreadsheet_extension(path)?;
            }
            return Ok(file_type);
        }
        match path_extension_lower(path).as_deref() {
            Some("xlsx") | Some("xlsm") => Ok(OutputFileType::Spreadsheet),
            Some(ext) if is_unsupported_spreadsheet_extension(ext) => Err(writecell_error_with(
                &WRITECELL_ERROR_OPTION,
                format!("writecell: unsupported spreadsheet file extension '.{ext}'"),
            )),
            _ => Ok(OutputFileType::DelimitedText),
        }
    }

    fn resolve_delimiter(&self, path: &Path) -> String {
        self.delimiter
            .clone()
            .unwrap_or_else(|| default_delimiter_for_path(path))
    }

    fn sheet_name(&self) -> String {
        match &self.sheet {
            SheetSelector::Default => "Sheet1".to_string(),
            SheetSelector::Name(name) => sanitize_sheet_name(name),
            SheetSelector::Index(index) => format!("Sheet{index}"),
        }
    }

    fn range_start(&self) -> RangeStart {
        self.range.unwrap_or_default()
    }
}

fn ensure_supported_spreadsheet_extension(path: &Path) -> BuiltinResult<()> {
    match path_extension_lower(path).as_deref() {
        Some("xlsx") | Some("xlsm") => Ok(()),
        Some(ext) => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            format!("writecell: unsupported spreadsheet file extension '.{ext}'"),
        )),
        None => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: spreadsheet output requires an .xlsx or .xlsm extension",
        )),
    }
}

fn is_unsupported_spreadsheet_extension(ext: &str) -> bool {
    matches!(ext, "xls" | "xlsb" | "ods")
}

async fn parse_options(args: &[Value]) -> BuiltinResult<WriteCellOptions> {
    if args.is_empty() {
        return Ok(WriteCellOptions::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err(writecell_error(&WRITECELL_ERROR_ARG_CONFIG));
    }

    let mut options = WriteCellOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let name_value = gather_if_needed_async(&args[index])
            .await
            .map_err(map_control_flow)?;
        let name = string_scalar_from_value(&name_value, "option name")
            .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
        let value = gather_if_needed_async(&args[index + 1])
            .await
            .map_err(map_control_flow)?;
        apply_option(&mut options, &name, &value)?;
        index += 2;
    }
    Ok(options)
}

fn apply_option(options: &mut WriteCellOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    if name.eq_ignore_ascii_case("Delimiter") {
        options.delimiter = Some(parse_delimiter(value)?);
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
    if name.eq_ignore_ascii_case("LineEnding") {
        options.line_ending = parse_line_ending(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("FileType") {
        options.file_type = Some(parse_file_type(value)?);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("Sheet") {
        options.sheet = parse_sheet(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("Range") {
        options.range = Some(parse_range_start(value)?);
        return Ok(());
    }
    Ok(())
}

fn parse_delimiter(value: &Value) -> BuiltinResult<String> {
    let text = string_scalar_from_value(value, "Delimiter")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    if text.is_empty() {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: Delimiter cannot be empty",
        ));
    }
    let trimmed = text.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "tab" => Ok("\t".to_string()),
        "space" | "whitespace" => Ok(" ".to_string()),
        "comma" => Ok(",".to_string()),
        "semicolon" => Ok(";".to_string()),
        "pipe" => Ok("|".to_string()),
        _ => Ok(trimmed.to_string()),
    }
}

fn parse_write_mode(value: &Value) -> BuiltinResult<WriteMode> {
    let text = string_scalar_from_value(value, "WriteMode")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: WriteMode must be 'overwrite' or 'append'",
        )),
    }
}

fn parse_bool_like(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => match i.to_i64() {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(writecell_error_with(
                &WRITECELL_ERROR_OPTION,
                format!("writecell: {context} must be logical (0 or 1)"),
            )),
        },
        Value::Num(n) if (*n - 0.0).abs() < f64::EPSILON => Ok(false),
        Value::Num(n) if (*n - 1.0).abs() < f64::EPSILON => Ok(true),
        _ => {
            let text = string_scalar_from_value(value, context)
                .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
            match text.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Ok(true),
                "off" | "false" | "no" | "0" => Ok(false),
                _ => Err(writecell_error_with(
                    &WRITECELL_ERROR_OPTION,
                    format!("writecell: {context} must be logical (true/on or false/off)"),
                )),
            }
        }
    }
}

fn parse_line_ending(value: &Value) -> BuiltinResult<LineEnding> {
    let text = string_scalar_from_value(value, "LineEnding")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(LineEnding::Auto),
        "unix" => Ok(LineEnding::Unix),
        "pc" | "windows" => Ok(LineEnding::Windows),
        "mac" => Ok(LineEnding::Mac),
        _ => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: LineEnding must be 'auto', 'unix', 'pc', or 'mac'",
        )),
    }
}

fn parse_file_type(value: &Value) -> BuiltinResult<OutputFileType> {
    let text = string_scalar_from_value(value, "FileType")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "text" | "delimitedtext" => Ok(OutputFileType::DelimitedText),
        "spreadsheet" => Ok(OutputFileType::Spreadsheet),
        _ => Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: FileType must be 'text', 'delimitedtext', or 'spreadsheet'",
        )),
    }
}

fn parse_sheet(value: &Value) -> BuiltinResult<SheetSelector> {
    match value {
        Value::Num(n) if n.is_finite() && *n >= 1.0 && n.fract() == 0.0 => {
            Ok(SheetSelector::Index(*n as usize))
        }
        Value::Int(i) if i.to_i64() >= 1 => Ok(SheetSelector::Index(i.to_i64() as usize)),
        _ => {
            let text = string_scalar_from_value(value, "Sheet")
                .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
            if text.trim().is_empty() {
                return Err(writecell_error_with(
                    &WRITECELL_ERROR_OPTION,
                    "writecell: Sheet name cannot be empty",
                ));
            }
            Ok(SheetSelector::Name(text))
        }
    }
}

fn parse_range_start(value: &Value) -> BuiltinResult<RangeStart> {
    let text = string_scalar_from_value(value, "Range")
        .map_err(|message| writecell_error_with(&WRITECELL_ERROR_OPTION, message))?;
    let start = text.split(':').next().unwrap_or("").trim();
    parse_a1_cell(start).ok_or_else(|| {
        writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: Range must start with an Excel A1 cell reference",
        )
    })
}

fn parse_a1_cell(value: &str) -> Option<RangeStart> {
    if value.is_empty() {
        return None;
    }
    let mut col = 0usize;
    let mut letters = 0usize;
    for ch in value.chars() {
        if ch.is_ascii_alphabetic() {
            if letters == 0 && col != 0 {
                return None;
            }
            col = col.checked_mul(26)?;
            col = col.checked_add((ch.to_ascii_uppercase() as u8 - b'A' + 1) as usize)?;
            letters += 1;
        } else {
            break;
        }
    }
    let row_text = &value[letters..];
    if letters == 0 || row_text.is_empty() || !row_text.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let row: usize = row_text.parse().ok()?;
    if row == 0 || col == 0 {
        return None;
    }
    Some(RangeStart {
        row: row - 1,
        col: col - 1,
    })
}

#[derive(Debug, Clone, PartialEq)]
enum CellValue {
    Empty,
    Number(f64),
    Boolean(bool),
    Text(String),
}

struct CellTable {
    rows: usize,
    cols: usize,
    data: Vec<CellValue>,
}

impl CellTable {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        let cell = match value {
            Value::Cell(cell) => cell,
            other => {
                return Err(writecell_error_with(
                    &WRITECELL_ERROR_DATA,
                    format!("writecell: input must be a cell array, got {other:?}"),
                ));
            }
        };
        ensure_cell_shape(&cell)?;

        let mut data = Vec::with_capacity(cell.data.len());
        for row in 0..cell.rows {
            for col in 0..cell.cols {
                let value = cell.get(row, col).map_err(|message| {
                    writecell_error_with(&WRITECELL_ERROR_DATA, format!("writecell: {message}"))
                })?;
                let gathered = gather_if_needed_async(&value)
                    .await
                    .map_err(map_control_flow)?;
                data.push(cell_value_from_value(gathered)?);
            }
        }
        Ok(Self {
            rows: cell.rows,
            cols: cell.cols,
            data,
        })
    }

    fn get(&self, row: usize, col: usize) -> &CellValue {
        &self.data[row * self.cols + col]
    }
}

fn ensure_cell_shape(cell: &CellArray) -> BuiltinResult<()> {
    if cell.shape.len() <= 2 || cell.shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(writecell_error_with(
        &WRITECELL_ERROR_DATA_SHAPE,
        "writecell: input cell array must be 2-D",
    ))
}

fn cell_value_from_value(value: Value) -> BuiltinResult<CellValue> {
    match value {
        Value::Num(n) => Ok(CellValue::Number(n)),
        Value::Int(i) => Ok(CellValue::Number(i.to_f64())),
        Value::Bool(b) => Ok(CellValue::Boolean(b)),
        Value::String(s) => Ok(CellValue::Text(s)),
        Value::CharArray(ca) if ca.rows == 1 => Ok(CellValue::Text(ca.data.iter().collect())),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(CellValue::Text(sa.data[0].clone())),
        Value::StringArray(sa) if sa.data.is_empty() => Ok(CellValue::Empty),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(CellValue::Number(tensor.data[0])),
        Value::Tensor(tensor) if tensor.data.is_empty() => Ok(CellValue::Empty),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(CellValue::Boolean(logical.data[0] != 0))
        }
        Value::LogicalArray(logical) if logical.data.is_empty() => Ok(CellValue::Empty),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_DATA,
            "writecell: complex values are not supported; split real and imaginary parts first",
        )),
        Value::Cell(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_DATA,
            "writecell: nested cell arrays are not supported",
        )),
        other => Err(writecell_error_with(
            &WRITECELL_ERROR_DATA,
            format!("writecell: unsupported cell value {other:?}"),
        )),
    }
}

async fn write_delimited_cells(
    path: &Path,
    table: &CellTable,
    options: &WriteCellOptions,
) -> BuiltinResult<usize> {
    let delimiter = options.resolve_delimiter(path);
    let line_ending = options.line_ending.as_str();
    let payload = build_delimited_payload(table, options, &delimiter, line_ending);
    if options.write_mode == WriteMode::Overwrite {
        safe_replace_file(path, &payload, "delimited text").await?;
        return Ok(payload.len());
    }

    let mut open_options = OpenOptions::new();
    open_options.create(true).write(true).append(true);

    let mut file = open_options.open_async(path).await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!(
                "writecell: unable to open \"{}\" for writing ({err})",
                path.display()
            ),
            err,
        )
    })?;

    let mut bytes_written = 0usize;
    if append_needs_line_ending(path).await? {
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                format!("writecell: failed to write append line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }
    file.write_all(&payload).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to write delimited text ({err})"),
            err,
        )
    })?;
    bytes_written += payload.len();
    file.flush_async().await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to flush output ({err})"),
            err,
        )
    })?;
    Ok(bytes_written)
}

fn build_delimited_payload(
    table: &CellTable,
    options: &WriteCellOptions,
    delimiter: &str,
    line_ending: &str,
) -> Vec<u8> {
    let mut payload = Vec::new();
    for row in 0..table.rows {
        for col in 0..table.cols {
            if col > 0 {
                payload.extend_from_slice(delimiter.as_bytes());
            }
            let rendered = format_cell_for_text(table.get(row, col), options, delimiter);
            if !rendered.is_empty() {
                payload.extend_from_slice(rendered.as_bytes());
            }
        }
        payload.extend_from_slice(line_ending.as_bytes());
    }
    payload
}

async fn append_needs_line_ending(path: &Path) -> BuiltinResult<bool> {
    let metadata = match runmat_filesystem::metadata_async(path).await {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => {
            return Err(writecell_error_with_source(
                &WRITECELL_ERROR_IO,
                format!(
                    "writecell: unable to inspect \"{}\" ({err})",
                    path.display()
                ),
                err,
            ));
        }
    };
    if metadata.is_empty() {
        return Ok(false);
    }
    let mut file = File::open_async(path).await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!(
                "writecell: unable to inspect \"{}\" ({err})",
                path.display()
            ),
            err,
        )
    })?;
    file.seek(SeekFrom::End(-1)).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: unable to inspect file ending ({err})"),
            err,
        )
    })?;
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: unable to read file ending ({err})"),
            err,
        )
    })?;
    Ok(!matches!(byte[0], b'\n' | b'\r'))
}

async fn write_spreadsheet_cells(
    path: &Path,
    table: &CellTable,
    options: &WriteCellOptions,
) -> BuiltinResult<usize> {
    if options.write_mode == WriteMode::Append {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: WriteMode 'append' is not supported for spreadsheet files",
        ));
    }
    let range_start = options.range_start();
    let end_row = range_start.row.checked_add(table.rows).ok_or_else(|| {
        writecell_error_with(&WRITECELL_ERROR_OPTION, "writecell: Range row overflow")
    })?;
    let end_col = range_start.col.checked_add(table.cols).ok_or_else(|| {
        writecell_error_with(&WRITECELL_ERROR_OPTION, "writecell: Range column overflow")
    })?;
    if end_row > MAX_EXCEL_ROW_INDEX + 1 || end_col > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_OPTION,
            "writecell: Range exceeds Excel worksheet limits",
        ));
    }

    let bytes = build_xlsx_workbook(table, &options.sheet_name(), range_start)?;
    safe_replace_file(path, &bytes, "spreadsheet").await?;
    Ok(bytes.len())
}

async fn safe_replace_file(path: &Path, bytes: &[u8], label: &str) -> BuiltinResult<()> {
    let temp_path = temporary_sibling_path(path);
    let mut open_options = OpenOptions::new();
    open_options.write(true).create_new(true);
    let mut file = open_options.open_async(&temp_path).await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!(
                "writecell: unable to create temporary {label} file \"{}\" ({err})",
                temp_path.display()
            ),
            err,
        )
    })?;
    file.write_all(bytes).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to write spreadsheet ({err})"),
            err,
        )
    })?;
    file.flush_async().await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to flush temporary {label} file ({err})"),
            err,
        )
    })?;
    file.sync_all_async().await.map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to sync temporary {label} file ({err})"),
            err,
        )
    })?;
    drop(file);
    if let Err(err) = runmat_filesystem::rename_async(&temp_path, path).await {
        let _ = runmat_filesystem::remove_file_async(&temp_path).await;
        return Err(writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!(
                "writecell: failed to replace \"{}\" with temporary {label} file ({err})",
                path.display()
            ),
            err,
        ));
    }
    Ok(())
}

fn temporary_sibling_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("writecell");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    parent.join(format!(".{name}.runmat-tmp-{}-{nanos}", std::process::id()))
}

fn build_xlsx_workbook(
    table: &CellTable,
    sheet_name: &str,
    start: RangeStart,
) -> BuiltinResult<Vec<u8>> {
    let cursor = Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    write_xlsx_part(
        &mut zip,
        "[Content_Types].xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "_rels/.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/workbook.xml",
        &format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#,
            xml_attr_escape(sheet_name)
        ),
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/_rels/workbook.xml.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/styles.xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>
</styleSheet>"#,
    )?;
    write_xlsx_part(
        &mut zip,
        "xl/worksheets/sheet1.xml",
        &build_sheet_xml(table, start),
    )?;
    let cursor = zip.finish().map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to finish spreadsheet package ({err})"),
            err,
        )
    })?;
    Ok(cursor.into_inner())
}

fn write_xlsx_part(
    zip: &mut zip::ZipWriter<Cursor<Vec<u8>>>,
    name: &str,
    contents: &str,
) -> BuiltinResult<()> {
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to start spreadsheet part {name} ({err})"),
            err,
        )
    })?;
    zip.write_all(contents.as_bytes()).map_err(|err| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            format!("writecell: failed to write spreadsheet part {name} ({err})"),
            err,
        )
    })?;
    Ok(())
}

fn build_sheet_xml(table: &CellTable, start: RangeStart) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
"#,
    );
    for row in 0..table.rows {
        let excel_row = start.row + row + 1;
        xml.push_str(&format!(r#"    <row r="{excel_row}">"#));
        xml.push('\n');
        for col in 0..table.cols {
            let cell = table.get(row, col);
            if *cell == CellValue::Empty {
                continue;
            }
            let reference = cell_reference(start.row + row, start.col + col);
            match cell {
                CellValue::Empty => {}
                CellValue::Number(value) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\"><v>{}</v></c>\n",
                        format_numeric(*value)
                    ));
                }
                CellValue::Boolean(value) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\" t=\"b\"><v>{}</v></c>\n",
                        if *value { 1 } else { 0 }
                    ));
                }
                CellValue::Text(text) => {
                    xml.push_str(&format!(
                        "      <c r=\"{reference}\" t=\"inlineStr\"><is><t>{}</t></is></c>\n",
                        xml_text_escape(text)
                    ));
                }
            }
        }
        xml.push_str("    </row>\n");
    }
    xml.push_str("  </sheetData>\n</worksheet>");
    xml
}

fn format_cell_for_text(cell: &CellValue, options: &WriteCellOptions, delimiter: &str) -> String {
    match cell {
        CellValue::Empty => String::new(),
        CellValue::Number(value) => format_numeric(*value),
        CellValue::Boolean(value) => {
            if *value {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        CellValue::Text(text) => format_string(text, options.quote_strings, delimiter),
    }
}

fn format_numeric(value: f64) -> String {
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
    trim_trailing_zeros(raw)
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
        if value == "-0" || value.is_empty() {
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

fn string_scalar_from_value(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(format!(
            "writecell: expected {context} as a string scalar or character vector"
        )),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::CharArray(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_FILENAME,
            "writecell: expected a 1-by-N character vector for the filename",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        Value::StringArray(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_FILENAME,
            "writecell: filename string array inputs must be scalar",
        )),
        other => Err(writecell_error_with(
            &WRITECELL_ERROR_FILENAME,
            format!(
                "writecell: expected filename as string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_FILENAME,
            "writecell: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| writecell_error_with(&WRITECELL_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn default_delimiter_for_path(path: &Path) -> String {
    match path_extension_lower(path).as_deref() {
        Some("csv") => ",".to_string(),
        Some("tsv") | Some("tab") => "\t".to_string(),
        Some("txt") | Some("dat") | Some("dlm") => " ".to_string(),
        _ => ",".to_string(),
    }
}

fn path_extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn sanitize_sheet_name(value: &str) -> String {
    let mut name: String = value
        .chars()
        .map(|ch| match ch {
            ':' | '\\' | '/' | '?' | '*' | '[' | ']' => '_',
            _ => ch,
        })
        .take(31)
        .collect();
    if name.trim().is_empty() {
        name = "Sheet1".to_string();
    }
    name
}

fn cell_reference(row: usize, col: usize) -> String {
    format!("{}{}", column_letters(col), row + 1)
}

fn column_letters(mut col: usize) -> String {
    let mut letters = Vec::new();
    col += 1;
    while col > 0 {
        let rem = (col - 1) % 26;
        letters.push((b'A' + rem as u8) as char);
        col = (col - 1) / 26;
    }
    letters.iter().rev().collect()
}

fn xml_text_escape(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            _ => ch.to_string(),
        })
        .collect()
}

fn xml_attr_escape(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            '\'' => "&apos;".to_string(),
            _ => ch.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use calamine::{open_workbook_auto, Data, Reader};
    use futures::executor::block_on;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use runmat_builtins::{CharArray, LogicalArray, Tensor};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_writecell_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    fn cell(values: Vec<Value>, rows: usize, cols: usize) -> Value {
        Value::Cell(CellArray::new(values, rows, cols).expect("cell array"))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = WRITECELL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"bytesWritten = writecell(C, filename)"));
        assert!(labels.contains(&"bytesWritten = writecell(C, filename, name, optionValue)"));
        assert!(labels.contains(&"bytesWritten = writecell(C, filename, nameValuePairs...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_writes_heterogeneous_csv() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let values = cell(
            vec![
                Value::Num(1.5),
                Value::from("alpha"),
                Value::Bool(true),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor")),
            ],
            2,
            2,
        );

        block_on(writecell_builtin(values, vec![Value::from(filename)])).expect("writecell");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1.5,\"alpha\"\n1,\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_honours_delimiter_quote_strings_and_append() {
        let path = temp_path("txt");
        let filename = path.to_string_lossy().into_owned();
        let first = cell(vec![Value::from("a,b"), Value::Num(2.0)], 1, 2);
        let second = cell(vec![Value::from("tail"), Value::Num(3.0)], 1, 2);

        block_on(writecell_builtin(
            first,
            vec![
                Value::from(filename.clone()),
                Value::from("Delimiter"),
                Value::from("|"),
                Value::from("QuoteStrings"),
                Value::Bool(false),
            ],
        ))
        .expect("initial write");
        block_on(writecell_builtin(
            second,
            vec![
                Value::from(filename.clone()),
                Value::from("Delimiter"),
                Value::from("|"),
                Value::from("QuoteStrings"),
                Value::Bool(false),
                Value::from("WriteMode"),
                Value::from("append"),
            ],
        ))
        .expect("append write");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "a,b|2\ntail|3\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_append_inserts_missing_row_boundary() {
        let path = temp_path("txt");
        fs::write(&path, "existing").expect("seed");
        let filename = path.to_string_lossy().into_owned();
        let values = cell(vec![Value::from("tail"), Value::Num(3.0)], 1, 2);

        block_on(writecell_builtin(
            values,
            vec![
                Value::from(filename),
                Value::from("Delimiter"),
                Value::from("|"),
                Value::from("QuoteStrings"),
                Value::Bool(false),
                Value::from("WriteMode"),
                Value::from("append"),
            ],
        ))
        .expect("append write");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "existing\ntail|3\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_accepts_scalar_char_tensor_and_logical_cells() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let values = cell(
            vec![
                Value::CharArray(CharArray::new_row("name")),
                Value::Tensor(Tensor::new(vec![42.0], vec![1, 1]).expect("scalar tensor")),
                Value::LogicalArray(LogicalArray::new(vec![0], vec![1, 1]).expect("logical")),
            ],
            1,
            3,
        );

        block_on(writecell_builtin(values, vec![Value::from(filename)])).expect("writecell");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "\"name\",42,0\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_rejects_nested_cells_and_nonscalar_arrays() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let nested = cell(vec![cell(vec![Value::Num(1.0)], 1, 1)], 1, 1);
        let err = block_on(writecell_builtin(
            nested,
            vec![Value::from(filename.clone())],
        ))
        .expect_err("nested cell error");
        assert!(err.message().contains("nested cell arrays"));

        let nonscalar = cell(
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor"),
            )],
            1,
            1,
        );
        let err = block_on(writecell_builtin(nonscalar, vec![Value::from(filename)]))
            .expect_err("nonscalar error");
        assert!(err.message().contains("unsupported cell value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_writes_xlsx_with_sheet_and_range() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        let values = cell(
            vec![Value::from("Voltage"), Value::Num(1.5), Value::Bool(true)],
            1,
            3,
        );

        block_on(writecell_builtin(
            values,
            vec![
                Value::from(filename),
                Value::from("Sheet"),
                Value::from("Measurements"),
                Value::from("Range"),
                Value::from("B2"),
            ],
        ))
        .expect("writecell xlsx");

        let mut workbook = open_workbook_auto(&path).expect("open workbook");
        assert_eq!(workbook.sheet_names()[0], "Measurements");
        let range = workbook
            .worksheet_range("Measurements")
            .expect("worksheet range");
        assert_eq!(
            range.get((0, 0)),
            Some(&Data::String("Voltage".to_string()))
        );
        assert_eq!(range.get((0, 1)), Some(&Data::Float(1.5)));
        assert_eq!(range.get((0, 2)), Some(&Data::Bool(true)));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writecell_rejects_unsupported_spreadsheet_extension() {
        let path = temp_path("xls");
        let filename = path.to_string_lossy().into_owned();
        let values = cell(vec![Value::from("A"), Value::Num(1.0)], 1, 2);
        let err = block_on(writecell_builtin(values, vec![Value::from(filename)]))
            .expect_err("unsupported extension");
        assert!(err
            .message()
            .contains("unsupported spreadsheet file extension"));
        let _ = fs::remove_file(path);
    }
}
