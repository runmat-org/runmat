//! MATLAB-compatible `xlswrite` builtin for legacy spreadsheet export.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto_from_rs, Data as SpreadsheetData, Reader as SpreadsheetReader};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, LogicalArray, SparseTensor, StringArray, StructValue, Tensor, Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::writecell::{self, CellTable, CellValue, RangeStart};

const BUILTIN_NAME: &str = "xlswrite";
const MAX_EXCEL_ROW_INDEX: usize = 1_048_575;
const MAX_EXCEL_COLUMN_INDEX: usize = 16_383;
const MAX_XLSWRITE_CELLS: usize = 5_000_000;
const MAX_XLSWRITE_SHEETS: usize = 1024;
const MAX_XLSWRITE_WORKBOOK_BYTES: u64 = 512 * 1024 * 1024;
const MAX_XLSWRITE_ZIP_ENTRY_UNCOMPRESSED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_XLSWRITE_ZIP_TOTAL_UNCOMPRESSED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_XLSWRITE_ZIP_ENTRIES: usize = 65_536;
const MAX_XLSWRITE_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
const XLSWRITE_MARKER_PART: &str = "docProps/app.xml";
const XLSWRITE_MARKER_TEXT: &str = "RunMat xlswrite";

const XLSWRITE_OUTPUT_STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the spreadsheet was written successfully.",
};
const XLSWRITE_OUTPUT_MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Diagnostic struct with message and identifier fields.",
};
const XLSWRITE_OUTPUTS_STATUS: [BuiltinParamDescriptor; 1] = [XLSWRITE_OUTPUT_STATUS];
const XLSWRITE_OUTPUTS_STATUS_MESSAGE: [BuiltinParamDescriptor; 2] =
    [XLSWRITE_OUTPUT_STATUS, XLSWRITE_OUTPUT_MESSAGE];

const XLSWRITE_INPUTS_FILENAME_A: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
];
const XLSWRITE_INPUTS_FILENAME_A_SELECTOR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
    BuiltinParamDescriptor {
        name: "sheetOrRange",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name/index or Excel A1 range start.",
    },
];
const XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based worksheet index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range start or range.",
    },
];
const XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE_BASIC: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, string, character, or cell data to write.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based worksheet index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range start or range.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"basic\""),
        description: "Legacy '-basic' compatibility flag.",
    },
];

const XLSWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A, sheetOrRange)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SELECTOR,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A, sheet, range)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = xlswrite(filename, A, sheet, range, '-basic')",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE_BASIC,
        outputs: &XLSWRITE_OUTPUTS_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A, sheetOrRange)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SELECTOR,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A, sheet, range)",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status,message] = xlswrite(filename, A, sheet, range, '-basic')",
        inputs: &XLSWRITE_INPUTS_FILENAME_A_SHEET_RANGE_BASIC,
        outputs: &XLSWRITE_OUTPUTS_STATUS_MESSAGE,
    },
];

const XLSWRITE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlswrite:InvalidArgument"),
    when: "Argument list does not match supported xlswrite call forms.",
    message: "xlswrite: invalid argument",
};
const XLSWRITE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.FILENAME",
    identifier: Some("RunMat:xlswrite:Filename"),
    when: "Filename is invalid or cannot be normalized.",
    message: "xlswrite: invalid filename",
};
const XLSWRITE_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.RANGE",
    identifier: Some("RunMat:xlswrite:Range"),
    when: "Range specification is malformed or exceeds worksheet limits.",
    message: "xlswrite: invalid range",
};
const XLSWRITE_ERROR_SHEET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.SHEET",
    identifier: Some("RunMat:xlswrite:Sheet"),
    when: "Worksheet selector is invalid.",
    message: "xlswrite: invalid sheet",
};
const XLSWRITE_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.DATA",
    identifier: Some("RunMat:xlswrite:Data"),
    when: "Input data cannot be represented as supported spreadsheet cells.",
    message: "xlswrite: invalid data",
};
const XLSWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.IO",
    identifier: Some("RunMat:xlswrite:Io"),
    when: "Spreadsheet cannot be written.",
    message: "xlswrite: unable to write spreadsheet",
};
const XLSWRITE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSWRITE.OUTPUT_COUNT",
    identifier: Some("RunMat:xlswrite:OutputCount"),
    when: "Caller requests more outputs than xlswrite supports.",
    message: "xlswrite: unsupported output count",
};
const XLSWRITE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    XLSWRITE_ERROR_INVALID_ARGUMENT,
    XLSWRITE_ERROR_FILENAME,
    XLSWRITE_ERROR_RANGE,
    XLSWRITE_ERROR_SHEET,
    XLSWRITE_ERROR_DATA,
    XLSWRITE_ERROR_IO,
    XLSWRITE_ERROR_OUTPUT_COUNT,
];

pub const XLSWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLSWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLSWRITE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::xlswrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "xlswrite",
    op_kind: GpuOpKind::Custom("io-xlswrite"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Writes spreadsheets on the host; gpuArray inputs are gathered before serialization.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::xlswrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "xlswrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; spreadsheet I/O executes as a standalone host operation.",
};

#[runtime_builtin(
    name = "xlswrite",
    category = "io/tabular",
    summary = "Write data to legacy Excel-compatible spreadsheet files.",
    keywords = "xlswrite,xls,xlsx,spreadsheet,excel,legacy export",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::xlswrite_type),
    descriptor(crate::builtins::io::tabular::xlswrite::XLSWRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::tabular::xlswrite"
)]
async fn xlswrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_OUTPUT_COUNT,
            "xlswrite: supports at most two output arguments",
        ));
    }

    match write_spreadsheet(filename, data, rest).await {
        Ok(_) => Ok(status_outputs(true, None)),
        Err(err) if captures_failure_as_outputs() => Ok(status_outputs(false, Some(&err))),
        Err(err) => Err(err),
    }
}

async fn write_spreadsheet(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<usize> {
    let path_value = gather_if_needed_async(&filename)
        .await
        .map_err(map_control_flow)?;
    let data_value = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let args = gather_arguments(&rest).await?;
    let request = XlsWriteRequest::parse(&args)?;
    let path = resolve_path(&path_value)?;
    let table = XlsTable::from_value(data_value).await?;
    let lock = writecell::write_lock_for_path(&path).await;
    let _guard = lock.lock().await;
    let mut workbook = load_workbook_or_new(&path).await?;
    workbook.overlay(&request, table)?;
    let bytes = workbook.to_xlsx_bytes()?;
    writecell::safe_replace_file(&path, &bytes, "spreadsheet")
        .await
        .map_err(map_writecell_error)?;
    Ok(bytes.len())
}

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

fn captures_failure_as_outputs() -> bool {
    matches!(crate::output_count::current_output_count(), Some(1 | 2))
}

fn status_outputs(success: bool, err: Option<&RuntimeError>) -> Value {
    match crate::output_count::current_output_count() {
        Some(0) => Value::OutputList(Vec::new()),
        Some(1) => Value::OutputList(vec![Value::Bool(success)]),
        Some(2) => Value::OutputList(vec![Value::Bool(success), message_value(err)]),
        None => Value::Bool(success),
        Some(_) => Value::Bool(success),
    }
}

fn message_value(err: Option<&RuntimeError>) -> Value {
    let mut message = StructValue::new();
    message.insert(
        "message",
        Value::String(err.map(|e| e.message().to_string()).unwrap_or_default()),
    );
    message.insert(
        "identifier",
        Value::String(
            err.and_then(RuntimeError::identifier)
                .unwrap_or("")
                .to_string(),
        ),
    );
    Value::Struct(message)
}

#[derive(Debug, Clone)]
struct XlsWriteRequest {
    sheet: SheetSelector,
    range: RangeStart,
}

impl Default for XlsWriteRequest {
    fn default() -> Self {
        Self {
            sheet: SheetSelector::Default,
            range: RangeStart::default(),
        }
    }
}

impl XlsWriteRequest {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut request = Self::default();
        match args {
            [] => {}
            [single] => parse_single_selector(single, &mut request)?,
            [sheet, range] => {
                request.sheet = parse_sheet_selector(sheet)?;
                request.range = parse_range_start(range)?;
            }
            [sheet, range, mode] => {
                request.sheet = parse_sheet_selector(sheet)?;
                request.range = parse_range_start(range)?;
                parse_basic_mode(mode)?;
            }
            _ => {
                return Err(xlswrite_error_with(
                    &XLSWRITE_ERROR_INVALID_ARGUMENT,
                    "xlswrite: expected filename, data, optional sheet, optional range, and optional '-basic'",
                ))
            }
        }
        Ok(request)
    }
}

#[derive(Debug, Clone)]
enum SheetSelector {
    Default,
    Name(String),
    Index(usize),
}

fn parse_single_selector(value: &Value, request: &mut XlsWriteRequest) -> BuiltinResult<()> {
    if let Ok(text) = value_to_string_scalar(value) {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("basic") || trimmed.eq_ignore_ascii_case("-basic") {
            return Ok(());
        }
        if text_looks_like_range(trimmed) {
            request.range = parse_range_text(trimmed)?;
        } else {
            request.sheet = SheetSelector::Name(nonempty_sheet_name(trimmed)?);
        }
        return Ok(());
    }
    request.sheet = parse_sheet_selector(value)?;
    Ok(())
}

fn parse_sheet_selector(value: &Value) -> BuiltinResult<SheetSelector> {
    match value {
        Value::Num(n) => numeric_sheet_index(*n),
        Value::Int(i) => {
            let index = i
                .try_to_usize()
                .and_then(|index| index.checked_sub(1))
                .ok_or_else(|| {
                    xlswrite_error_with(
                        &XLSWRITE_ERROR_SHEET,
                        "xlswrite: sheet index must be one-based",
                    )
                })?;
            if index >= MAX_XLSWRITE_SHEETS {
                return Err(xlswrite_error_with(
                    &XLSWRITE_ERROR_SHEET,
                    format!(
                        "xlswrite: sheet index exceeds supported limit of {MAX_XLSWRITE_SHEETS}"
                    ),
                ));
            }
            Ok(SheetSelector::Index(index))
        }
        Value::Tensor(t) if t.data.len() == 1 => numeric_sheet_index(t.data[0]),
        _ => {
            let text = value_to_string_scalar(value).map_err(|_| {
                xlswrite_error_with(
                    &XLSWRITE_ERROR_SHEET,
                    "xlswrite: sheet must be a name or one-based numeric index",
                )
            })?;
            Ok(SheetSelector::Name(nonempty_sheet_name(text.trim())?))
        }
    }
}

fn numeric_sheet_index(value: f64) -> BuiltinResult<SheetSelector> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet index must be a positive integer",
        ));
    }
    let index = value.round() as u128;
    let zero_based = index.checked_sub(1).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet index must be one-based",
        )
    })?;
    let index = usize::try_from(zero_based).map_err(|_| {
        xlswrite_error_with(&XLSWRITE_ERROR_SHEET, "xlswrite: sheet index is too large")
    })?;
    if index >= MAX_XLSWRITE_SHEETS {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            format!("xlswrite: sheet index exceeds supported limit of {MAX_XLSWRITE_SHEETS}"),
        ));
    }
    Ok(SheetSelector::Index(index))
}

fn nonempty_sheet_name(text: &str) -> BuiltinResult<String> {
    validate_sheet_name(text)
}

fn validate_sheet_name(text: &str) -> BuiltinResult<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet name must not be empty",
        ));
    }
    if trimmed.chars().count() > 31 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet name must be 31 characters or fewer",
        ));
    }
    if trimmed
        .chars()
        .any(|ch| matches!(ch, ':' | '\\' | '/' | '?' | '*' | '[' | ']'))
    {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_SHEET,
            "xlswrite: sheet name contains characters Excel does not allow",
        ));
    }
    Ok(trimmed.to_string())
}

fn parse_range_start(value: &Value) -> BuiltinResult<RangeStart> {
    if let Ok(text) = value_to_string_scalar(value) {
        return parse_range_text(&text);
    }
    match value {
        Value::Tensor(t) => parse_numeric_range(&t.data),
        Value::Num(n) => parse_numeric_range(&[*n]),
        _ => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range must be an A1 string or numeric [row col] vector",
        )),
    }
}

fn parse_numeric_range(values: &[f64]) -> BuiltinResult<RangeStart> {
    if values.len() < 2 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: numeric range must contain at least row and column",
        ));
    }
    let row = one_based_index(values[0], "row")?;
    let col = one_based_index(values[1], "column")?;
    Ok(RangeStart { row, col })
}

fn one_based_index(value: f64, label: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: range {label} must be a positive integer"),
        ));
    }
    let one_based = value.round() as u128;
    usize::try_from(one_based - 1).map_err(|_| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: range {label} is too large"),
        )
    })
}

fn parse_range_text(value: &str) -> BuiltinResult<RangeStart> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range string must not be empty",
        ));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            format!("xlswrite: invalid range specification '{value}'"),
        ));
    }
    let start = parse_a1_cell(parts[0].trim()).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range must start with an Excel A1 cell reference",
        )
    })?;
    if start.row > MAX_EXCEL_ROW_INDEX || start.col > MAX_EXCEL_COLUMN_INDEX {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range start exceeds Excel worksheet limits",
        ));
    }
    if parts.len() == 2 {
        let end = parse_a1_cell(parts[1].trim()).ok_or_else(|| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                "xlswrite: range end must be an Excel A1 cell reference",
            )
        })?;
        if end.row < start.row || end.col < start.col {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                "xlswrite: range end must be greater than or equal to range start",
            ));
        }
        if end.row > MAX_EXCEL_ROW_INDEX || end.col > MAX_EXCEL_COLUMN_INDEX {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_RANGE,
                "xlswrite: range end exceeds Excel worksheet limits",
            ));
        }
    }
    Ok(start)
}

fn parse_basic_mode(value: &Value) -> BuiltinResult<()> {
    let text = value_to_string_scalar(value)?;
    let normalized = text.trim().to_ascii_lowercase();
    if normalized == "basic" || normalized == "-basic" {
        return Ok(());
    }
    Err(xlswrite_error_with(
        &XLSWRITE_ERROR_INVALID_ARGUMENT,
        "xlswrite: only the legacy '-basic' mode flag is supported",
    ))
}

fn text_looks_like_range(value: &str) -> bool {
    parse_a1_cell(value.split(':').next().unwrap_or("")).is_some()
}

fn parse_a1_cell(value: &str) -> Option<RangeStart> {
    if value.is_empty() {
        return None;
    }
    let mut col = 0usize;
    let mut letters = 0usize;
    let stripped = value.chars().filter(|&ch| ch != '$').collect::<String>();
    for ch in stripped.chars() {
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
    let row_text = &stripped[letters..];
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

#[derive(Debug, Clone)]
struct XlsWorkbook {
    sheets: Vec<XlsSheet>,
}

#[derive(Debug, Clone)]
struct XlsSheet {
    name: String,
    table: XlsTable,
}

impl XlsWorkbook {
    fn new() -> Self {
        Self { sheets: Vec::new() }
    }

    fn overlay(&mut self, request: &XlsWriteRequest, table: XlsTable) -> BuiltinResult<()> {
        validate_write_bounds(&table, request.range)?;
        let sheet_index = self.resolve_sheet_index(&request.sheet);
        self.sheets[sheet_index]
            .table
            .overlay(request.range, &table)?;
        Ok(())
    }

    fn resolve_sheet_index(&mut self, selector: &SheetSelector) -> usize {
        match selector {
            SheetSelector::Default => {
                if self.sheets.is_empty() {
                    self.sheets.push(XlsSheet::empty("Sheet1"));
                }
                0
            }
            SheetSelector::Name(name) => {
                if let Some(index) = self
                    .sheets
                    .iter()
                    .position(|sheet| sheet.name.eq_ignore_ascii_case(name))
                {
                    index
                } else {
                    self.sheets.push(XlsSheet::empty(name));
                    self.sheets.len() - 1
                }
            }
            SheetSelector::Index(index) => {
                while self.sheets.len() <= *index {
                    let next = self.sheets.len() + 1;
                    let name = self.unique_sheet_name(&format!("Sheet{next}"));
                    self.sheets.push(XlsSheet::empty(&name));
                }
                *index
            }
        }
    }

    fn unique_sheet_name(&self, base: &str) -> String {
        if !self
            .sheets
            .iter()
            .any(|sheet| sheet.name.eq_ignore_ascii_case(base))
        {
            return base.to_string();
        }
        for suffix in 1..=MAX_XLSWRITE_SHEETS {
            let candidate = make_suffixed_sheet_name(base, suffix);
            if !self
                .sheets
                .iter()
                .any(|sheet| sheet.name.eq_ignore_ascii_case(&candidate))
            {
                return candidate;
            }
        }
        "Sheet".to_string()
    }

    fn to_xlsx_bytes(&self) -> BuiltinResult<Vec<u8>> {
        let sheets = if self.sheets.is_empty() {
            vec![XlsSheet::empty("Sheet1")]
        } else {
            self.sheets.clone()
        };
        let cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        writecell::write_xlsx_part(
            &mut zip,
            "[Content_Types].xml",
            &content_types_xml(sheets.len()),
        )
        .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, "_rels/.rels", ROOT_RELS_XML)
            .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, "xl/workbook.xml", &workbook_xml(&sheets))
            .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(
            &mut zip,
            "xl/_rels/workbook.xml.rels",
            &workbook_relationships_xml(sheets.len()),
        )
        .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, "xl/styles.xml", STYLES_XML)
            .map_err(map_writecell_error)?;
        writecell::write_xlsx_part(&mut zip, XLSWRITE_MARKER_PART, &marker_app_xml())
            .map_err(map_writecell_error)?;

        for (index, sheet) in sheets.iter().enumerate() {
            let cell_table = sheet.table.clone().into_cell_table()?;
            let xml = writecell::build_sheet_xml(&cell_table, RangeStart::default());
            if xml.len() > MAX_XLSWRITE_OUTPUT_BYTES {
                return Err(xlswrite_error_with(
                    &XLSWRITE_ERROR_IO,
                    "xlswrite: worksheet XML exceeds maximum supported size",
                ));
            }
            writecell::write_xlsx_part(
                &mut zip,
                &format!("xl/worksheets/sheet{}.xml", index + 1),
                &xml,
            )
            .map_err(map_writecell_error)?;
        }

        let cursor = zip.finish().map_err(|err| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                format!("xlswrite: failed to finish spreadsheet package ({err})"),
                err,
            )
        })?;
        let bytes = cursor.into_inner();
        if bytes.len() > MAX_XLSWRITE_OUTPUT_BYTES {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: workbook exceeds maximum supported output size of {MAX_XLSWRITE_OUTPUT_BYTES} bytes"
                ),
            ));
        }
        Ok(bytes)
    }
}

impl XlsSheet {
    fn empty(name: &str) -> Self {
        Self {
            name: sanitize_sheet_name(name),
            table: XlsTable::empty(),
        }
    }
}

const ROOT_RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"#;

const STYLES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>
</styleSheet>"#;

fn content_types_xml(sheet_count: usize) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
"#,
    );
    for index in 1..=sheet_count {
        xml.push_str(&format!(
            r#"  <Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
"#
        ));
    }
    xml.push_str(
        r#"  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"#,
    );
    xml
}

fn marker_app_xml() -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>{}</Application>
</Properties>"#,
        XLSWRITE_MARKER_TEXT
    )
}

fn workbook_xml(sheets: &[XlsSheet]) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
"#,
    );
    for (index, sheet) in sheets.iter().enumerate() {
        let sheet_id = index + 1;
        xml.push_str(&format!(
            r#"    <sheet name="{}" sheetId="{sheet_id}" r:id="rId{sheet_id}"/>
"#,
            writecell::xml_attr_escape(&sheet.name)
        ));
    }
    xml.push_str("  </sheets>\n</workbook>");
    xml
}

fn make_suffixed_sheet_name(base: &str, suffix: usize) -> String {
    let suffix = format!("_{suffix}");
    let max_base_chars = 31usize.saturating_sub(suffix.chars().count());
    let mut name: String = base.chars().take(max_base_chars).collect();
    name.push_str(&suffix);
    name
}

fn workbook_relationships_xml(sheet_count: usize) -> String {
    let mut xml = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
"#,
    );
    for index in 1..=sheet_count {
        xml.push_str(&format!(
            r#"  <Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>
"#
        ));
    }
    xml.push_str(&format!(
        r#"  <Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
        sheet_count + 1
    ));
    xml
}

async fn load_workbook_or_new(path: &Path) -> BuiltinResult<XlsWorkbook> {
    match read_existing_workbook_bytes(path).await? {
        Some(bytes) => workbook_from_bytes(path, bytes),
        None => Ok(XlsWorkbook::new()),
    }
}

async fn read_existing_workbook_bytes(path: &Path) -> BuiltinResult<Option<Vec<u8>>> {
    let file = match File::open_async(path).await {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: unable to open existing '{}': {err}",
                    path.display()
                ),
                err,
            ))
        }
    };
    let mut limited = file.take(MAX_XLSWRITE_WORKBOOK_BYTES + 1);
    let mut bytes = Vec::new();
    limited.read_to_end(&mut bytes).map_err(|err| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: unable to read existing '{}': {err}",
                path.display()
            ),
            err,
        )
    })?;
    if bytes.len() as u64 > MAX_XLSWRITE_WORKBOOK_BYTES {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: existing workbook exceeds maximum supported size of {MAX_XLSWRITE_WORKBOOK_BYTES} bytes"
            ),
        ));
    }
    Ok(Some(bytes))
}

fn workbook_from_bytes(path: &Path, bytes: Vec<u8>) -> BuiltinResult<XlsWorkbook> {
    preflight_existing_workbook(&bytes)?;
    ensure_runmat_xlswrite_workbook(path, &bytes)?;
    let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes)).map_err(|err| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: unable to open existing spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    let names = workbook.sheet_names().to_vec();
    let mut sheets = Vec::with_capacity(names.len());
    for name in names {
        let name = validate_sheet_name(&name)?;
        let range = workbook.worksheet_range(&name).map_err(|err| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!("xlswrite: unable to read existing sheet '{name}': {err:?}"),
            )
        })?;
        sheets.push(XlsSheet {
            name,
            table: XlsTable::from_calamine_range(&range)?,
        });
    }
    Ok(XlsWorkbook { sheets })
}

fn preflight_existing_workbook(bytes: &[u8]) -> BuiltinResult<()> {
    if !looks_like_zip(bytes) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            "xlswrite: updating existing legacy BIFF .xls workbooks is not supported",
        ));
    }
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|err| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: invalid ZIP workbook: {err}"),
        )
    })?;
    if archive.len() > MAX_XLSWRITE_ZIP_ENTRIES {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: ZIP workbook contains more than {MAX_XLSWRITE_ZIP_ENTRIES} entries"),
        ));
    }
    let mut total = 0u64;
    for idx in 0..archive.len() {
        let entry = archive.by_index(idx).map_err(|err| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!("xlswrite: unable to inspect ZIP workbook entry {idx}: {err}"),
            )
        })?;
        if entry.size() > MAX_XLSWRITE_ZIP_ENTRY_UNCOMPRESSED_BYTES {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: ZIP workbook entry exceeds maximum expanded size of {MAX_XLSWRITE_ZIP_ENTRY_UNCOMPRESSED_BYTES} bytes"
                ),
            ));
        }
        total = total.checked_add(entry.size()).ok_or_else(|| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                "xlswrite: ZIP workbook expanded size overflows supported bounds",
            )
        })?;
        if total > MAX_XLSWRITE_ZIP_TOTAL_UNCOMPRESSED_BYTES {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: ZIP workbook expanded size exceeds maximum of {MAX_XLSWRITE_ZIP_TOTAL_UNCOMPRESSED_BYTES} bytes"
                ),
            ));
        }
    }
    Ok(())
}

fn ensure_runmat_xlswrite_workbook(path: &Path, bytes: &[u8]) -> BuiltinResult<()> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|err| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: invalid ZIP workbook: {err}"),
        )
    })?;
    let mut marker = archive.by_name(XLSWRITE_MARKER_PART).map_err(|_| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: refusing to update existing workbook '{}' because it was not created by RunMat xlswrite",
                path.display()
            ),
        )
    })?;
    let mut contents = String::new();
    marker.read_to_string(&mut contents).map_err(|err| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: unable to read RunMat workbook marker: {err}"),
            err,
        )
    })?;
    if !contents.contains(XLSWRITE_MARKER_TEXT) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: refusing to update existing workbook '{}' because its RunMat marker is invalid",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
}

#[derive(Debug, Clone)]
struct XlsTable {
    rows: usize,
    cols: usize,
    cells: Vec<CellValue>,
}

impl XlsTable {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Cell(cell) => Self::from_cell_array(cell).await,
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::SparseTensor(sparse) => Self::from_sparse(sparse),
            Value::LogicalArray(logical) => Self::from_logical(logical),
            Value::StringArray(strings) => Self::from_string_array(strings),
            Value::String(text) => Ok(Self::single(CellValue::Text(text))),
            Value::CharArray(chars) => Self::from_char_array(chars),
            Value::Num(value) => Ok(Self::single(CellValue::Number(value))),
            Value::Int(value) => Ok(Self::single(CellValue::Number(value.to_f64()))),
            Value::Bool(value) => Ok(Self::single(CellValue::Boolean(value))),
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: complex values are not supported; split real and imaginary parts first",
            )),
            other => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                format!("xlswrite: unsupported data value {other:?}"),
            )),
        }
    }

    fn single(value: CellValue) -> Self {
        Self {
            rows: 1,
            cols: 1,
            cells: vec![value],
        }
    }

    fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            cells: Vec::new(),
        }
    }

    fn from_calamine_range(range: &calamine::Range<SpreadsheetData>) -> BuiltinResult<Self> {
        let Some((start_row, start_col)) = range.start() else {
            return Ok(Self::empty());
        };
        let Some((end_row, end_col)) = range.end() else {
            return Ok(Self::empty());
        };
        let rows = usize::try_from(end_row + 1).map_err(|_| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: existing sheet row count is too large",
            )
        })?;
        let cols = usize::try_from(end_col + 1).map_err(|_| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: existing sheet column count is too large",
            )
        })?;
        checked_cell_count(rows, cols, "existing worksheet")?;
        let mut cells = vec![CellValue::Empty; rows * cols];
        for row in start_row..=end_row {
            for col in start_col..=end_col {
                let row_usize = row as usize;
                let col_usize = col as usize;
                cells[row_usize * cols + col_usize] = spreadsheet_cell_to_cell_value(
                    range
                        .get_value((row, col))
                        .unwrap_or(&SpreadsheetData::Empty),
                );
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn overlay(&mut self, start: RangeStart, incoming: &XlsTable) -> BuiltinResult<()> {
        let required_rows = start.row.checked_add(incoming.rows).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
        })?;
        let required_cols = start.col.checked_add(incoming.cols).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
        })?;
        checked_cell_count(required_rows, required_cols, "worksheet")?;
        if required_rows > self.rows || required_cols > self.cols {
            self.resize(required_rows.max(self.rows), required_cols.max(self.cols))?;
        }
        for row in 0..incoming.rows {
            for col in 0..incoming.cols {
                let dest = (start.row + row) * self.cols + start.col + col;
                self.cells[dest] = incoming.cells[row * incoming.cols + col].clone();
            }
        }
        Ok(())
    }

    fn resize(&mut self, rows: usize, cols: usize) -> BuiltinResult<()> {
        checked_cell_count(rows, cols, "worksheet")?;
        let mut resized = vec![CellValue::Empty; rows * cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                resized[row * cols + col] = self.cells[row * self.cols + col].clone();
            }
        }
        self.rows = rows;
        self.cols = cols;
        self.cells = resized;
        Ok(())
    }

    async fn from_cell_array(cell: CellArray) -> BuiltinResult<Self> {
        ensure_shape_2d(&cell.shape, "cell array")?;
        checked_cell_count(cell.rows, cell.cols, "cell array")?;
        let mut cells = Vec::with_capacity(cell.data.len());
        for row in 0..cell.rows {
            for col in 0..cell.cols {
                let value = cell.get(row, col).map_err(|message| {
                    xlswrite_error_with(&XLSWRITE_ERROR_DATA, format!("xlswrite: {message}"))
                })?;
                let gathered = gather_if_needed_async(&value)
                    .await
                    .map_err(map_control_flow)?;
                cells.push(cell_value_from_scalar(gathered)?);
            }
        }
        Ok(Self {
            rows: cell.rows,
            cols: cell.cols,
            cells,
        })
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        ensure_shape_2d(&tensor.shape, "numeric array")?;
        let rows = tensor.rows();
        let cols = tensor.cols();
        checked_cell_count(rows, cols, "numeric array")?;
        let mut cells = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                cells.push(CellValue::Number(tensor.data[row + col * rows]));
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_sparse(sparse: SparseTensor) -> BuiltinResult<Self> {
        let rows = sparse.rows;
        let cols = sparse.cols;
        checked_cell_count(rows, cols, "sparse array")?;
        let mut cells = vec![CellValue::Number(0.0); rows * cols];
        for col in 0..cols {
            let start = sparse.col_ptrs[col];
            let end = sparse.col_ptrs[col + 1];
            for entry in start..end {
                let row = sparse.row_indices[entry];
                cells[row * cols + col] = CellValue::Number(sparse.values[entry]);
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_logical(logical: LogicalArray) -> BuiltinResult<Self> {
        ensure_shape_2d(&logical.shape, "logical array")?;
        let (rows, cols) = shape_rows_cols(&logical.shape);
        checked_cell_count(rows, cols, "logical array")?;
        let mut cells = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                cells.push(CellValue::Boolean(logical.data[row + col * rows] != 0));
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_string_array(strings: StringArray) -> BuiltinResult<Self> {
        ensure_shape_2d(&strings.shape, "string array")?;
        let rows = strings.rows();
        let cols = strings.cols();
        checked_cell_count(rows, cols, "string array")?;
        let mut cells = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                cells.push(CellValue::Text(strings.data[row + col * rows].clone()));
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_char_array(chars: CharArray) -> BuiltinResult<Self> {
        if chars.rows == 0 || chars.cols == 0 {
            return Ok(Self {
                rows: 0,
                cols: 0,
                cells: Vec::new(),
            });
        }
        if chars.rows == 1 {
            return Ok(Self::single(CellValue::Text(chars.data.iter().collect())));
        }
        checked_cell_count(chars.rows, 1, "character array")?;
        let mut cells = Vec::with_capacity(chars.rows);
        for row in 0..chars.rows {
            let mut text = String::with_capacity(chars.cols);
            for col in 0..chars.cols {
                text.push(chars.data[row * chars.cols + col]);
            }
            cells.push(CellValue::Text(text));
        }
        Ok(Self {
            rows: chars.rows,
            cols: 1,
            cells,
        })
    }

    fn into_cell_table(self) -> BuiltinResult<CellTable> {
        CellTable::from_cells(self.rows, self.cols, self.cells).map_err(map_writecell_error)
    }
}

fn cell_value_from_scalar(value: Value) -> BuiltinResult<CellValue> {
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
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: complex cell values are not supported",
        )),
        Value::Cell(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: nested cell arrays are not supported",
        )),
        other => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: unsupported cell value {other:?}"),
        )),
    }
}

fn ensure_shape_2d(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 || shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} must be 2-D"),
        ))
    }
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (1, 1),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn spreadsheet_cell_to_cell_value(cell: &SpreadsheetData) -> CellValue {
    match cell {
        SpreadsheetData::Empty => CellValue::Empty,
        SpreadsheetData::Int(value) => CellValue::Number(*value as f64),
        SpreadsheetData::Float(value) => CellValue::Number(*value),
        SpreadsheetData::String(value) => CellValue::Text(value.clone()),
        SpreadsheetData::Bool(value) => CellValue::Boolean(*value),
        SpreadsheetData::DateTime(value) => CellValue::Number(value.as_f64()),
        SpreadsheetData::DateTimeIso(value) | SpreadsheetData::DurationIso(value) => {
            CellValue::Text(value.clone())
        }
        SpreadsheetData::Error(value) => CellValue::Text(value.to_string()),
    }
}

fn checked_cell_count(rows: usize, cols: usize, context: &str) -> BuiltinResult<usize> {
    if rows > MAX_EXCEL_ROW_INDEX + 1 || cols > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} exceeds Excel worksheet limits"),
        ));
    }
    let cells = rows.checked_mul(cols).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} cell count overflows supported bounds"),
        )
    })?;
    if cells > MAX_XLSWRITE_CELLS {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!(
                "xlswrite: {context} exceeds maximum supported size of {MAX_XLSWRITE_CELLS} cells"
            ),
        ));
    }
    Ok(cells)
}

fn validate_write_bounds(table: &XlsTable, start: RangeStart) -> BuiltinResult<()> {
    let end_row = start.row.checked_add(table.rows).ok_or_else(|| {
        xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
    })?;
    let end_col = start.col.checked_add(table.cols).ok_or_else(|| {
        xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
    })?;
    if end_row > MAX_EXCEL_ROW_INDEX + 1 || end_col > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range exceeds Excel worksheet limits",
        ));
    }
    Ok(())
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = value_to_string_scalar(value).map_err(|_| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_FILENAME,
            "xlswrite: filename must be a string scalar or character vector",
        )
    })?;
    let path = normalize_spreadsheet_path(&text)?;
    expand_user_path(&path, BUILTIN_NAME)
        .map(PathBuf::from)
        .map_err(|msg| xlswrite_error_with(&XLSWRITE_ERROR_FILENAME, msg))
}

fn normalize_spreadsheet_path(text: &str) -> BuiltinResult<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_FILENAME,
            "xlswrite: filename must not be empty",
        ));
    }
    let path = Path::new(trimmed);
    if path.extension().is_some() {
        let ext = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "xlsx" | "xls") {
            return Ok(trimmed.to_string());
        }
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_FILENAME,
            format!("xlswrite: unsupported spreadsheet file extension '.{ext}'"),
        ));
    }
    Ok(format!("{trimmed}.xlsx"))
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_INVALID_ARGUMENT,
            "xlswrite: expected a string scalar or character vector",
        )),
    }
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

fn xlswrite_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn xlswrite_error_with_source<E>(
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

fn map_writecell_error(err: RuntimeError) -> RuntimeError {
    let detail = err.message().to_string();
    let builder = build_runtime_error(format!("xlswrite: {detail}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err)
        .with_identifier("RunMat:xlswrite:Io");
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use calamine::{open_workbook_auto, open_workbook_auto_from_rs, Data, Reader};
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, SparseTensor};
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_xlswrite_{}_{}_{}.{}",
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

    fn output_list(value: Value) -> Vec<Value> {
        let Value::OutputList(outputs) = value else {
            panic!("expected output list, got {value:?}");
        };
        outputs
    }

    #[test]
    fn xlswrite_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = XLSWRITE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = xlswrite(filename, A)"));
        assert!(labels.contains(&"status = xlswrite(filename, A, sheetOrRange)"));
        assert!(labels.contains(&"status = xlswrite(filename, A, sheet, range)"));
        assert!(labels.contains(&"status = xlswrite(filename, A, sheet, range, '-basic')"));
        assert!(labels.contains(&"[status,message] = xlswrite(filename, A)"));
        assert!(labels.contains(&"[status,message] = xlswrite(filename, A, sheetOrRange)"));
        assert!(
            labels.contains(&"[status,message] = xlswrite(filename, A, sheet, range, '-basic')")
        );
    }

    #[test]
    fn xlswrite_writes_numeric_matrix_with_sheet_and_range() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();

        let ok = block_on(xlswrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("Data"), Value::from("B2")],
        ))
        .expect("xlswrite");
        assert_eq!(ok, Value::Bool(true));

        let mut workbook = open_workbook_auto(&path).expect("open workbook");
        assert_eq!(workbook.sheet_names()[0], "Data");
        let range = workbook.worksheet_range("Data").expect("worksheet");
        assert_eq!(range.get((0, 0)), Some(&Data::Float(1.0)));
        assert_eq!(range.get((0, 1)), Some(&Data::Float(2.0)));
        assert_eq!(range.get((0, 2)), Some(&Data::Float(3.0)));
        assert_eq!(range.get((1, 0)), Some(&Data::Float(4.0)));
        assert_eq!(range.get((1, 1)), Some(&Data::Float(5.0)));
        assert_eq!(range.get((1, 2)), Some(&Data::Float(6.0)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_accepts_cell_strings_logicals_and_ints() {
        let path = temp_path("xls");
        let filename = path.to_string_lossy().into_owned();
        let values = cell(
            vec![
                Value::from("name"),
                Value::Bool(true),
                Value::Int(IntValue::I32(7)),
                Value::from("tail"),
            ],
            2,
            2,
        );

        let _status = block_on(xlswrite_builtin(
            Value::from(filename),
            values,
            vec![Value::from("A1")],
        ))
        .expect("xlswrite");

        let bytes = fs::read(&path).expect("read workbook bytes");
        let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes)).expect("open workbook");
        let range = workbook.worksheet_range("Sheet1").expect("worksheet");
        assert_eq!(range.get((0, 0)), Some(&Data::String("name".to_string())));
        assert_eq!(range.get((0, 1)), Some(&Data::Bool(true)));
        assert_eq!(range.get((1, 0)), Some(&Data::Float(7.0)));
        assert_eq!(range.get((1, 1)), Some(&Data::String("tail".to_string())));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_accepts_top_level_sparse_matrix() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        let sparse = SparseTensor {
            rows: 2,
            cols: 2,
            col_ptrs: vec![0, 1, 2],
            row_indices: vec![1, 0],
            values: vec![9.0, 8.0],
            integer_data: None,
        };

        block_on(xlswrite_builtin(
            Value::from(filename),
            Value::SparseTensor(sparse),
            Vec::new(),
        ))
        .expect("xlswrite sparse");

        let mut workbook = open_workbook_auto(&path).expect("open workbook");
        let range = workbook.worksheet_range("Sheet1").expect("worksheet");
        assert_eq!(range.get((0, 0)), Some(&Data::Float(0.0)));
        assert_eq!(range.get((0, 1)), Some(&Data::Float(8.0)));
        assert_eq!(range.get((1, 0)), Some(&Data::Float(9.0)));
        assert_eq!(range.get((1, 1)), Some(&Data::Float(0.0)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_preserves_existing_cells_outside_overlay() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        let initial = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        block_on(xlswrite_builtin(
            Value::from(filename.clone()),
            Value::Tensor(initial),
            vec![Value::from("Sheet1"), Value::from("A1")],
        ))
        .expect("initial xlswrite");

        block_on(xlswrite_builtin(
            Value::from(filename),
            Value::Num(9.0),
            vec![Value::from("Sheet1"), Value::from("B2")],
        ))
        .expect("overlay xlswrite");

        let mut workbook = open_workbook_auto(&path).expect("open workbook");
        let range = workbook.worksheet_range("Sheet1").expect("worksheet");
        assert_eq!(range.get_value((0, 0)), Some(&Data::Float(1.0)));
        assert_eq!(range.get_value((0, 1)), Some(&Data::Float(2.0)));
        assert_eq!(range.get_value((1, 0)), Some(&Data::Float(3.0)));
        assert_eq!(range.get_value((1, 1)), Some(&Data::Float(9.0)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_preserves_other_sheets_and_uses_numeric_sheet_ordinal() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        block_on(xlswrite_builtin(
            Value::from(filename.clone()),
            Value::Num(1.0),
            vec![Value::from("First"), Value::from("A1")],
        ))
        .expect("write first sheet");
        block_on(xlswrite_builtin(
            Value::from(filename),
            Value::Num(2.0),
            vec![Value::Num(2.0), Value::from("A1")],
        ))
        .expect("write second sheet");

        let mut workbook = open_workbook_auto(&path).expect("open workbook");
        assert_eq!(
            workbook.sheet_names(),
            &["First".to_string(), "Sheet2".to_string()]
        );
        assert_eq!(
            workbook
                .worksheet_range("First")
                .expect("first")
                .get_value((0, 0)),
            Some(&Data::Float(1.0))
        );
        assert_eq!(
            workbook
                .worksheet_range_at(1)
                .expect("second exists")
                .expect("second")
                .get_value((0, 0)),
            Some(&Data::Float(2.0))
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_numeric_sheet_placeholders_avoid_name_collisions() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        block_on(xlswrite_builtin(
            Value::from(filename.clone()),
            Value::Num(1.0),
            vec![Value::from("First"), Value::from("A1")],
        ))
        .expect("write first sheet");
        block_on(xlswrite_builtin(
            Value::from(filename.clone()),
            Value::Num(3.0),
            vec![Value::from("Sheet3"), Value::from("A1")],
        ))
        .expect("write named sheet3");
        block_on(xlswrite_builtin(
            Value::from(filename),
            Value::Num(2.0),
            vec![Value::Num(3.0), Value::from("B2")],
        ))
        .expect("write ordinal third sheet");

        let workbook = open_workbook_auto(&path).expect("open workbook");
        let names = workbook.sheet_names();
        assert_eq!(names.len(), 3);
        let unique: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
        assert_eq!(names[0], "First");
        assert_eq!(names[1], "Sheet3");
        assert_ne!(names[2], "Sheet3");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_accepts_scalar_tensor_sheet_selector_and_absolute_range() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        let sheet = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        block_on(xlswrite_builtin(
            Value::from(filename),
            Value::Num(42.0),
            vec![Value::Tensor(sheet), Value::from("$B$2:$B$2")],
        ))
        .expect("xlswrite");

        let mut workbook = open_workbook_auto(&path).expect("open workbook");
        let sheet2 = workbook
            .worksheet_range_at(1)
            .expect("second sheet exists")
            .expect("second sheet");
        assert_eq!(sheet2.get_value((1, 1)), Some(&Data::Float(42.0)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_rejects_malformed_range_end() {
        let err = block_on(xlswrite_builtin(
            Value::from("out.xlsx"),
            Value::Num(1.0),
            vec![Value::from("Sheet1"), Value::from("B2:not-a-cell")],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:xlswrite:Range"));
    }

    #[test]
    fn xlswrite_rejects_sparse_inputs_that_would_densify_too_large() {
        let sparse = SparseTensor {
            rows: MAX_XLSWRITE_CELLS + 1,
            cols: 1,
            col_ptrs: vec![0, 0],
            row_indices: Vec::new(),
            values: Vec::new(),
            integer_data: None,
        };
        let err = block_on(xlswrite_builtin(
            Value::from("out.xlsx"),
            Value::SparseTensor(sparse),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:xlswrite:Data"));
    }

    #[test]
    fn xlswrite_rejects_unmarked_existing_workbook_instead_of_flattening_it() {
        let path = temp_path("xlsx");
        let table = CellTable::from_cells(1, 1, vec![CellValue::Number(1.0)]).expect("table");
        let bytes = writecell::build_xlsx_workbook(&table, "External", RangeStart::default())
            .expect("workbook bytes");
        fs::write(&path, bytes).expect("write workbook");

        let err = block_on(xlswrite_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Value::Num(2.0),
            vec![Value::from("External"), Value::from("A1")],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:xlswrite:Io"));
        assert!(err.message().contains("not created by RunMat xlswrite"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_rejects_excessive_numeric_sheet_index() {
        let err = block_on(xlswrite_builtin(
            Value::from("out.xlsx"),
            Value::Num(1.0),
            vec![
                Value::Num((MAX_XLSWRITE_SHEETS + 1) as f64),
                Value::from("A1"),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:xlswrite:Sheet"));
    }

    #[test]
    fn xlswrite_rejects_invalid_sheet_names_instead_of_rewriting_them() {
        let err = block_on(xlswrite_builtin(
            Value::from("out.xlsx"),
            Value::Num(1.0),
            vec![Value::from("A/B"), Value::from("A1")],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:xlswrite:Sheet"));
    }

    #[test]
    fn xlswrite_returns_status_and_message_outputs() {
        let path = temp_path("xlsx");
        let filename = path.to_string_lossy().into_owned();
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(xlswrite_builtin(
            Value::from(filename),
            Value::Complex(1.0, 2.0),
            Vec::new(),
        ))
        .expect("xlswrite captures failure");
        let outputs = output_list(result);
        assert_eq!(outputs[0], Value::Bool(false));
        let Value::Struct(message) = &outputs[1] else {
            panic!("expected message struct");
        };
        assert!(matches!(
            message.fields.get("identifier"),
            Some(Value::String(id)) if id == "RunMat:xlswrite:Data"
        ));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlswrite_rejects_too_many_outputs() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(xlswrite_builtin(
            Value::from("out.xlsx"),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:xlswrite:OutputCount"));
    }
}
