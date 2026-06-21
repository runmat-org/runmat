//! MATLAB-compatible `xlsread` builtin for RunMat.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto_from_rs, Data as SpreadsheetData, Reader as SpreadsheetReader};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "xlsread";
const MAX_EXCEL_ROW_INDEX: usize = 1_048_575;
const MAX_EXCEL_COLUMN_INDEX: usize = 16_383;
const MAX_XLSREAD_SELECTED_CELLS: usize = 5_000_000;

const XLSREAD_OUTPUT_NUM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "num",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric worksheet data.",
};
const XLSREAD_OUTPUT_TXT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Text worksheet cells.",
};
const XLSREAD_OUTPUT_RAW: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "raw",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Raw worksheet cells.",
};
const XLSREAD_OUTPUTS_NUM: [BuiltinParamDescriptor; 1] = [XLSREAD_OUTPUT_NUM];
const XLSREAD_OUTPUTS_ALL: [BuiltinParamDescriptor; 3] =
    [XLSREAD_OUTPUT_NUM, XLSREAD_OUTPUT_TXT, XLSREAD_OUTPUT_RAW];

const XLSREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Spreadsheet file path.",
}];
const XLSREAD_INPUTS_FILENAME_SELECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "sheetOrRange",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name/index or Excel range.",
    },
];
const XLSREAD_INPUTS_FILENAME_SHEET_RANGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range.",
    },
];
const XLSREAD_INPUTS_FILENAME_SHEET_RANGE_BASIC: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "sheet",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Worksheet name or one-based index.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Excel A1 range.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"basic\""),
        description: "Legacy '-basic' compatibility flag.",
    },
];

const XLSREAD_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename)",
        inputs: &XLSREAD_INPUTS_FILENAME,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename, sheetOrRange)",
        inputs: &XLSREAD_INPUTS_FILENAME_SELECTOR,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename, sheet, range)",
        inputs: &XLSREAD_INPUTS_FILENAME_SHEET_RANGE,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "num = xlsread(filename, sheet, range, '-basic')",
        inputs: &XLSREAD_INPUTS_FILENAME_SHEET_RANGE_BASIC,
        outputs: &XLSREAD_OUTPUTS_NUM,
    },
    BuiltinSignatureDescriptor {
        label: "[num, txt, raw] = xlsread(___)",
        inputs: &XLSREAD_INPUTS_FILENAME_SHEET_RANGE_BASIC,
        outputs: &XLSREAD_OUTPUTS_ALL,
    },
];

const XLSREAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlsread:InvalidArgument"),
    when: "Argument list does not match supported xlsread call forms.",
    message: "xlsread: invalid argument",
};
const XLSREAD_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.FILENAME",
    identifier: Some("RunMat:xlsread:Filename"),
    when: "Filename is invalid or cannot be normalized.",
    message: "xlsread: invalid filename",
};
const XLSREAD_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.RANGE",
    identifier: Some("RunMat:xlsread:Range"),
    when: "Range specification is malformed or semantically invalid.",
    message: "xlsread: invalid range",
};
const XLSREAD_ERROR_SHEET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.SHEET",
    identifier: Some("RunMat:xlsread:Sheet"),
    when: "Requested worksheet is missing or invalid.",
    message: "xlsread: invalid sheet",
};
const XLSREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.IO",
    identifier: Some("RunMat:xlsread:Io"),
    when: "Spreadsheet cannot be opened or read.",
    message: "xlsread: unable to read spreadsheet",
};
const XLSREAD_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.OUTPUT_COUNT",
    identifier: Some("RunMat:xlsread:OutputCount"),
    when: "Caller requests more outputs than xlsread supports.",
    message: "xlsread: unsupported output count",
};
const XLSREAD_ERROR_TENSOR_BUILD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLSREAD.TENSOR_BUILD",
    identifier: Some("RunMat:xlsread:TensorBuild"),
    when: "Output arrays cannot be materialized.",
    message: "xlsread: unable to construct output",
};
const XLSREAD_ERRORS: [BuiltinErrorDescriptor; 7] = [
    XLSREAD_ERROR_INVALID_ARGUMENT,
    XLSREAD_ERROR_FILENAME,
    XLSREAD_ERROR_RANGE,
    XLSREAD_ERROR_SHEET,
    XLSREAD_ERROR_IO,
    XLSREAD_ERROR_OUTPUT_COUNT,
    XLSREAD_ERROR_TENSOR_BUILD,
];

pub const XLSREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLSREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLSREAD_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::xlsread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "xlsread",
    op_kind: GpuOpKind::Custom("io-xlsread"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reads spreadsheets on the host and creates CPU-resident values.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::xlsread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "xlsread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; spreadsheet I/O executes as a standalone host operation.",
};

#[runtime_builtin(
    name = "xlsread",
    category = "io/tabular",
    summary = "Read numeric, text, and raw data from spreadsheet files.",
    keywords = "xlsread,xls,xlsx,spreadsheet,excel,numeric import",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::xlsread_type),
    descriptor(crate::builtins::io::tabular::xlsread::XLSREAD_DESCRIPTOR),
    builtin_path = "crate::builtins::io::tabular::xlsread"
)]
async fn xlsread_builtin(path: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let request = parse_arguments(&rest).await?;
    let resolved = resolve_path(&path_value)?;
    let result = read_spreadsheet(&resolved, &request).await?;
    result.into_requested_value()
}

fn xlsread_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn xlsread_error_with_source<E>(
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

#[derive(Clone, Debug, Default)]
struct XlsReadRequest {
    sheet: Option<SheetSelector>,
    range: Option<RangeSpec>,
}

#[derive(Clone, Debug)]
enum SheetSelector {
    Name(String),
    Index(usize),
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<XlsReadRequest> {
    let gathered = gather_arguments(args).await?;
    let mut request = XlsReadRequest::default();
    match gathered.as_slice() {
        [] => {}
        [single] => parse_single_selector(single, &mut request)?,
        [sheet, range] => {
            request.sheet = Some(parse_sheet_selector(sheet)?);
            request.range = Some(parse_range(range)?);
        }
        [sheet, range, mode] => {
            request.sheet = Some(parse_sheet_selector(sheet)?);
            request.range = Some(parse_range(range)?);
            parse_basic_mode(mode)?;
        }
        _ => {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_INVALID_ARGUMENT,
                "xlsread: expected filename, optional sheet, optional range, and optional '-basic'",
            ))
        }
    }
    Ok(request)
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

fn parse_single_selector(value: &Value, request: &mut XlsReadRequest) -> BuiltinResult<()> {
    if let Ok(text) = value_to_string_scalar(value) {
        if text.trim().eq_ignore_ascii_case("basic") || text.trim().eq_ignore_ascii_case("-basic") {
            return Ok(());
        }
        match parse_range_string(&text) {
            Ok(range) if text_looks_like_range(&text) => {
                request.range = Some(range);
                return Ok(());
            }
            Ok(range) if range.end_row.is_some() || range.end_col.is_some() => {
                request.range = Some(range);
                return Ok(());
            }
            _ => {}
        }
        request.sheet = Some(SheetSelector::Name(text));
        return Ok(());
    }
    request.sheet = Some(parse_sheet_selector(value)?);
    Ok(())
}

fn parse_basic_mode(value: &Value) -> BuiltinResult<()> {
    let text = value_to_string_scalar(value)?;
    let normalized = text.trim().to_ascii_lowercase();
    if normalized == "basic" || normalized == "-basic" {
        return Ok(());
    }
    Err(xlsread_error_with(
        &XLSREAD_ERROR_INVALID_ARGUMENT,
        "xlsread: only the legacy '-basic' mode flag is supported",
    ))
}

fn parse_sheet_selector(value: &Value) -> BuiltinResult<SheetSelector> {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let text = value_to_string_scalar(value)?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_SHEET,
                    "xlsread: sheet name must not be empty",
                ));
            }
            Ok(SheetSelector::Name(trimmed.to_string()))
        }
        Value::Num(n) => numeric_sheet_index(*n),
        Value::Int(i) => {
            let index = i.to_i64();
            if index <= 0 {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_SHEET,
                    "xlsread: sheet index must be one-based",
                ));
            }
            usize::try_from(index - 1)
                .map(SheetSelector::Index)
                .map_err(|_| {
                    xlsread_error_with(&XLSREAD_ERROR_SHEET, "xlsread: sheet index is too large")
                })
        }
        Value::Tensor(t) if t.data.len() == 1 => numeric_sheet_index(t.data[0]),
        _ => Err(xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet must be a name or one-based numeric index",
        )),
    }
}

fn numeric_sheet_index(value: f64) -> BuiltinResult<SheetSelector> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet index must be a positive integer",
        ));
    }
    let index = value.round() as u128;
    let zero_based = index.checked_sub(1).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet index must be one-based",
        )
    })?;
    usize::try_from(zero_based)
        .map(SheetSelector::Index)
        .map_err(|_| xlsread_error_with(&XLSREAD_ERROR_SHEET, "xlsread: sheet index is too large"))
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(xlsread_error_with(
            &XLSREAD_ERROR_INVALID_ARGUMENT,
            "xlsread: expected a string scalar or character vector",
        )),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = value_to_string_scalar(value).map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_FILENAME,
            "xlsread: filename must be a string scalar or character vector",
        )
    })?;
    let path = normalize_spreadsheet_path(&text)?;
    expand_user_path(&path, BUILTIN_NAME)
        .map(PathBuf::from)
        .map_err(|msg| xlsread_error_with(&XLSREAD_ERROR_FILENAME, msg))
}

fn normalize_spreadsheet_path(text: &str) -> BuiltinResult<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_FILENAME,
            "xlsread: filename must not be empty",
        ));
    }
    let path = Path::new(trimmed);
    if path.extension().is_some() {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("{trimmed}.xlsx"))
    }
}

async fn read_spreadsheet(path: &Path, request: &XlsReadRequest) -> BuiltinResult<XlsReadResult> {
    let bytes = read_file_bytes(path).await?;
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|err| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            format!(
                "xlsread: unable to open spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    let range = match &request.sheet {
        Some(SheetSelector::Name(name)) => workbook.worksheet_range(name).map_err(|err| {
            xlsread_error_with(
                &XLSREAD_ERROR_SHEET,
                format!("xlsread: unable to read sheet '{name}': {err:?}"),
            )
        })?,
        Some(SheetSelector::Index(index)) => workbook
            .worksheet_range_at(*index)
            .ok_or_else(|| {
                xlsread_error_with(
                    &XLSREAD_ERROR_SHEET,
                    format!("xlsread: sheet index {} exceeds bounds", index + 1),
                )
            })?
            .map_err(|err| {
                xlsread_error_with(
                    &XLSREAD_ERROR_SHEET,
                    format!("xlsread: unable to read sheet {}: {err:?}", index + 1),
                )
            })?,
        None => workbook
            .worksheet_range_at(0)
            .ok_or_else(|| xlsread_error_with(&XLSREAD_ERROR_SHEET, "xlsread: no worksheets"))?
            .map_err(|err| {
                xlsread_error_with(
                    &XLSREAD_ERROR_SHEET,
                    format!("xlsread: unable to read first sheet: {err:?}"),
                )
            })?,
    };
    let cells = selected_cells(&range, request.range)?;
    XlsReadResult::from_cells(cells)
}

async fn read_file_bytes(path: &Path) -> BuiltinResult<Vec<u8>> {
    let mut file = File::open_async(path).await.map_err(|err| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            format!("xlsread: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|err| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            format!("xlsread: unable to read '{}': {err}", path.display()),
            err,
        )
    })?;
    Ok(bytes)
}

#[derive(Clone, Copy, Debug)]
struct RangeSpec {
    start_row: usize,
    start_col: usize,
    end_row: Option<usize>,
    end_col: Option<usize>,
}

impl RangeSpec {
    fn validate(&self) -> BuiltinResult<()> {
        if self.start_row > MAX_EXCEL_ROW_INDEX {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range start row exceeds Excel limits",
            ));
        }
        if self.start_col > MAX_EXCEL_COLUMN_INDEX {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range start column exceeds Excel limits",
            ));
        }
        if let Some(end_row) = self.end_row {
            if end_row < self.start_row {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end row must be >= start row",
                ));
            }
            if end_row > MAX_EXCEL_ROW_INDEX {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end row exceeds Excel limits",
                ));
            }
        }
        if let Some(end_col) = self.end_col {
            if end_col < self.start_col {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end column must be >= start column",
                ));
            }
            if end_col > MAX_EXCEL_COLUMN_INDEX {
                return Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    "xlsread: range end column exceeds Excel limits",
                ));
            }
        }
        Ok(())
    }
}

fn parse_range(value: &Value) -> BuiltinResult<RangeSpec> {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            parse_range_string(&value_to_string_scalar(value)?)
        }
        Value::Tensor(t) => parse_numeric_range(&t.data),
        _ => Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range must be a string or numeric vector",
        )),
    }
}

fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range string must not be empty",
        ));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: invalid range specification '{text}'"),
        ));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range must specify a starting column",
        ));
    }
    let has_explicit_end = parts.len() == 2;
    let end = if has_explicit_end {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(end_ref) = end {
        if end_ref.col.is_none() {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range end must include a column reference",
            ));
        }
    }
    let mut end_row = end.and_then(|cell| cell.row);
    let mut end_col = end.and_then(|cell| cell.col);
    if !has_explicit_end && start.row.is_some() && start.col.is_some() {
        end_row = start.row;
        end_col = start.col;
    }
    let spec = RangeSpec {
        start_row: start.row.unwrap_or(0),
        start_col: start.col.unwrap(),
        end_row,
        end_col,
    };
    spec.validate()?;
    Ok(spec)
}

fn parse_numeric_range(values: &[f64]) -> BuiltinResult<RangeSpec> {
    if values.len() != 2 && values.len() != 4 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: numeric range must contain 2 or 4 one-based indices",
        ));
    }
    let indices = values
        .iter()
        .enumerate()
        .map(|(idx, value)| positive_index(*value, idx))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let spec = RangeSpec {
        start_row: indices[0],
        start_col: indices[1],
        end_row: if indices.len() == 4 {
            Some(indices[2])
        } else {
            None
        },
        end_col: if indices.len() == 4 {
            Some(indices[3])
        } else {
            None
        },
    };
    spec.validate()?;
    Ok(spec)
}

fn positive_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range indices must be positive integers",
        ));
    }
    if value > usize::MAX as f64 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!("xlsread: range index {} is too large", position + 1),
        ));
    }
    let one_based = value.round() as usize;
    one_based.checked_sub(1).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range indices must be one-based",
        )
    })
}

#[derive(Clone, Copy)]
struct CellReference {
    row: Option<usize>,
    col: Option<usize>,
}

fn parse_cell_reference(token: &str) -> BuiltinResult<CellReference> {
    let mut letters = String::new();
    let mut digits = String::new();
    for ch in token.trim().chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                format!("xlsread: invalid range component '{token}'"),
            ));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: range references cannot be empty",
        ));
    }
    let col = if letters.is_empty() {
        None
    } else {
        Some(column_index_from_letters(&letters)?)
    };
    let row = if digits.is_empty() {
        None
    } else {
        let parsed = digits.parse::<usize>().map_err(|_| {
            xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                format!("xlsread: invalid row index '{digits}'"),
            )
        })?;
        if parsed == 0 {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                "xlsread: range rows must be one-based",
            ));
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

fn column_index_from_letters(letters: &str) -> BuiltinResult<usize> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(xlsread_error_with(
                &XLSREAD_ERROR_RANGE,
                format!("xlsread: invalid column designator '{letters}'"),
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|current| current.checked_add(digit))
            .ok_or_else(|| {
                xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: range column overflow")
            })?;
    }
    value
        .checked_sub(1)
        .ok_or_else(|| xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: range column underflow"))
        .and_then(|zero_based| {
            if zero_based <= MAX_EXCEL_COLUMN_INDEX {
                Ok(zero_based)
            } else {
                Err(xlsread_error_with(
                    &XLSREAD_ERROR_RANGE,
                    format!("xlsread: range column '{letters}' exceeds Excel limits"),
                ))
            }
        })
}

fn text_looks_like_range(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.contains(':')
        || trimmed.chars().any(|ch| ch.is_ascii_alphabetic())
            && trimmed.chars().any(|ch| ch.is_ascii_digit())
}

fn selected_cells(
    range: &calamine::Range<SpreadsheetData>,
    requested: Option<RangeSpec>,
) -> BuiltinResult<Vec<Vec<SpreadsheetData>>> {
    if range.is_empty() && requested.is_none() {
        return Ok(Vec::new());
    }
    let start = range.start().unwrap_or((0, 0));
    let end = range.end().unwrap_or((0, 0));
    let start_row = requested
        .map(|spec| checked_u32(spec.start_row, "range row"))
        .transpose()?
        .unwrap_or(start.0);
    let start_col = requested
        .map(|spec| checked_u32(spec.start_col, "range column"))
        .transpose()?
        .unwrap_or(start.1);
    let end_row = requested
        .and_then(|spec| spec.end_row)
        .map(|row| checked_u32(row, "range row"))
        .transpose()?
        .unwrap_or(end.0);
    let end_col = requested
        .and_then(|spec| spec.end_col)
        .map(|col| checked_u32(col, "range column"))
        .transpose()?
        .unwrap_or(end.1);
    if start_row > end_row || start_col > end_col {
        return Ok(Vec::new());
    }
    validate_selected_shape(start_row, start_col, end_row, end_col)?;

    let mut rows = Vec::new();
    for row_idx in start_row..=end_row {
        let mut row = Vec::new();
        for col_idx in start_col..=end_col {
            row.push(
                range
                    .get_value((row_idx, col_idx))
                    .cloned()
                    .unwrap_or(SpreadsheetData::Empty),
            );
        }
        rows.push(row);
    }
    if requested.is_some() {
        Ok(rows)
    } else {
        trim_empty_edges(rows)
    }
}

fn checked_u32(value: usize, context: &str) -> BuiltinResult<u32> {
    u32::try_from(value).map_err(|_| {
        xlsread_error_with(&XLSREAD_ERROR_RANGE, format!("xlsread: {context} overflow"))
    })
}

fn validate_selected_shape(
    start_row: u32,
    start_col: u32,
    end_row: u32,
    end_col: u32,
) -> BuiltinResult<()> {
    let row_count = usize::try_from(end_row - start_row + 1).map_err(|_| {
        xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: selected row count overflow")
    })?;
    let col_count = usize::try_from(end_col - start_col + 1).map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            "xlsread: selected column count overflow",
        )
    })?;
    let cell_count = row_count.checked_mul(col_count).ok_or_else(|| {
        xlsread_error_with(&XLSREAD_ERROR_RANGE, "xlsread: selected range is too large")
    })?;
    if cell_count > MAX_XLSREAD_SELECTED_CELLS {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_RANGE,
            format!(
                "xlsread: selected range has {cell_count} cells, exceeding the RunMat limit of {MAX_XLSREAD_SELECTED_CELLS}"
            ),
        ));
    }
    Ok(())
}

fn trim_empty_edges(rows: Vec<Vec<SpreadsheetData>>) -> BuiltinResult<Vec<Vec<SpreadsheetData>>> {
    if rows.is_empty() {
        return Ok(rows);
    }
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut first_row = None;
    let mut last_row = None;
    let mut first_col = None;
    let mut last_col = None;

    for row in 0..row_count {
        for col in 0..col_count {
            let cell = rows
                .get(row)
                .and_then(|values| values.get(col))
                .unwrap_or(&SpreadsheetData::Empty);
            if !matches!(cell, SpreadsheetData::Empty) {
                first_row = Some(first_row.map_or(row, |current: usize| current.min(row)));
                last_row = Some(last_row.map_or(row, |current: usize| current.max(row)));
                first_col = Some(first_col.map_or(col, |current: usize| current.min(col)));
                last_col = Some(last_col.map_or(col, |current: usize| current.max(col)));
            }
        }
    }

    let (Some(first_row), Some(last_row), Some(first_col), Some(last_col)) =
        (first_row, last_row, first_col, last_col)
    else {
        return Ok(Vec::new());
    };
    let mut trimmed = Vec::new();
    for row in rows.iter().take(last_row + 1).skip(first_row) {
        let mut out = Vec::new();
        for col in first_col..=last_col {
            out.push(row.get(col).cloned().unwrap_or(SpreadsheetData::Empty));
        }
        trimmed.push(out);
    }
    Ok(trimmed)
}

#[derive(Clone)]
struct XlsReadResult {
    num: Value,
    txt: Value,
    raw: Value,
}

impl XlsReadResult {
    fn from_cells(rows: Vec<Vec<SpreadsheetData>>) -> BuiltinResult<Self> {
        let num = build_numeric_output(&rows)?;
        let txt = build_text_output(&rows)?;
        let raw = build_raw_output(&rows)?;
        Ok(Self { num, txt, raw })
    }

    fn into_requested_value(self) -> BuiltinResult<Value> {
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![self.num])),
            Some(2) => Ok(Value::OutputList(vec![self.num, self.txt])),
            Some(3) => Ok(Value::OutputList(vec![self.num, self.txt, self.raw])),
            Some(n) => Err(xlsread_error_with(
                &XLSREAD_ERROR_OUTPUT_COUNT,
                format!("xlsread: expected at most 3 outputs, got {n}"),
            )),
            None => Ok(self.num),
        }
    }
}

fn build_numeric_output(rows: &[Vec<SpreadsheetData>]) -> BuiltinResult<Value> {
    let Some((first_row, last_row, first_col, last_col)) = numeric_bounds(rows) else {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
            xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}"))
        })?;
        return Ok(Value::Tensor(tensor));
    };
    let row_count = last_row - first_row + 1;
    let col_count = last_col - first_col + 1;
    let data_len = row_count.checked_mul(col_count).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_TENSOR_BUILD,
            "xlsread: numeric output is too large",
        )
    })?;
    let mut data = vec![f64::NAN; data_len];
    for out_row in 0..row_count {
        for out_col in 0..col_count {
            let source_row = first_row + out_row;
            let source_col = first_col + out_col;
            let value = rows
                .get(source_row)
                .and_then(|row| row.get(source_col))
                .and_then(cell_to_numeric)
                .unwrap_or(f64::NAN);
            data[out_row + out_col * row_count] = value;
        }
    }
    let tensor = Tensor::new(data, vec![row_count, col_count]).map_err(|err| {
        xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}"))
    })?;
    Ok(Value::Tensor(tensor))
}

fn numeric_bounds(rows: &[Vec<SpreadsheetData>]) -> Option<(usize, usize, usize, usize)> {
    let mut first_row = None;
    let mut last_row = None;
    let mut first_col = None;
    let mut last_col = None;
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, cell) in row.iter().enumerate() {
            if cell_to_numeric(cell).is_some() {
                first_row = Some(first_row.map_or(row_idx, |current: usize| current.min(row_idx)));
                last_row = Some(last_row.map_or(row_idx, |current: usize| current.max(row_idx)));
                first_col = Some(first_col.map_or(col_idx, |current: usize| current.min(col_idx)));
                last_col = Some(last_col.map_or(col_idx, |current: usize| current.max(col_idx)));
            }
        }
    }
    match (first_row, last_row, first_col, last_col) {
        (Some(first_row), Some(last_row), Some(first_col), Some(last_col)) => {
            Some((first_row, last_row, first_col, last_col))
        }
        _ => None,
    }
}

fn build_text_output(rows: &[Vec<SpreadsheetData>]) -> BuiltinResult<Value> {
    let (row_count, col_count) = rectangular_shape(rows);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row_idx in 0..row_count {
        for col_idx in 0..col_count {
            let text = rows
                .get(row_idx)
                .and_then(|row| row.get(col_idx))
                .and_then(cell_to_text)
                .unwrap_or_default();
            values.push(Value::from(text));
        }
    }
    crate::make_cell(values, row_count, col_count)
        .map_err(|err| xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}")))
}

fn build_raw_output(rows: &[Vec<SpreadsheetData>]) -> BuiltinResult<Value> {
    let (row_count, col_count) = rectangular_shape(rows);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row_idx in 0..row_count {
        for col_idx in 0..col_count {
            let value = rows
                .get(row_idx)
                .and_then(|row| row.get(col_idx))
                .map(cell_to_raw)
                .unwrap_or(Value::Num(f64::NAN));
            values.push(value);
        }
    }
    crate::make_cell(values, row_count, col_count)
        .map_err(|err| xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}")))
}

fn rectangular_shape(rows: &[Vec<SpreadsheetData>]) -> (usize, usize) {
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    (row_count, col_count)
}

fn cell_to_numeric(cell: &SpreadsheetData) -> Option<f64> {
    match cell {
        SpreadsheetData::Int(value) => Some(*value as f64),
        SpreadsheetData::Float(value) => Some(*value),
        SpreadsheetData::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        SpreadsheetData::DateTime(value) => Some(value.as_f64()),
        _ => None,
    }
}

fn cell_to_text(cell: &SpreadsheetData) -> Option<String> {
    match cell {
        SpreadsheetData::String(text)
        | SpreadsheetData::DateTimeIso(text)
        | SpreadsheetData::DurationIso(text) => Some(text.clone()),
        SpreadsheetData::Error(err) => Some(err.to_string()),
        _ => None,
    }
}

fn cell_to_raw(cell: &SpreadsheetData) -> Value {
    match cell {
        SpreadsheetData::Int(value) => Value::Num(*value as f64),
        SpreadsheetData::Float(value) => Value::Num(*value),
        SpreadsheetData::Bool(value) => Value::Bool(*value),
        SpreadsheetData::DateTime(value) => Value::Num(value.as_f64()),
        SpreadsheetData::String(text)
        | SpreadsheetData::DateTimeIso(text)
        | SpreadsheetData::DurationIso(text) => Value::from(text.clone()),
        SpreadsheetData::Error(err) => Value::from(err.to_string()),
        SpreadsheetData::Empty => Value::Num(f64::NAN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::fs;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_path(stem: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{stem}_{id}.xlsx"))
    }

    fn write_zip_file(zip: &mut zip::ZipWriter<std::fs::File>, name: &str, contents: &str) {
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file(name, options).expect("start xlsx part");
        zip.write_all(contents.as_bytes()).expect("write xlsx part");
    }

    fn write_minimal_xlsx(path: &Path) {
        write_named_xlsx(path, "Data");
    }

    fn write_named_xlsx(path: &Path, sheet_name: &str) {
        let file = std::fs::File::create(path).expect("create xlsx");
        let mut zip = zip::ZipWriter::new(file);
        write_zip_file(
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
        );
        write_zip_file(
            &mut zip,
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
        );
        write_zip_file(
            &mut zip,
            "xl/workbook.xml",
            &format!(
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{sheet_name}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#
            ),
        );
        write_zip_file(
            &mut zip,
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
        );
        write_zip_file(
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
        );
        write_zip_file(
            &mut zip,
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Voltage</t></is></c>
      <c r="B1" t="inlineStr"><is><t>Current</t></is></c>
      <c r="C1" t="inlineStr"><is><t>Label</t></is></c>
    </row>
    <row r="2">
      <c r="A2"><v>1.5</v></c>
      <c r="B2"><v>10</v></c>
      <c r="C2" t="inlineStr"><is><t>low</t></is></c>
    </row>
    <row r="3">
      <c r="A3"><v>2.5</v></c>
      <c r="B3"><v>20</v></c>
      <c r="C3" t="inlineStr"><is><t>high</t></is></c>
    </row>
    <row r="4">
      <c r="A4"><v>3.5</v></c>
      <c r="B4"><v>30</v></c>
      <c r="C4" t="b"><v>1</v></c>
    </row>
  </sheetData>
</worksheet>"#,
        );
        zip.finish().expect("finish xlsx");
    }

    fn direct_xlsread(path: &Path, args: Vec<Value>, outputs: Option<usize>) -> Value {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(xlsread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            args,
        ))
        .expect("xlsread")
    }

    #[test]
    fn xlsread_registers_public_descriptor() {
        assert!(runmat_builtins::builtin_function_by_name("xlsread").is_some());
        let labels = XLSREAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"num = xlsread(filename)"));
        assert!(labels.contains(&"num = xlsread(filename, sheetOrRange)"));
        assert!(labels.contains(&"[num, txt, raw] = xlsread(___)"));
    }

    #[test]
    fn xlsread_reads_numeric_range_as_default_output() {
        let path = unique_path("xlsread_numeric_range");
        write_minimal_xlsx(&path);
        let value = direct_xlsread(&path, vec![Value::from("A2:B4")], None);
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 2]);
                assert_eq!(tensor.data, vec![1.5, 2.5, 3.5, 10.0, 20.0, 30.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlsread_accepts_sheet_and_range() {
        let path = unique_path("xlsread_sheet_range");
        write_minimal_xlsx(&path);
        let value = direct_xlsread(&path, vec![Value::from("Data"), Value::from("B2:B3")], None);
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![10.0, 20.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn string_sheet_selector_is_not_numeric_index() {
        let path = unique_path("xlsread_numeric_sheet_name");
        write_named_xlsx(&path, "1");
        let value = direct_xlsread(&path, vec![Value::from("1"), Value::from("A2:A2")], None);
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![1.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn single_selector_prefers_sheet_name_when_not_valid_excel_range() {
        let mut request = XlsReadRequest::default();
        parse_single_selector(&Value::from("Data1"), &mut request).expect("selector");
        match request.sheet {
            Some(SheetSelector::Name(name)) => assert_eq!(name, "Data1"),
            other => panic!("expected sheet name, got {other:?}"),
        }
        assert!(request.range.is_none());
    }

    #[test]
    fn single_cell_range_selects_one_cell() {
        let spec = parse_range_string("B3").expect("single-cell range");
        assert_eq!(spec.start_row, 2);
        assert_eq!(spec.start_col, 1);
        assert_eq!(spec.end_row, Some(2));
        assert_eq!(spec.end_col, Some(1));
    }

    #[test]
    fn explicit_range_preserves_raw_shape_with_blank_borders() {
        let path = unique_path("xlsread_blank_border_range");
        write_minimal_xlsx(&path);
        let value = direct_xlsread(&path, vec![Value::from("B1:D5")], Some(3));
        let Value::OutputList(outputs) = value else {
            panic!("expected output list");
        };
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 5);
                assert_eq!(cell.cols, 3);
                assert_eq!(cell.get(0, 0).unwrap(), Value::from("Current"));
                match cell.get(4, 2).unwrap() {
                    Value::Num(value) => assert!(value.is_nan()),
                    other => panic!("expected NaN, got {other:?}"),
                }
            }
            other => panic!("expected raw cell, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn oversized_explicit_range_is_rejected_before_allocation() {
        let spec = parse_range_string("A1:XFD1048576").expect("maximum Excel range parses");
        let rows = calamine::Range::new((0, 0), (0, 0));
        let err = selected_cells(&rows, Some(spec)).expect_err("oversized range");
        assert!(err.message().contains("selected range"));
    }

    #[test]
    fn xlsread_returns_text_and_raw_outputs() {
        let path = unique_path("xlsread_multi_output");
        write_minimal_xlsx(&path);
        let value = direct_xlsread(&path, vec![Value::from("A1:C3")], Some(3));
        let Value::OutputList(outputs) = value else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 3);
        match &outputs[0] {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.data, vec![1.5, 2.5, 10.0, 20.0]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 3);
                assert_eq!(cell.get(0, 0).unwrap(), Value::from("Voltage"));
                assert_eq!(cell.get(2, 2).unwrap(), Value::from("high"));
            }
            other => panic!("expected text cell, got {other:?}"),
        }
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 3);
                assert_eq!(cell.get(0, 1).unwrap(), Value::from("Current"));
                assert_eq!(cell.get(1, 0).unwrap(), Value::Num(1.5));
                assert_eq!(cell.get(2, 2).unwrap(), Value::from("high"));
            }
            other => panic!("expected raw cell, got {other:?}"),
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn xlsread_rejects_too_many_outputs() {
        let path = unique_path("xlsread_too_many_outputs");
        write_minimal_xlsx(&path);
        let _guard = crate::output_count::push_output_count(Some(4));
        let err = block_on(xlsread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Vec::new(),
        ))
        .expect_err("too many outputs");
        assert!(err.message().contains("at most 3 outputs"));
        let _ = fs::remove_file(path);
    }
}
