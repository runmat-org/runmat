//! MATLAB-compatible `writecell` builtin for heterogeneous cell-array export.

use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[cfg(test)]
use runmat_builtins::WRITECELL_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, WRITECELL_BYTES_OUTPUT_EXTENSION, WRITECELL_ERROR_ARG_CONFIG,
    WRITECELL_ERROR_DATA, WRITECELL_ERROR_DATA_SHAPE, WRITECELL_ERROR_FILENAME, WRITECELL_ERROR_IO,
    WRITECELL_ERROR_OPTION, WRITECELL_EXPLICIT_GPU_EXTENSION,
};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, IntValue, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};
use super::write_owner;

const BUILTIN_NAME: &str = "writecell";
const MAX_EXCEL_ROW_INDEX: usize = 1_048_575;
const MAX_EXCEL_COLUMN_INDEX: usize = 16_383;

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
    E: crate::runtime_error::RuntimeErrorSourceBound,
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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::tabular::writecell"
)]
async fn writecell_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    if rest.is_empty() {
        return Err(writecell_error(&WRITECELL_ERROR_ARG_CONFIG));
    }
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(writecell_error_with(
            &WRITECELL_ERROR_ARG_CONFIG,
            "writecell: too many output arguments",
        ));
    }
    if requested_outputs.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITECELL_BYTES_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&data)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITECELL_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let path = resolve_path(&rest[0])?;
    let options_args = rest.range(1..)?;
    if !options_args.len().is_multiple_of(2) {
        return Err(writecell_error(&WRITECELL_ERROR_ARG_CONFIG));
    }
    controls::preflight_host_options(options_args.view())?;
    if let Value::Cell(cell) = &data {
        table::ensure_cell_shape(cell)?;
    }
    let owner = TabularFsOwner::capture(Some(&path)).map_err(|source| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: unable to capture filesystem authority",
            source,
        )
    })?;
    let selected = owner.path(path).map_err(|source| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: captured filesystem authority was revoked",
            source,
        )
    })?;
    let options = parse_options(runtime.as_ref(), &mut cooperation, options_args.view()).await?;

    let gathered = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&data),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let table = CellTable::from_value(runtime.as_ref(), &mut cooperation, gathered).await?;

    let bytes_written = match options.resolve_file_type(selected.path())? {
        OutputFileType::DelimitedText => {
            write_delimited_cells(
                runtime.as_ref(),
                &mut cooperation,
                &selected,
                &table,
                &options,
            )
            .await?
        }
        OutputFileType::Spreadsheet => {
            write_spreadsheet_cells(
                runtime.as_ref(),
                &mut cooperation,
                &selected,
                &table,
                &options,
            )
            .await?
        }
    };
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        writecell_error_with_source(
            &WRITECELL_ERROR_IO,
            "writecell: captured filesystem authority was revoked",
            source,
        )
    })?;

    match requested_outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Num(bytes_written as f64)])),
        Some(_) => Err(writecell_error(&WRITECELL_ERROR_ARG_CONFIG)),
        None => Ok(Value::Num(bytes_written as f64)),
    }
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
pub(super) struct RangeStart {
    pub(super) row: usize,
    pub(super) col: usize,
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

#[path = "writecell/controls.rs"]
mod controls;
use controls::*;

#[path = "writecell/table.rs"]
mod table;
pub(super) use table::{scalar_tensor_cell_value, CellTable, CellValue};

#[path = "writecell/writer.rs"]
mod writer;
use writer::*;

#[path = "writecell/xlsx.rs"]
mod xlsx;
#[cfg(test)]
pub(super) use xlsx::build_sheet_xml;
pub(super) use xlsx::{
    append_escaped_xml_text_character, append_sheet_cell_xml, append_sheet_text_end,
    append_sheet_text_start, build_xlsx_workbook, write_xlsx_part, SHEET_XML_FOOTER,
    SHEET_XML_HEADER,
};

fn format_cell_for_text(cell: &CellValue, options: &WriteCellOptions, delimiter: &str) -> String {
    match cell {
        CellValue::Empty => String::new(),
        CellValue::Number(value) => format_text_numeric(runmat_value::NumericScalar::F64(*value)),
        CellValue::Single(value) => format_text_numeric(runmat_value::NumericScalar::F32(*value)),
        CellValue::Integer(value) => value.decimal_string(),
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

fn format_text_numeric(value: runmat_value::NumericScalar) -> String {
    // Preserve this text writer's existing explicit signed-zero normalization.
    if value.is_zero() {
        "0".to_string()
    } else {
        crate::builtins::common::format::format_delimited_numeric(value)
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
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        format!("writecell: expected {context} as a string scalar or character vector")
    })
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
        return normalize_path(&text);
    }
    match value {
        Value::CharArray(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_FILENAME,
            "writecell: expected a 1-by-N character vector for the filename",
        )),
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

pub(super) fn cell_reference(row: usize, col: usize) -> String {
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

pub(super) fn xml_attr_escape(value: &str) -> String {
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
#[path = "writecell/tests.rs"]
mod tests;
