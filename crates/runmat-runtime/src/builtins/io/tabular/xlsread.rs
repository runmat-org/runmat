//! MATLAB-compatible `xlsread` builtin for RunMat.

use std::path::{Path, PathBuf};

#[cfg(test)]
use runmat_builtins::XLSREAD_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, XLSREAD_ERROR_FILENAME, XLSREAD_ERROR_INVALID_ARGUMENT,
    XLSREAD_ERROR_IO, XLSREAD_ERROR_OUTPUT_COUNT, XLSREAD_ERROR_RANGE, XLSREAD_ERROR_SHEET,
    XLSREAD_ERROR_TENSOR_BUILD, XLSREAD_EXPLICIT_GPU_EXTENSION, XLSREAD_NUMERIC_RANGE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};
use crate::builtins::common::fs::home_directory;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "xlsread";

#[runtime_builtin(
    name = "xlsread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::tabular::xlsread"
)]
async fn xlsread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    if crate::builtins::common::validation::value_contains_explicit_gpu(&path)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XLSREAD_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.len() >= 2 && is_numeric_range_value(&rest[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XLSREAD_NUMERIC_RANGE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let outputs = XlsReadOutputs::requested()?;
    preflight_host_arguments(rest.view())?;
    let captured_home = home_directory();
    let known_path = if matches!(path, Value::GpuTensor(_)) {
        None
    } else {
        Some(resolve_path(&path, captured_home.as_deref())?)
    };
    let owner = TabularFsOwner::capture(known_path.as_deref()).map_err(|source| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            "xlsread: unable to capture filesystem authority",
            source,
        )
    })?;
    let path_value = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&path),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let request = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        parse_arguments(runtime.as_ref(), &rest),
    )
    .await??;
    cooperation.completed(0).await?;
    let resolved = resolve_path(&path_value, captured_home.as_deref())?;
    let selected = owner.path(resolved).map_err(|source| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            "xlsread: captured filesystem authority was revoked",
            source,
        )
    })?;
    let result = read_spreadsheet(
        runtime.as_ref(),
        &mut cooperation,
        &selected,
        &request,
        outputs,
    )
    .await?;
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            "xlsread: captured filesystem authority was revoked",
            source,
        )
    })?;
    result.into_requested_value(outputs)
}

fn is_numeric_range_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
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

#[derive(Clone, Debug, Default)]
struct XlsReadRequest {
    sheet: Option<SheetSelector>,
    range: Option<RangeSpec>,
}

async fn checkpoint_cells(
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    work: usize,
) -> BuiltinResult<()> {
    cooperation.completed(work).await?;
    selected.check().map_err(|source| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            "xlsread: captured filesystem authority was revoked",
            source,
        )
    })
}

#[derive(Clone, Debug)]
enum SheetSelector {
    Name(String),
    Index(usize),
}

async fn parse_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<XlsReadRequest> {
    let gathered = gather_arguments(runtime, args).await?;
    let mut request = XlsReadRequest::default();
    match gathered.len() {
        0 => {}
        1 => parse_single_selector(&gathered[0], &mut request)?,
        2 => {
            let (sheet, range) = (&gathered[0], &gathered[1]);
            request.sheet = Some(parse_sheet_selector(sheet)?);
            request.range = Some(parse_range(range)?);
        }
        3 => {
            let (sheet, range, mode) = (&gathered[0], &gathered[1], &gathered[2]);
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

fn preflight_host_arguments(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if args.len() > 3 {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_INVALID_ARGUMENT,
            "xlsread: expected filename, optional sheet, optional range, and optional '-basic'",
        ));
    }
    let host = |value: &Value| !matches!(value, Value::GpuTensor(_));
    match args.len() {
        1 if host(&args[0]) => {
            let selector = &args[0];
            parse_single_selector(selector, &mut XlsReadRequest::default())?;
        }
        2 => {
            let (sheet, range) = (&args[0], &args[1]);
            if host(sheet) {
                parse_sheet_selector(sheet)?;
            }
            if host(range) {
                parse_range(range)?;
            }
        }
        3 => {
            let (sheet, range, mode) = (&args[0], &args[1], &args[2]);
            if host(sheet) {
                parse_sheet_selector(sheet)?;
            }
            if host(range) {
                parse_range(range)?;
            }
            if host(mode) {
                parse_basic_mode(mode)?;
            }
        }
        _ => {}
    }
    Ok(())
}

async fn gather_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        gather_if_needed_async(&args[index])
            .await
            .map_err(map_control_flow)
    })
    .await
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
        Value::Int(i) => integer_sheet_index(i),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                integer_sheet_index(&value)
            } else {
                numeric_sheet_index(tensor::tensor_value_f64(t, 0))
            }
        }
        _ => Err(xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet must be a name or one-based numeric index",
        )),
    }
}

fn integer_sheet_index(value: &IntValue) -> BuiltinResult<SheetSelector> {
    value
        .try_to_usize()
        .and_then(|index| index.checked_sub(1))
        .map(SheetSelector::Index)
        .ok_or_else(|| {
            xlsread_error_with(
                &XLSREAD_ERROR_SHEET,
                "xlsread: sheet index must be one-based",
            )
        })
}

fn numeric_sheet_index(value: f64) -> BuiltinResult<SheetSelector> {
    if !value.is_finite() || value < 1.0 || (value.round() - value).abs() > f64::EPSILON {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet index must be a positive integer",
        ));
    }
    let rounded = value.round();
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet index is too large",
        ));
    }
    let index = rounded as usize;
    let zero_based = index.checked_sub(1).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_SHEET,
            "xlsread: sheet index must be one-based",
        )
    })?;
    Ok(SheetSelector::Index(zero_based))
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_INVALID_ARGUMENT,
            "xlsread: expected a string scalar or character vector",
        )
    })
}

fn resolve_path(value: &Value, captured_home: Option<&Path>) -> BuiltinResult<PathBuf> {
    let text = value_to_string_scalar(value).map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_FILENAME,
            "xlsread: filename must be a string scalar or character vector",
        )
    })?;
    let path = normalize_spreadsheet_path(&text)?;
    if path == "~" || path.starts_with("~/") || path.starts_with("~\\") {
        let home = captured_home.ok_or_else(|| {
            xlsread_error_with(
                &XLSREAD_ERROR_FILENAME,
                "xlsread: unable to resolve home directory",
            )
        })?;
        let suffix = path.trim_start_matches('~').trim_start_matches(['/', '\\']);
        return Ok(home.join(suffix));
    }
    Ok(PathBuf::from(path))
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

mod range;
#[cfg(test)]
use range::MAX_EXCEL_ROW_INDEX;
use range::{parse_range, parse_range_string, text_looks_like_range, RangeSpec};
mod output;
use output::{XlsReadOutputs, XlsReadResult};
mod workbook;
use workbook::read_spreadsheet;
#[cfg(test)]
use workbook::{
    preflight_zip_workbook, validate_zip_entry_count, validate_zip_uncompressed_limits,
    MAX_XLSREAD_ZIP_ENTRIES, MAX_XLSREAD_ZIP_ENTRY_UNCOMPRESSED_BYTES,
};

#[cfg(test)]
#[path = "xlsread/tests.rs"]
mod tests;
#[cfg(test)]
#[path = "xlsread/tests/cooperation.rs"]
mod cooperation_tests;
