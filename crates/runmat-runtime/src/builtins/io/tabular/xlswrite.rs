//! MATLAB-compatible `xlswrite` builtin for legacy spreadsheet export.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto_from_rs, Data as SpreadsheetData, Reader as SpreadsheetReader};
#[cfg(test)]
use runmat_builtins::XLSWRITE_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, XLSWRITE_ERROR_DATA, XLSWRITE_ERROR_FILENAME,
    XLSWRITE_ERROR_INVALID_ARGUMENT, XLSWRITE_ERROR_IO, XLSWRITE_ERROR_OUTPUT_COUNT,
    XLSWRITE_ERROR_RANGE, XLSWRITE_ERROR_SHEET, XLSWRITE_EXPLICIT_GPU_EXTENSION,
    XLSWRITE_NUMERIC_RANGE_EXTENSION,
};
use runmat_filesystem::FsPublicationDisposition;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, IntValue, LogicalArray, SparseTensor, StringArray, Tensor, Value,
};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};
use super::write_owner;
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

#[runtime_builtin(
    name = "xlswrite",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::tabular::xlswrite"
)]
async fn xlswrite_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_OUTPUT_COUNT,
            "xlswrite: supports at most two output arguments",
        ));
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&filename)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&data)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XLSWRITE_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.len() >= 2 && is_numeric_range_value(&rest[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XLSWRITE_NUMERIC_RANGE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    match write_spreadsheet(runtime.as_ref(), filename, data, rest).await {
        Ok(_) => Ok(status_outputs(true, None)),
        Err(err)
            if captures_failure_as_outputs()
                && !is_cancelled(&err)
                && !matches!(
                    write_owner::publication_error_disposition(&err),
                    Some(
                        FsPublicationDisposition::Published
                            | FsPublicationDisposition::Indeterminate
                    )
                ) =>
        {
            Ok(status_outputs(false, Some(&err)))
        }
        Err(err) => Err(err),
    }
}

fn is_cancelled(error: &RuntimeError) -> bool {
    let mut current: &(dyn std::error::Error + 'static) = error;
    loop {
        if matches!(
            current.downcast_ref::<crate::execution::ExecutionServiceError>(),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ) {
            return true;
        }
        let Some(source) = current.source() else {
            return false;
        };
        current = source;
    }
}

fn is_numeric_range_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
}

async fn write_spreadsheet(
    runtime: Option<&crate::context::RuntimeContext>,
    filename: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<usize> {
    let mut cooperation = TabularCooperation::new(runtime, BUILTIN_NAME);
    let path = resolve_path(&filename)?;
    XlsWriteRequest::preflight_host_arguments(rest.view())?;
    XlsTable::preflight_host_known(&data, &mut cooperation).await?;
    let owner = TabularFsOwner::capture(Some(&path)).map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: unable to capture filesystem authority",
            source,
        )
    })?;
    let selected = owner.path(path).map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })?;
    let data_value = if matches!(&data, Value::GpuTensor(_)) {
        selected.check().map_err(|source| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                "xlswrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        await_io(runtime, BUILTIN_NAME, gather_if_needed_async(&data))
            .await?
            .map_err(map_control_flow)?
    } else {
        data
    };
    let args = gather_arguments(runtime, &mut cooperation, &rest).await?;
    let request = XlsWriteRequest::parse(args.view())?;
    let table =
        XlsTable::from_value_cooperative(data_value, runtime, &mut cooperation, &selected).await?;
    let lock = await_io(
        runtime,
        BUILTIN_NAME,
        write_owner::write_lock_for_path(&selected),
    )
    .await?
    .map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: write authority unavailable",
            source,
        )
    })?;
    let _guard = await_io(runtime, BUILTIN_NAME, lock.lock()).await?;
    let mut workbook = await_io(
        runtime,
        BUILTIN_NAME,
        load_workbook_or_new(&selected, &mut cooperation),
    )
    .await??;
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })?;
    workbook
        .overlay(&request, table, &mut cooperation, &selected)
        .await?;
    let bytes = workbook
        .into_xlsx_bytes(&mut cooperation, &selected)
        .await?;
    write_owner::safe_replace_file(runtime, &mut cooperation, BUILTIN_NAME, &selected, &bytes)
        .await
        .map_err(|source| match source {
            write_owner::StagedWriteError::Control(error) => error,
            write_owner::StagedWriteError::PublicationInterrupted(error, disposition) => {
                write_owner::publication_interruption_error(error, disposition)
            }
            write_owner::StagedWriteError::Publication(source) => xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                "xlswrite: staged publication failed or its effect could not be confirmed",
                source,
            ),
            write_owner::StagedWriteError::Io(source) => xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                "xlswrite: failed to replace spreadsheet file",
                source,
            ),
        })?;
    Ok(bytes.len())
}

async fn gather_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let gathered = args
        .try_map_preserving_roots_async(args.len(), runtime, async |index| {
            let value = await_io(runtime, BUILTIN_NAME, gather_if_needed_async(&args[index]))
                .await?
                .map_err(map_control_flow)?;
            cooperation.completed(0).await?;
            Ok(value)
        })
        .await?;
    Ok(gathered)
}

fn captures_failure_as_outputs() -> bool {
    matches!(crate::output_count::current_output_count(), Some(1 | 2))
}

fn status_outputs(success: bool, err: Option<&RuntimeError>) -> Value {
    match crate::output_count::current_output_count() {
        Some(0) => Value::output_list(Vec::new()),
        Some(1) => Value::output_list(vec![Value::Bool(success)]),
        Some(2) => Value::output_list(vec![Value::Bool(success), message_value(err)]),
        None => Value::Bool(success),
        Some(_) => Value::Bool(success),
    }
}

fn message_value(err: Option<&RuntimeError>) -> Value {
    let mut message = runmat_value::UnboundedStructValueBuilder::new();
    message.insert(
        "message",
        Value::string(err.map(|e| e.message().to_string()).unwrap_or_default()),
    );
    message.insert(
        "identifier",
        Value::string(
            err.and_then(RuntimeError::identifier)
                .unwrap_or("")
                .to_string(),
        ),
    );
    Value::Struct(message.finish())
}

#[path = "xlswrite/controls.rs"]
mod controls;
use controls::*;
#[path = "xlswrite/workbook.rs"]
mod workbook;
use workbook::*;
#[path = "xlswrite/table.rs"]
mod table;
use table::*;

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
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_INVALID_ARGUMENT,
            "xlswrite: expected a string scalar or character vector",
        )
    })
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

fn map_writecell_error(err: RuntimeError) -> RuntimeError {
    let detail = err.message().to_string();
    let mut builder = build_runtime_error(format!("xlswrite: {detail}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = XLSWRITE_ERROR_IO.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "xlswrite/tests.rs"]
mod tests;
