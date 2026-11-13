//! MATLAB-compatible `csvread` builtin for RunMat.
//!
//! `csvread` is largely superseded by `readmatrix`, but MATLAB users still rely on
//! its terse API for numeric CSV imports. This implementation mirrors MATLAB's
//! zero-based range semantics while integrating with the modern builtin template.

use std::io::BufReader;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinErrorDescriptor, CSVREAD_COLON_RANGE_EXTENSION, CSVREAD_ERROR_ARG_CONFIG,
    CSVREAD_ERROR_FILENAME, CSVREAD_ERROR_FILENAME_EMPTY, CSVREAD_ERROR_INDEX,
    CSVREAD_ERROR_IO_OPEN, CSVREAD_ERROR_IO_READ, CSVREAD_ERROR_NON_NUMERIC_TOKEN,
    CSVREAD_ERROR_RANGE, CSVREAD_ERROR_TENSOR_BUILD, CSVREAD_RESIDENT_CONTROL_EXTENSION,
    CSVREAD_TWO_VECTOR_RANGE_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{CSVREAD_DESCRIPTOR, CSVREAD_EXTENSIONS, CSVREAD_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "csvread";

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};

fn csvread_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn csvread_error_with_source<E>(
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
    name = "csvread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::tabular::csvread"
)]
async fn csvread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    if matches!(path, Value::GpuTensor(_)) || rest.iter().any(|v| matches!(v, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CSVREAD_RESIDENT_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_host_controls(rest.view())?;
    let known_path = if matches!(path, Value::GpuTensor(_)) {
        None
    } else {
        Some(resolve_path(&path)?)
    };
    let owner = TabularFsOwner::capture(known_path.as_deref()).map_err(|source| {
        csvread_error_with_source(
            &CSVREAD_ERROR_IO_OPEN,
            "csvread: unable to capture filesystem authority",
            source,
        )
    })?;
    let gathered_path = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&path),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let options = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        parse_arguments(runtime.as_ref(), &rest),
    )
    .await??;
    cooperation.completed(0).await?;
    let resolved = resolve_path(&gathered_path)?;
    let selected = owner.path(resolved).map_err(|source| {
        csvread_error_with_source(
            &CSVREAD_ERROR_IO_OPEN,
            "csvread: captured filesystem authority was revoked",
            source,
        )
    })?;
    let (rows, max_cols, skipped_rows) =
        read_csv_rows(runtime.as_ref(), &mut cooperation, &selected, &options).await?;
    let start_row = if options.range.is_none() {
        options.start_row.saturating_sub(skipped_rows)
    } else {
        options.start_row
    };
    let subset = if let Some(range) = options.range {
        let loaded_range = range.relative_to_loaded_rows();
        apply_range(
            &mut cooperation,
            &rows,
            max_cols,
            &loaded_range,
            ParsedNumber::ZERO,
        )
        .await?
    } else {
        apply_offsets(
            &mut cooperation,
            &rows,
            max_cols,
            start_row,
            options.start_col,
            ParsedNumber::ZERO,
        )
        .await?
    };
    let value = rows_to_value(
        &mut cooperation,
        subset.rows,
        subset.row_count,
        subset.col_count,
        ParsedNumber::ZERO,
    )
    .await?;
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        csvread_error_with_source(
            &CSVREAD_ERROR_IO_READ,
            "csvread: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok(value)
}

fn preflight_host_controls(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if !matches!(rest.len(), 0 | 2 | 3) {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_ARG_CONFIG,
            "csvread: expected filename, optional row and col, and optional range",
        ));
    }
    for (index, name) in [(0, "row"), (1, "col")] {
        if let Some(value) = rest.get(index) {
            if !matches!(value, Value::GpuTensor(_)) {
                value_to_start_index(value, name)?;
            }
        }
    }
    if let Some(range) = rest.get(2) {
        if !matches!(range, Value::GpuTensor(_)) {
            parse_range(range)?;
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct CsvReadOptions {
    start_row: usize,
    start_col: usize,
    range: Option<RangeSpec>,
}

async fn parse_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<CsvReadOptions> {
    let gathered = args
        .try_map_preserving_roots_async(args.len(), runtime, async |index| {
            gather_if_needed_async(&args[index])
                .await
                .map_err(map_control_flow)
        })
        .await?;
    match gathered.len() {
        0 => Ok(CsvReadOptions::default()),
        2 => {
            let start_row = value_to_start_index(&gathered[0], "row")?;
            let start_col = value_to_start_index(&gathered[1], "col")?;
            Ok(CsvReadOptions {
                start_row,
                start_col,
                range: None,
            })
        }
        3 => {
            let start_row = value_to_start_index(&gathered[0], "row")?;
            let start_col = value_to_start_index(&gathered[1], "col")?;
            let range = parse_range(&gathered[2])?;
            Ok(CsvReadOptions {
                start_row,
                start_col,
                range: Some(range),
            })
        }
        _ => Err(csvread_error_with(
            &CSVREAD_ERROR_ARG_CONFIG,
            "csvread: expected csvread(filename[, row, col[, range]])",
        )),
    }
}

fn value_to_start_index(value: &Value, name: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            csvread_error_with(
                &CSVREAD_ERROR_INDEX,
                format!("csvread: {name} must be a non-negative integer"),
            )
        });
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(csvread_error_with(
                    &CSVREAD_ERROR_INDEX,
                    format!("csvread: {name} must be a finite integer"),
                ));
            }
            if *n < 0.0 {
                return Err(csvread_error_with(
                    &CSVREAD_ERROR_INDEX,
                    format!("csvread: {name} must be a non-negative integer"),
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(csvread_error_with(
                    &CSVREAD_ERROR_INDEX,
                    format!("csvread: {name} must be an integer"),
                ));
            }
            if rounded > usize::MAX.saturating_sub(1) as f64 {
                return Err(csvread_error_with(
                    &CSVREAD_ERROR_INDEX,
                    format!("csvread: {name} is too large"),
                ));
            }
            let parsed = rounded as usize;
            if parsed as f64 != rounded || parsed == usize::MAX {
                return Err(csvread_error_with(
                    &CSVREAD_ERROR_INDEX,
                    format!("csvread: {name} is too large"),
                ));
            }
            Ok(parsed)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            value_to_start_index(&Value::Num(tensor::tensor_value_f64(t, 0)), name)
        }
        _ => Err(csvread_error_with(
            &CSVREAD_ERROR_INDEX,
            format!("csvread: expected {name} as a numeric scalar, got {value:?}"),
        )),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
        return normalize_path(&text);
    }
    match value {
        Value::StringArray(_) => Err(csvread_error_with(
            &CSVREAD_ERROR_FILENAME,
            "csvread: string array inputs must be scalar",
        )),
        Value::CharArray(_) => Err(csvread_error_with(
            &CSVREAD_ERROR_FILENAME,
            "csvread: expected a 1-by-N character vector for the file name",
        )),
        other => Err(csvread_error_with(
            &CSVREAD_ERROR_FILENAME,
            format!(
                "csvread: expected filename as string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(csvread_error_with(
            &CSVREAD_ERROR_FILENAME_EMPTY,
            CSVREAD_ERROR_FILENAME_EMPTY.message,
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| csvread_error_with(&CSVREAD_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

#[path = "csvread/reader.rs"]
mod reader;
use reader::*;

#[path = "csvread/number.rs"]
mod number;
use number::*;

#[path = "csvread/range.rs"]
mod range;
use range::*;

#[path = "csvread/output.rs"]
mod output;
use output::*;

#[cfg(test)]
#[path = "csvread/tests.rs"]
pub(crate) mod tests;
