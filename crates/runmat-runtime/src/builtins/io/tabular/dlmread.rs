//! MATLAB-compatible `dlmread` builtin for RunMat.
//!
//! `dlmread` predates `readmatrix` but is still widely used in MATLAB
//! codebases for quick numeric imports with custom delimiters. This
//! implementation mirrors MATLAB's zero-based range semantics and
//! accepts character and string delimiters. Numeric delimiter codes remain
//! available as an explicitly classified RunMat extension.

use std::char;
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[cfg(test)]
use runmat_builtins::DLMREAD_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, DLMREAD_COLON_RANGE_EXTENSION, DLMREAD_COMPOSED_RANGE_EXTENSION,
    DLMREAD_ERROR_ARG_CONFIG, DLMREAD_ERROR_DELIMITER, DLMREAD_ERROR_FILENAME,
    DLMREAD_ERROR_FILENAME_EMPTY, DLMREAD_ERROR_INDEX, DLMREAD_ERROR_IO_OPEN,
    DLMREAD_ERROR_IO_READ, DLMREAD_ERROR_NON_NUMERIC_TOKEN, DLMREAD_ERROR_RANGE,
    DLMREAD_ERROR_TENSOR_BUILD, DLMREAD_NUMERIC_DELIMITER_EXTENSION,
    DLMREAD_RESIDENT_ARGUMENT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericScalar, Tensor, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "dlmread";

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};

fn dlmread_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    dlmread_error_with(error, error.message)
}

fn dlmread_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn dlmread_error_with_source<E>(
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
    name = "dlmread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::tabular::dlmread"
)]
async fn dlmread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    if matches!(&path, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DLMREAD_RESIDENT_ARGUMENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.len() > 4 {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_ARG_CONFIG,
            "dlmread: expected dlmread(filename[, delimiter][, row, col][, range])",
        ));
    }
    if rest.iter().any(|value| {
        !matches!(
            value,
            Value::String(_)
                | Value::CharArray(_)
                | Value::StringArray(_)
                | Value::Int(_)
                | Value::Num(_)
                | Value::Tensor(_)
                | Value::GpuTensor(_)
        )
    }) {
        return Err(dlmread_error_with(
            &DLMREAD_ERROR_ARG_CONFIG,
            "dlmread: controls must be text or real numeric values",
        ));
    }
    let known_path = resolve_path(&path)?;
    let prepared_host_options = if rest
        .iter()
        .all(|value| !matches!(value, Value::GpuTensor(_)))
    {
        Some(parse_arguments(runtime.as_ref(), &mut cooperation, &rest).await?)
    } else {
        None
    };
    let owner = TabularFsOwner::capture(Some(&known_path)).map_err(|source| {
        dlmread_error_with_source(
            &DLMREAD_ERROR_IO_OPEN,
            "dlmread: unable to capture filesystem authority",
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
    let options = match prepared_host_options {
        Some(options) => options,
        None => parse_arguments(runtime.as_ref(), &mut cooperation, &rest).await?,
    };
    let resolved = resolve_path(&gathered_path)?;
    let selected = owner.path(resolved).map_err(|source| {
        dlmread_error_with_source(
            &DLMREAD_ERROR_IO_OPEN,
            "dlmread: captured filesystem authority was revoked",
            source,
        )
    })?;
    let (parse_start_row, parse_start_col) = if let Some(range) = options.range {
        (range.start_row, range.start_col)
    } else {
        (options.start_row, options.start_col)
    };
    let (rows, max_cols) = read_dlm_rows(
        runtime.as_ref(),
        &mut cooperation,
        &selected,
        &options.delimiter,
        parse_start_row,
        parse_start_col,
    )
    .await?;
    let subset = if let Some(range) = options.range {
        apply_range(&mut cooperation, &rows, max_cols, &range, 0.0).await?
    } else {
        apply_offsets(
            &mut cooperation,
            &rows,
            max_cols,
            options.start_row,
            options.start_col,
            0.0,
        )
        .await?
    };
    let tensor = rows_to_tensor(
        &mut cooperation,
        subset.rows,
        subset.row_count,
        subset.col_count,
        0.0,
    )
    .await?;
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        dlmread_error_with_source(
            &DLMREAD_ERROR_IO_READ,
            "dlmread: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok(Value::Tensor(tensor))
}

#[path = "dlmread/controls.rs"]
mod controls;
use controls::*;

#[path = "dlmread/reader.rs"]
mod reader;
use reader::*;

#[path = "dlmread/range.rs"]
mod range;
use range::*;

#[path = "dlmread/output.rs"]
mod output;
use output::*;

#[cfg(test)]
#[path = "dlmread/tests/mod.rs"]
pub(crate) mod tests;
