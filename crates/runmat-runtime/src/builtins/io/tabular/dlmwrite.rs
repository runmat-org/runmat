//! MATLAB-compatible `dlmwrite` builtin for delimiter-separated exports.

#[cfg(test)]
use runmat_builtins::DLMWRITE_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, DLMWRITE_BYTE_COUNT_OUTPUT_EXTENSION, DLMWRITE_ERROR_ARG_CONFIG,
    DLMWRITE_ERROR_DATA, DLMWRITE_ERROR_DATA_SHAPE, DLMWRITE_ERROR_FILENAME, DLMWRITE_ERROR_FORMAT,
    DLMWRITE_ERROR_IO, DLMWRITE_ERROR_OPTION, DLMWRITE_ERROR_SIZE,
    DLMWRITE_RESIDENT_INPUT_EXTENSION,
};
use runmat_filesystem::{File, OpenOptions};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "dlmwrite";
const MAX_NUMERIC_FORMAT_FIELD: usize = 4096;
const MAX_DLMWRITE_LAYOUT_FIELDS: usize = 50_000_000;

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};

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
    name = "dlmwrite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::tabular::dlmwrite"
)]
async fn dlmwrite_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 1) {
        return Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_ARG_CONFIG,
            "dlmwrite: too many output arguments",
        ));
    }
    if outputs.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DLMWRITE_BYTE_COUNT_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(&filename, Value::GpuTensor(_))
        || matches!(&data, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DLMWRITE_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_data(&data)?;
    let known_path = resolve_path(&filename)?;
    let prepared_host_options = if rest
        .iter()
        .all(|value| !matches!(value, Value::GpuTensor(_)))
    {
        Some(parse_arguments(rest.view())?)
    } else {
        None
    };
    let owner = TabularFsOwner::capture(Some(&known_path)).map_err(|source| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            "dlmwrite: unable to capture filesystem authority",
            source,
        )
    })?;
    let gathered_path = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&filename),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let path = owner
        .path(resolve_path(&gathered_path)?)
        .map_err(|source| {
            dlmwrite_error_with_source(
                &DLMWRITE_ERROR_IO,
                "dlmwrite: captured filesystem authority was revoked",
                source,
            )
        })?;

    let gathered_args = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
            let value = await_io(
                runtime.as_ref(),
                BUILTIN_NAME,
                gather_if_needed_async(&rest[index]),
            )
            .await?
            .map_err(map_control_flow)?;
            cooperation.completed(0).await?;
            Ok(value)
        })
        .await?;
    let options = match prepared_host_options {
        Some(options) => options,
        None => parse_arguments(gathered_args.view())?,
    };

    let gathered_data = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&data),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let tensor = tensor::value_into_tensor_for("dlmwrite", gathered_data)
        .map_err(|msg| dlmwrite_error_with(&DLMWRITE_ERROR_DATA, format!("dlmwrite: {msg}")))?;
    ensure_matrix_shape(&tensor)?;

    let bytes = write_dlm(runtime.as_ref(), &mut cooperation, &path, &tensor, &options).await?;
    cooperation.completed(0).await?;
    path.check().map_err(|source| {
        dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            "dlmwrite: captured filesystem authority was revoked",
            source,
        )
    })?;
    match outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Num(bytes as f64)])),
        Some(_) => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_ARG_CONFIG,
            "dlmwrite: too many output arguments",
        )),
        None => Ok(Value::Num(bytes as f64)),
    }
}

fn preflight_data(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(()),
        Value::Tensor(tensor) => ensure_matrix_shape(tensor),
        Value::GpuTensor(handle) => {
            if handle.physical().storage()
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_DATA,
                    "dlmwrite: complex input is not supported",
                ));
            }
            if handle.shape().iter().skip(2).any(|extent| *extent != 1) {
                return Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_DATA_SHAPE,
                    "dlmwrite: input must be 2-D",
                ));
            }
            Ok(())
        }
        Value::LogicalArray(array) => {
            if array.shape.iter().skip(2).any(|extent| *extent != 1) {
                Err(dlmwrite_error_with(
                    &DLMWRITE_ERROR_DATA_SHAPE,
                    "dlmwrite: input must be 2-D",
                ))
            } else {
                Ok(())
            }
        }
        _ => Err(dlmwrite_error_with(
            &DLMWRITE_ERROR_DATA,
            "dlmwrite: input must be real numeric or logical",
        )),
    }
}

#[path = "dlmwrite/controls.rs"]
mod controls;
use controls::*;

#[path = "dlmwrite/writer.rs"]
mod writer;
use writer::*;

#[path = "dlmwrite/output.rs"]
mod output;

#[path = "dlmwrite/format.rs"]
mod format;
use format::*;

#[cfg(test)]
#[path = "dlmwrite/tests/mod.rs"]
pub(crate) mod tests;
