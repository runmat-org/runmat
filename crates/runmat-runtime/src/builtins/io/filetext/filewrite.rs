//! RunMat-native `filewrite` builtin.

use std::io::Write;

use runmat_builtins::{
    BuiltinErrorDescriptor, FILEWRITE_ERROR_ENCODE, FILEWRITE_ERROR_INTERNAL,
    FILEWRITE_ERROR_INVALID_DATA, FILEWRITE_ERROR_INVALID_INPUT, FILEWRITE_ERROR_INVALID_OPTION,
    FILEWRITE_ERROR_IO, FILEWRITE_NATIVE_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerBackendRule, FILEWRITE_DESCRIPTOR, FILEWRITE_EXTENSIONS,
    FILEWRITE_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::OpenOptions;

const BUILTIN_NAME: &str = "filewrite";

fn filewrite_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filewrite_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn filewrite_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn filewrite_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FILEWRITE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

mod options;
mod payload;
mod target;

use options::{parse_options, resolve_path, WriteMode};
use payload::prepare_payload;
#[cfg(test)]
use payload::tensor_to_bytes;
use target::OutputTarget;

#[runtime_builtin(
    name = "filewrite",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::filetext::filewrite"
)]
async fn filewrite_builtin(
    path: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &FILEWRITE_NATIVE_EXTENSION,
        BUILTIN_NAME,
    )?;
    // Path and option forms are host text only. Reject them, and data classes
    // known to be invalid, before any resident data is materialized.
    let options = parse_options(rest.view())?;
    let resolved = resolve_path(&path)?;
    preflight_data_kind(&data)?;
    let target = OutputTarget::capture(resolved)?;
    let data = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let payload = prepare_payload(&data, options.encoding)?;
    let written = write_bytes(&target, &payload, options.write_mode).await?;
    Ok(Value::Num(written as f64))
}

fn preflight_data_kind(data: &Value) -> BuiltinResult<()> {
    match data {
        Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => Ok(()),
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            Ok(())
        }
        _ => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            "expected text or real uint8-compatible numeric data",
        )),
    }
}

async fn write_bytes(
    target: &OutputTarget,
    payload: &[u8],
    mode: WriteMode,
) -> BuiltinResult<usize> {
    let mut options = OpenOptions::new();
    options.create(true);
    match mode {
        WriteMode::Overwrite => {
            options.write(true).truncate(true);
        }
        WriteMode::Append => {
            options.write(true).append(true);
        }
    }

    let mut file = target.open(&options).await.map_err(|err| {
        filewrite_error_with_source(
            format!(
                "{}: unable to open '{}': {}",
                FILEWRITE_ERROR_IO.message,
                target.path().display(),
                err
            ),
            &FILEWRITE_ERROR_IO,
            err,
        )
    })?;

    file.write_all(payload).map_err(|err| {
        filewrite_error_with_source(
            format!(
                "{}: unable to write to '{}': {}",
                FILEWRITE_ERROR_IO.message,
                target.path().display(),
                err
            ),
            &FILEWRITE_ERROR_IO,
            err,
        )
    })?;

    file.flush_async().await.map_err(|err| {
        filewrite_error_with_source(
            format!(
                "{}: unable to flush '{}': {}",
                FILEWRITE_ERROR_IO.message,
                target.path().display(),
                err
            ),
            &FILEWRITE_ERROR_IO,
            err,
        )
    })?;

    Ok(payload.len())
}

#[cfg(test)]
pub(crate) mod tests;
