//! MATLAB-compatible `frewind` builtin for RunMat.
//!
//! Rewinds the file position indicator to the beginning of an open file.
//! Equivalent to `fseek(fid, 0, 'bof')`.

use std::io::{Seek, SeekFrom};

use runmat_builtins::{
    BuiltinErrorDescriptor, FREWIND_ERROR_INTERNAL, FREWIND_ERROR_INVALID_IDENTIFIER,
    FREWIND_ERROR_INVALID_INPUT, FREWIND_ERROR_IO, FREWIND_INTEGER_ID_EXTENSION,
    FREWIND_LOGICAL_ID_EXTENSION, FREWIND_RESIDENT_ID_EXTENSION, FREWIND_SINGLE_ID_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{FREWIND_DESCRIPTOR, FREWIND_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const IDENTIFIER_TYPE_ERROR_DETAIL: &str = "file identifier must be a numeric scalar";
const BUILTIN_NAME: &str = "frewind";

fn frewind_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    frewind_error_with_message(error.message, error)
}

fn frewind_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    frewind_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn frewind_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn frewind_error_with_source(
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

fn frewind_error_with_source_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    frewind_error_with_source(
        format!("{}: {}", error.message, detail.as_ref()),
        error,
        source,
    )
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let message = err.message().to_string();
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {message}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FREWIND_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "frewind",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::frewind"
)]
async fn frewind_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    fid: Value,
) -> crate::BuiltinResult<Value> {
    evaluate_in(runtime.as_ref(), &fid).await?;
    Ok(Value::output_list(Vec::new()))
}

/// Evaluate the `frewind` builtin without invoking the runtime dispatcher.
pub async fn evaluate(fid_value: &Value) -> BuiltinResult<()> {
    evaluate_in(None, fid_value).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    fid_value: &Value,
) -> BuiltinResult<()> {
    preflight_fid(fid_value)?;
    let fid_host =
        super::cooperation::await_io(runtime, BUILTIN_NAME, gather_if_needed_async(fid_value))
            .await?
            .map_err(map_control_flow)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Ok(());
    }

    let mut operation = registry::acquire_file_in(fid, runtime).map_err(|err| match err {
        registry::FileOperationError::InvalidIdentifier
        | registry::FileOperationError::ForeignSession => {
            frewind_error(&FREWIND_ERROR_INVALID_IDENTIFIER)
        }
        other => {
            let source = other.into_io_error();
            frewind_error_with_source_detail(&FREWIND_ERROR_IO, source.to_string(), source)
        }
    })?;

    operation
        .check_authority(runtime)
        .map_err(|err| frewind_error_with_source_detail(&FREWIND_ERROR_IO, err.to_string(), err))?;
    super::cooperation::checkpoint(runtime, BUILTIN_NAME).await?;
    operation
        .file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(|err| {
            frewind_error_with_source_detail(
                &FREWIND_ERROR_IO,
                format!("failed to rewind file: {err}"),
                err,
            )
        })?;
    operation
        .check_authority(runtime)
        .map_err(|err| frewind_error_with_source_detail(&FREWIND_ERROR_IO, err.to_string(), err))?;
    super::cooperation::checkpoint(runtime, BUILTIN_NAME).await?;
    drop(operation);
    registry::clear_eof_encountered_in(fid, runtime);

    Ok(())
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FREWIND_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREWIND_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREWIND_LOGICAL_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREWIND_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    != Some(1)
            {
                return Err(frewind_error_with_detail(
                    &FREWIND_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Num(n) => parse_scalar_fid(*n),
        Value::Int(int) => int.try_to_i32().ok_or_else(|| {
            frewind_error_with_detail(
                &FREWIND_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(storage) = t.integer_storage() {
                    let value = storage.value_at(0).expect("one-element integer storage");
                    return value.try_to_i32().ok_or_else(|| {
                        frewind_error_with_detail(
                            &FREWIND_ERROR_INVALID_INPUT,
                            "file identifier is out of range",
                        )
                    });
                }
                parse_scalar_fid(tensor::tensor_value_f64(t, 0))
            } else {
                Err(frewind_error_with_detail(
                    &FREWIND_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ))
            }
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let v = if la.data[0] != 0 { 1.0 } else { 0.0 };
            parse_scalar_fid(v)
        }
        Value::LogicalArray(_) => Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
        Value::Bool(b) => parse_scalar_fid(if *b { 1.0 } else { 0.0 }),
        _ => Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
    }
}

fn parse_scalar_fid(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

#[cfg(test)]
pub(crate) mod tests;
