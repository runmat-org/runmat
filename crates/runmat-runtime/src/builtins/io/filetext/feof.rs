//! MATLAB-compatible `feof` builtin for RunMat.
//!
//! Provides end-of-file detection for file identifiers opened via `fopen`.
//! The implementation mirrors MATLAB semantics for host-side files and
//! integrates with the shared registry used by the other text I/O builtins.

#[cfg(test)]
use runmat_builtins::FEOF_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, FEOF_ERROR_INTERNAL, FEOF_ERROR_INVALID_IDENTIFIER,
    FEOF_ERROR_INVALID_INPUT, FEOF_INTEGER_ID_EXTENSION, FEOF_LOGICAL_ID_EXTENSION,
    FEOF_RESIDENT_ID_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const IDENTIFIER_TYPE_ERROR_DETAIL: &str = "file identifier must be a numeric scalar";
const BUILTIN_NAME: &str = "feof";

fn feof_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn feof_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    feof_error_with_message(error.message, error)
}

fn feof_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    feof_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FEOF_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "feof",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::feof"
)]
async fn feof_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    fid: Value,
) -> crate::BuiltinResult<Value> {
    let at_end = evaluate_in(runtime.as_ref(), &fid).await?;
    Ok(Value::Bool(at_end))
}

/// Evaluate the `feof` builtin without invoking the runtime dispatcher.
pub async fn evaluate(fid_value: &Value) -> BuiltinResult<bool> {
    evaluate_in(None, fid_value).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    fid_value: &Value,
) -> BuiltinResult<bool> {
    preflight_fid(fid_value)?;
    let fid_host =
        super::cooperation::await_io(runtime, BUILTIN_NAME, gather_if_needed_async(fid_value))
            .await?
            .map_err(map_control_flow)?;
    preflight_fid(&fid_host)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Ok(false);
    }

    registry::eof_encountered_in(fid, runtime)
        .ok_or_else(|| feof_error(&FEOF_ERROR_INVALID_IDENTIFIER))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FEOF_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => Ok(()),
        Value::Tensor(_) => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer-valued double scalar",
        )),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FEOF_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FEOF_LOGICAL_ID_EXTENSION,
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
                return Err(feof_error_with_detail(
                    &FEOF_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ));
            }
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    != Some(runmat_accelerate_api::ProviderPrecision::F64)
            {
                return Err(feof_error_with_detail(
                    &FEOF_ERROR_INVALID_INPUT,
                    "file identifier must be an integer-valued double scalar",
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
            feof_error_with_detail(&FEOF_ERROR_INVALID_INPUT, "file identifier is out of range")
        }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(storage) = t.integer_storage() {
                    let value = storage.value_at(0).expect("one-element integer storage");
                    return value.try_to_i32().ok_or_else(|| {
                        feof_error_with_detail(
                            &FEOF_ERROR_INVALID_INPUT,
                            "file identifier is out of range",
                        )
                    });
                }
                if t.numeric_dtype() != NumericDType::F64 {
                    return Err(feof_error_with_detail(
                        &FEOF_ERROR_INVALID_INPUT,
                        "file identifier must be an integer-valued double scalar",
                    ));
                }
                parse_scalar_fid(tensor::tensor_value_f64(t, 0))
            } else {
                Err(feof_error_with_detail(
                    &FEOF_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ))
            }
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let v = if la.data[0] != 0 { 1.0 } else { 0.0 };
            parse_scalar_fid(v)
        }
        Value::LogicalArray(_) => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
        Value::Bool(b) => parse_scalar_fid(if *b { 1.0 } else { 0.0 }),
        _ => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
    }
}

fn parse_scalar_fid(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    if value.fract() != 0.0 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    let rounded = value;
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

#[cfg(test)]
pub(crate) mod tests;
