//! MATLAB-compatible `fgetl` builtin for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, FGETL_ERROR_INTERNAL, FGETL_ERROR_INVALID_IDENTIFIER,
    FGETL_ERROR_INVALID_INPUT, FGETL_ERROR_IO, FGETL_INTEGER_ID_EXTENSION,
    FGETL_RESIDENT_ID_EXTENSION, FGETL_SINGLE_ID_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerInputAvailability, FGETL_DESCRIPTOR,
    FGETL_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{
    helpers::{bytes_to_char_array, read_text_line},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fgetl";

fn fgetl_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fgetl_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fgetl_error_with_message(error.message, error)
}

fn fgetl_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fgetl_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FGETL_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fgetl",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::filetext::fgetl"
)]
async fn fgetl_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.first_output()],
        ));
    }
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct FgetlEval {
    line: Value,
}

impl FgetlEval {
    fn new(line: Value) -> Self {
        Self { line }
    }

    fn end_of_file() -> Self {
        Self {
            line: Value::Num(-1.0),
        }
    }

    pub fn first_output(&self) -> Value {
        self.line.clone()
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetlEval> {
    if !rest.is_empty() {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }

    preflight_fid(fid_value)?;
    let fid_host = gather_value(fid_value).await?;
    preflight_fid(&fid_host)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info =
        registry::info_for(fid).ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;
    if !info.access.read {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let mut operation = registry::acquire_file(fid).map_err(|err| match err {
        registry::FileOperationError::InvalidIdentifier => {
            fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER)
        }
        other => fgetl_error_with_detail(&FGETL_ERROR_IO, other.into_io_error().to_string()),
    })?;
    let read = read_text_line(operation.file_mut(), None, BUILTIN_NAME)
        .map_err(|e| fgetl_error_with_detail(&FGETL_ERROR_IO, e.message()))?;
    drop(operation);
    if read.encountered_eof {
        registry::mark_eof_encountered(fid);
    }
    if read.eof_before_any {
        return Ok(FgetlEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_len = read.data.len().saturating_sub(read.terminators.len());
    let line_value = bytes_to_char_array(&read.data[..line_len], &encoding, BUILTIN_NAME)
        .map_err(|e| fgetl_error_with_detail(&FGETL_ERROR_IO, e.message()))?;
    Ok(FgetlEval::new(line_value))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FGETL_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(_) => Ok(()),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FGETL_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FGETL_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_is_logical(handle)
                || handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    != Some(1)
            {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be a numeric scalar",
                ));
            }
            Ok(())
        }
        _ => Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be a numeric scalar",
        )),
    }
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    fn checked_f64_to_i32(n: f64) -> BuiltinResult<i32> {
        if n < i32::MIN as f64 || n > i32::MAX as f64 {
            return Err(fgetl_error_with_detail(
                &FGETL_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            ));
        }
        Ok(n as i32)
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(*n)
        }
        Value::Int(i) => i.try_to_i32().ok_or_else(|| {
            fgetl_error_with_detail(
                &FGETL_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return value.try_to_i32().ok_or_else(|| {
                    fgetl_error_with_detail(
                        &FGETL_ERROR_INVALID_INPUT,
                        "file identifier is out of range",
                    )
                });
            }
            let n = tensor::tensor_value_f64(t, 0);
            if !n.is_finite() {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(n)
        }
        _ => Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be a numeric scalar",
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests;
