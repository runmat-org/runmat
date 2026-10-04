//! MATLAB-compatible `fprintf` builtin enabling formatted text output to files and standard streams.

use std::io::Write;

#[cfg(test)]
use runmat_builtins::FPRINTF_DESCRIPTOR;
#[cfg(test)]
use runmat_builtins::FPRINTF_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES;
use runmat_builtins::{
    BuiltinErrorDescriptor, FPRINTF_ERROR_ENCODE, FPRINTF_ERROR_FORMAT, FPRINTF_ERROR_INTERNAL,
    FPRINTF_ERROR_INVALID_IDENTIFIER, FPRINTF_ERROR_INVALID_INPUT, FPRINTF_ERROR_IO,
    FPRINTF_INTEGER_FORMAT_EXTENSION, FPRINTF_INTEGER_ID_EXTENSION,
    FPRINTF_NUMERIC_FORMAT_EXTENSION, FPRINTF_RESIDENT_FORMAT_EXTENSION,
    FPRINTF_RESIDENT_ID_EXTENSION, FPRINTF_SINGLE_ID_EXTENSION, FPRINTF_STREAM_LABEL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Value};

use crate::builtins::common::format::{
    decode_escape_sequences, flatten_arguments, format_variadic_with_cursor, ArgCursor,
};
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry::{self, FileInfo, SharedFileHandle};
use crate::console::{record_console_output, ConsoleStream};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fprintf";

mod admission;
mod encoding;
mod target;

use admission::{
    coerce_single_integer_string_argument, coerce_to_format_string, preflight_special_roles,
};
use encoding::encode_output;
use target::{match_stream_label, parse_fid, target_from_fid, OutputTarget, SpecialStream};

fn fprintf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fprintf_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fprintf_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fprintf_error_with_source(
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
    if let Some(identifier) = FPRINTF_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(
    result: Result<T, String>,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    result.map_err(|message| fprintf_error_with_detail(error, message))
}

/// Result of evaluating `fprintf`.
#[derive(Debug)]
pub struct FprintfEval {
    bytes_written: usize,
}

impl FprintfEval {
    /// Number of bytes emitted by the write.
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }
}

/// Evaluate the `fprintf` builtin without going through the dispatcher.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<FprintfEval> {
    if args.is_empty() {
        return Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_INPUT,
            "not enough input arguments",
        ));
    }

    preflight_special_roles(args)?;
    // Gather all arguments to host first
    let mut all: Vec<Value> = Vec::with_capacity(args.len());
    for v in args {
        all.push(gather_value(v).await?);
    }

    // Locate the first valid formatSpec anywhere in the list
    let mut fmt_idx: Option<usize> = None;
    let mut format_string_val: Option<String> = None;
    for (i, value) in all.iter().enumerate() {
        // Never interpret a stream label ('stdout'/'stderr') as the format string
        if match_stream_label(value).is_some() {
            continue;
        }
        if let Some(Value::String(s)) =
            map_string_result(coerce_to_format_string(value), &FPRINTF_ERROR_INVALID_INPUT)?
        {
            fmt_idx = Some(i);
            format_string_val = Some(s);
            break;
        }
    }
    let fmt_idx = fmt_idx.ok_or_else(|| {
        fprintf_error_with_detail(&FPRINTF_ERROR_INVALID_INPUT, "missing format string")
    })?;
    let raw_format = format_string_val.unwrap();

    // Determine output target by scanning only arguments BEFORE the format
    let mut target_idx: Option<usize> = None;
    let mut target: OutputTarget = OutputTarget::Stdout;
    // Prefer explicit stream labels over numeric fids if both appear
    let mut first_stream: Option<(usize, SpecialStream)> = None;
    for (i, value) in all.iter().enumerate().take(fmt_idx) {
        if let Some(stream) = match_stream_label(value) {
            first_stream = Some((i, stream));
            break;
        }
    }
    if let Some((idx, stream)) = first_stream {
        target_idx = Some(idx);
        target = match stream {
            SpecialStream::Stdout => OutputTarget::Stdout,
            SpecialStream::Stderr => OutputTarget::Stderr,
        };
    } else {
        // Try to parse a numeric fid that appears before the format
        for (i, value) in all.iter().enumerate().take(fmt_idx) {
            if matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
                if let Ok(fid) = parse_fid(value) {
                    target_idx = Some(i);
                    target = target_from_fid(fid)?;
                    break;
                }
            }
        }
    }

    // Remaining arguments are data, excluding the chosen target and the format
    let mut data_args: Vec<Value> = Vec::with_capacity(all.len().saturating_sub(1));
    for (i, v) in all.into_iter().enumerate() {
        if i == fmt_idx {
            continue;
        }
        if let Some(tidx) = target_idx {
            if i == tidx {
                continue;
            }
        }
        data_args.push(v);
    }

    let format_string =
        decode_escape_sequences("fprintf", &raw_format).map_err(map_control_flow)?;
    let data_args = coerce_single_integer_string_argument(&format_string, data_args)?;
    let flattened_args = flatten_arguments(&data_args, "fprintf")
        .await
        .map_err(map_control_flow)?;
    let rendered = format_with_repetition(&format_string, &flattened_args)?;
    let bytes = map_string_result(
        encode_output(&rendered, target.encoding_label()),
        &FPRINTF_ERROR_ENCODE,
    )?;
    target.write(&bytes)?;
    Ok(FprintfEval {
        bytes_written: bytes.len(),
    })
}

// kind_of was used for debugging logs; removed to avoid dead code in release builds.

#[runtime_builtin(
    name = "fprintf",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::filetext::fprintf"
)]
async fn fprintf_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mut args = Vec::with_capacity(rest.len() + 1);
    args.push(first);
    args.extend(rest);
    let eval = evaluate(&args).await?;
    Ok(Value::Num(eval.bytes_written() as f64))
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn format_with_repetition(format: &str, args: &[Value]) -> BuiltinResult<String> {
    let mut cursor = ArgCursor::new(args);
    let mut out = String::new();
    loop {
        let step = format_variadic_with_cursor(format, &mut cursor).map_err(remap_format_error)?;
        out.push_str(&step.output);
        if step.consumed == 0 {
            if cursor.remaining() > 0 {
                return Err(fprintf_error_with_detail(
                    &FPRINTF_ERROR_FORMAT,
                    "formatSpec contains no conversion specifiers but additional arguments were supplied",
                ));
            }
            break;
        }
        if cursor.remaining() == 0 {
            break;
        }
    }
    Ok(out)
}

fn remap_format_error(err: RuntimeError) -> RuntimeError {
    let message = err.message().replace("sprintf", "fprintf");
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FPRINTF_ERROR_FORMAT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests;
