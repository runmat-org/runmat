//! MATLAB-compatible `fprintf` builtin enabling formatted text output to files and standard streams.

#[cfg(test)]
use runmat_builtins::FPRINTF_DESCRIPTOR;
#[cfg(test)]
use runmat_builtins::FPRINTF_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES;
use runmat_builtins::{
    BuiltinErrorDescriptor, FPRINTF_ERROR_ENCODE, FPRINTF_ERROR_FORMAT, FPRINTF_ERROR_INTERNAL,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::format::{
    decode_escape_sequences, flatten_argument_elements, format_variadic_with_cursor, ArgCursor,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fprintf";

mod admission;
mod encoding;
mod prepared;
mod target;

#[cfg(test)]
use admission::coerce_single_integer_string_argument;
use encoding::encode_output;

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
    evaluate_in(None, args).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &[Value],
) -> BuiltinResult<FprintfEval> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, runtime)?;
    evaluate_retained(runtime, args).await
}

async fn evaluate_retained(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<FprintfEval> {
    let prepared = prepared::prepare(runtime, args).await?;
    let raw_format = prepared.format;
    let mut target = prepared.target;
    let format_string =
        decode_escape_sequences("fprintf", &raw_format).map_err(map_control_flow)?;
    let format_summary = crate::builtins::common::format::validate_format_syntax(&format_string)
        .map_err(remap_format_error)?;
    let input = prepared.data_args;
    let data_args = input
        .try_map_preserving_roots_async(input.len(), runtime, async |index| {
            let value =
                super::cooperation::await_io(runtime, BUILTIN_NAME, gather_value(&input[index]))
                    .await??;
            if input.len() == 1 && format_summary.sole_string_conversion_without_star() {
                admission::coerce_single_integer_string_value(value)
            } else {
                Ok(value)
            }
        })
        .await?;
    let flattened_args = super::cooperation::await_io(
        runtime,
        BUILTIN_NAME,
        flatten_argument_elements(data_args.view(), "fprintf"),
    )
    .await?
    .map_err(map_control_flow)?;
    let rendered = format_with_repetition(&format_string, &flattened_args)?;
    let bytes = map_string_result(
        encode_output(&rendered, target.encoding_label()),
        &FPRINTF_ERROR_ENCODE,
    )?;
    super::cooperation::checkpoint(runtime, BUILTIN_NAME).await?;
    target.write(runtime, &bytes)?;
    super::cooperation::checkpoint(runtime, BUILTIN_NAME).await?;
    Ok(FprintfEval {
        bytes_written: bytes.len(),
    })
}

#[runtime_builtin(
    name = "fprintf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::fprintf"
)]
async fn fprintf_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    use crate::context::host_allocation::{host_allocation_runtime_error, HostAllocationReceipt};
    HostAllocationReceipt::capture_owner(runtime.as_ref()).map_err(|source| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:fprintf:Allocation",
            "fprintf prefix policy unavailable",
            source,
        )
    })?;
    let first = match runtime.as_ref() {
        Some(runtime) => crate::execution::RootedValue::in_context(first, runtime)?,
        None => crate::execution::RootedValue::new(first).map_err(|e| e.into_runtime_error())?,
    };
    let args = crate::builtins::common::mapped_callable::prepend_rooted_argument(
        &rest,
        first,
        runtime.as_ref(),
    )
    .await?;
    let eval = evaluate_retained(runtime.as_ref(), args).await?;
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
