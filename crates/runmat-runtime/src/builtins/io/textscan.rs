//! MATLAB-compatible `textscan` builtin for formatted text imports.

use runmat_value::{NumericDType, NumericScalar, NumericStorage};
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};

#[cfg(test)]
use runmat_builtins::TEXTSCAN_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, TEXTSCAN_ERROR_ARGUMENT, TEXTSCAN_ERROR_FILE, TEXTSCAN_ERROR_FORMAT,
    TEXTSCAN_ERROR_PARSE, TEXTSCAN_TYPED_INTEGER_CONTROL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "textscan";

fn textscan_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn textscan_error_with_source<E>(
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
    name = "textscan",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::textscan"
)]
async fn textscan_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    input: Value,
    format_spec: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            "textscan: at most one output is available",
        ));
    }
    if is_typed_integer_value(&input) || rest.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TEXTSCAN_TYPED_INTEGER_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let admitted = admission::preflight(runtime.as_ref(), &input, &format_spec, rest.view())?;
    let input = crate::builtins::io::filetext::cooperation::await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&input),
    )
    .await?
    .map_err(map_control_flow)?;
    let parsed_format = admitted.format;
    let (repeat, options) = match admitted.controls {
        Some(controls) => controls,
        None => {
            let gathered = crate::builtins::io::filetext::cooperation::await_io(
                runtime.as_ref(),
                BUILTIN_NAME,
                rest.try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
                    gather_if_needed_async(&rest[index])
                        .await
                        .map_err(map_control_flow)
                }),
            )
            .await??;
            parse_args(gathered.view())?
        }
    };
    let mut cooperation = TextscanCooperation::new(runtime.as_ref());
    cooperation.completed(0).await?;
    let mut parsed = parse_input(
        runtime.as_ref(),
        &input,
        &parsed_format,
        repeat,
        &options,
        &mut cooperation,
    )
    .await?;
    cooperation.completed(0).await?;
    let output = build_output(
        std::mem::take(&mut parsed.columns),
        &options,
        &mut cooperation,
    )
    .await?;
    cooperation.completed(0).await?;
    parsed.check_authority(runtime.as_ref())?;
    parsed.commit_file_position();
    Ok(output)
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

mod admission;
mod column;
mod cooperation;
mod format;
mod input;
mod numeric;
mod options;
mod output;
mod scanner;
use self::{
    column::*, cooperation::*, format::*, input::*, numeric::*, options::*, output::*, scanner::*,
};

#[cfg(test)]
#[path = "textscan/tests.rs"]
mod tests;
