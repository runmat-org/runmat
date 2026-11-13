//! MATLAB-compatible `fwrite` builtin for RunMat.
use std::io::{Seek, SeekFrom, Write};

use runmat_builtins::{
    BuiltinErrorDescriptor, FWRITE_ARROW_PRECISION_EXTENSION, FWRITE_ERROR_INTERNAL,
    FWRITE_ERROR_INVALID_IDENTIFIER, FWRITE_ERROR_INVALID_INPUT, FWRITE_ERROR_INVALID_OPTION,
    FWRITE_ERROR_IO, FWRITE_GPU_INPUT_EXTENSION, FWRITE_INTEGER_ID_EXTENSION,
    FWRITE_INTEGER_SKIP_EXTENSION, FWRITE_LOGICAL_CONTROL_EXTENSION,
    FWRITE_RESIDENT_CONTROL_EXTENSION, FWRITE_SINGLE_CONTROL_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{FWRITE_DESCRIPTOR, FWRITE_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntValue, NumericDType, Value};

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fwrite";

fn fwrite_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("fwrite: ").unwrap_or(detail);
    fwrite_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn fwrite_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fwrite_error_with_source(
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
    if let Some(identifier) = FWRITE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(
    result: Result<T, String>,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    result.map_err(|detail| fwrite_error_with_detail(error, detail))
}

#[runtime_builtin(
    name = "fwrite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::fwrite"
)]
async fn fwrite_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    fid: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let eval = evaluate_in_view(runtime.as_ref(), &fid, &data, rest.view()).await?;
    Ok(Value::Num(eval.count as f64))
}

/// Result of an `fwrite` evaluation.
#[derive(Debug, Clone)]
pub struct FwriteEval {
    count: usize,
}

impl FwriteEval {
    fn new(count: usize) -> Self {
        Self { count }
    }

    /// Number of elements successfully written.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Evaluate the `fwrite` builtin without invoking the runtime dispatcher.
pub async fn evaluate(
    fid_value: &Value,
    data_value: &Value,
    rest: &[Value],
) -> BuiltinResult<FwriteEval> {
    evaluate_in(None, fid_value, data_value, rest).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    fid_value: &Value,
    data_value: &Value,
    rest: &[Value],
) -> BuiltinResult<FwriteEval> {
    evaluate_in_view(
        runtime,
        fid_value,
        data_value,
        crate::call::FunctionArgumentView::from_slice(rest),
    )
    .await
}

pub async fn evaluate_in_view(
    runtime: Option<&crate::context::RuntimeContext>,
    fid_value: &Value,
    data_value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FwriteEval> {
    let mut cooperation = super::cooperation::FileCooperation::new(runtime, BUILTIN_NAME);
    preflight_data(data_value)?;
    if matches!(data_value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FWRITE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_control(fid_value, ControlRole::FileId)?;
    let (raw_precision, raw_skip, raw_machine) =
        map_string_result(classify_arguments(rest), &FWRITE_ERROR_INVALID_INPUT)?;
    if let Some(skip) = raw_skip {
        preflight_control(skip, ControlRole::Skip)?;
    }
    if let Some(precision) = raw_precision {
        if scalar_string(
            precision,
            "fwrite: precision argument must be a string scalar or character vector",
        )
        .is_ok_and(|value| value.contains("=>"))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FWRITE_ARROW_PRECISION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let precision_spec = parse_precision(raw_precision)?;
    if let Some(skip) = raw_skip.filter(|value| !matches!(value, Value::GpuTensor(_))) {
        map_string_result(parse_skip(Some(skip)), &FWRITE_ERROR_INVALID_OPTION)?;
    }
    if let Some(machine) = raw_machine {
        parse_machine_format(Some(machine), "native")?;
    }
    let fid_host = cooperation.await_io(gather_value(fid_value)).await??;
    let fid = map_string_result(parse_fid(&fid_host), &FWRITE_ERROR_INVALID_INPUT)?;
    if fid < 0 {
        return Err(fwrite_error_with_detail(
            &FWRITE_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fwrite_error_with_detail(
            &FWRITE_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let mut operation = registry::acquire_file_in(fid, runtime).map_err(|err| match err {
        registry::FileOperationError::InvalidIdentifier
        | registry::FileOperationError::ForeignSession => fwrite_error_with_message(
            FWRITE_ERROR_INVALID_IDENTIFIER.message,
            &FWRITE_ERROR_INVALID_IDENTIFIER,
        ),
        other => {
            let source = other.into_io_error();
            fwrite_error_with_source(source.to_string(), &FWRITE_ERROR_IO, source)
        }
    })?;
    if !operation.info.access.write {
        return Err(fwrite_error_with_detail(
            &FWRITE_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for writing",
        ));
    }
    let skip_host = match raw_skip {
        Some(value @ Value::GpuTensor(_)) => {
            Some(cooperation.await_io(gather_value(value)).await??)
        }
        Some(value) => Some(value.clone()),
        None => None,
    };
    let skip_bytes =
        map_string_result(parse_skip(skip_host.as_ref()), &FWRITE_ERROR_INVALID_OPTION)?;
    let machine_format = parse_machine_format(raw_machine, &operation.info.machinefmt)?;
    let data_host = cooperation.await_io(gather_value(data_value)).await??;
    let elements = map_string_result(borrow_elements(&data_host), &FWRITE_ERROR_INVALID_INPUT)?;

    operation
        .check_authority(runtime)
        .map_err(|source| fwrite_error_with_source(source.to_string(), &FWRITE_ERROR_IO, source))?;
    cooperation.completed(0).await?;
    let count = write_elements(
        operation.file_mut(),
        elements,
        precision_spec,
        skip_bytes,
        machine_format,
        &mut cooperation,
    )
    .await?;
    operation
        .check_authority(runtime)
        .map_err(|source| fwrite_error_with_source(source.to_string(), &FWRITE_ERROR_IO, source))?;
    cooperation.completed(0).await?;
    Ok(FwriteEval::new(count))
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn preflight_data(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => Ok(()),
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            Ok(())
        }
        _ => Err(fwrite_error_with_detail(
            &FWRITE_ERROR_INVALID_INPUT,
            "data must be real numeric, logical, or text",
        )),
    }
}

mod controls;
mod writing;

use controls::*;
use writing::*;

#[cfg(test)]
pub(crate) mod tests;
