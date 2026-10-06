//! MATLAB-compatible `fread` builtin for RunMat.

#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use std::io::{ErrorKind, Read, Seek, SeekFrom};

use runmat_builtins::{
    BuiltinErrorDescriptor, FREAD_ERROR_INTERNAL, FREAD_ERROR_INVALID_IDENTIFIER,
    FREAD_ERROR_INVALID_INPUT, FREAD_ERROR_INVALID_OPTION, FREAD_ERROR_IO,
    FREAD_INTEGER_ID_EXTENSION, FREAD_INTEGER_SIZE_EXTENSION, FREAD_INTEGER_SKIP_EXTENSION,
    FREAD_LIKE_EXTENSION, FREAD_LOGICAL_CONTROL_EXTENSION, FREAD_RESIDENT_CONTROL_EXTENSION,
    FREAD_SINGLE_CONTROL_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{FREAD_DESCRIPTOR, FREAD_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericScalar, NumericStorage,
    Tensor, Value,
};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::io::filetext::{helpers::extract_scalar_string, registry};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fread";

fn fread_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("fread: ").unwrap_or(detail);
    fread_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn fread_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FREAD_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(
    result: Result<T, String>,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    result.map_err(|detail| fread_error_with_detail(error, detail))
}

#[runtime_builtin(
    name = "fread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::fread"
)]
async fn fread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    fid: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let eval = evaluate_in(runtime.as_ref(), &fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FreadEval {
    data: Value,
    count: usize,
}

impl FreadEval {
    fn new(data: Value, count: usize) -> Self {
        Self { data, count }
    }

    pub fn first_output(&self) -> Value {
        self.data.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.data.clone(), Value::Num(self.count as f64)]
    }

    fn apply_like(
        &mut self,
        like_proto: Option<&Value>,
        precision: PrecisionSpec,
    ) -> BuiltinResult<()> {
        if let Some(proto) = like_proto {
            let adjusted = adjust_output_for_like(self.data.clone(), proto, precision)?;
            self.data = adjusted;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn data(&self) -> &Value {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
    evaluate_in(None, fid_value, rest).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    fid_value: &Value,
    rest: &[Value],
) -> BuiltinResult<FreadEval> {
    let mut cooperation = super::cooperation::FileCooperation::new(runtime, BUILTIN_NAME);
    let prepared = PreparedFreadCall::new(fid_value, rest)?;
    let fid = cooperation.await_io(prepared.resolve_fid()).await??;

    let mut operation = registry::acquire_file_in(fid, runtime).map_err(|err| match err {
        registry::FileOperationError::InvalidIdentifier
        | registry::FileOperationError::ForeignSession => fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        ),
        other => {
            let source = other.into_io_error();
            let mut builder = build_runtime_error(format!("{}: {source}", FREAD_ERROR_IO.message))
                .with_builtin(BUILTIN_NAME)
                .with_source(source);
            if let Some(identifier) = FREAD_ERROR_IO.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        }
    })?;
    if !operation.info.access.read {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let like = prepared.like;
    let controls = cooperation
        .await_io(prepared.resolve_controls(&operation.info.machinefmt))
        .await??;

    operation.check_authority(runtime).map_err(|source| {
        let mut builder = build_runtime_error(format!("{}: {source}", FREAD_ERROR_IO.message))
            .with_builtin(BUILTIN_NAME)
            .with_source(source);
        if let Some(identifier) = FREAD_ERROR_IO.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })?;
    cooperation.completed(0).await?;

    let (mut eval, encountered_eof) = read_from_handle(
        operation.file_mut(),
        &controls.size,
        &controls.precision,
        controls.skip,
        controls.machine,
        &mut cooperation,
    )
    .await
    .map_err(|error| match error {
        ReadFailure::Execution(source) => *source,
        other => {
            let detail = other.to_string();
            let detail = detail.strip_prefix("fread: ").unwrap_or(&detail);
            let mut builder =
                build_runtime_error(format!("{}: {}", FREAD_ERROR_IO.message, detail))
                    .with_builtin(BUILTIN_NAME)
                    .with_source(other);
            if let Some(identifier) = FREAD_ERROR_IO.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        }
    })?;
    operation.check_authority(runtime).map_err(|source| {
        let mut builder = build_runtime_error(format!("{}: {source}", FREAD_ERROR_IO.message))
            .with_builtin(BUILTIN_NAME)
            .with_source(source);
        if let Some(identifier) = FREAD_ERROR_IO.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })?;
    if encountered_eof {
        registry::mark_eof_encountered_in(fid, runtime);
    }
    eval.apply_like(like, controls.precision)?;
    operation.check_authority(runtime).map_err(|source| {
        let mut builder = build_runtime_error(format!("{}: {source}", FREAD_ERROR_IO.message))
            .with_builtin(BUILTIN_NAME)
            .with_source(source);
        if let Some(identifier) = FREAD_ERROR_IO.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })?;
    cooperation.completed(0).await?;
    Ok(eval)
}

fn validate_like_prototype(prototype: &Value) -> Result<(), String> {
    match prototype {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::provider_for_handle(handle)
                .is_none_or(|owner| owner.device_id() != handle.device_id())
            {
                return Err(
                    "fread: resident 'like' prototype has no registered owning provider"
                        .to_string(),
                );
            }
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(
                    "fread: complex resident 'like' prototypes are not supported".to_string(),
                );
            }
            let integer = runmat_accelerate_api::handle_integer_type(handle).is_some();
            let logical = runmat_accelerate_api::handle_is_logical(handle);
            if !integer && !logical && runmat_accelerate_api::handle_precision(handle).is_none() {
                return Err(
                    "fread: resident 'like' prototype is missing precision metadata".to_string(),
                );
            }
            Ok(())
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err("fread: complex prototypes are not supported yet".to_string())
        }
        Value::Cell(_) => Err("fread: cell prototypes are not supported".to_string()),
        Value::LogicalArray(_)
        | Value::Bool(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Tensor(_)
        | Value::Int(_)
        | Value::Num(_) => Ok(()),
        _ => Err("fread: unsupported 'like' prototype".to_string()),
    }
}

fn validate_like_output_kind(prototype: &Value, output: OutputKind) -> Result<(), String> {
    let character_prototype = matches!(
        prototype,
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_)
    );
    match (character_prototype, output) {
        (true, OutputKind::Char) | (false, OutputKind::Numeric(_)) => Ok(()),
        (true, OutputKind::Numeric(_)) => Err(
            "fread: character prototypes require a character precision such as '*char'".to_string(),
        ),
        (false, OutputKind::Char) => {
            Err("fread: character output requires a character prototype".to_string())
        }
    }
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

mod admission;
mod controls;
mod decode;
mod output;
mod precision;
mod reader;

use admission::*;
use controls::*;
use decode::*;
use output::*;
use precision::*;
use reader::*;

#[cfg(test)]
pub(crate) mod tests;
