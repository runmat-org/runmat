//! MATLAB-compatible `write` builtin for TCP/IP clients in RunMat.

use crate::context::RuntimeTcpInvocation;
use crate::tcp::{TcpTransportError, TcpTransportErrorKind, TcpWriteProgress};
use runmat_builtins::{
    BuiltinErrorDescriptor, NetDataType as DataType, NET_ERROR_CANCELLED,
    WRITE_ERROR_CONNECTION_CLOSED, WRITE_ERROR_INTERNAL, WRITE_ERROR_INVALID_CLIENT,
    WRITE_ERROR_INVALID_DATA, WRITE_ERROR_INVALID_DATATYPE, WRITE_ERROR_INVALID_INPUT,
    WRITE_ERROR_NOT_CONNECTED, WRITE_ERROR_TIMEOUT, WRITE_EXPLICIT_GPU_EXTENSION,
    WRITE_LOGICAL_EXTENSION, WRITE_NONVECTOR_EXTENSION, WRITE_SPARSE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "write";

fn write_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn write_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    write_error_with_message(error.message, error)
}

fn write_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("write: ").unwrap_or(detail);
    write_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn write_flow(error: &'static BuiltinErrorDescriptor, message: impl AsRef<str>) -> RuntimeError {
    write_error_with_detail(error, message)
}

fn map_write_flow(err: RuntimeError, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_write_await_failure(failure: super::transport::TcpAwaitFailure) -> RuntimeError {
    if failure.cause != crate::execution::ExecutionServiceError::Cancelled {
        return failure.into_runtime_error(BUILTIN_NAME);
    }
    let (message, progress) = if failure.launched {
        (
            "write: cancelled while sending; delivery may be partial",
            TcpWriteProgress::Indeterminate,
        )
    } else {
        (
            "write: cancelled before sending",
            TcpWriteProgress::NoBytesSent,
        )
    };
    let source = TcpTransportError::with_source(TcpTransportErrorKind::Io, message, failure.cause)
        .with_write_progress(progress);
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = NET_ERROR_CANCELLED.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "write",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::net::write"
)]
pub(super) async fn write_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    client: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ensure_write_extensions(&data)?;
    let datatype = preflight_datatype(rest)?;
    let authority = super::receiver::authority(&client, runmat_builtins::TCPCLIENT_CLASS)
        .ok_or_else(|| {
            write_flow(
                &WRITE_ERROR_INVALID_CLIENT,
                "write: expected a live tcpclient object",
            )
        })?;
    if !matches!(
        &data,
        Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::CharArray(_)
            | Value::String(_)
            | Value::StringArray(_)
    ) {
        return Err(write_flow(
            &WRITE_ERROR_INVALID_DATA,
            "write: unsupported input type",
        ));
    }
    let service = super::transport::service(runtime.as_ref(), BUILTIN_NAME)?;
    let mut work = super::transport::NetCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let data = gather_if_needed_async(&data)
        .await
        .map_err(|flow| map_write_flow(flow, &WRITE_ERROR_INVALID_DATA))?;
    let datatype = match datatype {
        PreparedWriteDatatype::Default => None,
        PreparedWriteDatatype::Known(datatype) => Some(datatype),
        PreparedWriteDatatype::Resident(value) => Some(parse_datatype(
            &gather_if_needed_async(&value)
                .await
                .map_err(|flow| map_write_flow(flow, &WRITE_ERROR_INVALID_DATATYPE))?,
        )?),
    };

    let status = work
        .await_service(service.describe_client(
            RuntimeTcpInvocation::from_context(runtime.as_ref()),
            authority.clone(),
        ))
        .await
        .map_err(|failure| failure.into_runtime_error(BUILTIN_NAME))?
        .map_err(|err| {
            let descriptor = if err.kind() == TcpTransportErrorKind::InvalidAuthority {
                &WRITE_ERROR_INVALID_CLIENT
            } else {
                &WRITE_ERROR_INTERNAL
            };
            super::error::with_source(BUILTIN_NAME, descriptor, format!("write: {err}"), err)
        })?;
    if !status.connected {
        return Err(write_error(&WRITE_ERROR_NOT_CONNECTED));
    }
    let byte_order = match status.byte_order {
        crate::context::RuntimeTcpByteOrder::Little => ByteOrder::Little,
        crate::context::RuntimeTcpByteOrder::Big => ByteOrder::Big,
    };

    let payload = match datatype {
        Some(datatype) => prepare_payload(&data, datatype, byte_order, &mut work).await?,
        None => prepare_default_payload(&data, byte_order, &mut work).await?,
    };
    if payload.bytes.is_empty() {
        work.completed(0).await?;
        return Ok(Value::Num(0.0));
    }

    work.completed(0).await?;
    let expected_bytes = payload.bytes.len();
    let accepted_bytes = work
        .await_service(service.write(
            RuntimeTcpInvocation::from_context(runtime.as_ref()),
            authority,
            payload.bytes,
        ))
        .await
        .map_err(map_write_await_failure)?
        .map_err(|err| {
            let descriptor = match err.kind() {
                TcpTransportErrorKind::InvalidAuthority => &WRITE_ERROR_INVALID_CLIENT,
                TcpTransportErrorKind::Closed => &WRITE_ERROR_CONNECTION_CLOSED,
                TcpTransportErrorKind::Timeout => &WRITE_ERROR_TIMEOUT,
                _ => &WRITE_ERROR_INTERNAL,
            };
            super::error::with_source(BUILTIN_NAME, descriptor, format!("write: {err}"), err)
        })?;
    if accepted_bytes != expected_bytes {
        let progress = if accepted_bytes <= expected_bytes {
            TcpWriteProgress::PrefixSent(accepted_bytes)
        } else {
            TcpWriteProgress::Indeterminate
        };
        let cause = TcpTransportError::new(
            TcpTransportErrorKind::Io,
            format!(
                "TCP service reported {accepted_bytes} accepted bytes for a {expected_bytes}-byte write"
            ),
        )
        .with_write_progress(progress);
        return Err(super::error::with_source(
            BUILTIN_NAME,
            &WRITE_ERROR_INTERNAL,
            format!("write: {cause}"),
            cause,
        ));
    }
    Ok(Value::Num(payload.elements as f64))
}

fn ensure_write_extensions(data: &Value) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(data) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITE_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(data, Value::SparseTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITE_SPARSE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(data, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(data, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITE_LOGICAL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let is_nonvector = match data {
        Value::Tensor(tensor) => !is_vector_shape(tensor.shape()),
        Value::SparseTensor(tensor) => !is_vector_shape(&[tensor.rows, tensor.cols]),
        Value::GpuTensor(handle) => !is_vector_shape(handle.shape()),
        _ => false,
    };
    if is_nonvector {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITE_NONVECTOR_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.len() <= 2 && shape.iter().filter(|extent| **extent > 1).count() <= 1
}

#[path = "write/codec.rs"]
mod codec;
#[cfg(all(test, not(target_arch = "wasm32")))]
use codec::{extend_f64, extend_u64};
use codec::{
    parse_datatype, preflight_datatype, prepare_default_payload, prepare_payload, ByteOrder,
    PreparedWriteDatatype,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "write/tests.rs"]
pub(crate) mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "write/count_tests.rs"]
mod count_tests;
