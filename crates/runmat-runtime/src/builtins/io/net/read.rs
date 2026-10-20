//! MATLAB-compatible `read` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, NetDataType as DataType, READ_ERROR_CONNECTION_CLOSED,
    READ_ERROR_INTERNAL, READ_ERROR_INVALID_CLIENT, READ_ERROR_INVALID_COUNT,
    READ_ERROR_INVALID_DATATYPE, READ_ERROR_INVALID_INPUT, READ_ERROR_NOT_CONNECTED,
    READ_ERROR_TIMEOUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::{RuntimeTcpByteOrder, RuntimeTcpInvocation, RuntimeTcpReadMode};
use crate::tcp::TcpTransportErrorKind;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "read";

fn read_error_with_message(
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

fn read_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    read_error_with_message(error.message, error)
}

fn read_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("read: ").unwrap_or(detail);
    read_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn read_flow(error: &'static BuiltinErrorDescriptor, message: impl AsRef<str>) -> RuntimeError {
    read_error_with_detail(error, message)
}

#[runtime_builtin(
    name = "read",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::net::read"
)]
pub(super) async fn read_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    client: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let known = preflight_known_arguments(rest.view())?;
    let authority = super::receiver::authority(&client, runmat_builtins::TCPCLIENT_CLASS)
        .ok_or_else(|| {
            read_flow(
                &READ_ERROR_INVALID_CLIENT,
                "read: expected a live tcpclient object",
            )
        })?;
    let service = super::transport::service(runtime.as_ref(), BUILTIN_NAME)?;
    let mut work = super::transport::NetCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let options = parse_arguments(rest, known).await?;
    let element_size = options.datatype.element_size();
    let mode = match options.mode {
        ReadMode::Available => RuntimeTcpReadMode::Available,
        ReadMode::Count(count) => {
            RuntimeTcpReadMode::Count(count.checked_mul(element_size).ok_or_else(|| {
                read_flow(
                    &READ_ERROR_INVALID_COUNT,
                    "read: count and datatype size exceed the maximum supported byte count",
                )
            })?)
        }
    };
    let read_result = work
        .await_service(service.read(
            RuntimeTcpInvocation::from_context(runtime.as_ref()),
            authority,
            mode,
        ))
        .await
        .map_err(|failure| failure.into_runtime_error(BUILTIN_NAME))?
        .map_err(|err| {
            let error = match err.kind() {
                TcpTransportErrorKind::InvalidAuthority => &READ_ERROR_INVALID_CLIENT,
                TcpTransportErrorKind::Closed => &READ_ERROR_NOT_CONNECTED,
                TcpTransportErrorKind::Timeout => &READ_ERROR_TIMEOUT,
                _ => &READ_ERROR_INTERNAL,
            };
            super::error::with_source(BUILTIN_NAME, error, format!("read: {err}"), err)
        })?;

    if let ReadMode::Count(count) = options.mode {
        if read_result.bytes().is_empty() && count > 0 {
            return Err(read_error(&READ_ERROR_CONNECTION_CLOSED));
        }
        let expected = count.saturating_mul(element_size);
        if read_result.bytes().len() != expected {
            return Err(read_error(&READ_ERROR_CONNECTION_CLOSED));
        }
    }

    let byte_order = match read_result.byte_order() {
        RuntimeTcpByteOrder::Little => ByteOrder::Little,
        RuntimeTcpByteOrder::Big => ByteOrder::Big,
    };
    let value =
        bytes_to_value(read_result.bytes(), options.datatype, byte_order, &mut work).await?;
    work.completed(0).await?;
    read_result.publish();
    Ok(value)
}

#[derive(Clone, Copy)]
enum ReadMode {
    Available,
    Count(usize),
}

#[derive(Clone, Copy)]
struct ReadOptions {
    mode: ReadMode,
    datatype: DataType,
}

#[path = "read/decode.rs"]
mod decode;
use decode::{bytes_to_value, ByteOrder};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "read/codec_tests.rs"]
mod codec_tests;

async fn parse_arguments(
    rest: crate::call::BuiltinArguments,
    known: KnownReadControls,
) -> BuiltinResult<ReadOptions> {
    match rest.len() {
        0 => Ok(ReadOptions {
            mode: ReadMode::Available,
            datatype: DataType::UInt8,
        }),
        1 => {
            let count = match known.count {
                Some(count) => count,
                None => parse_count(&gather_if_needed_async(&rest[0]).await?)?,
            };
            Ok(ReadOptions {
                mode: ReadMode::Count(count),
                datatype: DataType::UInt8,
            })
        }
        2 => {
            let count = match known.count {
                Some(count) => count,
                None => parse_count(&gather_if_needed_async(&rest[0]).await?)?,
            };
            let datatype = match known.datatype {
                Some(datatype) => datatype,
                None => parse_datatype(&gather_if_needed_async(&rest[1]).await?)?,
            };
            Ok(ReadOptions {
                mode: ReadMode::Count(count),
                datatype,
            })
        }
        _ => Err(read_error(&READ_ERROR_INVALID_INPUT)),
    }
}

struct KnownReadControls {
    count: Option<usize>,
    datatype: Option<DataType>,
}

fn preflight_known_arguments(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<KnownReadControls> {
    if rest.len() > 2 {
        return Err(read_error(&READ_ERROR_INVALID_INPUT));
    }
    let count = match rest.first() {
        Some(Value::GpuTensor(_)) | None => None,
        Some(count) => Some(parse_count(count)?),
    };
    let datatype = match rest.get(1) {
        Some(Value::GpuTensor(_)) | None => None,
        Some(datatype) => Some(parse_datatype(datatype)?),
    };
    Ok(KnownReadControls { count, datatype })
}

fn parse_count(value: &Value) -> BuiltinResult<usize> {
    if let Value::Int(int) = value {
        return parse_integer_count(int);
    }

    let numeric = match value {
        Value::Num(n) => *n,
        Value::Tensor(t) if crate::builtins::common::tensor::is_scalar_tensor(t) => {
            let value = t
                .numeric_value_at(0)
                .expect("scalar tensor has authoritative numeric value");
            if let Some(int) = value.into_int_value() {
                return parse_integer_count(&int);
            }
            value.materialize_f64()
        }
        _ => {
            return Err(read_flow(
                &READ_ERROR_INVALID_COUNT,
                "read: count must be a numeric scalar",
            ))
        }
    };
    if numeric.is_nan() || numeric <= 0.0 {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be a positive finite integer",
        ));
    }
    if numeric.is_infinite() {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be finite",
        ));
    }
    if numeric > usize::MAX as f64 || (usize::BITS == 64 && numeric == usize::MAX as f64) {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count exceeds the maximum supported size",
        ));
    }
    let parsed = numeric as usize;
    if parsed as f64 != numeric {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be a positive finite integer",
        ));
    }
    Ok(parsed)
}

fn parse_integer_count(int: &runmat_value::IntValue) -> BuiltinResult<usize> {
    let unsigned = int.try_to_u64().ok_or_else(|| {
        read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be a positive finite integer",
        )
    })?;
    if unsigned == 0 {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be a positive finite integer",
        ));
    }
    usize::try_from(unsigned).map_err(|_| {
        read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count exceeds the maximum supported size",
        )
    })
}

fn parse_datatype(value: &Value) -> BuiltinResult<DataType> {
    let text =
        crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
            read_flow(
                &READ_ERROR_INVALID_DATATYPE,
                "read: datatype must be a string scalar",
            )
        })?;
    DataType::parse(&text).ok_or_else(|| {
        read_flow(
            &READ_ERROR_INVALID_DATATYPE,
            format!("read: unsupported datatype '{text}'"),
        )
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "read/tests.rs"]
pub(crate) mod tests;
