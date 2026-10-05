//! MATLAB-compatible `read` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, NetDataType as DataType, READ_ERROR_CONNECTION_CLOSED,
    READ_ERROR_INTERNAL, READ_ERROR_INVALID_CLIENT, READ_ERROR_INVALID_COUNT,
    READ_ERROR_INVALID_DATATYPE, READ_ERROR_INVALID_INPUT, READ_ERROR_NOT_CONNECTED,
    READ_ERROR_TIMEOUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericStorage;
use runmat_value::{CharArray, IntegerStorage, Tensor, Value};

use crate::context::{RuntimeTcpByteOrder, RuntimeTcpIssuer, RuntimeTcpReadMode};
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
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let known = preflight_known_arguments(&rest)?;
    let authority = super::receiver::authority(&client, runmat_builtins::TCPCLIENT_CLASS)
        .ok_or_else(|| {
            read_flow(
                &READ_ERROR_INVALID_CLIENT,
                "read: expected a live tcpclient object",
            )
        })?;
    let service = super::transport::service(runtime.as_ref(), BUILTIN_NAME)?;
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
    let read_result = service
        .read(
            RuntimeTcpIssuer::from_context(runtime.as_ref()),
            authority,
            mode,
        )
        .await
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
    let value = bytes_to_value(read_result.bytes(), options.datatype, byte_order)?;
    if let Some(runtime) = &runtime {
        runtime.cpu_cooperation().completed(0).await?;
    }
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

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

fn bytes_to_value(bytes: &[u8], datatype: DataType, order: ByteOrder) -> BuiltinResult<Value> {
    match datatype {
        DataType::Char => Ok(char_row(bytes)),
        DataType::String => Ok(Value::String(bytes.iter().map(|&b| b as char).collect())),
        DataType::UInt8
        | DataType::Int8
        | DataType::UInt16
        | DataType::Int16
        | DataType::UInt32
        | DataType::Int32
        | DataType::UInt64
        | DataType::Int64 => {
            let storage = integer_storage_from_bytes(bytes, datatype, order)?;
            let tensor =
                Tensor::new_integer(storage, vec![1, bytes.len() / datatype.element_size()])
                    .map_err(|err| read_flow(&READ_ERROR_INTERNAL, format!("read: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
        DataType::Single | DataType::Double => {
            let storage = floating_storage_from_bytes(bytes, datatype, order)?;
            let cols = storage.len();
            let tensor = Tensor::from_numeric_storage(storage, vec![1, cols])
                .map_err(|err| read_flow(&READ_ERROR_INTERNAL, format!("read: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
    }
}

fn char_row(bytes: &[u8]) -> Value {
    let chars: Vec<char> = bytes.iter().map(|&b| b as char).collect();
    let len = chars.len();
    let array = CharArray::new(chars, 1, len).unwrap_or_else(|_| CharArray::new_row(""));
    Value::CharArray(array)
}

fn floating_storage_from_bytes(
    bytes: &[u8],
    datatype: DataType,
    order: ByteOrder,
) -> BuiltinResult<NumericStorage> {
    let size = datatype.element_size();
    if size == 0 {
        return Ok(match datatype {
            DataType::Single => NumericStorage::F32(Vec::new()),
            DataType::Double => NumericStorage::F64(Vec::new()),
            _ => unreachable!("floating storage requested for nonfloating datatype"),
        });
    }
    if !bytes.len().is_multiple_of(size) {
        return Err(read_flow(
            &READ_ERROR_INTERNAL,
            "read: received byte count does not align with datatype size",
        ));
    }
    Ok(match datatype {
        DataType::Single => NumericStorage::F32(
            bytes
                .chunks_exact(size)
                .map(|chunk| f32_from(chunk, order))
                .collect(),
        ),
        DataType::Double => NumericStorage::F64(
            bytes
                .chunks_exact(size)
                .map(|chunk| f64_from(chunk, order))
                .collect(),
        ),
        _ => {
            return Err(read_flow(
                &READ_ERROR_INTERNAL,
                "read: expected a floating-point datatype",
            ))
        }
    })
}

fn integer_storage_from_bytes(
    bytes: &[u8],
    datatype: DataType,
    order: ByteOrder,
) -> BuiltinResult<IntegerStorage> {
    let size = datatype.element_size();
    if !bytes.len().is_multiple_of(size) {
        return Err(read_flow(
            &READ_ERROR_INTERNAL,
            "read: received byte count does not align with datatype size",
        ));
    }
    Ok(match datatype {
        DataType::UInt8 => IntegerStorage::U8(bytes.to_vec()),
        DataType::Int8 => IntegerStorage::I8(bytes.iter().map(|value| *value as i8).collect()),
        DataType::UInt16 => IntegerStorage::U16(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16_from(chunk, order))
                .collect(),
        ),
        DataType::Int16 => IntegerStorage::I16(
            bytes
                .chunks_exact(2)
                .map(|chunk| i16_from(chunk, order))
                .collect(),
        ),
        DataType::UInt32 => IntegerStorage::U32(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32_from(chunk, order))
                .collect(),
        ),
        DataType::Int32 => IntegerStorage::I32(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32_from(chunk, order))
                .collect(),
        ),
        DataType::UInt64 => IntegerStorage::U64(
            bytes
                .chunks_exact(8)
                .map(|chunk| u64_from(chunk, order))
                .collect(),
        ),
        DataType::Int64 => IntegerStorage::I64(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64_from(chunk, order))
                .collect(),
        ),
        DataType::Single | DataType::Double | DataType::Char | DataType::String => {
            return Err(read_flow(
                &READ_ERROR_INTERNAL,
                "read: expected an integer datatype",
            ));
        }
    })
}

fn u16_from(bytes: &[u8], order: ByteOrder) -> u16 {
    match order {
        ByteOrder::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        ByteOrder::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    }
}

fn i16_from(bytes: &[u8], order: ByteOrder) -> i16 {
    u16_from(bytes, order) as i16
}

fn u32_from(bytes: &[u8], order: ByteOrder) -> u32 {
    match order {
        ByteOrder::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        ByteOrder::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

fn i32_from(bytes: &[u8], order: ByteOrder) -> i32 {
    u32_from(bytes, order) as i32
}

fn u64_from(bytes: &[u8], order: ByteOrder) -> u64 {
    match order {
        ByteOrder::Little => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        ByteOrder::Big => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

fn i64_from(bytes: &[u8], order: ByteOrder) -> i64 {
    u64_from(bytes, order) as i64
}

fn f32_from(bytes: &[u8], order: ByteOrder) -> f32 {
    match order {
        ByteOrder::Little => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        ByteOrder::Big => f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

fn f64_from(bytes: &[u8], order: ByteOrder) -> f64 {
    match order {
        ByteOrder::Little => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        ByteOrder::Big => f64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

async fn parse_arguments(rest: Vec<Value>, known: KnownReadControls) -> BuiltinResult<ReadOptions> {
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

fn preflight_known_arguments(rest: &[Value]) -> BuiltinResult<KnownReadControls> {
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
