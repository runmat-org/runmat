//! MATLAB-compatible `read` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, NetDataType as DataType, READ_ERROR_CONNECTION_CLOSED,
    READ_ERROR_INTERNAL, READ_ERROR_INVALID_CLIENT, READ_ERROR_INVALID_COUNT,
    READ_ERROR_INVALID_DATATYPE, READ_ERROR_INVALID_INPUT, READ_ERROR_NOT_CONNECTED,
    READ_ERROR_TIMEOUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericStorage;
use runmat_value::{CharArray, IntegerStorage, StructValue, Tensor, Value};
use std::io::{self, Read};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::client_state::{client_handle, CLIENT_HANDLE_FIELD};
use super::control::configure_stream;

const BUILTIN_NAME: &str = "read";

fn read_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
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

fn map_control_flow(err: RuntimeError, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "read",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::net::read"
)]
pub(super) async fn read_builtin(client: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let known = preflight_known_arguments(&rest)?;
    super::transport::require_native_tcp(BUILTIN_NAME)?;
    let options = parse_arguments(rest, known).await?;
    let client = gather_if_needed_async(&client)
        .await
        .map_err(|err| map_control_flow(err, &READ_ERROR_INVALID_CLIENT))?;

    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(read_flow(
                &READ_ERROR_INVALID_CLIENT,
                "read: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        read_flow(
            &READ_ERROR_INVALID_CLIENT,
            "read: tcpclient handle is no longer valid",
        )
    })?;

    let (stream, timeout, byte_order, connected) = {
        let guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(read_error(&READ_ERROR_NOT_CONNECTED));
        }
        let timeout = guard.timeout;
        let byte_order = parse_byte_order(&guard.byte_order);
        let stream = guard.stream.try_clone().map_err(|err| {
            read_flow(&READ_ERROR_INTERNAL, format!("read: clone failed ({err})"))
        })?;
        (stream, timeout, byte_order, guard.connected)
    };

    // Ensure cloned descriptor uses the configured timeout.
    if connected {
        if let Err(err) = configure_stream(&stream, timeout) {
            return Err(read_flow(
                &READ_ERROR_INTERNAL,
                format!("read: unable to configure socket timeout ({err})"),
            ));
        }
    }

    let element_size = options.datatype.element_size();
    if let ReadMode::Count(count) = options.mode {
        count.checked_mul(element_size).ok_or_else(|| {
            read_flow(
                &READ_ERROR_INVALID_COUNT,
                "read: count and datatype size exceed the maximum supported byte count",
            )
        })?;
    }
    let mut stream = stream;
    let read_result = perform_read(&mut stream, &options.mode, element_size)?;

    if read_result.connection_closed {
        if let Ok(mut guard) = handle.lock() {
            guard.connected = false;
        }
    }

    if let ReadMode::Count(count) = options.mode {
        if read_result.bytes.is_empty() && count > 0 {
            return Err(read_error(&READ_ERROR_CONNECTION_CLOSED));
        }
        let expected = count.saturating_mul(element_size);
        if read_result.bytes.len() != expected {
            return Err(read_error(&READ_ERROR_CONNECTION_CLOSED));
        }
    }

    let value = bytes_to_value(&read_result.bytes, options.datatype, byte_order)?;
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

struct ReadOutcome {
    bytes: Vec<u8>,
    connection_closed: bool,
}

enum ReadError {
    Timeout,
    ConnectionClosed,
    Io(io::Error),
}

fn perform_read(
    stream: &mut TcpStream,
    mode: &ReadMode,
    element_size: usize,
) -> BuiltinResult<ReadOutcome> {
    match read_from_stream(stream, mode, element_size) {
        Ok(outcome) => Ok(outcome),
        Err(ReadError::Timeout) => Err(read_error(&READ_ERROR_TIMEOUT)),
        Err(ReadError::ConnectionClosed) => Err(read_error(&READ_ERROR_CONNECTION_CLOSED)),
        Err(ReadError::Io(err)) => Err(read_flow(
            &READ_ERROR_INTERNAL,
            format!("read: socket error ({err})"),
        )),
    }
}

fn read_from_stream(
    stream: &mut TcpStream,
    mode: &ReadMode,
    element_size: usize,
) -> Result<ReadOutcome, ReadError> {
    match mode {
        ReadMode::Available => read_all_available(stream),
        ReadMode::Count(count) => {
            let total = count.saturating_mul(element_size);
            read_exact_bytes(stream, total)
        }
    }
}

fn read_all_available(stream: &mut TcpStream) -> Result<ReadOutcome, ReadError> {
    let mut buffer = [0u8; 4096];
    let mut data = Vec::new();
    let mut connection_closed = false;

    loop {
        match stream.read(&mut buffer) {
            Ok(0) => {
                connection_closed = true;
                return Ok(ReadOutcome {
                    bytes: Vec::new(),
                    connection_closed,
                });
            }
            Ok(n) => {
                data.extend_from_slice(&buffer[..n]);
                break;
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) if is_timeout(&err) => return Err(ReadError::Timeout),
            Err(err) => return Err(ReadError::Io(err)),
        }
    }

    let guard = NonBlockingGuard::enter(stream).map_err(ReadError::Io)?;
    let mut guard = Some(guard);
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => {
                connection_closed = true;
                break;
            }
            Ok(n) => {
                data.extend_from_slice(&buffer[..n]);
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => break,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) if is_timeout(&err) => break,
            Err(err) => return Err(ReadError::Io(err)),
        }
    }
    drop(guard.take());

    Ok(ReadOutcome {
        bytes: data,
        connection_closed,
    })
}

fn read_exact_bytes(stream: &mut TcpStream, total: usize) -> Result<ReadOutcome, ReadError> {
    if total == 0 {
        return Ok(ReadOutcome {
            bytes: Vec::new(),
            connection_closed: false,
        });
    }

    let mut buf = vec![0u8; total];
    let mut offset = 0;
    let timeout = stream.read_timeout().ok().flatten();
    let start = Instant::now();
    let _guard = if timeout.is_some() {
        Some(NonBlockingGuard::enter(stream).map_err(ReadError::Io)?)
    } else {
        None
    };
    while offset < total {
        match stream.read(&mut buf[offset..]) {
            Ok(0) => return Err(ReadError::ConnectionClosed),
            Ok(n) => {
                offset += n;
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) if is_timeout(&err) => {
                if let Some(timeout) = timeout {
                    if start.elapsed() < timeout {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                }
                return Err(ReadError::Timeout);
            }
            Err(err) => return Err(ReadError::Io(err)),
        }
    }
    Ok(ReadOutcome {
        bytes: buf,
        connection_closed: false,
    })
}

fn is_timeout(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
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
    let text = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(read_flow(
                &READ_ERROR_INVALID_DATATYPE,
                "read: datatype must be a string scalar",
            ))
        }
    };
    DataType::parse(&text).ok_or_else(|| {
        read_flow(
            &READ_ERROR_INVALID_DATATYPE,
            format!("read: unsupported datatype '{text}'"),
        )
    })
}

fn extract_client_id(struct_value: &StructValue) -> BuiltinResult<u64> {
    let id_value = struct_field(struct_value, CLIENT_HANDLE_FIELD).ok_or_else(|| {
        read_flow(
            &READ_ERROR_INVALID_CLIENT,
            "read: tcpclient struct is missing internal handle",
        )
    })?;
    match id_value {
        Value::Int(iv) => iv.try_to_u64().ok_or_else(|| {
            read_flow(
                &READ_ERROR_INVALID_CLIENT,
                "read: tcpclient struct has invalid handle field",
            )
        }),
        _ => Err(read_flow(
            &READ_ERROR_INVALID_CLIENT,
            "read: tcpclient struct has invalid handle field",
        )),
    }
}

fn struct_field<'a>(value: &'a StructValue, name: &str) -> Option<&'a Value> {
    value.fields.get(name)
}

fn parse_byte_order(text: &str) -> ByteOrder {
    if text.eq_ignore_ascii_case("big-endian") || text.eq_ignore_ascii_case("big endian") {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    }
}

struct NonBlockingGuard {
    stream: *const TcpStream,
}

impl NonBlockingGuard {
    fn enter(stream: &TcpStream) -> io::Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(NonBlockingGuard {
            stream: stream as *const TcpStream,
        })
    }
}

impl Drop for NonBlockingGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(stream) = self.stream.as_ref() {
                let _ = stream.set_nonblocking(false);
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "read/tests.rs"]
pub(crate) mod tests;
