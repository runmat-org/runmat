//! MATLAB-compatible `read` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{CharArray, IntValue, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::io::{self, Read};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::accept::{client_handle, configure_stream, CLIENT_HANDLE_FIELD};

const MESSAGE_ID_INVALID_CLIENT: &str = "MATLAB:read:InvalidTcpClient";
const MESSAGE_ID_NOT_CONNECTED: &str = "MATLAB:read:NotConnected";
const MESSAGE_ID_TIMEOUT: &str = "MATLAB:read:Timeout";
const MESSAGE_ID_CONNECTION_CLOSED: &str = "MATLAB:read:ConnectionClosed";
const MESSAGE_ID_INVALID_COUNT: &str = "MATLAB:read:InvalidCount";
const MESSAGE_ID_INVALID_DATATYPE: &str = "MATLAB:read:InvalidDataType";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:read:InternalError";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::read")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "read",
    op_kind: GpuOpKind::Custom("network"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Socket reads always execute on the host CPU; GPU providers are never consulted.",
};

fn read_error(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(message_id)
        .with_builtin("read")
        .build()
}

fn read_flow(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    read_error(message_id, message)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::read")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "read",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "read",
    category = "io/net",
    summary = "Read numeric or text data from a TCP/IP client.",
    keywords = "read,tcpclient,networking",
    builtin_path = "crate::builtins::io::net::read"
)]
async fn read_builtin(client: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let client = gather_if_needed_async(&client).await?;
    let options = parse_arguments(rest).await?;

    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(read_flow(
                MESSAGE_ID_INVALID_CLIENT,
                "read: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        read_flow(
            MESSAGE_ID_INVALID_CLIENT,
            "read: tcpclient handle is no longer valid",
        )
    })?;

    let (stream, timeout, byte_order, connected) = {
        let guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(read_flow(
                MESSAGE_ID_NOT_CONNECTED,
                "read: tcpclient is disconnected",
            ));
        }
        let timeout = guard.timeout;
        let byte_order = parse_byte_order(&guard.byte_order);
        let stream = guard
            .stream
            .try_clone()
            .map_err(|err| read_flow(MESSAGE_ID_INTERNAL, format!("read: clone failed ({err})")))?;
        (stream, timeout, byte_order, guard.connected)
    };

    // Ensure cloned descriptor uses the configured timeout.
    if connected {
        if let Err(err) = configure_stream(&stream, timeout) {
            return Err(read_flow(
                MESSAGE_ID_INTERNAL,
                format!("read: unable to configure socket timeout ({err})"),
            ));
        }
    }

    let element_size = options.datatype.element_size();
    let mut stream = stream;
    let read_result = perform_read(&mut stream, &options.mode, element_size)?;

    if read_result.connection_closed {
        if let Ok(mut guard) = handle.lock() {
            guard.connected = false;
        }
    }

    if let ReadMode::Count(count) = options.mode {
        if read_result.bytes.is_empty() && count > 0 {
            return Err(read_flow(
                MESSAGE_ID_CONNECTION_CLOSED,
                "read: connection closed before the requested data was received",
            ));
        }
        let expected = count.saturating_mul(element_size);
        if read_result.bytes.len() != expected {
            return Err(read_flow(
                MESSAGE_ID_CONNECTION_CLOSED,
                "read: connection closed before the requested data was received",
            ));
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
enum DataType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Single,
    Double,
    Char,
    String,
}

impl DataType {
    fn element_size(self) -> usize {
        match self {
            DataType::UInt8 | DataType::Int8 | DataType::Char | DataType::String => 1,
            DataType::UInt16 | DataType::Int16 => 2,
            DataType::UInt32 | DataType::Int32 | DataType::Single => 4,
            DataType::UInt64 | DataType::Int64 | DataType::Double => 8,
        }
    }
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
        Err(ReadError::Timeout) => Err(read_flow(
            MESSAGE_ID_TIMEOUT,
            "read: timed out waiting for data",
        )),
        Err(ReadError::ConnectionClosed) => Err(read_flow(
            MESSAGE_ID_CONNECTION_CLOSED,
            "read: connection closed before the requested data was received",
        )),
        Err(ReadError::Io(err)) => Err(read_flow(
            MESSAGE_ID_INTERNAL,
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
        | DataType::Int64
        | DataType::Single
        | DataType::Double => {
            let values = numeric_from_bytes(bytes, datatype, order)?;
            let cols = values.len();
            let tensor = Tensor::new(values, vec![1, cols])
                .map_err(|err| read_flow(MESSAGE_ID_INTERNAL, format!("read: {err}")))?;
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

fn numeric_from_bytes(
    bytes: &[u8],
    datatype: DataType,
    order: ByteOrder,
) -> BuiltinResult<Vec<f64>> {
    let size = datatype.element_size();
    if size == 0 {
        return Ok(Vec::new());
    }
    if !bytes.len().is_multiple_of(size) {
        return Err(read_flow(
            MESSAGE_ID_INTERNAL,
            "read: received byte count does not align with datatype size",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / size);
    let chunks = bytes.chunks_exact(size);
    for chunk in chunks {
        let value = match datatype {
            DataType::UInt8 => chunk[0] as f64,
            DataType::Int8 => (chunk[0] as i8) as f64,
            DataType::UInt16 => u16_from(chunk, order) as f64,
            DataType::Int16 => i16_from(chunk, order) as f64,
            DataType::UInt32 => u32_from(chunk, order) as f64,
            DataType::Int32 => i32_from(chunk, order) as f64,
            DataType::UInt64 => u64_from(chunk, order) as f64,
            DataType::Int64 => i64_from(chunk, order) as f64,
            DataType::Single => f32_from(chunk, order) as f64,
            DataType::Double => f64_from(chunk, order),
            DataType::Char | DataType::String => unreachable!(),
        };
        out.push(value);
    }
    Ok(out)
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

async fn parse_arguments(rest: Vec<Value>) -> BuiltinResult<ReadOptions> {
    match rest.len() {
        0 => Ok(ReadOptions {
            mode: ReadMode::Available,
            datatype: DataType::UInt8,
        }),
        1 => {
            let count_value = gather_if_needed_async(&rest[0]).await?;
            let count = parse_count(&count_value)?;
            Ok(ReadOptions {
                mode: ReadMode::Count(count),
                datatype: DataType::UInt8,
            })
        }
        2 => {
            let count_value = gather_if_needed_async(&rest[0]).await?;
            let dtype_value = gather_if_needed_async(&rest[1]).await?;
            let count = parse_count(&count_value)?;
            let datatype = parse_datatype(&dtype_value)?;
            Ok(ReadOptions {
                mode: ReadMode::Count(count),
                datatype,
            })
        }
        _ => Err(read_flow(
            MESSAGE_ID_INVALID_CLIENT,
            "read: invalid argument list",
        )),
    }
}

fn parse_count(value: &Value) -> BuiltinResult<usize> {
    let numeric = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            return Err(read_flow(
                MESSAGE_ID_INVALID_COUNT,
                "read: count must be a numeric scalar",
            ))
        }
    };
    if numeric.is_nan() || numeric.is_sign_negative() {
        return Err(read_flow(
            MESSAGE_ID_INVALID_COUNT,
            "read: count must be a non-negative finite value",
        ));
    }
    if numeric.is_infinite() {
        return Err(read_flow(
            MESSAGE_ID_INVALID_COUNT,
            "read: count must be finite",
        ));
    }
    if numeric > usize::MAX as f64 {
        return Err(read_flow(
            MESSAGE_ID_INVALID_COUNT,
            "read: count exceeds the maximum supported size",
        ));
    }
    Ok(numeric.trunc() as usize)
}

fn parse_datatype(value: &Value) -> BuiltinResult<DataType> {
    let text = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(read_flow(
                MESSAGE_ID_INVALID_DATATYPE,
                "read: datatype must be a string scalar",
            ))
        }
    };
    let lowered = text.trim().to_ascii_lowercase();
    let dtype = match lowered.as_str() {
        "uint8" => DataType::UInt8,
        "int8" => DataType::Int8,
        "uint16" => DataType::UInt16,
        "int16" => DataType::Int16,
        "uint32" => DataType::UInt32,
        "int32" => DataType::Int32,
        "uint64" => DataType::UInt64,
        "int64" => DataType::Int64,
        "single" => DataType::Single,
        "double" => DataType::Double,
        "char" => DataType::Char,
        "string" => DataType::String,
        _ => {
            return Err(read_flow(
                MESSAGE_ID_INVALID_DATATYPE,
                format!("read: unsupported datatype '{text}'"),
            ))
        }
    };
    Ok(dtype)
}

fn extract_client_id(struct_value: &StructValue) -> BuiltinResult<u64> {
    let id_value = struct_field(struct_value, CLIENT_HANDLE_FIELD).ok_or_else(|| {
        read_flow(
            MESSAGE_ID_INVALID_CLIENT,
            "read: tcpclient struct is missing internal handle",
        )
    })?;
    match id_value {
        Value::Int(IntValue::U64(id)) => Ok(*id),
        Value::Int(iv) => Ok(iv.to_i64() as u64),
        _ => Err(read_flow(
            MESSAGE_ID_INVALID_CLIENT,
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::net::accept::{
        configure_stream, insert_client, remove_client_for_test,
    };
    use runmat_builtins::{IntValue, StructValue, Value};
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn make_client(stream: TcpStream, timeout: f64) -> Value {
        let peer_addr = stream.peer_addr().expect("peer addr");
        configure_stream(&stream, timeout).expect("configure stream");
        let client_id = insert_client(stream, 0, peer_addr, timeout, "little-endian".to_string());
        let mut st = StructValue::new();
        st.fields.insert(
            CLIENT_HANDLE_FIELD.to_string(),
            Value::Int(IntValue::U64(client_id)),
        );
        Value::Struct(st)
    }

    fn client_id(client: &Value) -> u64 {
        match client {
            Value::Struct(st) => match st.fields.get(CLIENT_HANDLE_FIELD) {
                Some(Value::Int(IntValue::U64(id))) => *id,
                Some(Value::Int(iv)) => iv.to_i64() as u64,
                other => panic!("unexpected id field {other:?}"),
            },
            other => panic!("expected struct, got {other:?}"),
        }
    }

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    fn run_read(client: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(read_builtin(client, rest))
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::builtins::io::net::accept::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn read_reads_requested_uint8_values() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let payload: Vec<u8> = (1..=10).collect();
            stream.write_all(&payload).expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let data = run_read(client.clone(), vec![Value::Num(6.0)]).expect("read");
        let tensor = match data {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![1, 6]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        handle.join().expect("server thread");
        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn read_without_count_drains_available_bytes() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            std::thread::sleep(Duration::from_millis(50));
            let payload: Vec<u8> = vec![42, 43, 44];
            stream.write_all(&payload).expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let data = run_read(client.clone(), Vec::new()).expect("read");
        let tensor = match data {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.data, vec![42.0, 43.0, 44.0]);

        handle.join().expect("server thread");
        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn read_respects_timeout() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let _handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            // Keep the connection open well beyond the client timeout so the
            // read call experiences a timeout rather than a clean closure.
            std::thread::sleep(Duration::from_millis(500));
            drop(stream);
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 0.1);

        let err = run_read(client.clone(), vec![Value::Num(4.0)]).unwrap_err();
        assert_error_identifier(err, MESSAGE_ID_TIMEOUT);

        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let _guard = net_guard();
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
