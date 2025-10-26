//! MATLAB-compatible `read` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{CharArray, IntValue, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::io::{self, Read};
use std::net::TcpStream;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

use super::accept::{client_handle, configure_stream, CLIENT_HANDLE_FIELD};

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

const MESSAGE_ID_INVALID_CLIENT: &str = "MATLAB:read:InvalidTcpClient";
const MESSAGE_ID_NOT_CONNECTED: &str = "MATLAB:read:NotConnected";
const MESSAGE_ID_TIMEOUT: &str = "MATLAB:read:Timeout";
const MESSAGE_ID_CONNECTION_CLOSED: &str = "MATLAB:read:ConnectionClosed";
const MESSAGE_ID_INVALID_COUNT: &str = "MATLAB:read:InvalidCount";
const MESSAGE_ID_INVALID_DATATYPE: &str = "MATLAB:read:InvalidDataType";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:read:InternalError";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "read"
category: "io/net"
keywords: ["read", "tcpclient", "networking", "socket", "binary data"]
summary: "Read numeric or text data from a remote host through a MATLAB-compatible tcpclient struct."
references:
  - https://www.mathworks.com/help/matlab/ref/tcpclient.read.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "All TCP reads run on the host CPU. GPU-resident arguments are gathered before any socket I/O; providers do not expose networking hooks."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::read::tests"
---

# What does the `read` function do in MATLAB / RunMat?
`read(t)` consumes data waiting on the TCP/IP client returned by `tcpclient` (or `accept`). The builtin
mimics MATLAB’s behaviour for `read` so existing code that exchanges bytes with remote services behaves
identically. It honours the client’s configured `Timeout`, respects the `ByteOrder` property when
materialising multi-byte numeric types, and interprets the optional `datatype` argument just like MATLAB.

## How does the `read` function behave in MATLAB / RunMat?
- `data = read(t)` waits until at least one byte becomes available (subject to the client timeout) and
  then drains the socket buffer, returning a **row vector of doubles** whose values come from the received
  bytes. If the peer closes the connection without sending data, the result is an empty row vector.
- `data = read(t, count)` blocks until exactly `count` values are available (again honouring `Timeout`).
  When the peer closes the socket before satisfying the request the builtin raises
  `MATLAB:read:ConnectionClosed`.
- `data = read(t, count, datatype)` interprets the values using the requested MATLAB datatype. Supported
  tokens mirror MATLAB: `"uint8"` (default), `"int8"`, `"uint16"`, `"int16"`, `"uint32"`, `"int32"`,
  `"uint64"`, `"int64"`, `"single"`, `"double"`, `"char"`, and `"string"`. Numeric outputs are returned
  as doubles; `"char"` produces a MATLAB-style character row vector and `"string"` returns a scalar string.
- Every call honours the client’s `ByteOrder` property when decoding multi-byte numbers. Little-endian is
  the default, but `"big-endian"` is respected for data written in network byte order.
- The builtin gathers GPU-resident arguments automatically, executes the socket read on the CPU, and
  returns host values.
- Errors are raised with MATLAB-compatible identifiers: invalid client structs trigger
  `MATLAB:read:InvalidTcpClient`, timeouts emit `MATLAB:read:Timeout`, and connection closures before a
  requested count is met raise `MATLAB:read:ConnectionClosed`.

## `read` Function GPU Execution Behaviour
Networking is a host-only subsystem. When a client struct or argument arrives from the GPU, RunMat gathers
the value back to the CPU before reading from the socket. No acceleration-provider hooks participate in
the operation, and the result is always a host value (double tensor, char array, or string). Future GPU
providers continue to gather metadata automatically so networking remains CPU-bound.

## Examples of using the `read` function in MATLAB / RunMat

### Reading a fixed number of bytes from a TCP echo service
```matlab
client = tcpclient("127.0.0.1", 50000);
write(client, uint8(1:6));
payload = read(client, 6);
```
Expected output (double row vector):
```matlab
payload =
     1     2     3     4     5     6
```

### Reading ASCII text as characters
```matlab
client = tcpclient("127.0.0.1", 50001);
write(client, "RunMat TCP");
chars = read(client, 10, "char");
```
Expected output:
```matlab
chars =
    'RunMat TCP'
```

### Reading doubles written in big-endian byte order
```matlab
client = tcpclient("localhost", 50002, "ByteOrder", "big-endian");
write(client, swapbytes([1 2 3], "double"));
values = read(client, 3, "double");
```
Expected output:
```matlab
values =
     1     2     3
```

### Reading all available data without specifying a count
```matlab
client = tcpclient("127.0.0.1", 50003);
write(client, uint8([10 20 30]));
burst = read(client);
```
Expected output:
```matlab
burst =
    10    20    30
```

### Handling read timeouts gracefully
```matlab
client = tcpclient("example.com", 12345, "Timeout", 0.5);
try
    data = read(client, 64);
catch err
    disp(err.identifier)
end
```
Expected output:
```matlab
MATLAB:read:Timeout
```

## FAQ

### Does `read` modify the client struct?
No. The builtin interacts with the socket stored in RunMat’s internal registry. The visible struct returned
from `tcpclient` is passed by value and is not mutated in place.

### What happens when the remote host closes the connection?
If the peer closes the connection before the requested count is satisfied, `read` raises
`MATLAB:read:ConnectionClosed`. When no specific count is requested (`read(t)`), the builtin returns whatever
data was available before the closure (possibly an empty vector).

### Does `read` support infinite timeouts?
Yes. Setting `t.Timeout = Inf` (or passing `"Timeout", inf` when constructing the client) leaves the socket in
blocking mode. The builtin waits indefinitely until enough data arrives or the peer closes the connection.

### How are multibyte integers decoded?
RunMat honours the client’s `ByteOrder` property (`"little-endian"` or `"big-endian"`). For example,
`read(t, 4, "uint16")` consumes eight bytes and interprets each pair in the configured byte order.

### Can I read UTF-8 strings directly?
Use the `"string"` datatype. The builtin converts each received byte directly into a MATLAB string scalar
assuming UTF-8 (non-UTF-8 sequences fall back to byte-wise decoding).

## See also
[tcpclient](./tcpclient), [accept](./accept), [write](./write), [readline](./readline)

## Source & feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/net/read.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/net/read.rs)
- Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with repro steps if you observe a behavioural difference from MATLAB.
"#;

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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "read",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("read", DOC_MD);

#[runtime_builtin(
    name = "read",
    category = "io/net",
    summary = "Read numeric or text data from a TCP/IP client.",
    keywords = "read,tcpclient,networking"
)]
fn read_builtin(client: Value, rest: Vec<Value>) -> Result<Value, String> {
    let client = gather_if_needed(&client)?;
    let options = parse_arguments(rest)?;

    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(runtime_error(
                MESSAGE_ID_INVALID_CLIENT,
                "read: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        runtime_error(
            MESSAGE_ID_INVALID_CLIENT,
            "read: tcpclient handle is no longer valid",
        )
    })?;

    let (stream, timeout, byte_order, connected) = {
        let guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(runtime_error(
                MESSAGE_ID_NOT_CONNECTED,
                "read: tcpclient is disconnected",
            ));
        }
        let timeout = guard.timeout;
        let byte_order = parse_byte_order(&guard.byte_order);
        let stream = guard.stream.try_clone().map_err(|err| {
            runtime_error(MESSAGE_ID_INTERNAL, format!("read: clone failed ({err})"))
        })?;
        (stream, timeout, byte_order, guard.connected)
    };

    // Ensure cloned descriptor uses the configured timeout.
    if connected {
        if let Err(err) = configure_stream(&stream, timeout) {
            return Err(runtime_error(
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
            return Err(runtime_error(
                MESSAGE_ID_CONNECTION_CLOSED,
                "read: connection closed before the requested data was received",
            ));
        }
        let expected = count.saturating_mul(element_size);
        if read_result.bytes.len() != expected {
            return Err(runtime_error(
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
) -> Result<ReadOutcome, String> {
    match read_from_stream(stream, mode, element_size) {
        Ok(outcome) => Ok(outcome),
        Err(ReadError::Timeout) => Err(runtime_error(
            MESSAGE_ID_TIMEOUT,
            "read: timed out waiting for data",
        )),
        Err(ReadError::ConnectionClosed) => Err(runtime_error(
            MESSAGE_ID_CONNECTION_CLOSED,
            "read: connection closed before the requested data was received",
        )),
        Err(ReadError::Io(err)) => Err(runtime_error(
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
                break;
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

    if !data.is_empty() {
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
    }

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
    while offset < total {
        match stream.read(&mut buf[offset..]) {
            Ok(0) => return Err(ReadError::ConnectionClosed),
            Ok(n) => {
                offset += n;
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) if is_timeout(&err) => return Err(ReadError::Timeout),
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

fn bytes_to_value(bytes: &[u8], datatype: DataType, order: ByteOrder) -> Result<Value, String> {
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
            let tensor =
                Tensor::new(values, vec![1, cols]).map_err(|err| format!("read: {err}"))?;
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
) -> Result<Vec<f64>, String> {
    let size = datatype.element_size();
    if size == 0 {
        return Ok(Vec::new());
    }
    if bytes.len() % size != 0 {
        return Err(runtime_error(
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

fn parse_arguments(rest: Vec<Value>) -> Result<ReadOptions, String> {
    match rest.len() {
        0 => Ok(ReadOptions {
            mode: ReadMode::Available,
            datatype: DataType::UInt8,
        }),
        1 => {
            let count_value = gather_if_needed(&rest[0])?;
            let count = parse_count(&count_value)?;
            Ok(ReadOptions {
                mode: ReadMode::Count(count),
                datatype: DataType::UInt8,
            })
        }
        2 => {
            let count_value = gather_if_needed(&rest[0])?;
            let dtype_value = gather_if_needed(&rest[1])?;
            let count = parse_count(&count_value)?;
            let datatype = parse_datatype(&dtype_value)?;
            Ok(ReadOptions {
                mode: ReadMode::Count(count),
                datatype,
            })
        }
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_CLIENT,
            "read: invalid argument list",
        )),
    }
}

fn parse_count(value: &Value) -> Result<usize, String> {
    let numeric = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            return Err(runtime_error(
                MESSAGE_ID_INVALID_COUNT,
                "read: count must be a numeric scalar",
            ))
        }
    };
    if numeric.is_nan() || numeric.is_sign_negative() {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_COUNT,
            "read: count must be a non-negative finite value",
        ));
    }
    if numeric.is_infinite() {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_COUNT,
            "read: count must be finite",
        ));
    }
    if numeric > usize::MAX as f64 {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_COUNT,
            "read: count exceeds the maximum supported size",
        ));
    }
    Ok(numeric.trunc() as usize)
}

fn parse_datatype(value: &Value) -> Result<DataType, String> {
    let text = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(runtime_error(
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
            return Err(runtime_error(
                MESSAGE_ID_INVALID_DATATYPE,
                format!("read: unsupported datatype '{text}'"),
            ))
        }
    };
    Ok(dtype)
}

fn extract_client_id(struct_value: &StructValue) -> Result<u64, String> {
    let id_value = struct_field(struct_value, CLIENT_HANDLE_FIELD).ok_or_else(|| {
        runtime_error(
            MESSAGE_ID_INVALID_CLIENT,
            "read: tcpclient struct is missing internal handle",
        )
    })?;
    match id_value {
        Value::Int(IntValue::U64(id)) => Ok(*id),
        Value::Int(iv) => Ok(iv.to_i64() as u64),
        _ => Err(runtime_error(
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

fn runtime_error(message_id: &'static str, message: impl Into<String>) -> String {
    format!("{message_id}: {}", message.into())
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
mod tests {
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;
    use crate::builtins::io::net::accept::{
        configure_stream, insert_client, remove_client_for_test,
    };
    use runmat_builtins::{IntValue, StructValue, Value};
    use std::collections::HashMap;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn make_client(stream: TcpStream, timeout: f64) -> Value {
        let peer_addr = stream.peer_addr().expect("peer addr");
        configure_stream(&stream, timeout).expect("configure stream");
        let client_id = insert_client(stream, 0, peer_addr, timeout, "little-endian".to_string());
        let mut fields = HashMap::new();
        fields.insert(
            CLIENT_HANDLE_FIELD.to_string(),
            Value::Int(IntValue::U64(client_id)),
        );
        Value::Struct(StructValue { fields })
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

    #[test]
    fn read_reads_requested_uint8_values() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let payload: Vec<u8> = (1..=10).collect();
            stream.write_all(&payload).expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let data = read_builtin(client.clone(), vec![Value::Num(6.0)]).expect("read");
        let tensor = match data {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![1, 6]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        handle.join().expect("server thread");
        remove_client_for_test(client_id(&client));
    }

    #[test]
    fn read_without_count_drains_available_bytes() {
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

        let data = read_builtin(client.clone(), Vec::new()).expect("read");
        let tensor = match data {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.data, vec![42.0, 43.0, 44.0]);

        handle.join().expect("server thread");
        remove_client_for_test(client_id(&client));
    }

    #[test]
    fn read_respects_timeout() {
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

        let err = read_builtin(client.clone(), vec![Value::Num(4.0)]).unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_TIMEOUT));

        remove_client_for_test(client_id(&client));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
