//! MATLAB-compatible `write` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;
use std::io::{self, Write};
use std::net::TcpStream;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

use super::accept::{client_handle, configure_stream, CLIENT_HANDLE_FIELD};

const MESSAGE_ID_INVALID_CLIENT: &str = "MATLAB:write:InvalidTcpClient";
const MESSAGE_ID_INVALID_DATA: &str = "MATLAB:write:InvalidData";
const MESSAGE_ID_INVALID_DATATYPE: &str = "MATLAB:write:InvalidDataType";
const MESSAGE_ID_NOT_CONNECTED: &str = "MATLAB:write:NotConnected";
const MESSAGE_ID_TIMEOUT: &str = "MATLAB:write:Timeout";
const MESSAGE_ID_CONNECTION_CLOSED: &str = "MATLAB:write:ConnectionClosed";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:write:InternalError";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "write")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "write"
category: "io/net"
keywords: ["write", "tcpclient", "networking", "socket", "binary data", "text"]
summary: "Write numeric or text data to a remote host through a MATLAB-compatible tcpclient struct."
references:
  - https://www.mathworks.com/help/matlab/ref/tcpclient.write.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "All TCP writes execute on the host CPU. GPU-resident arguments are gathered automatically before socket I/O."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::write::tests"
---

# What does the `write` function do in MATLAB / RunMat?
`write(t, data)` transmits binary or textual data over the TCP/IP client returned by `tcpclient` (or `accept`).
The builtin mirrors MATLAB’s `write` behaviour so existing socket code continues working without modification.
It honours the client’s configured `Timeout`, applies the `ByteOrder` property when encoding multi-byte values,
and accepts the optional `datatype` argument used throughout MATLAB’s I/O APIs.

## How does the `write` function behave in MATLAB / RunMat?
- `write(t, data)` converts `data` to unsigned 8-bit integers (the MATLAB default) and sends the bytes to the peer.
  The return value is the number of elements written when the caller requests an output argument.
- `write(t, data, datatype)` encodes the payload using the supplied MATLAB datatype token.
  Supported values mirror MATLAB: `"uint8"` (default), `"int8"`, `"uint16"`, `"int16"`, `"uint32"`, `"int32"`,
  `"uint64"`, `"int64"`, `"single"`, `"double"`, `"char"`, and `"string"`. Numeric conversions saturate to the
  destination range just like MATLAB cast operations. `"char"` treats values as single-byte character codes and
  `"string"` encodes UTF-8 text.
- The client’s `ByteOrder` property controls how multi-byte numeric values are serialised. `"little-endian"` is
  the default, while `"big-endian"` matches the traditional network byte order.
- When the socket cannot send the entire payload before the timeout expires, `write` raises `MATLAB:write:Timeout`.
  If the peer closes the connection before or during the transfer the builtin raises `MATLAB:write:ConnectionClosed`
  and marks the client as disconnected.
- Inputs that originate on the GPU are gathered back to the host automatically before any bytes are written.

## `write` Function GPU Execution Behaviour
Networking occurs on the host CPU. If `data` or the tcpclient struct resides on the GPU, RunMat gathers the values
to host memory before converting them to bytes. Acceleration providers are not involved and the resulting payload
remains on the CPU. Providers that support residency tracking automatically mark any gathered tensors as released.

## Examples of using the `write` function in MATLAB / RunMat

### Sending an array of bytes to an echo service
```matlab
client = tcpclient("127.0.0.1", 50000);
count = write(client, uint8(1:4));
```
Expected output when an output argument is requested:
```matlab
count =
     4
```

### Writing doubles with explicit byte order
```matlab
client = tcpclient("localhost", 50001, "ByteOrder", "big-endian");
values = [1.5 2.5 3.5];
write(client, values, "double");
```
The remote peer receives 24 bytes representing the doubles in big-endian order.

### Transmitting ASCII text
```matlab
client = tcpclient("127.0.0.1", 50002);
write(client, "RunMat TCP", "char");
```
Expected payload (one byte per character):
```
52 117 110 77 97 116 32 84 67 80
```

### Sending UTF-8 encoded strings
```matlab
client = tcpclient("127.0.0.1", 50003);
write(client, "Διακριτό", "string");
```
The builtin encodes the Unicode text as UTF-8 before sending it across the socket.

### Handling connection closures
```matlab
client = tcpclient("example.com", 12345, "Timeout", 0.25);
try
    write(client, uint8([1 2 3 4]));
catch err
    disp(err.identifier)
end
```
Expected output when the peer closes the connection abruptly:
```matlab
MATLAB:write:ConnectionClosed
```

## FAQ

### How many output values does `write` return?
When the caller requests an output, the builtin returns the number of elements written (after datatype conversion).
This mirrors the behaviour of MATLAB’s numeric I/O routines. If no output is requested, the value is discarded.

### Does `write` support complex numbers?
No. The input must be real. Pass separate real and imaginary parts or convert to a byte representation manually.

### How are values rounded when converting to integer datatypes?
Floating-point inputs are rounded to the nearest integer and then saturated to the target range, matching MATLAB
casts (for example `uint8(255.7)` becomes `256 → 255`, `int8(-128.2)` becomes `-128`).

### What happens to GPU-resident tensors?
They are gathered automatically before the write. Networking is a CPU-only subsystem, so the resulting data is sent
from host memory and any temporary handles are released after the transfer.

### Can I stream large payloads?
Yes. `write` loops until the entire payload has been sent or an error occurs. Large payloads honour the client’s
timeout and byte-order settings.

## See also
[tcpclient](./tcpclient), [accept](./accept), [read](./read), [readline](./readline)

## Source & feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/net/write.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/net/write.rs)
- Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) if you encounter behavioural
  differences from MATLAB.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "write",
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
    notes: "Socket writes always execute on the host CPU; GPU providers are never consulted.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "write",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "write",
    category = "io/net",
    summary = "Write numeric or text data to a TCP/IP client.",
    keywords = "write,tcpclient,networking"
)]
fn write_builtin(client: Value, data: Value, rest: Vec<Value>) -> Result<Value, String> {
    let client = gather_if_needed(&client)
        .map_err(|err| runtime_error(MESSAGE_ID_INVALID_CLIENT, format!("write: {err}")))?;
    let data = gather_if_needed(&data)
        .map_err(|err| runtime_error(MESSAGE_ID_INVALID_DATA, format!("write: {err}")))?;

    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in rest {
        gathered_rest.push(
            gather_if_needed(&value).map_err(|err| {
                runtime_error(MESSAGE_ID_INVALID_DATATYPE, format!("write: {err}"))
            })?,
        );
    }
    let datatype = parse_arguments(&gathered_rest)?;

    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(runtime_error(
                MESSAGE_ID_INVALID_CLIENT,
                "write: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        runtime_error(
            MESSAGE_ID_INVALID_CLIENT,
            "write: tcpclient handle is no longer valid",
        )
    })?;

    let (mut stream, timeout, byte_order) = {
        let guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(runtime_error(
                MESSAGE_ID_NOT_CONNECTED,
                "write: tcpclient is disconnected",
            ));
        }
        let timeout = guard.timeout;
        let byte_order = parse_byte_order(&guard.byte_order);
        let stream = guard.stream.try_clone().map_err(|err| {
            runtime_error(MESSAGE_ID_INTERNAL, format!("write: clone failed ({err})"))
        })?;
        (stream, timeout, byte_order)
    };

    if let Err(err) = configure_stream(&stream, timeout) {
        return Err(runtime_error(
            MESSAGE_ID_INTERNAL,
            format!("write: unable to configure socket timeout ({err})"),
        ));
    }

    let payload = prepare_payload(&data, datatype, byte_order)?;
    if payload.bytes.is_empty() {
        return Ok(Value::Num(0.0));
    }

    match write_bytes(&mut stream, &payload.bytes) {
        Ok(_) => Ok(Value::Num(payload.elements as f64)),
        Err(WriteError::Timeout) => Err(runtime_error(
            MESSAGE_ID_TIMEOUT,
            "write: timed out while sending data",
        )),
        Err(WriteError::ConnectionClosed) => {
            if let Ok(mut guard) = handle.lock() {
                guard.connected = false;
            }
            Err(runtime_error(
                MESSAGE_ID_CONNECTION_CLOSED,
                "write: connection closed before all data was sent",
            ))
        }
        Err(WriteError::Io(err)) => Err(runtime_error(
            MESSAGE_ID_INTERNAL,
            format!("write: socket error ({err})"),
        )),
    }
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
    fn default() -> Self {
        DataType::UInt8
    }

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

struct Payload {
    bytes: Vec<u8>,
    elements: usize,
}

fn parse_arguments(args: &[Value]) -> Result<DataType, String> {
    match args.len() {
        0 => Ok(DataType::default()),
        1 => parse_datatype(&args[0]),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_DATATYPE,
            "write: expected at most one datatype argument",
        )),
    }
}

fn parse_datatype(value: &Value) -> Result<DataType, String> {
    let text =
        scalar_string(value).map_err(|err| runtime_error(MESSAGE_ID_INVALID_DATATYPE, err))?;
    let lowered = text.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_DATATYPE,
            "write: datatype must not be empty",
        ));
    }
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
                format!("write: unsupported datatype '{text}'"),
            ))
        }
    };
    Ok(dtype)
}

fn prepare_payload(data: &Value, datatype: DataType, order: ByteOrder) -> Result<Payload, String> {
    match datatype {
        DataType::Char => char_payload(data),
        DataType::String => string_payload(data),
        _ => numeric_payload(data, datatype, order),
    }
}

fn numeric_payload(data: &Value, datatype: DataType, order: ByteOrder) -> Result<Payload, String> {
    let values = flatten_numeric(data)?;
    let mut bytes = Vec::with_capacity(values.len() * datatype.element_size());
    for value in values.iter().copied() {
        match datatype {
            DataType::UInt8 => bytes.push(cast_to_u8(value)),
            DataType::Int8 => bytes.push(cast_to_i8(value) as u8),
            DataType::UInt16 => extend_u16(&mut bytes, cast_to_u16(value), order),
            DataType::Int16 => extend_i16(&mut bytes, cast_to_i16(value), order),
            DataType::UInt32 => extend_u32(&mut bytes, cast_to_u32(value), order),
            DataType::Int32 => extend_i32(&mut bytes, cast_to_i32(value), order),
            DataType::UInt64 => extend_u64(&mut bytes, cast_to_u64(value), order),
            DataType::Int64 => extend_i64(&mut bytes, cast_to_i64(value), order),
            DataType::Single => extend_f32(&mut bytes, cast_to_f32(value), order),
            DataType::Double => extend_f64(&mut bytes, value, order),
            DataType::Char | DataType::String => unreachable!(),
        }
    }
    Ok(Payload {
        bytes,
        elements: values.len(),
    })
}

fn char_payload(data: &Value) -> Result<Payload, String> {
    let bytes = match data {
        Value::CharArray(ca) => ca.data.iter().map(|&ch| (ch as u32 & 0xFF) as u8).collect(),
        Value::String(text) => text.bytes().collect(),
        Value::StringArray(sa) => {
            if sa.data.len() != 1 {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_DATA,
                    "write: string array input must be scalar when using 'char'",
                ));
            }
            sa.data[0].as_bytes().to_vec()
        }
        Value::Tensor(t) => t.data.iter().map(|&v| cast_to_u8(v)).collect::<Vec<u8>>(),
        Value::Num(n) => vec![cast_to_u8(*n)],
        Value::Int(iv) => vec![cast_to_u8(iv.to_f64())],
        Value::Bool(b) => vec![if *b { 1 } else { 0 }],
        Value::LogicalArray(la) => la
            .data
            .iter()
            .map(|&b| if b != 0 { 1 } else { 0 })
            .collect(),
        _ => {
            return Err(runtime_error(
                MESSAGE_ID_INVALID_DATA,
                "write: unsupported input for 'char' datatype",
            ))
        }
    };
    Ok(Payload {
        elements: bytes.len(),
        bytes,
    })
}

fn string_payload(data: &Value) -> Result<Payload, String> {
    match data {
        Value::String(text) => Ok(Payload {
            elements: 1,
            bytes: text.as_bytes().to_vec(),
        }),
        Value::CharArray(ca) => {
            let string: String = ca.data.iter().collect();
            Ok(Payload {
                elements: 1,
                bytes: string.into_bytes(),
            })
        }
        Value::StringArray(sa) => {
            if sa.data.is_empty() {
                return Ok(Payload {
                    elements: 0,
                    bytes: Vec::new(),
                });
            }
            if sa.data.len() != 1 {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_DATA,
                    "write: string array input must be scalar when using 'string'",
                ));
            }
            Ok(Payload {
                elements: 1,
                bytes: sa.data[0].as_bytes().to_vec(),
            })
        }
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_DATA,
            "write: expected text input when using 'string' datatype",
        )),
    }
}

fn flatten_numeric(value: &Value) -> Result<Vec<f64>, String> {
    match value {
        Value::Tensor(t) => Ok(t.data.clone()),
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(iv) => Ok(vec![iv.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::LogicalArray(la) => Ok(la
            .data
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::CharArray(ca) => Ok(ca
            .data
            .iter()
            .map(|&ch| (ch as u32 & 0xFF) as f64)
            .collect()),
        Value::String(text) => Ok(text.chars().map(|ch| (ch as u32) as f64).collect()),
        Value::StringArray(sa) => {
            if sa.data.len() != 1 {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_DATA,
                    "write: string array input must be scalar",
                ));
            }
            Ok(sa.data[0].chars().map(|ch| (ch as u32) as f64).collect())
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(runtime_error(
            MESSAGE_ID_INVALID_DATA,
            "write: complex data is not supported",
        )),
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(runtime_error(
            MESSAGE_ID_INVALID_DATA,
            "write: unsupported input type",
        )),
        Value::GpuTensor(_) => Err(runtime_error(
            MESSAGE_ID_INVALID_DATA,
            "write: GPU tensor should have been gathered before encoding",
        )),
    }
}

fn cast_to_u8(value: f64) -> u8 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            0
        } else {
            u8::MAX
        };
    }
    if rounded < 0.0 {
        0
    } else if rounded > u8::MAX as f64 {
        u8::MAX
    } else {
        rounded as u8
    }
}

fn cast_to_i8(value: f64) -> i8 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            i8::MIN
        } else {
            i8::MAX
        };
    }
    if rounded < i8::MIN as f64 {
        i8::MIN
    } else if rounded > i8::MAX as f64 {
        i8::MAX
    } else {
        rounded as i8
    }
}

fn cast_to_u16(value: f64) -> u16 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            0
        } else {
            u16::MAX
        };
    }
    if rounded < 0.0 {
        0
    } else if rounded > u16::MAX as f64 {
        u16::MAX
    } else {
        rounded as u16
    }
}

fn cast_to_i16(value: f64) -> i16 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            i16::MIN
        } else {
            i16::MAX
        };
    }
    if rounded < i16::MIN as f64 {
        i16::MIN
    } else if rounded > i16::MAX as f64 {
        i16::MAX
    } else {
        rounded as i16
    }
}

fn cast_to_u32(value: f64) -> u32 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            0
        } else {
            u32::MAX
        };
    }
    if rounded < 0.0 {
        0
    } else if rounded > u32::MAX as f64 {
        u32::MAX
    } else {
        rounded as u32
    }
}

fn cast_to_i32(value: f64) -> i32 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            i32::MIN
        } else {
            i32::MAX
        };
    }
    if rounded < i32::MIN as f64 {
        i32::MIN
    } else if rounded > i32::MAX as f64 {
        i32::MAX
    } else {
        rounded as i32
    }
}

fn cast_to_u64(value: f64) -> u64 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            0
        } else {
            u64::MAX
        };
    }
    if rounded < 0.0 {
        0
    } else if rounded > u64::MAX as f64 {
        u64::MAX
    } else {
        rounded as u64
    }
}

fn cast_to_i64(value: f64) -> i64 {
    let rounded = rounded_scalar(value);
    if !rounded.is_finite() {
        return if rounded.is_sign_negative() {
            i64::MIN
        } else {
            i64::MAX
        };
    }
    if rounded < i64::MIN as f64 {
        i64::MIN
    } else if rounded > i64::MAX as f64 {
        i64::MAX
    } else {
        rounded as i64
    }
}

fn cast_to_f32(value: f64) -> f32 {
    value as f32
}

fn rounded_scalar(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.round()
    }
}

fn extend_u16(buffer: &mut Vec<u8>, value: u16, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_i16(buffer: &mut Vec<u8>, value: i16, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_u32(buffer: &mut Vec<u8>, value: u32, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_i32(buffer: &mut Vec<u8>, value: i32, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_u64(buffer: &mut Vec<u8>, value: u64, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_i64(buffer: &mut Vec<u8>, value: i64, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_f32(buffer: &mut Vec<u8>, value: f32, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn extend_f64(buffer: &mut Vec<u8>, value: f64, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

fn parse_byte_order(text: &str) -> ByteOrder {
    if text.eq_ignore_ascii_case("big-endian") || text.eq_ignore_ascii_case("big endian") {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    }
}

fn scalar_string(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(
            "write: datatype argument must be a string scalar or character row vector".to_string(),
        ),
    }
}

fn extract_client_id(struct_value: &StructValue) -> Result<u64, String> {
    let id_value = struct_value
        .fields
        .get(CLIENT_HANDLE_FIELD)
        .ok_or_else(|| {
            runtime_error(
                MESSAGE_ID_INVALID_CLIENT,
                "write: tcpclient struct is missing internal handle",
            )
        })?;
    match id_value {
        Value::Int(IntValue::U64(id)) => Ok(*id),
        Value::Int(iv) => Ok(iv.to_i64() as u64),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_CLIENT,
            "write: tcpclient struct has invalid handle field",
        )),
    }
}

fn runtime_error(message_id: &'static str, message: impl Into<String>) -> String {
    format!("{message_id}: {}", message.into())
}

enum WriteError {
    Timeout,
    ConnectionClosed,
    Io(io::Error),
}

fn write_bytes(stream: &mut TcpStream, bytes: &[u8]) -> Result<(), WriteError> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        match stream.write(&bytes[offset..]) {
            Ok(0) => return Err(WriteError::ConnectionClosed),
            Ok(n) => offset += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) if is_timeout(&err) => return Err(WriteError::Timeout),
            Err(err) if is_connection_closed_error(&err) => {
                return Err(WriteError::ConnectionClosed)
            }
            Err(err) => return Err(WriteError::Io(err)),
        }
    }
    Ok(())
}

fn is_timeout(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}

fn is_connection_closed_error(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::BrokenPipe
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::NotConnected
            | io::ErrorKind::UnexpectedEof
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::net::accept::{
        configure_stream, insert_client, remove_client_for_test,
    };
    use runmat_builtins::{CharArray, IntValue, StructValue, Tensor};
    use std::io::Read;
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn make_client(stream: TcpStream, timeout: f64, byte_order: &str) -> Value {
        let peer_addr = stream.peer_addr().expect("peer addr");
        configure_stream(&stream, timeout).expect("configure stream");
        let client_id = insert_client(stream, 0, peer_addr, timeout, byte_order.to_string());
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

    #[test]
    fn write_default_uint8_sends_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut received = Vec::new();
            stream.read_to_end(&mut received).unwrap_or_default();
            received
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0, "little-endian");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result =
            write_builtin(client.clone(), Value::Tensor(tensor), Vec::new()).expect("write");
        match result {
            Value::Num(count) => assert_eq!(count, 4.0),
            other => panic!("expected numeric result, got {other:?}"),
        }
        remove_client_for_test(client_id(&client));
        let received = handle.join().expect("join");
        assert_eq!(received, vec![1, 2, 3, 4]);
    }

    #[test]
    fn write_double_big_endian_encodes_correctly() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 24];
            stream.read_exact(&mut buf).expect("read");
            buf
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0, "big-endian");
        let tensor = Tensor::new(vec![1.5, 2.5, 3.5], vec![1, 3]).unwrap();
        let result = write_builtin(
            client.clone(),
            Value::Tensor(tensor),
            vec![Value::from("double")],
        )
        .expect("write");
        match result {
            Value::Num(count) => assert_eq!(count, 3.0),
            other => panic!("expected numeric count, got {other:?}"),
        }
        remove_client_for_test(client_id(&client));

        let received = handle.join().expect("join");
        let mut expected = Vec::new();
        extend_f64(&mut expected, 1.5, ByteOrder::Big);
        extend_f64(&mut expected, 2.5, ByteOrder::Big);
        extend_f64(&mut expected, 3.5, ByteOrder::Big);
        assert_eq!(received.to_vec(), expected);
    }

    #[test]
    fn write_char_payload_encodes_ascii() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).unwrap_or_default();
            buf
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0, "little-endian");
        let chars = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = write_builtin(
            client.clone(),
            Value::CharArray(chars),
            vec![Value::from("char")],
        )
        .expect("write");
        match result {
            Value::Num(count) => assert_eq!(count, 6.0),
            other => panic!("expected numeric count, got {other:?}"),
        }
        remove_client_for_test(client_id(&client));
        let received = handle.join().expect("join");
        assert_eq!(received, b"RunMat");
    }

    #[test]
    fn write_errors_when_client_disconnected() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let barrier = Arc::new(Barrier::new(2));
        let thread_barrier = barrier.clone();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            thread_barrier.wait();
            drop(stream);
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0, "little-endian");
        let id = client_id(&client);
        if let Some(handle_ref) = client_handle(id) {
            if let Ok(mut guard) = handle_ref.lock() {
                guard.connected = false;
            }
        }

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err =
            write_builtin(client.clone(), Value::Tensor(tensor), Vec::new()).expect_err("write");
        assert!(
            err.starts_with(MESSAGE_ID_NOT_CONNECTED),
            "unexpected error {err}"
        );

        remove_client_for_test(id);
        barrier.wait();
        handle.join().expect("join");
    }

    #[test]
    fn doc_examples_compile() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
