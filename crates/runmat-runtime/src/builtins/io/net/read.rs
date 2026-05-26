//! MATLAB-compatible `read` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, IntValue, StructValue, Tensor, Value,
};
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

const BUILTIN_NAME: &str = "read";

const READ_OUTPUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Available bytes or requested typed payload from the TCP client.",
}];
const READ_INPUTS_CLIENT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "client",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpclient handle struct.",
}];
const READ_INPUTS_CLIENT_COUNT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "client",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpclient handle struct.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Requested element count.",
    },
];
const READ_INPUTS_CLIENT_COUNT_DATATYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "client",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpclient handle struct.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Requested element count.",
    },
    BuiltinParamDescriptor {
        name: "datatype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Data type label (for example \"uint8\", \"double\", \"char\", \"string\").",
    },
];
const READ_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "data = read(client)",
        inputs: &READ_INPUTS_CLIENT,
        outputs: &READ_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = read(client, count)",
        inputs: &READ_INPUTS_CLIENT_COUNT,
        outputs: &READ_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = read(client, count, datatype)",
        inputs: &READ_INPUTS_CLIENT_COUNT_DATATYPE,
        outputs: &READ_OUTPUT_DATA,
    },
];

const READ_ERROR_INVALID_CLIENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_CLIENT",
    identifier: Some("RunMat:read:InvalidTcpClient"),
    when: "Client handle is missing, malformed, invalid, or disconnected.",
    message: "read: invalid tcpclient handle",
};
const READ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_INPUT",
    identifier: Some("RunMat:read:InvalidInput"),
    when: "Argument list shape is unsupported for read.",
    message: "read: invalid argument list",
};
const READ_ERROR_NOT_CONNECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.NOT_CONNECTED",
    identifier: Some("RunMat:read:NotConnected"),
    when: "Client has no active socket connection.",
    message: "read: tcpclient is disconnected",
};
const READ_ERROR_TIMEOUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.TIMEOUT",
    identifier: Some("RunMat:read:Timeout"),
    when: "Socket read exceeds configured timeout.",
    message: "read: timed out waiting for data",
};
const READ_ERROR_CONNECTION_CLOSED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.CONNECTION_CLOSED",
    identifier: Some("RunMat:read:ConnectionClosed"),
    when: "Peer closes socket before requested payload is fully available.",
    message: "read: connection closed before the requested data was received",
};
const READ_ERROR_INVALID_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_COUNT",
    identifier: Some("RunMat:read:InvalidCount"),
    when: "Requested count is non-scalar, negative, non-finite, or out of range.",
    message: "read: invalid count argument",
};
const READ_ERROR_INVALID_DATATYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_DATATYPE",
    identifier: Some("RunMat:read:InvalidDataType"),
    when: "Datatype argument is not a supported scalar text label.",
    message: "read: invalid datatype argument",
};
const READ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INTERNAL",
    identifier: Some("RunMat:read:InternalError"),
    when: "Internal socket/control-flow conversion fails.",
    message: "read: internal socket error",
};
const READ_ERRORS: [BuiltinErrorDescriptor; 8] = [
    READ_ERROR_INVALID_CLIENT,
    READ_ERROR_INVALID_INPUT,
    READ_ERROR_NOT_CONNECTED,
    READ_ERROR_TIMEOUT,
    READ_ERROR_CONNECTION_CLOSED,
    READ_ERROR_INVALID_COUNT,
    READ_ERROR_INVALID_DATATYPE,
    READ_ERROR_INTERNAL,
];
pub const READ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &READ_ERRORS,
};

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
    type_resolver(crate::builtins::io::type_resolvers::read_type),
    descriptor(crate::builtins::io::net::read::READ_DESCRIPTOR),
    builtin_path = "crate::builtins::io::net::read"
)]
async fn read_builtin(client: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let client = gather_if_needed_async(&client)
        .await
        .map_err(|err| map_control_flow(err, &READ_ERROR_INVALID_CLIENT))?;
    let options = parse_arguments(rest).await?;

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
            return Err(read_flow(
                &READ_ERROR_NOT_CONNECTED,
                "read: tcpclient is disconnected",
            ));
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
                &READ_ERROR_CONNECTION_CLOSED,
                "read: connection closed before the requested data was received",
            ));
        }
        let expected = count.saturating_mul(element_size);
        if read_result.bytes.len() != expected {
            return Err(read_flow(
                &READ_ERROR_CONNECTION_CLOSED,
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
            &READ_ERROR_TIMEOUT,
            "read: timed out waiting for data",
        )),
        Err(ReadError::ConnectionClosed) => Err(read_flow(
            &READ_ERROR_CONNECTION_CLOSED,
            "read: connection closed before the requested data was received",
        )),
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
        | DataType::Int64
        | DataType::Single
        | DataType::Double => {
            let values = numeric_from_bytes(bytes, datatype, order)?;
            let cols = values.len();
            let tensor = Tensor::new(values, vec![1, cols])
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
            &READ_ERROR_INTERNAL,
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
            &READ_ERROR_INVALID_INPUT,
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
                &READ_ERROR_INVALID_COUNT,
                "read: count must be a numeric scalar",
            ))
        }
    };
    if numeric.is_nan() || numeric.is_sign_negative() {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be a non-negative finite value",
        ));
    }
    if numeric.is_infinite() {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
            "read: count must be finite",
        ));
    }
    if numeric > usize::MAX as f64 {
        return Err(read_flow(
            &READ_ERROR_INVALID_COUNT,
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
                &READ_ERROR_INVALID_DATATYPE,
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
                &READ_ERROR_INVALID_DATATYPE,
                format!("read: unsupported datatype '{text}'"),
            ))
        }
    };
    Ok(dtype)
}

fn extract_client_id(struct_value: &StructValue) -> BuiltinResult<u64> {
    let id_value = struct_field(struct_value, CLIENT_HANDLE_FIELD).ok_or_else(|| {
        read_flow(
            &READ_ERROR_INVALID_CLIENT,
            "read: tcpclient struct is missing internal handle",
        )
    })?;
    match id_value {
        Value::Int(IntValue::U64(id)) => Ok(*id),
        Value::Int(iv) => Ok(iv.to_i64() as u64),
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
    fn read_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = READ_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"data = read(client)"));
        assert!(labels.contains(&"data = read(client, count)"));
        assert!(labels.contains(&"data = read(client, count, datatype)"));
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
        assert_error_identifier(err, READ_ERROR_TIMEOUT.identifier.unwrap());

        remove_client_for_test(client_id(&client));
    }
}
