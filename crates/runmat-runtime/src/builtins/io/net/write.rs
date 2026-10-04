//! MATLAB-compatible `write` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, NetDataType as DataType, WRITE_ERROR_CONNECTION_CLOSED,
    WRITE_ERROR_INTERNAL, WRITE_ERROR_INVALID_CLIENT, WRITE_ERROR_INVALID_DATA,
    WRITE_ERROR_INVALID_DATATYPE, WRITE_ERROR_INVALID_INPUT, WRITE_ERROR_NOT_CONNECTED,
    WRITE_ERROR_TIMEOUT, WRITE_EXPLICIT_GPU_EXTENSION, WRITE_LOGICAL_EXTENSION,
    WRITE_NONVECTOR_EXTENSION, WRITE_SPARSE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericDType;
use runmat_value::{IntValue, StructValue, Value};
use std::io::{self, Write};
use std::net::TcpStream;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::client_state::{client_handle, CLIENT_HANDLE_FIELD};
use super::control::configure_stream;

const BUILTIN_NAME: &str = "write";

fn write_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
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

#[runtime_builtin(
    name = "write",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::net::write"
)]
pub(super) async fn write_builtin(
    client: Value,
    data: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    ensure_write_extensions(&data)?;
    let datatype = preflight_datatype(rest)?;
    super::transport::require_native_tcp(BUILTIN_NAME)?;
    let client = gather_if_needed_async(&client)
        .await
        .map_err(|flow| map_write_flow(flow, &WRITE_ERROR_INVALID_CLIENT))?;
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

    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_CLIENT,
                "write: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        write_flow(
            &WRITE_ERROR_INVALID_CLIENT,
            "write: tcpclient handle is no longer valid",
        )
    })?;

    let (mut stream, timeout, byte_order) = {
        let guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(write_error(&WRITE_ERROR_NOT_CONNECTED));
        }
        let timeout = guard.timeout;
        let byte_order = parse_byte_order(&guard.byte_order);
        let stream = guard.stream.try_clone().map_err(|err| {
            super::error::with_source(
                BUILTIN_NAME,
                &WRITE_ERROR_INTERNAL,
                format!("write: clone failed ({err})"),
                err,
            )
        })?;
        (stream, timeout, byte_order)
    };

    if let Err(err) = configure_stream(&stream, timeout) {
        return Err(super::error::with_source(
            BUILTIN_NAME,
            &WRITE_ERROR_INTERNAL,
            format!("write: unable to configure socket timeout ({err})"),
            err,
        ));
    }

    let payload = match datatype {
        Some(datatype) => prepare_payload(&data, datatype, byte_order)?,
        None => prepare_default_payload(&data, byte_order)?,
    };
    if payload.bytes.is_empty() {
        return Ok(Value::Num(0.0));
    }

    match write_bytes(&mut stream, &payload.bytes) {
        Ok(_) => Ok(Value::Num(payload.elements as f64)),
        Err(WriteError::Timeout) => Err(write_error(&WRITE_ERROR_TIMEOUT)),
        Err(WriteError::ConnectionClosed) => {
            if let Ok(mut guard) = handle.lock() {
                guard.connected = false;
            }
            Err(write_error(&WRITE_ERROR_CONNECTION_CLOSED))
        }
        Err(WriteError::Io(err)) => Err(super::error::with_source(
            BUILTIN_NAME,
            &WRITE_ERROR_INTERNAL,
            format!("write: socket error ({err})"),
            err,
        )),
    }
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
        Value::Tensor(tensor) => !is_vector_shape(&tensor.shape),
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

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

struct Payload {
    bytes: Vec<u8>,
    elements: usize,
}

#[derive(Clone, Debug)]
enum WriteElement {
    Floating(f64),
    Integer(IntValue),
}

impl WriteElement {
    fn as_f64(&self) -> f64 {
        match self {
            Self::Floating(value) => *value,
            Self::Integer(value) => value.to_f64(),
        }
    }
}

enum PreparedWriteDatatype {
    Default,
    Known(DataType),
    Resident(Value),
}

fn preflight_datatype(args: Vec<Value>) -> BuiltinResult<PreparedWriteDatatype> {
    match args.len() {
        0 => Ok(PreparedWriteDatatype::Default),
        1 => {
            let value = args.into_iter().next().expect("one datatype argument");
            if matches!(value, Value::GpuTensor(_)) {
                Ok(PreparedWriteDatatype::Resident(value))
            } else {
                parse_datatype(&value).map(PreparedWriteDatatype::Known)
            }
        }
        _ => Err(write_flow(
            &WRITE_ERROR_INVALID_INPUT,
            "write: expected at most one datatype argument",
        )),
    }
}

fn prepare_default_payload(data: &Value, order: ByteOrder) -> BuiltinResult<Payload> {
    prepare_payload(data, default_datatype(data)?, order)
}

fn default_datatype(data: &Value) -> BuiltinResult<DataType> {
    let dtype = match data {
        Value::Num(_) => DataType::Double,
        Value::Int(value) => datatype_for_int(value),
        Value::Tensor(tensor) => datatype_for_numeric_dtype(tensor.numeric_dtype()),
        Value::SparseTensor(tensor) if tensor.is_logical() => DataType::UInt8,
        Value::SparseTensor(tensor) => {
            let Some(dtype) = tensor.numeric_dtype() else {
                return Err(write_flow(
                    &WRITE_ERROR_INVALID_DATA,
                    "write: unsupported sparse input type",
                ));
            };
            datatype_for_numeric_dtype(dtype)
        }
        Value::Bool(_) | Value::LogicalArray(_) => DataType::UInt8,
        Value::CharArray(_) => DataType::Char,
        Value::String(_) | Value::StringArray(_) => DataType::String,
        _ => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: unsupported input type",
            ))
        }
    };
    Ok(dtype)
}

fn datatype_for_int(value: &IntValue) -> DataType {
    match value {
        IntValue::I8(_) => DataType::Int8,
        IntValue::I16(_) => DataType::Int16,
        IntValue::I32(_) => DataType::Int32,
        IntValue::I64(_) => DataType::Int64,
        IntValue::U8(_) => DataType::UInt8,
        IntValue::U16(_) => DataType::UInt16,
        IntValue::U32(_) => DataType::UInt32,
        IntValue::U64(_) => DataType::UInt64,
    }
}

fn datatype_for_numeric_dtype(dtype: NumericDType) -> DataType {
    match dtype {
        NumericDType::F64 => DataType::Double,
        NumericDType::F32 => DataType::Single,
        NumericDType::I8 => DataType::Int8,
        NumericDType::I16 => DataType::Int16,
        NumericDType::I32 => DataType::Int32,
        NumericDType::I64 => DataType::Int64,
        NumericDType::U8 => DataType::UInt8,
        NumericDType::U16 => DataType::UInt16,
        NumericDType::U32 => DataType::UInt32,
        NumericDType::U64 => DataType::UInt64,
    }
}

fn parse_datatype(value: &Value) -> BuiltinResult<DataType> {
    let text = scalar_string(value)?;
    if text.trim().is_empty() {
        return Err(write_flow(
            &WRITE_ERROR_INVALID_DATATYPE,
            "write: datatype must not be empty",
        ));
    }
    DataType::parse(&text).ok_or_else(|| {
        write_flow(
            &WRITE_ERROR_INVALID_DATATYPE,
            format!("write: unsupported datatype '{text}'"),
        )
    })
}

fn prepare_payload(data: &Value, datatype: DataType, order: ByteOrder) -> BuiltinResult<Payload> {
    match datatype {
        DataType::Char => char_payload(data),
        DataType::String => string_payload(data),
        _ => numeric_payload(data, datatype, order),
    }
}

fn numeric_payload(data: &Value, datatype: DataType, order: ByteOrder) -> BuiltinResult<Payload> {
    let values = flatten_numeric(data)?;
    let mut bytes = Vec::with_capacity(values.len() * datatype.element_size());
    for value in &values {
        let floating = value.as_f64();
        match datatype {
            DataType::UInt8 => bytes.push(
                integer_unsigned(value, u8::MAX as u64)
                    .map_or_else(|| cast_to_u8(floating), |v| v as u8),
            ),
            DataType::Int8 => bytes.push(
                integer_signed(value, i8::MIN as i64, i8::MAX as i64)
                    .map_or_else(|| cast_to_i8(floating), |v| v as i8) as u8,
            ),
            DataType::UInt16 => {
                let encoded = integer_unsigned(value, u16::MAX as u64)
                    .map_or_else(|| cast_to_u16(floating), |v| v as u16);
                extend_u16(&mut bytes, encoded, order)
            }
            DataType::Int16 => {
                let encoded = integer_signed(value, i16::MIN as i64, i16::MAX as i64)
                    .map_or_else(|| cast_to_i16(floating), |v| v as i16);
                extend_i16(&mut bytes, encoded, order)
            }
            DataType::UInt32 => {
                let encoded = integer_unsigned(value, u32::MAX as u64)
                    .map_or_else(|| cast_to_u32(floating), |v| v as u32);
                extend_u32(&mut bytes, encoded, order)
            }
            DataType::Int32 => {
                let encoded = integer_signed(value, i32::MIN as i64, i32::MAX as i64)
                    .map_or_else(|| cast_to_i32(floating), |v| v as i32);
                extend_i32(&mut bytes, encoded, order)
            }
            DataType::UInt64 => {
                let encoded =
                    integer_unsigned(value, u64::MAX).unwrap_or_else(|| cast_to_u64(floating));
                extend_u64(&mut bytes, encoded, order)
            }
            DataType::Int64 => {
                let encoded = integer_signed(value, i64::MIN, i64::MAX)
                    .unwrap_or_else(|| cast_to_i64(floating));
                extend_i64(&mut bytes, encoded, order)
            }
            DataType::Single => extend_f32(&mut bytes, cast_to_f32(floating), order),
            DataType::Double => extend_f64(&mut bytes, floating, order),
            DataType::Char | DataType::String => unreachable!(),
        }
    }
    Ok(Payload {
        bytes,
        elements: values.len(),
    })
}

fn char_payload(data: &Value) -> BuiltinResult<Payload> {
    let bytes = match data {
        Value::CharArray(ca) => ca.data.iter().map(|&ch| (ch as u32 & 0xFF) as u8).collect(),
        Value::String(text) => text.bytes().collect(),
        Value::StringArray(sa) => {
            if sa.data.len() != 1 {
                return Err(write_flow(
                    &WRITE_ERROR_INVALID_DATA,
                    "write: string array input must be scalar when using 'char'",
                ));
            }
            sa.data[0].as_bytes().to_vec()
        }
        Value::Tensor(_) => flatten_numeric(data)?
            .iter()
            .map(|value| {
                let floating = value.as_f64();
                integer_unsigned(value, u8::MAX as u64)
                    .map_or_else(|| cast_to_u8(floating), |v| v as u8)
            })
            .collect(),
        Value::Num(n) => vec![cast_to_u8(*n)],
        Value::Int(iv) => vec![
            integer_unsigned(&WriteElement::Integer(iv.clone()), u8::MAX as u64)
                .expect("integer conversion is always available for integer input")
                as u8,
        ],
        Value::Bool(b) => vec![if *b { 1 } else { 0 }],
        Value::LogicalArray(la) => la
            .data
            .iter()
            .map(|&b| if b != 0 { 1 } else { 0 })
            .collect(),
        _ => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: unsupported input for 'char' datatype",
            ))
        }
    };
    Ok(Payload {
        elements: bytes.len(),
        bytes,
    })
}

fn string_payload(data: &Value) -> BuiltinResult<Payload> {
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
                return Err(write_flow(
                    &WRITE_ERROR_INVALID_DATA,
                    "write: string array input must be scalar when using 'string'",
                ));
            }
            Ok(Payload {
                elements: 1,
                bytes: sa.data[0].as_bytes().to_vec(),
            })
        }
        _ => Err(write_flow(
            &WRITE_ERROR_INVALID_DATA,
            "write: expected text input when using 'string' datatype",
        )),
    }
}

fn flatten_numeric(value: &Value) -> BuiltinResult<Vec<WriteElement>> {
    match value {
        Value::Tensor(t) => Ok((0..t.len())
            .map(|index| {
                let value = t
                    .numeric_value_at(index)
                    .expect("index within authoritative numeric storage");
                value.into_int_value().map_or_else(
                    || WriteElement::Floating(value.materialize_f64()),
                    WriteElement::Integer,
                )
            })
            .collect()),
        Value::SparseTensor(s) => {
            let total_elements = s.rows.checked_mul(s.cols).ok_or_else(|| {
                write_error_with_message(
                    "write: sparse matrix dimensions overflow",
                    &WRITE_ERROR_INTERNAL,
                )
            })?;
            if total_elements > 10_000_000 {
                return Err(write_error_with_message(
                    format!("write: cannot densify sparse tensor {}x{} ({} elements exceeds safe threshold)", s.rows, s.cols, total_elements),
                    &WRITE_ERROR_INTERNAL,
                ));
            }
            if s.is_logical() {
                let dense = s.to_dense_logical().map_err(|err| {
                    write_error_with_message(format!("write: {err}"), &WRITE_ERROR_INTERNAL)
                })?;
                return Ok(dense
                    .data
                    .into_iter()
                    .map(|value| WriteElement::Floating(f64::from(value)))
                    .collect());
            }
            let dense = s.to_dense().map_err(|err| {
                write_error_with_message(format!("write: {err}"), &WRITE_ERROR_INTERNAL)
            })?;
            Ok((0..dense.len())
                .map(|index| {
                    let value = dense
                        .numeric_value_at(index)
                        .expect("index within authoritative numeric storage");
                    value.into_int_value().map_or_else(
                        || WriteElement::Floating(value.materialize_f64()),
                        WriteElement::Integer,
                    )
                })
                .collect())
        }
        Value::Num(n) => Ok(vec![WriteElement::Floating(*n)]),
        Value::Int(iv) => Ok(vec![WriteElement::Integer(iv.clone())]),
        Value::Bool(b) => Ok(vec![WriteElement::Floating(if *b { 1.0 } else { 0.0 })]),
        Value::LogicalArray(la) => Ok(la
            .data
            .iter()
            .map(|&b| WriteElement::Floating(if b != 0 { 1.0 } else { 0.0 }))
            .collect()),
        Value::CharArray(ca) => Ok(ca
            .data
            .iter()
            .map(|&ch| WriteElement::Floating((ch as u32 & 0xFF) as f64))
            .collect()),
        Value::String(text) => Ok(text
            .chars()
            .map(|ch| WriteElement::Floating((ch as u32) as f64))
            .collect()),
        Value::StringArray(sa) => {
            if sa.data.len() != 1 {
                return Err(write_flow(
                    &WRITE_ERROR_INVALID_DATA,
                    "write: string array input must be scalar",
                ));
            }
            Ok(sa.data[0]
                .chars()
                .map(|ch| WriteElement::Floating((ch as u32) as f64))
                .collect())
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(write_flow(
            &WRITE_ERROR_INVALID_DATA,
            "write: complex data is not supported",
        )),
        Value::Symbolic(_) | Value::SymbolicArray(_) => Err(write_flow(
            &WRITE_ERROR_INVALID_DATA,
            "write: symbolic data is not supported",
        )),
        Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(write_flow(
            &WRITE_ERROR_INVALID_DATA,
            "write: unsupported input type",
        )),
        Value::GpuTensor(_) => Err(write_flow(
            &WRITE_ERROR_INVALID_DATA,
            "write: GPU tensor should have been gathered before encoding",
        )),
    }
}

fn integer_raw(value: &WriteElement) -> Option<i128> {
    match value {
        WriteElement::Integer(IntValue::I8(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::I16(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::I32(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::I64(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::U8(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::U16(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::U32(v)) => Some(*v as i128),
        WriteElement::Integer(IntValue::U64(v)) => Some(*v as i128),
        WriteElement::Floating(_) => None,
    }
}

fn integer_unsigned(value: &WriteElement, max: u64) -> Option<u64> {
    integer_raw(value).map(|raw| raw.clamp(0, max as i128) as u64)
}

fn integer_signed(value: &WriteElement, min: i64, max: i64) -> Option<i64> {
    integer_raw(value).map(|raw| raw.clamp(min as i128, max as i128) as i64)
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

fn scalar_string(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(write_flow(
            &WRITE_ERROR_INVALID_DATATYPE,
            "write: datatype argument must be a string scalar or character row vector",
        )),
    }
}

fn extract_client_id(struct_value: &StructValue) -> BuiltinResult<u64> {
    let id_value = struct_value
        .fields
        .get(CLIENT_HANDLE_FIELD)
        .ok_or_else(|| {
            write_flow(
                &WRITE_ERROR_INVALID_CLIENT,
                "write: tcpclient struct is missing internal handle",
            )
        })?;
    match id_value {
        Value::Int(iv) => iv.try_to_u64().ok_or_else(|| {
            write_flow(
                &WRITE_ERROR_INVALID_CLIENT,
                "write: tcpclient struct has invalid handle field",
            )
        }),
        _ => Err(write_flow(
            &WRITE_ERROR_INVALID_CLIENT,
            "write: tcpclient struct has invalid handle field",
        )),
    }
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

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::net::client_state::{insert_client, remove_client_for_test};
    use crate::builtins::io::net::control::configure_stream;
    use crate::builtins::io::net::test_support::CountingDownloadProvider;
    use runmat_builtins::WRITE_DESCRIPTOR;
    use runmat_value::{CharArray, IntValue, IntegerStorage, SparseTensor, StructValue, Tensor};
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

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    #[test]
    fn typed_negative_client_handle_is_rejected() {
        let mut client = StructValue::new();
        client.fields.insert(
            CLIENT_HANDLE_FIELD.to_string(),
            Value::Int(IntValue::I8(-1)),
        );
        let err = extract_client_id(&client).unwrap_err();
        assert_error_identifier(err, WRITE_ERROR_INVALID_CLIENT.identifier.unwrap());
    }

    fn run_write(client: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(write_builtin(client, data, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn known_invalid_datatype_and_arity_reject_before_payload_download() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let payload = Value::GpuTensor(owner.resident());
        let invalid_datatype = run_write(
            Value::Num(0.0),
            payload.clone(),
            vec![Value::from("uint128")],
        )
        .expect_err("known invalid datatype rejects before resident payload download");
        assert_eq!(
            invalid_datatype.identifier(),
            WRITE_ERROR_INVALID_DATATYPE.identifier
        );
        assert_eq!(owner.downloads(), 0);
        let overarity = run_write(
            Value::Num(0.0),
            payload,
            vec![Value::from("uint8"), Value::from("double")],
        )
        .expect_err("extra controls reject before resident payload download");
        assert_eq!(overarity.identifier(), WRITE_ERROR_INVALID_INPUT.identifier);
        assert_eq!(owner.downloads(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn write_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = WRITE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"count = write(client, data)"));
        assert!(labels.contains(&"count = write(client, data, datatype)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn write_default_preserves_double_datatype() {
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
        let result = run_write(client.clone(), Value::Tensor(tensor), Vec::new()).expect("write");
        match result {
            Value::Num(count) => assert_eq!(count, 4.0),
            other => panic!("expected numeric result, got {other:?}"),
        }
        remove_client_for_test(client_id(&client));
        let received = handle.join().expect("join");
        let expected = [1.0_f64, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(received, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let result = run_write(
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn write_uint64_payload_reads_typed_integer_storage_exactly() {
        let wide = (1_u64 << 53) + 1;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
            .expect("typed integer tensor");

        let payload =
            prepare_payload(&Value::Tensor(tensor), DataType::UInt64, ByteOrder::Little).unwrap();
        let mut expected = Vec::new();
        extend_u64(&mut expected, wide, ByteOrder::Little);
        extend_u64(&mut expected, u64::MAX, ByteOrder::Little);
        assert_eq!(payload.elements, 2);
        assert_eq!(payload.bytes, expected);
    }

    #[test]
    fn write_default_payload_preserves_uint64_datatype_exactly() {
        let wide = (1_u64 << 53) + 1;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
            .expect("typed integer tensor");

        let payload = prepare_default_payload(&Value::Tensor(tensor), ByteOrder::Big).unwrap();
        let expected = [wide, u64::MAX]
            .into_iter()
            .flat_map(u64::to_be_bytes)
            .collect::<Vec<_>>();
        assert_eq!(payload.elements, 2);
        assert_eq!(payload.bytes, expected);
    }

    #[test]
    fn write_extensions_are_gated_before_client_access() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let data = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
        let error = run_write(Value::Num(0.0), data, Vec::new())
            .expect_err("strict mode rejects nonvector data before client access");
        assert_eq!(
            error.identifier(),
            WRITE_NONVECTOR_EXTENSION.error_identifier
        );

        let sparse =
            SparseTensor::new_integer(2, 1, vec![0, 1], vec![0], IntegerStorage::U8(vec![1]))
                .unwrap();
        let error = run_write(Value::Num(0.0), Value::SparseTensor(sparse), Vec::new())
            .expect_err("strict mode rejects sparse data before client access");
        assert_eq!(error.identifier(), WRITE_SPARSE_EXTENSION.error_identifier);

        let error = run_write(Value::Num(0.0), Value::Bool(true), Vec::new())
            .expect_err("strict mode rejects logical data before client access");
        assert_eq!(error.identifier(), WRITE_LOGICAL_EXTENSION.error_identifier);

        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let original = allocation.handle.wire_descriptor();
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let handle = allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit immutable view");
        let error = run_write(Value::Num(0.0), Value::GpuTensor(handle), Vec::new())
            .expect_err("strict mode rejects explicit GPU data before provider access");
        assert_eq!(
            error.identifier(),
            WRITE_EXPLICIT_GPU_EXTENSION.error_identifier
        );

        assert_eq!(allocation.handle.wire_descriptor(), original);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn write_sparse_integer_payload_uses_exact_storage() {
        let wide = (1_u64 << 53) + 1;
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            IntegerStorage::U64(vec![wide, u64::MAX]),
        )
        .expect("typed sparse tensor");

        let payload = prepare_payload(
            &Value::SparseTensor(sparse),
            DataType::UInt64,
            ByteOrder::Little,
        )
        .expect("sparse uint64 payload");
        let mut expected = Vec::new();
        extend_u64(&mut expected, 0, ByteOrder::Little);
        extend_u64(&mut expected, wide, ByteOrder::Little);
        extend_u64(&mut expected, u64::MAX, ByteOrder::Little);
        extend_u64(&mut expected, 0, ByteOrder::Little);
        assert_eq!(payload.elements, 4);
        assert_eq!(payload.bytes, expected);

        let signed = SparseTensor::new_integer(
            2,
            1,
            vec![0, 2],
            vec![0, 1],
            IntegerStorage::I16(vec![-7, 42]),
        )
        .expect("signed sparse tensor");
        let payload = prepare_payload(
            &Value::SparseTensor(signed),
            DataType::Int16,
            ByteOrder::Little,
        )
        .expect("sparse int16 payload");
        assert_eq!(
            payload.bytes,
            [-7_i16, 42]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn write_integer_payload_narrowing_uses_exact_integer_bounds() {
        let value = Value::Int(IntValue::U64((1_u64 << 53) + 1));
        let payload = prepare_payload(&value, DataType::UInt32, ByteOrder::Big).unwrap();
        assert_eq!(payload.bytes, u32::MAX.to_be_bytes());

        let value = Value::Int(IntValue::I64(i64::MIN + 1));
        let payload = prepare_payload(&value, DataType::Int16, ByteOrder::Little).unwrap();
        assert_eq!(payload.bytes, i16::MIN.to_le_bytes());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn write_char_payload_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![82, 77]), vec![1, 2]).expect("chars");

        let payload = prepare_payload(&Value::Tensor(tensor), DataType::Char, ByteOrder::Little)
            .expect("char payload");
        assert_eq!(payload.elements, 2);
        assert_eq!(payload.bytes, b"RM");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let result = run_write(
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let err = run_write(client.clone(), Value::Tensor(tensor), Vec::new()).expect_err("write");
        assert_error_identifier(err, WRITE_ERROR_NOT_CONNECTED.identifier.unwrap());

        remove_client_for_test(id);
        barrier.wait();
        handle.join().expect("join");
    }
}
