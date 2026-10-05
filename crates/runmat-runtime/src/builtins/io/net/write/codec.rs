use super::*;
use runmat_value::{IntValue, NumericDType};

#[derive(Clone, Copy)]
pub(super) enum ByteOrder {
    Little,
    Big,
}

pub(super) struct Payload {
    pub(super) bytes: Vec<u8>,
    pub(super) elements: usize,
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

pub(super) enum PreparedWriteDatatype {
    Default,
    Known(DataType),
    Resident(Value),
}

pub(super) fn preflight_datatype(args: Vec<Value>) -> BuiltinResult<PreparedWriteDatatype> {
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

pub(super) fn prepare_default_payload(data: &Value, order: ByteOrder) -> BuiltinResult<Payload> {
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

pub(super) fn parse_datatype(value: &Value) -> BuiltinResult<DataType> {
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

pub(super) fn prepare_payload(
    data: &Value,
    datatype: DataType,
    order: ByteOrder,
) -> BuiltinResult<Payload> {
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

#[path = "codec/cast.rs"]
mod cast;
pub(super) use cast::*;

fn scalar_string(value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        write_flow(
            &WRITE_ERROR_INVALID_DATATYPE,
            "write: datatype argument must be a string scalar or character row vector",
        )
    })
}
