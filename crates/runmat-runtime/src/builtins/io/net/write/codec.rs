use super::super::transport::NetCooperation;
use super::{
    write_error_with_message, write_flow, BuiltinResult, DataType, Value, WRITE_ERROR_INTERNAL,
    WRITE_ERROR_INVALID_DATA, WRITE_ERROR_INVALID_DATATYPE, WRITE_ERROR_INVALID_INPUT,
};
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
pub(super) enum WriteElement {
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

pub(super) fn preflight_datatype(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<PreparedWriteDatatype> {
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

pub(super) async fn prepare_default_payload(
    data: &Value,
    order: ByteOrder,
    work: &mut NetCooperation,
) -> BuiltinResult<Payload> {
    prepare_payload(data, default_datatype(data)?, order, work).await
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

pub(super) async fn prepare_payload(
    data: &Value,
    datatype: DataType,
    order: ByteOrder,
    work: &mut NetCooperation,
) -> BuiltinResult<Payload> {
    match datatype {
        DataType::Char => char_payload(data, work).await,
        DataType::String => string_payload(data, work).await,
        _ => numeric_payload(data, datatype, order, work).await,
    }
}

async fn numeric_payload(
    data: &Value,
    datatype: DataType,
    order: ByteOrder,
    work: &mut NetCooperation,
) -> BuiltinResult<Payload> {
    let values = numeric_elements(data)?;
    let mut bytes = Vec::new();
    let mut elements = 0usize;
    for value in values {
        if elements.is_multiple_of(1024) {
            let additional = datatype.element_size().checked_mul(1024).ok_or_else(|| {
                write_flow(
                    &WRITE_ERROR_INTERNAL,
                    "write: encoded payload size overflow",
                )
            })?;
            bytes.try_reserve(additional).map_err(|source| {
                write_flow(
                    &WRITE_ERROR_INTERNAL,
                    format!("write: encoded payload allocation failed ({source})"),
                )
            })?;
            work.completed(0).await?;
        }
        let floating = value.as_f64();
        match datatype {
            DataType::UInt8 => bytes.push(
                integer_unsigned(&value, u8::MAX as u64)
                    .map_or_else(|| cast_to_u8(floating), |v| v as u8),
            ),
            DataType::Int8 => bytes.push(
                integer_signed(&value, i8::MIN as i64, i8::MAX as i64)
                    .map_or_else(|| cast_to_i8(floating), |v| v as i8) as u8,
            ),
            DataType::UInt16 => {
                let encoded = integer_unsigned(&value, u16::MAX as u64)
                    .map_or_else(|| cast_to_u16(floating), |v| v as u16);
                extend_u16(&mut bytes, encoded, order)
            }
            DataType::Int16 => {
                let encoded = integer_signed(&value, i16::MIN as i64, i16::MAX as i64)
                    .map_or_else(|| cast_to_i16(floating), |v| v as i16);
                extend_i16(&mut bytes, encoded, order)
            }
            DataType::UInt32 => {
                let encoded = integer_unsigned(&value, u32::MAX as u64)
                    .map_or_else(|| cast_to_u32(floating), |v| v as u32);
                extend_u32(&mut bytes, encoded, order)
            }
            DataType::Int32 => {
                let encoded = integer_signed(&value, i32::MIN as i64, i32::MAX as i64)
                    .map_or_else(|| cast_to_i32(floating), |v| v as i32);
                extend_i32(&mut bytes, encoded, order)
            }
            DataType::UInt64 => {
                let encoded =
                    integer_unsigned(&value, u64::MAX).unwrap_or_else(|| cast_to_u64(floating));
                extend_u64(&mut bytes, encoded, order)
            }
            DataType::Int64 => {
                let encoded = integer_signed(&value, i64::MIN, i64::MAX)
                    .unwrap_or_else(|| cast_to_i64(floating));
                extend_i64(&mut bytes, encoded, order)
            }
            DataType::Single => extend_f32(&mut bytes, cast_to_f32(floating), order),
            DataType::Double => extend_f64(&mut bytes, floating, order),
            DataType::Char | DataType::String => unreachable!(),
        }
        elements = elements.checked_add(1).ok_or_else(|| {
            write_flow(
                &WRITE_ERROR_INTERNAL,
                "write: encoded element count overflow",
            )
        })?;
        if elements.is_multiple_of(1024) {
            work.completed(1024 * datatype.element_size()).await?;
        }
    }
    work.completed((elements % 1024) * datatype.element_size())
        .await?;
    Ok(Payload { bytes, elements })
}

async fn char_payload(data: &Value, work: &mut NetCooperation) -> BuiltinResult<Payload> {
    let mut bytes = Vec::new();
    if matches!(data, Value::Num(_) | Value::Int(_) | Value::Bool(_)) {
        bytes.try_reserve(1).map_err(|source| {
            write_flow(
                &WRITE_ERROR_INTERNAL,
                format!("write: encoded payload allocation failed ({source})"),
            )
        })?;
    }
    match data {
        Value::CharArray(array) => {
            append_bounded(
                &mut bytes,
                array.data().iter().map(|ch| (u32::from(*ch) & 0xff) as u8),
                work,
            )
            .await?
        }
        Value::String(runmat_value::StringElement::Text(text)) => {
            append_bounded(&mut bytes, text.bytes(), work).await?
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            append_bounded(
                &mut bytes,
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .bytes(),
                work,
            )
            .await?
        }
        Value::StringArray(_) => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: string array input must be scalar when using 'char'",
            ))
        }
        Value::Tensor(_) => {
            append_bounded(
                &mut bytes,
                numeric_elements(data)?.map(|value| {
                    integer_unsigned(&value, u8::MAX as u64)
                        .map_or_else(|| cast_to_u8(value.as_f64()), |v| v as u8)
                }),
                work,
            )
            .await?
        }
        Value::Num(number) => bytes.push(cast_to_u8(*number)),
        Value::Int(integer) => bytes.push(
            integer_unsigned(&WriteElement::Integer(integer.clone()), u8::MAX as u64)
                .expect("integer input converts exactly") as u8,
        ),
        Value::Bool(boolean) => bytes.push(u8::from(*boolean)),
        Value::LogicalArray(array) => {
            append_bounded(
                &mut bytes,
                array.data.iter().map(|byte| u8::from(*byte != 0)),
                work,
            )
            .await?
        }
        _ => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: unsupported input for 'char' datatype",
            ))
        }
    }
    work.completed(0).await?;
    Ok(Payload {
        elements: bytes.len(),
        bytes,
    })
}

async fn string_payload(data: &Value, work: &mut NetCooperation) -> BuiltinResult<Payload> {
    let mut bytes = Vec::new();
    let elements = match data {
        Value::String(runmat_value::StringElement::Text(text)) => {
            append_bounded(&mut bytes, text.bytes(), work).await?;
            1
        }
        Value::CharArray(array) => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                write_flow(
                    &WRITE_ERROR_INVALID_DATA,
                    "write: string input contains malformed UTF-16 text",
                )
            })?;
            append_bounded(&mut bytes, text.bytes(), work).await?;
            1
        }
        Value::StringArray(array) if array.data().is_empty() => 0,
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            append_bounded(
                &mut bytes,
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .bytes(),
                work,
            )
            .await?;
            1
        }
        Value::StringArray(_) => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: string array input must be scalar when using 'string'",
            ))
        }
        _ => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: expected text input when using 'string' datatype",
            ))
        }
    };
    work.completed(0).await?;
    Ok(Payload { elements, bytes })
}

async fn append_bounded(
    bytes: &mut Vec<u8>,
    values: impl Iterator<Item = u8>,
    work: &mut NetCooperation,
) -> BuiltinResult<()> {
    let mut pending = 0usize;
    for value in values {
        if pending == 0 {
            bytes.try_reserve(4096).map_err(|source| {
                write_flow(
                    &WRITE_ERROR_INTERNAL,
                    format!("write: encoded payload allocation failed ({source})"),
                )
            })?;
        }
        bytes.push(value);
        pending += 1;
        if pending == 4096 {
            work.completed(pending).await?;
            pending = 0;
        }
    }
    work.completed(pending).await
}

#[path = "codec/cast.rs"]
mod cast;
pub(super) use cast::*;

#[path = "codec/elements.rs"]
mod elements;
use elements::numeric_elements;

fn scalar_string(value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        write_flow(
            &WRITE_ERROR_INVALID_DATATYPE,
            "write: datatype argument must be a string scalar or character row vector",
        )
    })
}
