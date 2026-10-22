use super::super::transport::NetCooperation;
use super::{read_flow, BuiltinResult, DataType, Value, READ_ERROR_INTERNAL};
use runmat_value::{CharArray, IntegerStorage, NumericStorage, Tensor, Utf16CodeUnit};

#[derive(Clone, Copy)]
pub(super) enum ByteOrder {
    Little,
    Big,
}

pub(super) async fn bytes_to_value(
    bytes: &[u8],
    datatype: DataType,
    order: ByteOrder,
    work: &mut NetCooperation,
) -> BuiltinResult<Value> {
    match datatype {
        DataType::Char => char_row(bytes, work).await,
        DataType::String => latin1_string(bytes, work).await.map(Value::string),
        DataType::UInt8
        | DataType::Int8
        | DataType::UInt16
        | DataType::Int16
        | DataType::UInt32
        | DataType::Int32
        | DataType::UInt64
        | DataType::Int64 => {
            let storage = integer_storage_from_bytes(bytes, datatype, order, work).await?;
            let tensor =
                Tensor::new_integer(storage, vec![1, bytes.len() / datatype.element_size()])
                    .map_err(|err| read_flow(&READ_ERROR_INTERNAL, format!("read: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
        DataType::Single | DataType::Double => {
            let storage = floating_storage_from_bytes(bytes, datatype, order, work).await?;
            let cols = storage.len();
            let tensor = Tensor::from_numeric_storage(storage, vec![1, cols])
                .map_err(|err| read_flow(&READ_ERROR_INTERNAL, format!("read: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
    }
}

async fn decode_elements<T>(
    bytes: &[u8],
    width: usize,
    work: &mut NetCooperation,
    mut decode: impl FnMut(&[u8]) -> T,
) -> BuiltinResult<Vec<T>> {
    if width == 0 || !bytes.len().is_multiple_of(width) {
        return Err(read_flow(
            &READ_ERROR_INTERNAL,
            "read: received byte count does not align with datatype size",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / width)
        .map_err(|source| {
            read_flow(
                &READ_ERROR_INTERNAL,
                format!("read: decoded payload allocation failed ({source})"),
            )
        })?;
    for group in bytes.chunks(width * 1024) {
        for element in group.chunks_exact(width) {
            values.push(decode(element));
        }
        work.completed(group.len()).await?;
    }
    Ok(values)
}

async fn char_row(bytes: &[u8], work: &mut NetCooperation) -> BuiltinResult<Value> {
    let chars = decode_elements(bytes, 1, work, |byte| {
        Utf16CodeUnit::from_raw(u16::from(byte[0]))
    })
    .await?;
    let len = chars.len();
    CharArray::from_code_units(chars, 1, len)
        .map(Value::CharArray)
        .map_err(|source| read_flow(&READ_ERROR_INTERNAL, format!("read: {source}")))
}

async fn latin1_string(bytes: &[u8], work: &mut NetCooperation) -> BuiltinResult<String> {
    let mut text = String::new();
    for group in bytes.chunks(4096) {
        text.try_reserve(group.len() * 2).map_err(|source| {
            read_flow(
                &READ_ERROR_INTERNAL,
                format!("read: string allocation failed ({source})"),
            )
        })?;
        for byte in group {
            text.push(char::from(*byte));
        }
        work.completed(group.len()).await?;
    }
    Ok(text)
}

async fn floating_storage_from_bytes(
    bytes: &[u8],
    datatype: DataType,
    order: ByteOrder,
    work: &mut NetCooperation,
) -> BuiltinResult<NumericStorage> {
    match datatype {
        DataType::Single => Ok(NumericStorage::F32(
            decode_elements(bytes, 4, work, |chunk| f32_from(chunk, order)).await?,
        )),
        DataType::Double => Ok(NumericStorage::F64(
            decode_elements(bytes, 8, work, |chunk| f64_from(chunk, order)).await?,
        )),
        _ => Err(read_flow(
            &READ_ERROR_INTERNAL,
            "read: expected a floating-point datatype",
        )),
    }
}

async fn integer_storage_from_bytes(
    bytes: &[u8],
    datatype: DataType,
    order: ByteOrder,
    work: &mut NetCooperation,
) -> BuiltinResult<IntegerStorage> {
    Ok(match datatype {
        DataType::UInt8 => {
            IntegerStorage::U8(decode_elements(bytes, 1, work, |chunk| chunk[0]).await?)
        }
        DataType::Int8 => {
            IntegerStorage::I8(decode_elements(bytes, 1, work, |chunk| chunk[0] as i8).await?)
        }
        DataType::UInt16 => IntegerStorage::U16(
            decode_elements(bytes, 2, work, |chunk| u16_from(chunk, order)).await?,
        ),
        DataType::Int16 => IntegerStorage::I16(
            decode_elements(bytes, 2, work, |chunk| i16_from(chunk, order)).await?,
        ),
        DataType::UInt32 => IntegerStorage::U32(
            decode_elements(bytes, 4, work, |chunk| u32_from(chunk, order)).await?,
        ),
        DataType::Int32 => IntegerStorage::I32(
            decode_elements(bytes, 4, work, |chunk| i32_from(chunk, order)).await?,
        ),
        DataType::UInt64 => IntegerStorage::U64(
            decode_elements(bytes, 8, work, |chunk| u64_from(chunk, order)).await?,
        ),
        DataType::Int64 => IntegerStorage::I64(
            decode_elements(bytes, 8, work, |chunk| i64_from(chunk, order)).await?,
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
