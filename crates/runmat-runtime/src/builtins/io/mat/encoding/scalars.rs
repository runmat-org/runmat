use super::*;

const TEXT_WORK_CHUNK: usize = 4096;

pub(super) async fn text_to_utf16_cooperative(
    text: &str,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u16>> {
    let mut data = Vec::new();
    let mut work = 0usize;
    for unit in text.encode_utf16() {
        if data.capacity() == data.len() {
            data.try_reserve(TEXT_WORK_CHUNK).map_err(|error| {
                owner.error(
                    MatEncodingFailure::Io,
                    format!("UTF-16 data cannot be allocated: {error}"),
                )
            })?;
        }
        data.push(unit);
        work += 1;
        if work == TEXT_WORK_CHUNK {
            completed(cooperation, work).await?;
            work = 0;
        }
    }
    completed(cooperation, work).await?;
    Ok(data)
}

pub(super) async fn char_array_to_utf16_cooperative(
    chars: &CharArray,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u16>> {
    let mut data = Vec::new();
    let mut work = 0usize;
    let page_len = chars.rows.checked_mul(chars.cols).ok_or_else(|| {
        owner.error(
            MatEncodingFailure::Io,
            "MAT character shape exceeds platform limits",
        )
    })?;
    let pages = if page_len == 0 {
        0
    } else if chars.shape.len() <= 2 {
        1
    } else if chars.shape[2..].contains(&0) {
        0
    } else {
        chars.shape[2..]
            .iter()
            .try_fold(1usize, |product, &size| product.checked_mul(size))
            .ok_or_else(|| {
                owner.error(
                    MatEncodingFailure::Io,
                    "MAT character shape exceeds platform limits",
                )
            })?
    };
    if page_len.checked_mul(pages) != Some(chars.data().len()) {
        return Err(owner.error(
            MatEncodingFailure::Io,
            "MAT character payload does not match its shape",
        ));
    }
    for page in 0..pages {
        let page_offset = page * page_len;
        for col in 0..chars.cols {
            for row in 0..chars.rows {
                if data.capacity() == data.len() {
                    data.try_reserve(TEXT_WORK_CHUNK).map_err(|error| {
                        owner.error(
                            MatEncodingFailure::Io,
                            format!("MAT characters cannot be allocated: {error}"),
                        )
                    })?;
                }
                data.push(chars.data()[page_offset + row * chars.cols + col].raw());
                work += 1;
                if work == TEXT_WORK_CHUNK {
                    completed(cooperation, work).await?;
                    work = 0;
                }
            }
        }
    }
    completed(cooperation, work).await?;
    Ok(data)
}

pub(super) fn int_value_mat_class(value: &IntValue) -> MatClass {
    match value {
        IntValue::I8(_) => MatClass::Int8,
        IntValue::I16(_) => MatClass::Int16,
        IntValue::I32(_) => MatClass::Int32,
        IntValue::I64(_) => MatClass::Int64,
        IntValue::U8(_) => MatClass::UInt8,
        IntValue::U16(_) => MatClass::UInt16,
        IntValue::U32(_) => MatClass::UInt32,
        IntValue::U64(_) => MatClass::UInt64,
    }
}

pub(super) fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

pub(super) fn integer_storage_mat_class(storage: &IntegerStorage) -> MatClass {
    match storage {
        IntegerStorage::I8(_) => MatClass::Int8,
        IntegerStorage::I16(_) => MatClass::Int16,
        IntegerStorage::I32(_) => MatClass::Int32,
        IntegerStorage::I64(_) => MatClass::Int64,
        IntegerStorage::U8(_) => MatClass::UInt8,
        IntegerStorage::U16(_) => MatClass::UInt16,
        IntegerStorage::U32(_) => MatClass::UInt32,
        IntegerStorage::U64(_) => MatClass::UInt64,
    }
}

pub(super) fn tensor_dtype_mat_class(dtype: NumericDType) -> MatClass {
    match dtype {
        NumericDType::F64 => MatClass::Double,
        NumericDType::F32 => MatClass::Single,
        NumericDType::I8 => MatClass::Int8,
        NumericDType::I16 => MatClass::Int16,
        NumericDType::I32 => MatClass::Int32,
        NumericDType::I64 => MatClass::Int64,
        NumericDType::U8 => MatClass::UInt8,
        NumericDType::U16 => MatClass::UInt16,
        NumericDType::U32 => MatClass::UInt32,
        NumericDType::U64 => MatClass::UInt64,
    }
}
