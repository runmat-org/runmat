use super::*;

pub(super) fn decode_numeric_scalar(
    bytes: &[u8],
    input: InputType,
    endianness: Endianness,
) -> Result<NumericScalar, String> {
    Ok(match input {
        InputType::UInt8 => NumericScalar::U8(bytes[0]),
        InputType::Int8 => NumericScalar::I8(bytes[0] as i8),
        InputType::UInt16 => NumericScalar::U16(read_u16(bytes, endianness)),
        InputType::Int16 => NumericScalar::I16(read_u16(bytes, endianness) as i16),
        InputType::UInt32 => NumericScalar::U32(read_u32(bytes, endianness)),
        InputType::Int32 => NumericScalar::I32(read_u32(bytes, endianness) as i32),
        InputType::UInt64 => NumericScalar::U64(read_u64(bytes, endianness)),
        InputType::Int64 => NumericScalar::I64(read_u64(bytes, endianness) as i64),
        InputType::Float32 => {
            let bits = read_u32(bytes, endianness);
            NumericScalar::F32(f32::from_bits(bits))
        }
        InputType::Float64 => {
            let bits = read_u64(bytes, endianness);
            NumericScalar::F64(f64::from_bits(bits))
        }
    })
}

#[cfg(test)]
pub(super) async fn numeric_storage_from_scalars(
    input: InputType,
    values: Vec<NumericScalar>,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> BuiltinResult<NumericStorage> {
    macro_rules! collect_variant {
        ($scalar_variant:ident, $storage_variant:ident) => {{
            let mut output = Vec::with_capacity(values.len());
            for value in values {
                let NumericScalar::$scalar_variant(value) = value else {
                    return Err(ReadFailure::Semantic(
                        "fread: decoded numeric source class mismatch".to_string(),
                    )
                    .into());
                };
                output.push(value);
                cooperation
                    .completed(std::mem::size_of::<NumericScalar>())
                    .await?;
            }
            NumericStorage::$storage_variant(output)
        }};
    }
    Ok(match input {
        InputType::UInt8 => collect_variant!(U8, U8),
        InputType::Int8 => collect_variant!(I8, I8),
        InputType::UInt16 => collect_variant!(U16, U16),
        InputType::Int16 => collect_variant!(I16, I16),
        InputType::UInt32 => collect_variant!(U32, U32),
        InputType::Int32 => collect_variant!(I32, I32),
        InputType::UInt64 => collect_variant!(U64, U64),
        InputType::Int64 => collect_variant!(I64, I64),
        InputType::Float32 => collect_variant!(F32, F32),
        InputType::Float64 => collect_variant!(F64, F64),
    })
}

#[cfg(test)]
pub(super) fn decode_to_code_unit(
    bytes: &[u8],
    input: InputType,
    endianness: Endianness,
) -> Result<runmat_value::Utf16CodeUnit, String> {
    let scalar = decode_numeric_scalar(bytes, input, endianness)?;
    let code = match scalar {
        NumericScalar::F64(value) => floating_char_code(value)?,
        NumericScalar::F32(value) => floating_char_code(f64::from(value))?,
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_u64())
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| {
                "fread: character value is outside the UTF-16 code-unit range".to_string()
            })?,
    };
    Ok(runmat_value::Utf16CodeUnit::from_raw(code))
}

pub(super) fn floating_char_code(value: f64) -> Result<u16, String> {
    if !value.is_finite() || value < 0.0 || value > f64::from(u16::MAX) || value.round() != value {
        return Err("fread: character value is outside the UTF-16 code-unit range".to_string());
    }
    Ok(value as u16)
}

pub(super) fn read_u16(bytes: &[u8], endianness: Endianness) -> u16 {
    match endianness {
        Endianness::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        Endianness::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    }
}

pub(super) fn read_u32(bytes: &[u8], endianness: Endianness) -> u32 {
    match endianness {
        Endianness::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        Endianness::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

pub(super) fn read_u64(bytes: &[u8], endianness: Endianness) -> u64 {
    match endianness {
        Endianness::Little => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        Endianness::Big => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}
