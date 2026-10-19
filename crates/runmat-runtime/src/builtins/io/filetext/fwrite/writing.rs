use super::*;

mod elements;
pub(super) use elements::{borrow_elements, WriteElement};

pub(super) async fn write_elements<W: Write + Seek>(
    file: &mut W,
    values: impl IntoIterator<Item = WriteElement>,
    spec: WriteSpec,
    skip: usize,
    machine: MachineFormat,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> BuiltinResult<usize> {
    let endianness = machine.to_endianness();
    let skip_offset = skip as i64;
    let mut count = 0;
    for value in values {
        let value = &value;
        match spec.input {
            InputType::UInt8 => {
                let byte = match integer_unsigned(value, u8::MAX as u64) {
                    Some(value) => value as u8,
                    None => to_u8(value.as_f64()),
                };
                write_bytes(file, &[byte])?;
            }
            InputType::Int8 => {
                let byte = match integer_signed(value, i8::MIN as i64, i8::MAX as i64) {
                    Some(value) => value as i8,
                    None => to_i8(value.as_f64()),
                } as u8;
                write_bytes(file, &[byte])?;
            }
            InputType::UInt16 => {
                let bytes = match integer_unsigned(value, u16::MAX as u64) {
                    Some(value) => endian_u16(value as u16, endianness),
                    None => encode_u16(value.as_f64(), endianness),
                };
                write_bytes(file, &bytes)?;
            }
            InputType::Int16 => {
                let bytes = match integer_signed(value, i16::MIN as i64, i16::MAX as i64) {
                    Some(value) => endian_i16(value as i16, endianness),
                    None => encode_i16(value.as_f64(), endianness),
                };
                write_bytes(file, &bytes)?;
            }
            InputType::UInt32 => {
                let bytes = match integer_unsigned(value, u32::MAX as u64) {
                    Some(value) => endian_u32(value as u32, endianness),
                    None => encode_u32(value.as_f64(), endianness),
                };
                write_bytes(file, &bytes)?;
            }
            InputType::Int32 => {
                let bytes = match integer_signed(value, i32::MIN as i64, i32::MAX as i64) {
                    Some(value) => endian_i32(value as i32, endianness),
                    None => encode_i32(value.as_f64(), endianness),
                };
                write_bytes(file, &bytes)?;
            }
            InputType::UInt64 => {
                let bytes = match value {
                    WriteElement::Integer(_) => {
                        endian_u64(integer_unsigned(value, u64::MAX).unwrap(), endianness)
                    }
                    _ => encode_u64(value.as_f64(), endianness),
                };
                write_bytes(file, &bytes)?;
            }
            InputType::Int64 => {
                let bytes = match value {
                    WriteElement::Integer(_) => endian_i64(
                        integer_signed(value, i64::MIN, i64::MAX).unwrap(),
                        endianness,
                    ),
                    _ => encode_i64(value.as_f64(), endianness),
                };
                write_bytes(file, &bytes)?;
            }
            InputType::Float32 => {
                let bytes = encode_f32(value.as_f64(), endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Float64 => {
                let bytes = encode_f64(value.as_f64(), endianness);
                write_bytes(file, &bytes)?;
            }
        }

        if skip > 0 {
            file.seek(SeekFrom::Current(skip_offset)).map_err(|err| {
                fwrite_error_with_source(
                    format!("fwrite: failed to seek while applying skip ({err})"),
                    &FWRITE_ERROR_IO,
                    err,
                )
            })?;
        }
        count += 1;
        cooperation
            .completed(spec.input.byte_len().saturating_add(skip))
            .await?;
    }
    Ok(count)
}

pub(super) fn integer_raw(value: &WriteElement) -> Option<i128> {
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
pub(super) fn integer_unsigned(value: &WriteElement, max: u64) -> Option<u64> {
    integer_raw(value).map(|v| v.clamp(0, max as i128) as u64)
}
pub(super) fn integer_signed(value: &WriteElement, min: i64, max: i64) -> Option<i64> {
    integer_raw(value).map(|v| v.clamp(min as i128, max as i128) as i64)
}
pub(super) fn endian_u16(value: u16, endian: Endianness) -> [u8; 2] {
    match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    }
}
pub(super) fn endian_i16(value: i16, endian: Endianness) -> [u8; 2] {
    match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    }
}
pub(super) fn endian_u32(value: u32, endian: Endianness) -> [u8; 4] {
    match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    }
}
pub(super) fn endian_i32(value: i32, endian: Endianness) -> [u8; 4] {
    match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    }
}
pub(super) fn endian_u64(value: u64, endian: Endianness) -> [u8; 8] {
    match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    }
}
pub(super) fn endian_i64(value: i64, endian: Endianness) -> [u8; 8] {
    match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    }
}

pub(super) fn write_bytes<W: Write>(file: &mut W, bytes: &[u8]) -> BuiltinResult<()> {
    file.write_all(bytes).map_err(|err| {
        fwrite_error_with_source(
            format!("fwrite: failed to write to file ({err})"),
            &FWRITE_ERROR_IO,
            err,
        )
    })
}

pub(super) fn to_u8(value: f64) -> u8 {
    if value.is_nan() {
        return 0;
    }
    if !value.is_finite() {
        return if value.is_sign_negative() { 0 } else { u8::MAX };
    }
    let mut rounded = value.round();
    if rounded.is_nan() {
        return 0;
    }
    if rounded < 0.0 {
        rounded = 0.0;
    }
    if rounded > u8::MAX as f64 {
        rounded = u8::MAX as f64;
    }
    rounded as u8
}

pub(super) fn to_i8(value: f64) -> i8 {
    saturating_round(value, i8::MIN as f64, i8::MAX as f64) as i8
}

pub(super) fn encode_u16(value: f64, endianness: Endianness) -> [u8; 2] {
    let rounded = saturating_round(value, 0.0, u16::MAX as f64) as u16;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

pub(super) fn encode_i16(value: f64, endianness: Endianness) -> [u8; 2] {
    let rounded = saturating_round(value, i16::MIN as f64, i16::MAX as f64) as i16;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

pub(super) fn encode_u32(value: f64, endianness: Endianness) -> [u8; 4] {
    let rounded = saturating_round(value, 0.0, u32::MAX as f64) as u32;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

pub(super) fn encode_i32(value: f64, endianness: Endianness) -> [u8; 4] {
    let rounded = saturating_round(value, i32::MIN as f64, i32::MAX as f64) as i32;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

pub(super) fn encode_u64(value: f64, endianness: Endianness) -> [u8; 8] {
    let rounded = saturating_round(value, 0.0, u64::MAX as f64);
    let as_u64 = if rounded.is_finite() {
        rounded as u64
    } else if rounded.is_sign_negative() {
        0
    } else {
        u64::MAX
    };
    match endianness {
        Endianness::Little => as_u64.to_le_bytes(),
        Endianness::Big => as_u64.to_be_bytes(),
    }
}

pub(super) fn encode_i64(value: f64, endianness: Endianness) -> [u8; 8] {
    let rounded = saturating_round(value, i64::MIN as f64, i64::MAX as f64);
    let as_i64 = if rounded.is_finite() {
        rounded as i64
    } else if rounded.is_sign_negative() {
        i64::MIN
    } else {
        i64::MAX
    };
    match endianness {
        Endianness::Little => as_i64.to_le_bytes(),
        Endianness::Big => as_i64.to_be_bytes(),
    }
}

pub(super) fn encode_f32(value: f64, endianness: Endianness) -> [u8; 4] {
    let as_f32 = value as f32;
    let bits = as_f32.to_bits();
    match endianness {
        Endianness::Little => bits.to_le_bytes(),
        Endianness::Big => bits.to_be_bytes(),
    }
}

pub(super) fn encode_f64(value: f64, endianness: Endianness) -> [u8; 8] {
    let bits = value.to_bits();
    match endianness {
        Endianness::Little => bits.to_le_bytes(),
        Endianness::Big => bits.to_be_bytes(),
    }
}

pub(super) fn saturating_round(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    if !value.is_finite() {
        return if value.is_sign_negative() { min } else { max };
    }
    let mut rounded = value.round();
    if rounded.is_nan() {
        return 0.0;
    }
    if rounded < min {
        rounded = min;
    }
    if rounded > max {
        rounded = max;
    }
    rounded
}
