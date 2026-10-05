use super::{ByteOrder, WriteElement};
use runmat_value::IntValue;

pub(in super::super) fn integer_raw(value: &WriteElement) -> Option<i128> {
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

pub(in super::super) fn integer_unsigned(value: &WriteElement, max: u64) -> Option<u64> {
    integer_raw(value).map(|raw| raw.clamp(0, max as i128) as u64)
}

pub(in super::super) fn integer_signed(value: &WriteElement, min: i64, max: i64) -> Option<i64> {
    integer_raw(value).map(|raw| raw.clamp(min as i128, max as i128) as i64)
}

pub(in super::super) fn cast_to_u8(value: f64) -> u8 {
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

pub(in super::super) fn cast_to_i8(value: f64) -> i8 {
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

pub(in super::super) fn cast_to_u16(value: f64) -> u16 {
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

pub(in super::super) fn cast_to_i16(value: f64) -> i16 {
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

pub(in super::super) fn cast_to_u32(value: f64) -> u32 {
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

pub(in super::super) fn cast_to_i32(value: f64) -> i32 {
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

pub(in super::super) fn cast_to_u64(value: f64) -> u64 {
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

pub(in super::super) fn cast_to_i64(value: f64) -> i64 {
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

pub(in super::super) fn cast_to_f32(value: f64) -> f32 {
    value as f32
}

pub(in super::super) fn rounded_scalar(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.round()
    }
}

pub(in super::super) fn extend_u16(buffer: &mut Vec<u8>, value: u16, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_i16(buffer: &mut Vec<u8>, value: i16, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_u32(buffer: &mut Vec<u8>, value: u32, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_i32(buffer: &mut Vec<u8>, value: i32, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_u64(buffer: &mut Vec<u8>, value: u64, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_i64(buffer: &mut Vec<u8>, value: i64, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_f32(buffer: &mut Vec<u8>, value: f32, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}

pub(in super::super) fn extend_f64(buffer: &mut Vec<u8>, value: f64, order: ByteOrder) {
    match order {
        ByteOrder::Little => buffer.extend_from_slice(&value.to_le_bytes()),
        ByteOrder::Big => buffer.extend_from_slice(&value.to_be_bytes()),
    }
}
