use super::*;

#[derive(Debug, Clone)]
pub(super) enum ParsedValue {
    Number(NumericScalar),
    Text(String),
}

pub(super) fn parse_float(token: &str) -> BuiltinResult<f64> {
    match token.trim().to_ascii_lowercase().as_str() {
        "" => Ok(f64::NAN),
        "nan" => Ok(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
        "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
        _ => token.trim().parse::<f64>().map_err(|_| {
            textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                format!("textscan: cannot parse '{token}' as a floating-point value"),
            )
        }),
    }
}

pub(super) fn parse_float_scalar(token: &str, dtype: NumericDType) -> BuiltinResult<NumericScalar> {
    let value = parse_float(token)?;
    Ok(match dtype {
        NumericDType::F32 => NumericScalar::F32(value as f32),
        NumericDType::F64 => NumericScalar::F64(value),
        _ => unreachable!("float parser requested an integer dtype"),
    })
}

pub(super) fn parse_signed_scalar(
    token: &str,
    dtype: NumericDType,
) -> BuiltinResult<NumericScalar> {
    let (negative, magnitude) = parse_integer_magnitude(token, 10)?;
    Ok(signed_scalar_from_magnitude(dtype, negative, magnitude))
}

pub(super) fn parse_unsigned_scalar(
    token: &str,
    dtype: NumericDType,
) -> BuiltinResult<NumericScalar> {
    let (negative, magnitude) = parse_integer_magnitude(token, 10)?;
    Ok(unsigned_scalar_from_magnitude(dtype, negative, magnitude))
}

pub(super) fn parse_radix_scalar(
    token: &str,
    dtype: NumericDType,
    radix: u32,
) -> BuiltinResult<NumericScalar> {
    let (negative, magnitude) = parse_integer_magnitude(token, radix)?;
    Ok(
        if matches!(
            dtype,
            NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64
        ) {
            signed_scalar_from_magnitude(dtype, negative, magnitude)
        } else {
            unsigned_scalar_from_magnitude(dtype, negative, magnitude)
        },
    )
}

pub(super) fn empty_numeric_scalar(dtype: NumericDType) -> NumericScalar {
    match dtype {
        NumericDType::F64 => NumericScalar::F64(f64::NAN),
        NumericDType::F32 => NumericScalar::F32(f32::NAN),
        NumericDType::I8 => NumericScalar::I8(0),
        NumericDType::I16 => NumericScalar::I16(0),
        NumericDType::I32 => NumericScalar::I32(0),
        NumericDType::I64 => NumericScalar::I64(0),
        NumericDType::U8 => NumericScalar::U8(0),
        NumericDType::U16 => NumericScalar::U16(0),
        NumericDType::U32 => NumericScalar::U32(0),
        NumericDType::U64 => NumericScalar::U64(0),
    }
}

pub(super) fn parse_integer_magnitude(
    token: &str,
    radix: u32,
) -> BuiltinResult<(bool, Option<u128>)> {
    let trimmed = token.trim();
    let (negative, unsigned) = if let Some(rest) = trimmed.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = trimmed.strip_prefix('+') {
        (false, rest)
    } else {
        (false, trimmed)
    };
    let digits = match radix {
        16 => unsigned
            .strip_prefix("0x")
            .or_else(|| unsigned.strip_prefix("0X"))
            .unwrap_or(unsigned),
        2 => unsigned
            .strip_prefix("0b")
            .or_else(|| unsigned.strip_prefix("0B"))
            .unwrap_or(unsigned),
        _ => unsigned,
    };
    if digits.is_empty() {
        return Err(integer_parse_error(token));
    }
    match u128::from_str_radix(digits, radix) {
        Ok(value) => Ok((negative, Some(value))),
        Err(error) if matches!(error.kind(), std::num::IntErrorKind::PosOverflow) => {
            Ok((negative, None))
        }
        Err(_) => Err(integer_parse_error(token)),
    }
}

pub(super) fn integer_parse_error(token: &str) -> RuntimeError {
    textscan_error_with(
        &TEXTSCAN_ERROR_PARSE,
        format!("textscan: cannot parse '{token}' as an integer value"),
    )
}

pub(super) fn signed_scalar_from_magnitude(
    dtype: NumericDType,
    negative: bool,
    magnitude: Option<u128>,
) -> NumericScalar {
    let signed_value = |maximum: u128| -> i128 {
        if negative {
            let minimum_magnitude = maximum + 1;
            match magnitude {
                Some(value) if value < minimum_magnitude => -(value as i128),
                Some(value) if value == minimum_magnitude => -(minimum_magnitude as i128),
                _ => -(minimum_magnitude as i128),
            }
        } else {
            magnitude.unwrap_or(maximum).min(maximum) as i128
        }
    };
    match dtype {
        NumericDType::I8 => NumericScalar::I8(signed_value(i8::MAX as u128) as i8),
        NumericDType::I16 => NumericScalar::I16(signed_value(i16::MAX as u128) as i16),
        NumericDType::I32 => NumericScalar::I32(signed_value(i32::MAX as u128) as i32),
        NumericDType::I64 => NumericScalar::I64(signed_value(i64::MAX as u128) as i64),
        _ => unreachable!("signed conversion requested non-signed dtype"),
    }
}

pub(super) fn unsigned_scalar_from_magnitude(
    dtype: NumericDType,
    negative: bool,
    magnitude: Option<u128>,
) -> NumericScalar {
    let unsigned_value = |maximum: u128| {
        if negative {
            0
        } else {
            magnitude.unwrap_or(maximum).min(maximum)
        }
    };
    match dtype {
        NumericDType::U8 => NumericScalar::U8(unsigned_value(u8::MAX as u128) as u8),
        NumericDType::U16 => NumericScalar::U16(unsigned_value(u16::MAX as u128) as u16),
        NumericDType::U32 => NumericScalar::U32(unsigned_value(u32::MAX as u128) as u32),
        NumericDType::U64 => NumericScalar::U64(unsigned_value(u64::MAX as u128) as u64),
        _ => unreachable!("unsigned conversion requested non-unsigned dtype"),
    }
}
