use super::*;

pub(crate) fn map_index(value: NumericScalar, map_rows: usize) -> BuiltinResult<usize> {
    let index = match value {
        NumericScalar::F64(value) => floating_map_index(value)?,
        NumericScalar::F32(value) => floating_map_index(f64::from(value))?,
        NumericScalar::U8(value) => usize::from(value),
        NumericScalar::U16(value) => usize::from(value),
        NumericScalar::I8(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::I16(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::I32(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::I64(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::U32(value) => one_based_unsigned_index(u128::from(value))?,
        NumericScalar::U64(value) => one_based_unsigned_index(u128::from(value))?,
    };
    if index >= map_rows {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!(
                "indexed image value {} is outside the colormap",
                numeric_scalar_text(value)
            ),
        ));
    }
    Ok(index)
}

fn floating_map_index(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image values must be finite",
        ));
    }
    let rounded = value.round();
    if rounded < 1.0 || rounded > usize::MAX as f64 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        ));
    }
    Ok(rounded as usize - 1)
}

fn one_based_signed_index(value: i128) -> BuiltinResult<usize> {
    if value < 1 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        ));
    }
    usize::try_from(value - 1).map_err(|_| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        )
    })
}

fn one_based_unsigned_index(value: u128) -> BuiltinResult<usize> {
    if value == 0 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image value 0 is outside the colormap",
        ));
    }
    usize::try_from(value - 1).map_err(|_| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        )
    })
}

pub(crate) fn value_to_u8(value: NumericScalar) -> u8 {
    match value {
        NumericScalar::F64(value) => normalized_float_to_u8(value),
        NumericScalar::F32(value) => normalized_float_to_u8(f64::from(value)),
        integer => scaled_integer(integer, u8::MAX as u128) as u8,
    }
}

pub(crate) fn value_to_u16(value: NumericScalar) -> u16 {
    match value {
        NumericScalar::F64(value) => normalized_float_to_u16(value),
        NumericScalar::F32(value) => normalized_float_to_u16(f64::from(value)),
        integer => scaled_integer(integer, u16::MAX as u128) as u16,
    }
}

fn normalized_float_to_u8(value: f64) -> u8 {
    let scaled = value.clamp(0.0, 1.0) * u8::MAX as f64;
    if scaled.is_nan() {
        0
    } else {
        scaled.round() as u8
    }
}

fn normalized_float_to_u16(value: f64) -> u16 {
    let scaled = value.clamp(0.0, 1.0) * u16::MAX as f64;
    if scaled.is_nan() {
        0
    } else {
        scaled.round() as u16
    }
}

fn scaled_integer(value: NumericScalar, output_max: u128) -> u128 {
    let (offset, input_max) = match value {
        NumericScalar::I8(value) => (
            (i128::from(value) - i128::from(i8::MIN)) as u128,
            u128::from(u8::MAX),
        ),
        NumericScalar::I16(value) => (
            (i128::from(value) - i128::from(i16::MIN)) as u128,
            u128::from(u16::MAX),
        ),
        NumericScalar::I32(value) => (
            (i128::from(value) - i128::from(i32::MIN)) as u128,
            u128::from(u32::MAX),
        ),
        NumericScalar::I64(value) => (
            (i128::from(value) - i128::from(i64::MIN)) as u128,
            u128::from(u64::MAX),
        ),
        NumericScalar::U8(value) => (u128::from(value), u128::from(u8::MAX)),
        NumericScalar::U16(value) => (u128::from(value), u128::from(u16::MAX)),
        NumericScalar::U32(value) => (u128::from(value), u128::from(u32::MAX)),
        NumericScalar::U64(value) => (u128::from(value), u128::from(u64::MAX)),
        NumericScalar::F64(_) | NumericScalar::F32(_) => {
            unreachable!("floating image samples use normalized conversion")
        }
    };
    (offset * output_max + input_max / 2) / input_max
}

fn numeric_scalar_text(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => value.to_string(),
        NumericScalar::F32(value) => value.to_string(),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}
