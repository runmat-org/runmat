use super::*;

pub(super) fn value_from_integer_values(
    values: Vec<IntValue>,
    shape: Vec<usize>,
    class: IntegerClass,
    name: &'static str,
) -> BuiltinResult<Value> {
    value_from_exact_integers(values, shape, class)
        .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))
}

pub(super) fn value_from_bits_with_classes(
    data: Vec<u64>,
    shape: Vec<usize>,
    compute_class: Option<IntegerClass>,
    output_class: Option<IntegerClass>,
    name: &'static str,
) -> BuiltinResult<Value> {
    match output_class {
        Some(class) => {
            let values = data
                .into_iter()
                .map(|bits| class.value_from_bits(bits))
                .collect::<Vec<_>>();
            value_from_integer_values(values, shape, class, name)
        }
        None => {
            let double_value = |bits| match compute_class {
                Some(class) => class.value_from_bits(bits).to_i128() as f64,
                None => bits as f64,
            };
            if data.len() == 1 && tensor::element_count(&shape) == 1 {
                Ok(Value::Num(double_value(data[0])))
            } else {
                Tensor::new(data.into_iter().map(double_value).collect(), shape)
                    .map(Value::Tensor)
                    .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))
            }
        }
    }
}

pub(super) fn binary_output_class(
    name: &'static str,
    left: &BitBuffer,
    right: &BitBuffer,
) -> BuiltinResult<Option<IntegerClass>> {
    match (left.output_class, right.output_class) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Ok(Some(lhs)),
        (Some(_), Some(_)) => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "integer operands must have the same class unless the other operand is a scalar double",
        )),
        (Some(class), None) if right.is_scalar => Ok(Some(class)),
        (None, Some(class)) if left.is_scalar => Ok(Some(class)),
        (Some(_), None) | (None, Some(_)) => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "an integer array can only be combined with a scalar double",
        )),
        (None, None) => Ok(None),
    }
}

pub(super) fn apply_shift(value: u64, shift: i128, class: Option<IntegerClass>) -> u64 {
    let width = class.map_or(64, IntegerClass::bit_width);
    let mask = class.map_or(u64::MAX, IntegerClass::bit_mask);
    let value = value & mask;
    if shift >= width as i128 {
        return 0;
    }
    if shift <= -(width as i128) {
        return if class.is_some_and(IntegerClass::is_signed) && value & (1_u64 << (width - 1)) != 0
        {
            mask
        } else {
            0
        };
    }
    if shift >= 0 {
        return (value << shift as u32) & mask;
    }

    let amount = (-shift) as u32;
    let shifted = value >> amount;
    if class.is_some_and(IntegerClass::is_signed) && value & (1_u64 << (width - 1)) != 0 {
        (shifted | (mask << (width - amount))) & mask
    } else {
        shifted
    }
}

pub(super) fn double_to_u64(name: &'static str, value: f64) -> BuiltinResult<u64> {
    if !value.is_finite()
        || value.fract() != 0.0
        || !(0.0..18_446_744_073_709_551_616.0).contains(&value)
    {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("{name}: input values must be finite nonnegative integers smaller than 2^64"),
        ));
    }
    Ok(value as u64)
}

pub(super) fn double_to_bits(
    name: &'static str,
    value: f64,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<u64> {
    let Some(class) = assumed else {
        return double_to_u64(name, value);
    };
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "input values must be finite integers",
        ));
    }
    let (min, max) = class.range();
    let fits = match class {
        IntegerClass::Int64 => (-(2_f64.powi(63))..2_f64.powi(63)).contains(&value),
        IntegerClass::UInt64 => (0.0..2_f64.powi(64)).contains(&value),
        _ => (min as f64..=max as f64).contains(&value),
    };
    if !fits {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("input value {value} is outside assumedtype range"),
        ));
    }
    let integer = class.value_from_i128(value as i128).ok_or_else(|| {
        error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("input value {value} is outside assumedtype range"),
        )
    })?;
    Ok(int_to_bits(&integer))
}

pub(super) fn require_assumed_class(
    name: &'static str,
    native_class: IntegerClass,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Option<IntegerClass>> {
    if let Some(assumed) = assumed {
        if assumed != native_class {
            return Err(error_with_detail(
                name,
                &ERROR_INVALID_INPUT,
                "assumedtype must match the class of integer inputs",
            ));
        }
    }
    Ok(Some(native_class))
}

pub(super) fn double_to_shift(value: f64) -> BuiltinResult<i128> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            "bitshift: shift counts must be finite integers",
        ));
    }
    Ok(value.clamp(-128.0, 128.0) as i128)
}

pub(super) fn int_to_bits(value: &IntValue) -> u64 {
    match value {
        IntValue::I8(value) => *value as u8 as u64,
        IntValue::I16(value) => *value as u16 as u64,
        IntValue::I32(value) => *value as u32 as u64,
        IntValue::I64(value) => *value as u64,
        IntValue::U8(value) => *value as u64,
        IntValue::U16(value) => *value as u64,
        IntValue::U32(value) => *value as u64,
        IntValue::U64(value) => *value,
    }
}
