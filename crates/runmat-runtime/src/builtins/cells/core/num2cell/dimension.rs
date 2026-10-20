use runmat_value::{IntValue, NumericScalar, Value};

use super::error;

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    let mut dimensions = Vec::new();
    visit(value, |dimension| {
        dimensions.push(dimension);
        Ok(())
    })?;
    Ok(dimensions)
}

pub(super) fn visit(
    value: &Value,
    mut accept: impl FnMut(usize) -> crate::BuiltinResult<()>,
) -> crate::BuiltinResult<()> {
    match value {
        Value::Num(value) => accept(floating(*value)?),
        Value::Int(value) => accept(integer(value)?),
        Value::Tensor(tensor) if is_vector(tensor.shape()) => {
            for index in 0..tensor.len() {
                let value = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| error::invalid_input("dimension storage is inconsistent"))?;
                accept(numeric(value)?)?;
            }
            Ok(())
        }
        Value::Tensor(_) => Err(error::invalid_input(
            "dimensions must be a positive integer scalar or vector",
        )),
        _ => Err(error::invalid_input(
            "dimensions must be a positive integer scalar or vector",
        )),
    }
}

fn is_vector(shape: &[usize]) -> bool {
    shape.iter().filter(|&&extent| extent > 1).count() <= 1
}

fn numeric(value: NumericScalar) -> crate::BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => floating(value),
        NumericScalar::F32(value) => floating(f64::from(value)),
        NumericScalar::I8(value) => integer(&IntValue::I8(value)),
        NumericScalar::I16(value) => integer(&IntValue::I16(value)),
        NumericScalar::I32(value) => integer(&IntValue::I32(value)),
        NumericScalar::I64(value) => integer(&IntValue::I64(value)),
        NumericScalar::U8(value) => integer(&IntValue::U8(value)),
        NumericScalar::U16(value) => integer(&IntValue::U16(value)),
        NumericScalar::U32(value) => integer(&IntValue::U32(value)),
        NumericScalar::U64(value) => integer(&IntValue::U64(value)),
    }
}

fn floating(value: f64) -> crate::BuiltinResult<usize> {
    if !value.is_finite()
        || value < 1.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        return Err(error::invalid_input(
            "dimensions must contain positive integers",
        ));
    }
    Ok(value as usize)
}

fn integer(value: &IntValue) -> crate::BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value >= 1)
        .ok_or_else(|| error::invalid_input("dimensions must contain positive integers"))
}
