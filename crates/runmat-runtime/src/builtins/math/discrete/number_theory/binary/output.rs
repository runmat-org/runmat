use runmat_types::NumericClass;
use runmat_value::{IntValue, IntegerStorage, NumericDType, Tensor, Value};

use super::{binary_error, plan::element_count, BinaryContext, BinaryOutput};
use crate::BuiltinResult;

pub(in crate::builtins::math::discrete) fn magnitude_value(
    data: Vec<u128>,
    shape: Vec<usize>,
    output: BinaryOutput,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    if data
        .iter()
        .any(|&value| value > maximum_magnitude(output.class))
    {
        return Err(binary_error(
            context,
            context.overflow,
            "result exceeds output class range",
        ));
    }
    if data.len() == 1 && element_count(&shape) == 1 {
        return scalar_magnitude(data[0], shape, output.class, context);
    }
    if output.class.integer_class().is_some() {
        return Tensor::new_integer(integer_storage(data, output.class), shape)
            .map(Value::Tensor)
            .map_err(|error| binary_error(context, context.internal, error));
    }
    floating_tensor(
        data.into_iter().map(|value| value as f64),
        shape,
        output.class,
        context,
    )
}

pub(in crate::builtins::math::discrete) fn coefficient_value(
    data: Vec<i128>,
    shape: Vec<usize>,
    output: BinaryOutput,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    if let Some((minimum, maximum)) = signed_bounds(output.class) {
        if data.iter().any(|&value| value < minimum || value > maximum) {
            return Err(binary_error(
                context,
                context.overflow,
                "Bézout coefficient exceeds output class range",
            ));
        }
    }
    if data.len() == 1 && element_count(&shape) == 1 {
        return scalar_coefficient(data[0], shape, output.class, context);
    }
    if output.class.integer_class().is_some() {
        let storage = signed_integer_storage(data, output.class);
        return Tensor::new_integer(storage, shape)
            .map(Value::Tensor)
            .map_err(|error| binary_error(context, context.internal, error));
    }
    floating_tensor(
        data.into_iter().map(|value| value as f64),
        shape,
        output.class,
        context,
    )
}

fn scalar_magnitude(
    value: u128,
    shape: Vec<usize>,
    class: NumericClass,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    match class {
        NumericClass::Double => Ok(Value::Num(value as f64)),
        NumericClass::Single => Tensor::from_f32(vec![value as f32], shape)
            .map(Value::Tensor)
            .map_err(|error| binary_error(context, context.internal, error)),
        NumericClass::Int8 => Ok(Value::Int(IntValue::I8(value as i8))),
        NumericClass::Int16 => Ok(Value::Int(IntValue::I16(value as i16))),
        NumericClass::Int32 => Ok(Value::Int(IntValue::I32(value as i32))),
        NumericClass::Int64 => Ok(Value::Int(IntValue::I64(value as i64))),
        NumericClass::UInt8 => Ok(Value::Int(IntValue::U8(value as u8))),
        NumericClass::UInt16 => Ok(Value::Int(IntValue::U16(value as u16))),
        NumericClass::UInt32 => Ok(Value::Int(IntValue::U32(value as u32))),
        NumericClass::UInt64 => Ok(Value::Int(IntValue::U64(value as u64))),
    }
}

fn scalar_coefficient(
    value: i128,
    shape: Vec<usize>,
    class: NumericClass,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    match class {
        NumericClass::Double => Ok(Value::Num(value as f64)),
        NumericClass::Single => Tensor::from_f32(vec![value as f32], shape)
            .map(Value::Tensor)
            .map_err(|error| binary_error(context, context.internal, error)),
        NumericClass::Int8 => Ok(Value::Int(IntValue::I8(value as i8))),
        NumericClass::Int16 => Ok(Value::Int(IntValue::I16(value as i16))),
        NumericClass::Int32 => Ok(Value::Int(IntValue::I32(value as i32))),
        NumericClass::Int64 => Ok(Value::Int(IntValue::I64(value as i64))),
        NumericClass::UInt8
        | NumericClass::UInt16
        | NumericClass::UInt32
        | NumericClass::UInt64 => {
            unreachable!("unsigned classes do not support Bézout coefficients")
        }
    }
}

fn floating_tensor(
    values: impl IntoIterator<Item = f64>,
    shape: Vec<usize>,
    class: NumericClass,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    let dtype = floating_dtype(class).expect("floating output classes have a tensor dtype");
    let data = values
        .into_iter()
        .map(|value| match class {
            NumericClass::Single => (value as f32) as f64,
            _ => value,
        })
        .collect();
    Tensor::new_with_dtype(data, shape, dtype)
        .map(Value::Tensor)
        .map_err(|error| binary_error(context, context.internal, error))
}

fn integer_storage(data: Vec<u128>, class: NumericClass) -> IntegerStorage {
    match class {
        NumericClass::Int8 => {
            IntegerStorage::I8(data.into_iter().map(|value| value as i8).collect())
        }
        NumericClass::Int16 => {
            IntegerStorage::I16(data.into_iter().map(|value| value as i16).collect())
        }
        NumericClass::Int32 => {
            IntegerStorage::I32(data.into_iter().map(|value| value as i32).collect())
        }
        NumericClass::Int64 => {
            IntegerStorage::I64(data.into_iter().map(|value| value as i64).collect())
        }
        NumericClass::UInt8 => {
            IntegerStorage::U8(data.into_iter().map(|value| value as u8).collect())
        }
        NumericClass::UInt16 => {
            IntegerStorage::U16(data.into_iter().map(|value| value as u16).collect())
        }
        NumericClass::UInt32 => {
            IntegerStorage::U32(data.into_iter().map(|value| value as u32).collect())
        }
        NumericClass::UInt64 => {
            IntegerStorage::U64(data.into_iter().map(|value| value as u64).collect())
        }
        NumericClass::Double | NumericClass::Single => {
            unreachable!("integer output class required")
        }
    }
}

fn signed_integer_storage(data: Vec<i128>, class: NumericClass) -> IntegerStorage {
    match class {
        NumericClass::Int8 => {
            IntegerStorage::I8(data.into_iter().map(|value| value as i8).collect())
        }
        NumericClass::Int16 => {
            IntegerStorage::I16(data.into_iter().map(|value| value as i16).collect())
        }
        NumericClass::Int32 => {
            IntegerStorage::I32(data.into_iter().map(|value| value as i32).collect())
        }
        NumericClass::Int64 => {
            IntegerStorage::I64(data.into_iter().map(|value| value as i64).collect())
        }
        _ => unreachable!("signed integer output class required"),
    }
}

fn maximum_magnitude(class: NumericClass) -> u128 {
    class
        .integer_class()
        .map_or(u128::MAX, |integer| integer.range().1 as u128)
}

fn signed_bounds(class: NumericClass) -> Option<(i128, i128)> {
    class
        .integer_class()
        .filter(|integer| integer.is_signed())
        .map(|integer| integer.range())
}

fn floating_dtype(class: NumericClass) -> Option<NumericDType> {
    match class {
        NumericClass::Double => Some(NumericDType::F64),
        NumericClass::Single => Some(NumericDType::F32),
        _ => None,
    }
}
