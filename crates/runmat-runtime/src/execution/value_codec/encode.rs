use runmat_builtins::{IntValue, Value};
use runmat_execution::value::{InlineValue, ValuePayload};

use super::ValueCodecError;

pub fn encode_inline_value(value: &Value) -> Result<ValuePayload, ValueCodecError> {
    let inline = match value {
        Value::Int(value) => match value {
            IntValue::I8(value) => InlineValue::I8(*value),
            IntValue::I16(value) => InlineValue::I16(*value),
            IntValue::I32(value) => InlineValue::I32(*value),
            IntValue::I64(value) => InlineValue::I64(*value),
            IntValue::U8(value) => InlineValue::U8(*value),
            IntValue::U16(value) => InlineValue::U16(*value),
            IntValue::U32(value) => InlineValue::U32(*value),
            IntValue::U64(value) => InlineValue::U64(*value),
        },
        Value::Num(value) => InlineValue::F64Bits(value.to_bits()),
        Value::Complex(real, imaginary) => InlineValue::ComplexF64Bits {
            real: real.to_bits(),
            imaginary: imaginary.to_bits(),
        },
        Value::Bool(value) => InlineValue::Logical(*value),
        Value::String(value) => InlineValue::String(value.clone()),
        Value::Cell(value) => InlineValue::Cell(
            value
                .data
                .iter()
                .map(encode_inline_value)
                .collect::<Result<_, _>>()?,
        ),
        Value::OutputList(value) => InlineValue::OutputList(
            value
                .iter()
                .map(encode_inline_value)
                .collect::<Result<_, _>>()?,
        ),
        Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => {
            return Err(ValueCodecError::Unsupported("execution handle"))
        }
        _ => return Err(ValueCodecError::Unsupported("runtime value")),
    };
    Ok(ValuePayload::Inline(Box::new(inline)))
}
