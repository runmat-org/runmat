use runmat_builtins::{CellArray, IntValue, Value};
use runmat_execution::value::{InlineValue, ValuePayload};

use super::ValueCodecError;

pub fn decode_inline_value(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    let ValuePayload::Inline(value) = payload else {
        return Err(ValueCodecError::Unsupported("object reference"));
    };
    match value.as_ref() {
        InlineValue::Logical(value) => Ok(Value::Bool(*value)),
        InlineValue::F64Bits(value) => Ok(Value::Num(f64::from_bits(*value))),
        InlineValue::I8(value) => Ok(Value::Int(IntValue::I8(*value))),
        InlineValue::I16(value) => Ok(Value::Int(IntValue::I16(*value))),
        InlineValue::I32(value) => Ok(Value::Int(IntValue::I32(*value))),
        InlineValue::I64(value) => Ok(Value::Int(IntValue::I64(*value))),
        InlineValue::U8(value) => Ok(Value::Int(IntValue::U8(*value))),
        InlineValue::U16(value) => Ok(Value::Int(IntValue::U16(*value))),
        InlineValue::U32(value) => Ok(Value::Int(IntValue::U32(*value))),
        InlineValue::U64(value) => Ok(Value::Int(IntValue::U64(*value))),
        InlineValue::ComplexF64Bits { real, imaginary } => Ok(Value::Complex(
            f64::from_bits(*real),
            f64::from_bits(*imaginary),
        )),
        InlineValue::String(value) => Ok(Value::String(value.clone())),
        InlineValue::Cell(values) => Ok(Value::Cell(
            CellArray::new(
                values
                    .iter()
                    .map(decode_inline_value)
                    .collect::<Result<_, _>>()?,
                1,
                values.len(),
            )
            .map_err(ValueCodecError::Invalid)?,
        )),
        InlineValue::OutputList(values) => Ok(Value::OutputList(
            values
                .iter()
                .map(decode_inline_value)
                .collect::<Result<_, _>>()?,
        )),
        _ => Err(ValueCodecError::Unsupported("inline payload")),
    }
}
