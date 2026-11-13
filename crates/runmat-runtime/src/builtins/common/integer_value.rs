//! Exact construction at the boundary between integer classes and runtime values.

use runmat_types::IntegerClass;
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

use super::tensor;

pub(crate) fn value_from_exact_integers(
    values: Vec<IntValue>,
    shape: Vec<usize>,
    class: IntegerClass,
) -> Result<Value, String> {
    if values.len() == 1 && tensor::element_count(&shape) == 1 {
        return Ok(Value::Int(values[0].clone()));
    }
    let prototype = IntegerStorage::from_scalar(class.value_from_bits(0));
    let storage = prototype.from_exact_values_like(values)?;
    Tensor::new_integer(storage, shape).map(Value::Tensor)
}

pub(crate) trait IntegerClassValueExt {
    fn value_from_i128(self, value: i128) -> Option<IntValue>;
    fn value_from_bits(self, bits: u64) -> IntValue;
}

impl IntegerClassValueExt for IntegerClass {
    fn value_from_i128(self, value: i128) -> Option<IntValue> {
        let (minimum, maximum) = self.range();
        if !(minimum..=maximum).contains(&value) {
            return None;
        }
        Some(match self {
            Self::Int8 => IntValue::I8(value as i8),
            Self::Int16 => IntValue::I16(value as i16),
            Self::Int32 => IntValue::I32(value as i32),
            Self::Int64 => IntValue::I64(value as i64),
            Self::UInt8 => IntValue::U8(value as u8),
            Self::UInt16 => IntValue::U16(value as u16),
            Self::UInt32 => IntValue::U32(value as u32),
            Self::UInt64 => IntValue::U64(value as u64),
        })
    }

    fn value_from_bits(self, bits: u64) -> IntValue {
        match self {
            Self::Int8 => IntValue::I8(bits as u8 as i8),
            Self::Int16 => IntValue::I16(bits as u16 as i16),
            Self::Int32 => IntValue::I32(bits as u32 as i32),
            Self::Int64 => IntValue::I64(bits as i64),
            Self::UInt8 => IntValue::U8(bits as u8),
            Self::UInt16 => IntValue::U16(bits as u16),
            Self::UInt32 => IntValue::U32(bits as u32),
            Self::UInt64 => IntValue::U64(bits),
        }
    }
}
