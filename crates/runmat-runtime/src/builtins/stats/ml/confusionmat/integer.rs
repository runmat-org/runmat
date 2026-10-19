use super::*;
pub(super) fn int_value_dtype(value: &IntValue) -> NumericDType {
    IntegerStorage::from_scalar(value.clone()).numeric_dtype()
}

pub(super) fn integer_sort_key(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(v) => *v as i128,
        IntValue::I16(v) => *v as i128,
        IntValue::I32(v) => *v as i128,
        IntValue::I64(v) => *v as i128,
        IntValue::U8(v) => *v as i128,
        IntValue::U16(v) => *v as i128,
        IntValue::U32(v) => *v as i128,
        IntValue::U64(v) => *v as i128,
    }
}

pub(super) fn integer_storage_from_values(
    dtype: NumericDType,
    values: Vec<IntValue>,
) -> BuiltinResult<IntegerStorage> {
    macro_rules! collect_variant {
        ($variant:ident, $ty:ty) => {{
            values
                .into_iter()
                .map(|value| match value {
                    IntValue::$variant(value) => Ok(value),
                    _ => Err(internal("confusionmat: integer order class mismatch")),
                })
                .collect::<BuiltinResult<Vec<$ty>>>()
                .map(IntegerStorage::$variant)
        }};
    }
    match dtype {
        NumericDType::I8 => collect_variant!(I8, i8),
        NumericDType::I16 => collect_variant!(I16, i16),
        NumericDType::I32 => collect_variant!(I32, i32),
        NumericDType::I64 => collect_variant!(I64, i64),
        NumericDType::U8 => collect_variant!(U8, u8),
        NumericDType::U16 => collect_variant!(U16, u16),
        NumericDType::U32 => collect_variant!(U32, u32),
        NumericDType::U64 => collect_variant!(U64, u64),
        NumericDType::F32 | NumericDType::F64 => {
            Err(internal("confusionmat: floating dtype in integer order"))
        }
    }
}
