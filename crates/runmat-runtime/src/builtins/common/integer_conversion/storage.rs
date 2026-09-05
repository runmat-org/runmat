use runmat_value::{IntValue, IntegerStorage};

pub(crate) fn integer_values(storage: IntegerStorage) -> Vec<IntValue> {
    match storage {
        IntegerStorage::I8(values) => values.into_iter().map(IntValue::I8).collect(),
        IntegerStorage::I16(values) => values.into_iter().map(IntValue::I16).collect(),
        IntegerStorage::I32(values) => values.into_iter().map(IntValue::I32).collect(),
        IntegerStorage::I64(values) => values.into_iter().map(IntValue::I64).collect(),
        IntegerStorage::U8(values) => values.into_iter().map(IntValue::U8).collect(),
        IntegerStorage::U16(values) => values.into_iter().map(IntValue::U16).collect(),
        IntegerStorage::U32(values) => values.into_iter().map(IntValue::U32).collect(),
        IntegerStorage::U64(values) => values.into_iter().map(IntValue::U64).collect(),
    }
}
