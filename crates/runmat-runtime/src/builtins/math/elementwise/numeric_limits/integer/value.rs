use runmat_builtins::IntegerLimitKind;
use runmat_value::{IntValue, NumericDType};

pub(super) fn scalar(dtype: NumericDType, kind: IntegerLimitKind) -> IntValue {
    use IntegerLimitKind::{Maximum, Minimum};

    match (dtype, kind) {
        (NumericDType::I8, Minimum) => IntValue::I8(i8::MIN),
        (NumericDType::I8, Maximum) => IntValue::I8(i8::MAX),
        (NumericDType::I16, Minimum) => IntValue::I16(i16::MIN),
        (NumericDType::I16, Maximum) => IntValue::I16(i16::MAX),
        (NumericDType::I32, Minimum) => IntValue::I32(i32::MIN),
        (NumericDType::I32, Maximum) => IntValue::I32(i32::MAX),
        (NumericDType::I64, Minimum) => IntValue::I64(i64::MIN),
        (NumericDType::I64, Maximum) => IntValue::I64(i64::MAX),
        (NumericDType::U8, Minimum) => IntValue::U8(0),
        (NumericDType::U8, Maximum) => IntValue::U8(u8::MAX),
        (NumericDType::U16, Minimum) => IntValue::U16(0),
        (NumericDType::U16, Maximum) => IntValue::U16(u16::MAX),
        (NumericDType::U32, Minimum) => IntValue::U32(0),
        (NumericDType::U32, Maximum) => IntValue::U32(u32::MAX),
        (NumericDType::U64, Minimum) => IntValue::U64(0),
        (NumericDType::U64, Maximum) => IntValue::U64(u64::MAX),
        (NumericDType::F32 | NumericDType::F64, _) => {
            unreachable!("integer limit operation carries an integer class")
        }
    }
}
