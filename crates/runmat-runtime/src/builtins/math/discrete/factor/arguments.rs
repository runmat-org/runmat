use runmat_value::{IntValue, NumericDType, NumericScalar, Value};

use super::{factor_error, FactorError};
use crate::BuiltinResult;

#[derive(Clone, Copy)]
pub(super) enum OutputClass {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

pub(super) struct FactorInput {
    pub(super) value: u64,
    pub(super) class: OutputClass,
}

pub(super) fn parse_input(value: Value) -> BuiltinResult<FactorInput> {
    match value {
        Value::Num(value) => parse_float(value)
            .map(|value| FactorInput {
                value,
                class: OutputClass::F64,
            })
            .ok_or_else(invalid),
        Value::Int(value) => parse_int(value),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let class = class_from_dtype(tensor.numeric_dtype());
            parse_scalar(tensor.numeric_value_at(0).ok_or_else(invalid)?, class)
        }
        _ => Err(invalid()),
    }
}

fn parse_scalar(value: NumericScalar, class: OutputClass) -> BuiltinResult<FactorInput> {
    let value = match value {
        NumericScalar::F64(value) => parse_float(value),
        NumericScalar::F32(value) => parse_float(f64::from(value)),
        NumericScalar::I8(value) => u64::try_from(value).ok(),
        NumericScalar::I16(value) => u64::try_from(value).ok(),
        NumericScalar::I32(value) => u64::try_from(value).ok(),
        NumericScalar::I64(value) => u64::try_from(value).ok(),
        NumericScalar::U8(value) => Some(u64::from(value)),
        NumericScalar::U16(value) => Some(u64::from(value)),
        NumericScalar::U32(value) => Some(u64::from(value)),
        NumericScalar::U64(value) => Some(value),
    };
    value
        .map(|value| FactorInput { value, class })
        .ok_or_else(invalid)
}

fn parse_int(value: IntValue) -> BuiltinResult<FactorInput> {
    let class = match value {
        IntValue::I8(_) => OutputClass::I8,
        IntValue::I16(_) => OutputClass::I16,
        IntValue::I32(_) => OutputClass::I32,
        IntValue::I64(_) => OutputClass::I64,
        IntValue::U8(_) => OutputClass::U8,
        IntValue::U16(_) => OutputClass::U16,
        IntValue::U32(_) => OutputClass::U32,
        IntValue::U64(_) => OutputClass::U64,
    };
    parse_scalar(NumericScalar::from(value), class)
}

fn parse_float(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64)
        .then_some(value as u64)
}

fn class_from_dtype(dtype: NumericDType) -> OutputClass {
    match dtype {
        NumericDType::F64 => OutputClass::F64,
        NumericDType::F32 => OutputClass::F32,
        NumericDType::I8 => OutputClass::I8,
        NumericDType::I16 => OutputClass::I16,
        NumericDType::I32 => OutputClass::I32,
        NumericDType::I64 => OutputClass::I64,
        NumericDType::U8 => OutputClass::U8,
        NumericDType::U16 => OutputClass::U16,
        NumericDType::U32 => OutputClass::U32,
        NumericDType::U64 => OutputClass::U64,
    }
}

fn invalid() -> crate::RuntimeError {
    factor_error(
        FactorError::InvalidInput,
        "expected a real nonnegative integer scalar",
    )
}
