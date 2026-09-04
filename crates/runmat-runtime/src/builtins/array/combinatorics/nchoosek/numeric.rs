use runmat_value::{IntValue, NumericDType, NumericScalar, Tensor};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) enum CoefficientClass {
    Double,
    Single,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

pub(super) fn nonnegative_f64(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let rounded = value.round();
    if value != rounded
        || rounded > usize::MAX as f64
        || (usize::BITS == 64 && rounded == usize::MAX as f64)
    {
        return None;
    }
    Some(rounded as usize)
}

pub(super) fn nonnegative_int(value: &IntValue) -> Option<usize> {
    match value {
        IntValue::I8(value) => (*value >= 0).then_some(*value as usize),
        IntValue::I16(value) => (*value >= 0).then_some(*value as usize),
        IntValue::I32(value) => (*value >= 0).then_some(*value as usize),
        IntValue::I64(value) => usize::try_from(*value).ok(),
        IntValue::U8(value) => Some(*value as usize),
        IntValue::U16(value) => Some(*value as usize),
        IntValue::U32(value) => usize::try_from(*value).ok(),
        IntValue::U64(value) => usize::try_from(*value).ok(),
    }
}

pub(super) fn nonnegative_scalar(value: NumericScalar) -> Option<usize> {
    match value {
        NumericScalar::F64(value) => nonnegative_f64(value),
        NumericScalar::F32(value) => nonnegative_f64(f64::from(value)),
        value => nonnegative_int(&value.into_int_value()?),
    }
}

pub(super) fn int_class(value: &IntValue) -> CoefficientClass {
    match value {
        IntValue::I8(_) => CoefficientClass::I8,
        IntValue::I16(_) => CoefficientClass::I16,
        IntValue::I32(_) => CoefficientClass::I32,
        IntValue::I64(_) => CoefficientClass::I64,
        IntValue::U8(_) => CoefficientClass::U8,
        IntValue::U16(_) => CoefficientClass::U16,
        IntValue::U32(_) => CoefficientClass::U32,
        IntValue::U64(_) => CoefficientClass::U64,
    }
}

pub(super) fn tensor_class(dtype: NumericDType) -> CoefficientClass {
    match dtype {
        NumericDType::F64 => CoefficientClass::Double,
        NumericDType::F32 => CoefficientClass::Single,
        NumericDType::I8 => CoefficientClass::I8,
        NumericDType::I16 => CoefficientClass::I16,
        NumericDType::I32 => CoefficientClass::I32,
        NumericDType::I64 => CoefficientClass::I64,
        NumericDType::U8 => CoefficientClass::U8,
        NumericDType::U16 => CoefficientClass::U16,
        NumericDType::U32 => CoefficientClass::U32,
        NumericDType::U64 => CoefficientClass::U64,
    }
}

pub(super) fn tensor_element_len(tensor: &Tensor) -> usize {
    crate::builtins::common::tensor::tensor_element_len(tensor)
}
