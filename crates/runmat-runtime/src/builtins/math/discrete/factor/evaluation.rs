use runmat_value::{NumericStorage, Tensor, Value};

use super::arguments::{FactorInput, OutputClass};
use super::{factor_error, FactorError};
use crate::builtins::math::discrete::number_theory::prime_factors;
use crate::BuiltinResult;

pub(super) fn result_value(input: FactorInput) -> BuiltinResult<Value> {
    let values = prime_factors(input.value);
    macro_rules! cast {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(values.iter().map(|&value| value as $ty).collect())
        };
    }
    let storage = match input.class {
        OutputClass::F64 => cast!(F64, f64),
        OutputClass::F32 => cast!(F32, f32),
        OutputClass::I8 => cast!(I8, i8),
        OutputClass::I16 => cast!(I16, i16),
        OutputClass::I32 => cast!(I32, i32),
        OutputClass::I64 => cast!(I64, i64),
        OutputClass::U8 => cast!(U8, u8),
        OutputClass::U16 => cast!(U16, u16),
        OutputClass::U32 => cast!(U32, u32),
        OutputClass::U64 => NumericStorage::U64(values),
    };
    let len = storage.len();
    Tensor::from_numeric_storage(storage, vec![1, len])
        .map(Value::Tensor)
        .map_err(|detail| factor_error(FactorError::Internal, detail))
}
