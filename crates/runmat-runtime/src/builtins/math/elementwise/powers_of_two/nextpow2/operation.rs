use runmat_value::{NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::{errors, BUILTIN_NAME};

pub(super) fn host(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(errors::invalid)?;
    Ok(tensor::tensor_into_value(transform(tensor)?))
}

pub(super) fn transform(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(errors::internal)?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64_exponent).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32_exponent).collect())
        }
        NumericStorage::I8(values) => NumericStorage::I8(signed_values(values)),
        NumericStorage::I16(values) => NumericStorage::I16(signed_values(values)),
        NumericStorage::I32(values) => NumericStorage::I32(signed_values(values)),
        NumericStorage::I64(values) => NumericStorage::I64(signed_values(values)),
        NumericStorage::U8(values) => NumericStorage::U8(unsigned_values(values)),
        NumericStorage::U16(values) => NumericStorage::U16(unsigned_values(values)),
        NumericStorage::U32(values) => NumericStorage::U32(unsigned_values(values)),
        NumericStorage::U64(values) => NumericStorage::U64(unsigned_values(values)),
    };
    Tensor::from_numeric_storage(output, shape).map_err(errors::internal)
}

fn f64_exponent(value: f64) -> f64 {
    let magnitude = value.abs();
    if magnitude == 0.0 {
        0.0
    } else {
        magnitude.log2().ceil()
    }
}

fn f32_exponent(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude == 0.0 {
        0.0
    } else {
        magnitude.log2().ceil()
    }
}

trait UnsignedValue: Copy {
    fn magnitude(self) -> u128;
    fn from_exponent(value: u128) -> Self;
}

trait SignedValue: Copy {
    fn magnitude(self) -> u128;
    fn from_exponent(value: u128) -> Self;
}

macro_rules! impl_unsigned_value {
    ($($ty:ty),+ $(,)?) => {$(
        impl UnsignedValue for $ty {
            fn magnitude(self) -> u128 { self as u128 }
            fn from_exponent(value: u128) -> Self { value as $ty }
        }
    )+};
}

macro_rules! impl_signed_value {
    ($(($signed:ty, $unsigned:ty)),+ $(,)?) => {$(
        impl SignedValue for $signed {
            fn magnitude(self) -> u128 { self.unsigned_abs() as $unsigned as u128 }
            fn from_exponent(value: u128) -> Self { value as $signed }
        }
    )+};
}

impl_unsigned_value!(u8, u16, u32, u64);
impl_signed_value!((i8, u8), (i16, u16), (i32, u32), (i64, u64));

fn unsigned_values<T: UnsignedValue>(values: Vec<T>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| T::from_exponent(integer_exponent(value.magnitude())))
        .collect()
}

fn signed_values<T: SignedValue>(values: Vec<T>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| T::from_exponent(integer_exponent(value.magnitude())))
        .collect()
}

fn integer_exponent(magnitude: u128) -> u128 {
    if magnitude == 0 {
        0
    } else {
        u128::from(u128::BITS - (magnitude - 1).leading_zeros())
    }
}
