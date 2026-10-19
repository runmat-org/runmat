use runmat_builtins::PRIMES_ERROR_INTERNAL;
use runmat_types::NumericClass;
use runmat_value::{IntegerStorage, NumericDType, Tensor, Value};

use super::arguments::PrimeRequest;
use super::error::primes_error;
use super::sieve::primes_through;
use crate::BuiltinResult;

pub(super) fn evaluate(request: PrimeRequest) -> BuiltinResult<Value> {
    let primes = primes_through(request.limit);
    let shape = vec![1, primes.len()];
    let tensor = match request.output_class {
        NumericClass::Double => Tensor::new_with_dtype(
            primes.iter().map(|&prime| prime as f64).collect(),
            shape,
            NumericDType::F64,
        ),
        NumericClass::Single => {
            Tensor::from_f32(primes.iter().map(|&prime| prime as f32).collect(), shape)
        }
        NumericClass::Int8 => Tensor::new_integer(
            IntegerStorage::I8(primes.iter().map(|&prime| prime as i8).collect()),
            shape,
        ),
        NumericClass::Int16 => Tensor::new_integer(
            IntegerStorage::I16(primes.iter().map(|&prime| prime as i16).collect()),
            shape,
        ),
        NumericClass::Int32 => Tensor::new_integer(
            IntegerStorage::I32(primes.iter().map(|&prime| prime as i32).collect()),
            shape,
        ),
        NumericClass::Int64 => Tensor::new_integer(
            IntegerStorage::I64(primes.iter().map(|&prime| prime as i64).collect()),
            shape,
        ),
        NumericClass::UInt8 => Tensor::new_integer(
            IntegerStorage::U8(primes.iter().map(|&prime| prime as u8).collect()),
            shape,
        ),
        NumericClass::UInt16 => Tensor::new_integer(
            IntegerStorage::U16(primes.iter().map(|&prime| prime as u16).collect()),
            shape,
        ),
        NumericClass::UInt32 => Tensor::new_integer(
            IntegerStorage::U32(primes.iter().map(|&prime| prime as u32).collect()),
            shape,
        ),
        NumericClass::UInt64 => Tensor::new_integer(IntegerStorage::U64(primes), shape),
    };
    tensor
        .map(Value::Tensor)
        .map_err(|error| primes_error(&PRIMES_ERROR_INTERNAL, error))
}
