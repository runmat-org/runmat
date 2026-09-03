use once_cell::sync::Lazy;
use runmat_value::{NumericStorage, Tensor};

use super::{factorial_error, integer, FactorialError};
use crate::BuiltinResult;

const MAX_DOUBLE_FACTORIAL_N: usize = 170;
const FIRST_SINGLE_OVERFLOW_N: usize = 35;

static DOUBLE_TABLE: Lazy<[f64; MAX_DOUBLE_FACTORIAL_N + 1]> = Lazy::new(|| {
    let mut table = [1.0f64; MAX_DOUBLE_FACTORIAL_N + 1];
    let mut accumulator = 1.0;
    for (n, slot) in table.iter_mut().enumerate().skip(1) {
        accumulator *= n as f64;
        *slot = accumulator;
    }
    table
});

pub(super) fn factorial_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| factorial_error(FactorialError::Internal, error))?;
    let output = match storage {
        NumericStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(evaluate_double)
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(evaluate_single)
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        NumericStorage::I8(values) => NumericStorage::I8(integer::evaluate_signed(values)?),
        NumericStorage::I16(values) => NumericStorage::I16(integer::evaluate_signed(values)?),
        NumericStorage::I32(values) => NumericStorage::I32(integer::evaluate_signed(values)?),
        NumericStorage::I64(values) => NumericStorage::I64(integer::evaluate_signed(values)?),
        NumericStorage::U8(values) => NumericStorage::U8(integer::evaluate_unsigned(values)),
        NumericStorage::U16(values) => NumericStorage::U16(integer::evaluate_unsigned(values)),
        NumericStorage::U32(values) => NumericStorage::U32(integer::evaluate_unsigned(values)),
        NumericStorage::U64(values) => NumericStorage::U64(integer::evaluate_unsigned(values)),
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|error| factorial_error(FactorialError::Internal, error))
}

fn evaluate_double(value: f64) -> BuiltinResult<f64> {
    let n = validate_double(value)?;
    Ok(if n > MAX_DOUBLE_FACTORIAL_N {
        f64::INFINITY
    } else {
        DOUBLE_TABLE[n]
    })
}

fn evaluate_single(value: f32) -> BuiltinResult<f32> {
    let n = validate_single(value)?;
    Ok(if n >= FIRST_SINGLE_OVERFLOW_N {
        f32::INFINITY
    } else {
        DOUBLE_TABLE[n] as f32
    })
}

fn validate_double(value: f64) -> BuiltinResult<usize> {
    validate_domain(value)?;
    Ok(if value > MAX_DOUBLE_FACTORIAL_N as f64 {
        MAX_DOUBLE_FACTORIAL_N + 1
    } else {
        value as usize
    })
}

fn validate_single(value: f32) -> BuiltinResult<usize> {
    validate_domain(value)?;
    Ok(if value >= FIRST_SINGLE_OVERFLOW_N as f32 {
        FIRST_SINGLE_OVERFLOW_N
    } else {
        value as usize
    })
}

fn validate_domain<T>(value: T) -> BuiltinResult<()>
where
    T: num_traits::Float,
{
    if !value.is_finite() || value < T::zero() || value.fract() != T::zero() {
        return Err(factorial_error(
            FactorialError::InvalidInput,
            "input values must be real, finite, nonnegative integers",
        ));
    }
    Ok(())
}
