mod floating;
mod integer;
mod operand;
mod shape;

use runmat_builtins::COMPLEX_ERROR_INTERNAL;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::BuiltinResult;

use super::error;

pub(super) use operand::{from_value, RealInput};

pub(super) fn unary(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        other => lift_real(from_value(other)?),
    }
}

pub(super) fn binary(real: Value, imaginary: Value) -> BuiltinResult<Value> {
    compose(&from_value(real)?, &from_value(imaginary)?)
}

pub(super) fn compose(real: &RealInput, imaginary: &RealInput) -> BuiltinResult<Value> {
    if real.tensor.integer_storage().is_some() || imaginary.tensor.integer_storage().is_some() {
        integer::compose(real, imaginary)
    } else {
        floating::compose(&real.tensor, &imaginary.tensor)
    }
}

fn lift_real(input: RealInput) -> BuiltinResult<Value> {
    let shape = input.tensor.shape.clone();
    let scalar = shape::is_scalar(&input.tensor);
    let storage = input
        .tensor
        .into_numeric_storage()
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))?;
    match storage {
        NumericStorage::F64(values) if scalar => values
            .first()
            .copied()
            .map(|value| Value::Complex(value, 0.0))
            .ok_or_else(|| error(&COMPLEX_ERROR_INTERNAL, "scalar input had no storage")),
        NumericStorage::F64(values) => complex_value(
            ComplexStorage::F64(values.into_iter().map(|value| (value, 0.0)).collect()),
            shape,
        ),
        NumericStorage::F32(values) => complex_value(
            ComplexStorage::F32(values.into_iter().map(|value| (value, 0.0)).collect()),
            shape,
        ),
        integer_storage => integer::lift(integer_storage, shape),
    }
}

pub(super) fn complex_value(storage: ComplexStorage, shape: Vec<usize>) -> BuiltinResult<Value> {
    ComplexTensor::from_complex_storage(storage, shape)
        .map(complex_tensor_into_value)
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))
}
