use super::super::super::errors;
use super::super::OPERATION;
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{CharArray, NumericStorage, Tensor, Value};

pub(super) fn characters(chars: CharArray) -> BuiltinResult<(Value, Value)> {
    let values = chars
        .data
        .iter()
        .map(|character| *character as u32 as f64)
        .collect();
    let tensor = Tensor::new(values, vec![chars.rows, chars.cols])
        .map_err(|error| errors::internal(OPERATION, &error))?;
    tensor_values(tensor)
}

pub(super) fn numeric(value: Value) -> BuiltinResult<(Value, Value)> {
    let tensor = tensor::value_into_tensor_for(OPERATION.name(), value)
        .map_err(|error| errors::invalid(OPERATION, &error))?;
    tensor_values(tensor)
}

fn tensor_values(tensor: Tensor) -> BuiltinResult<(Value, Value)> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| errors::internal(OPERATION, &error))?;
    match storage {
        NumericStorage::F32(values) => {
            let (fractions, exponents): (Vec<_>, Vec<_>) =
                values.into_iter().map(dissection_f32).unzip();
            values_to_output(
                NumericStorage::F32(fractions),
                NumericStorage::F32(exponents),
                shape,
            )
        }
        NumericStorage::F64(values) => {
            let (fractions, exponents): (Vec<_>, Vec<_>) =
                values.into_iter().map(dissection_f64).unzip();
            values_to_output(
                NumericStorage::F64(fractions),
                NumericStorage::F64(exponents),
                shape,
            )
        }
        storage => {
            let values = storage
                .into_integer_storage()
                .expect("log2 dissection received non-integer storage")
                .to_f64_vec();
            let (fractions, exponents): (Vec<_>, Vec<_>) =
                values.into_iter().map(dissection_f64).unzip();
            values_to_output(
                NumericStorage::F64(fractions),
                NumericStorage::F64(exponents),
                shape,
            )
        }
    }
}

fn dissection_f64(value: f64) -> (f64, f64) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0.0);
    }
    let (fraction, exponent) = libm::frexp(value);
    (fraction, f64::from(exponent))
}

fn dissection_f32(value: f32) -> (f32, f32) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0.0);
    }
    let (fraction, exponent) = libm::frexpf(value);
    (fraction, exponent as f32)
}

fn values_to_output(
    fractions: NumericStorage,
    exponents: NumericStorage,
    shape: Vec<usize>,
) -> BuiltinResult<(Value, Value)> {
    let fractions = Tensor::from_numeric_storage(fractions, shape.clone())
        .map_err(|error| errors::internal(OPERATION, &error))?;
    let exponents = Tensor::from_numeric_storage(exponents, shape)
        .map_err(|error| errors::internal(OPERATION, &error))?;
    Ok((
        tensor::tensor_into_value(fractions),
        tensor::tensor_into_value(exponents),
    ))
}
