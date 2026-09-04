use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::errors;

pub(super) fn evaluate(value: Value) -> BuiltinResult<Value> {
    let tensor =
        tensor::value_into_tensor_for(super::BUILTIN_NAME, value).map_err(errors::internal)?;
    evaluate_tensor(tensor)
}

pub(super) fn evaluate_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(errors::internal)?;
    match storage {
        NumericStorage::F64(values) => evaluate_f64(values, shape),
        NumericStorage::F32(values) => evaluate_f32(values, shape),
        storage => evaluate_f64(
            storage
                .into_integer_storage()
                .expect("sqrt integer boundary received floating storage")
                .to_f64_vec(),
            shape,
        ),
    }
}

pub(super) fn evaluate_characters(chars: CharArray) -> BuiltinResult<Value> {
    let values = chars
        .data
        .into_iter()
        .map(|value| super::complex::zero_f64((value as u32 as f64).sqrt()))
        .collect();
    let tensor = Tensor::new(values, vec![chars.rows, chars.cols]).map_err(errors::internal)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn evaluate_f64(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if values.iter().all(|value| *value >= 0.0) {
        let output = values
            .into_iter()
            .map(|value| super::complex::zero_f64(value.sqrt()))
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(output), shape)
            .map_err(errors::internal)?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let output = values
        .into_iter()
        .map(|value| {
            if value < 0.0 {
                (0.0, super::complex::zero_f64((-value).sqrt()))
            } else {
                (super::complex::zero_f64(value.sqrt()), 0.0)
            }
        })
        .collect();
    let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(output), shape)
        .map_err(errors::internal)?;
    Ok(complex_tensor_into_value(tensor))
}

fn evaluate_f32(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if values.iter().all(|value| *value >= 0.0) {
        let output = values
            .into_iter()
            .map(|value| super::complex::zero_f32(value.sqrt()))
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(output), shape)
            .map_err(errors::internal)?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let output = values
        .into_iter()
        .map(|value| {
            if value < 0.0 {
                (0.0, super::complex::zero_f32((-value).sqrt()))
            } else {
                (super::complex::zero_f32(value.sqrt()), 0.0)
            }
        })
        .collect();
    let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(output), shape)
        .map_err(errors::internal)?;
    Ok(complex_tensor_into_value(tensor))
}
