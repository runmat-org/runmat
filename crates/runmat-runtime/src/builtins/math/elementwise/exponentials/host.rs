use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) fn evaluate(operation: ExponentialOperation, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(real, imag) => {
            let (real, imag) = operation.apply_complex_f64(real, imag);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(tensor) => evaluate_complex(operation, tensor),
        Value::CharArray(chars) => evaluate_characters(operation, chars),
        Value::String(_) | Value::StringArray(_) => {
            Err(super::errors::invalid(operation, "expected numeric input"))
        }
        value => evaluate_real(operation, value),
    }
}

pub(super) fn evaluate_real(operation: ExponentialOperation, value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(operation.name(), value)
        .map_err(|error| super::errors::invalid(operation, error))?;
    evaluate_tensor(operation, tensor).map(tensor::tensor_into_value)
}

pub(super) fn evaluate_tensor(
    operation: ExponentialOperation,
    tensor: Tensor,
) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| super::errors::internal(operation, error))?;
    let output = match storage {
        NumericStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|value| operation.apply_f64(value))
                .collect(),
        ),
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| operation.apply_f32(value))
                .collect(),
        ),
        storage => {
            let integers = storage
                .into_integer_storage()
                .expect("exponential integer boundary received floating storage");
            ensure_exact_integer(operation, &integers)?;
            NumericStorage::F64(
                integers
                    .to_f64_vec()
                    .into_iter()
                    .map(|value| operation.apply_f64(value))
                    .collect(),
            )
        }
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|error| super::errors::internal(operation, error))
}

fn evaluate_characters(operation: ExponentialOperation, chars: CharArray) -> BuiltinResult<Value> {
    let values = chars
        .data
        .into_iter()
        .map(|value| operation.apply_f64(value as u32 as f64))
        .collect();
    Tensor::new(values, vec![chars.rows, chars.cols])
        .map(Value::Tensor)
        .map_err(|error| super::errors::internal(operation, error))
}

fn evaluate_complex(
    operation: ExponentialOperation,
    tensor: ComplexTensor,
) -> BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer_tensor(
        &tensor,
        operation.name(),
    )?;
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| operation.apply_complex_f64(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| operation.apply_complex_f32(real, imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => unreachable!("complex integer input rejected above"),
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(|error| super::errors::internal(operation, error))
}

pub(super) fn ensure_exact_integer(
    operation: ExponentialOperation,
    storage: &runmat_value::IntegerStorage,
) -> BuiltinResult<()> {
    if super::super::exact_integer::is_exact_binary64(storage) {
        Ok(())
    } else {
        Err(super::errors::invalid(
            operation,
            "integer input lies outside the inclusive exact binary64 interval [-2^53, 2^53]",
        ))
    }
}
