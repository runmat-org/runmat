use super::operation::LogarithmOperation;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

pub(super) fn evaluate(operation: LogarithmOperation, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(real, imag) => {
            let (real, imag) = operation.complex_f64(real, imag);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(tensor) => evaluate_complex(operation, tensor),
        Value::SparseTensor(_) => Err(super::errors::invalid(
            operation,
            "sparse input is not currently supported",
        )),
        Value::CharArray(chars) => evaluate_characters(operation, chars),
        Value::String(_) | Value::StringArray(_) => {
            Err(super::errors::invalid(operation, "expected numeric input"))
        }
        value => {
            let tensor = tensor::value_into_tensor_for(operation.name(), value)
                .map_err(|error| super::errors::invalid(operation, &error))?;
            evaluate_tensor(operation, tensor)
        }
    }
}

pub(super) fn evaluate_tensor(
    operation: LogarithmOperation,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| super::errors::internal(operation, &error))?;
    match storage {
        NumericStorage::F64(values) => evaluate_f64(operation, values, shape),
        NumericStorage::F32(values) => evaluate_f32(operation, values, shape),
        storage => evaluate_f64(
            operation,
            storage
                .into_integer_storage()
                .expect("logarithm integer boundary received floating storage")
                .to_f64_vec(),
            shape,
        ),
    }
}

fn evaluate_f64(
    operation: LogarithmOperation,
    values: Vec<f64>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    if values
        .iter()
        .any(|value| *value < operation.real_boundary())
    {
        let values: Vec<_> = values
            .into_iter()
            .map(|value| operation.complex_f64(value, 0.0))
            .collect();
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(values.into()), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = values
            .into_iter()
            .map(|value| operation.real_f64(value))
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(values), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn evaluate_f32(
    operation: LogarithmOperation,
    values: Vec<f32>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    if values
        .iter()
        .any(|value| f64::from(*value) < operation.real_boundary())
    {
        let values: Vec<_> = values
            .into_iter()
            .map(|value| operation.complex_f32(value, 0.0))
            .collect();
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(values.into()), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = values
            .into_iter()
            .map(|value| operation.real_f32(value))
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn evaluate_complex(operation: LogarithmOperation, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| operation.complex_f64(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| operation.complex_f32(real, imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(super::errors::invalid(
                operation,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|error| super::errors::internal(operation, &error))?;
    Ok(complex_tensor_into_value(tensor))
}

fn evaluate_characters(operation: LogarithmOperation, chars: CharArray) -> BuiltinResult<Value> {
    let values = chars
        .to_column_major()
        .iter()
        .map(|&ch| f64::from(ch.raw()))
        .collect();
    let tensor = Tensor::new(values, chars.shape)
        .map_err(|error| super::errors::internal(operation, &error))?;
    evaluate_tensor(operation, tensor)
}
