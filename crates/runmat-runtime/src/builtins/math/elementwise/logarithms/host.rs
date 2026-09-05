use super::operation::LogarithmOperation;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

const EPSILON: f64 = 1e-12;

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
    let shape = tensor.shape.clone();
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
    let values: Vec<_> = values
        .into_iter()
        .map(|value| normalize_f64(operation, operation.complex_f64(value, 0.0)))
        .collect();
    if values.iter().any(|(_, imag)| *imag != 0.0) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(values.into()), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = values.into_iter().map(|(real, _)| real).collect();
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
    let values: Vec<_> = values
        .into_iter()
        .map(|value| normalize_f32(operation, operation.complex_f32(value, 0.0)))
        .collect();
    if values.iter().any(|(_, imag)| *imag != 0.0) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(values.into()), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = values.into_iter().map(|(real, _)| real).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|error| super::errors::internal(operation, &error))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn evaluate_complex(operation: LogarithmOperation, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| normalize_f64(operation, operation.complex_f64(real, imag)))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| normalize_f32(operation, operation.complex_f32(real, imag)))
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
    let values = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(values, vec![chars.rows, chars.cols])
        .map_err(|error| super::errors::internal(operation, &error))?;
    evaluate_tensor(operation, tensor)
}

fn normalize_f64(operation: LogarithmOperation, (mut real, mut imag): (f64, f64)) -> (f64, f64) {
    if operation.normalizes_near_zero_real() && real.is_finite() && real.abs() < EPSILON {
        real = 0.0;
    }
    if !imag.is_finite() || imag.abs() < EPSILON {
        imag = 0.0;
    }
    (real, imag)
}
fn normalize_f32(operation: LogarithmOperation, (mut real, mut imag): (f32, f32)) -> (f32, f32) {
    if operation.normalizes_near_zero_real() && real.is_finite() && real.abs() < EPSILON as f32 {
        real = 0.0;
    }
    if !imag.is_finite() || imag.abs() < EPSILON as f32 {
        imag = 0.0;
    }
    (real, imag)
}
