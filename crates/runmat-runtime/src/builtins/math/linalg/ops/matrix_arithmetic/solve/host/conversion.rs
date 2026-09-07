use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};
use super::NumericInput;

pub(super) fn classify(orientation: SolveOrientation, value: Value) -> BuiltinResult<NumericInput> {
    match value {
        Value::ComplexTensor(tensor) => {
            super::validation::ensure_matrix(orientation, &tensor.shape)?;
            Ok(NumericInput::Complex(tensor))
        }
        Value::Complex(real, imag) => ComplexTensor::new(vec![(real, imag)], vec![1, 1])
            .map(NumericInput::Complex)
            .map_err(|error| {
                errors::invalid_input(orientation, format!("{}: {error}", orientation.name()))
            }),
        other => {
            let tensor = tensor::value_into_tensor_for(orientation.name(), other)
                .map_err(|error| errors::invalid_input(orientation, error))?;
            super::validation::ensure_matrix(orientation, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
    }
}

pub(super) fn real_tensor(
    orientation: SolveOrientation,
    matrix: DMatrix<f64>,
    dtype: NumericDType,
) -> BuiltinResult<Tensor> {
    let shape = vec![matrix.nrows(), matrix.ncols()];
    Tensor::new_with_dtype(matrix.as_slice().to_vec(), shape, dtype)
        .map_err(|error| errors::internal(orientation, format!("{}: {error}", orientation.name())))
}

pub(super) fn complex_tensor(
    orientation: SolveOrientation,
    matrix: DMatrix<Complex64>,
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let shape = vec![matrix.nrows(), matrix.ncols()];
    let data = matrix
        .as_slice()
        .iter()
        .map(|value| (value.re, value.im))
        .collect();
    ComplexTensor::from_f64_values_with_dtype(data, shape, dtype)
        .map_err(|error| errors::internal(orientation, format!("{}: {error}", orientation.name())))
}

pub(super) fn promote_real(
    orientation: SolveOrientation,
    tensor: &Tensor,
) -> BuiltinResult<ComplexTensor> {
    let data = tensor::tensor_values_f64_cow(tensor)
        .iter()
        .map(|&real| (real, 0.0))
        .collect();
    let dtype = storage_dtype(tensor.numeric_dtype());
    ComplexTensor::from_f64_values_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|error| errors::internal(orientation, format!("{}: {error}", orientation.name())))
}

pub(super) fn scale_real(
    orientation: SolveOrientation,
    tensor: &Tensor,
    scalar: f64,
) -> BuiltinResult<Tensor> {
    let values = tensor::tensor_values_f64_cow(tensor)
        .iter()
        .map(|value| value * scalar)
        .collect();
    Tensor::new_with_dtype(
        values,
        tensor.shape.clone(),
        storage_dtype(tensor.numeric_dtype()),
    )
    .map_err(|error| errors::internal(orientation, error))
}

pub(super) fn scale_complex(
    orientation: SolveOrientation,
    tensor: &ComplexTensor,
    scalar: Complex64,
) -> BuiltinResult<ComplexTensor> {
    let data = tensor
        .materialize_f64()
        .iter()
        .map(|&(real, imag)| {
            let value = Complex64::new(real, imag) * scalar;
            (value.re, value.im)
        })
        .collect();
    ComplexTensor::from_f64_values_with_dtype(
        data,
        tensor.shape.clone(),
        storage_dtype(tensor.numeric_dtype()),
    )
    .map_err(|error| errors::internal(orientation, format!("{}: {error}", orientation.name())))
}

pub(super) fn complex_values(tensor: &ComplexTensor) -> Vec<Complex64> {
    tensor
        .materialize_f64()
        .iter()
        .map(|&(real, imag)| Complex64::new(real, imag))
        .collect()
}

pub(super) fn is_complex_scalar(tensor: &ComplexTensor) -> bool {
    tensor.materialize_f64().len() == 1
}

fn storage_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}
