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
            super::validation::ensure_matrix(orientation, tensor.shape())?;
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
            super::validation::ensure_matrix(orientation, tensor.shape())?;
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
    ComplexTensor::from_f64_values_with_dtype(data, tensor.shape().to_vec(), dtype)
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
        tensor.shape().to_vec(),
        storage_dtype(tensor.numeric_dtype()),
    )
    .map_err(|error| errors::internal(orientation, error))
}

pub(super) fn divide_complex_scalar(
    orientation: SolveOrientation,
    values: &ComplexTensor,
    scalar: &ComplexTensor,
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let scalar_f32 = scalar.as_f32_slice();
    let scalar_f64 = scalar.as_f64_slice();
    if let (Some(numerators), Some(divisors), NumericDType::F32) =
        (values.as_f32_slice(), scalar_f32, dtype)
    {
        let divisor = (divisors[0].0, divisors[0].1);
        let data = numerators
            .iter()
            .map(|value| {
                let numerator = (value.0, value.1);
                if [numerator.0, numerator.1, divisor.0, divisor.1]
                    .into_iter()
                    .all(f32::is_finite)
                    && (divisor.0 != 0.0 || divisor.1 != 0.0)
                {
                    runmat_value::divide_complex_f32(numerator, divisor)
                } else {
                    // Preserve the previous F64 reciprocal/product exceptional policy.
                    let reciprocal = Complex64::new(1.0, 0.0)
                        / Complex64::new(f64::from(divisor.0), f64::from(divisor.1));
                    let quotient =
                        Complex64::new(f64::from(numerator.0), f64::from(numerator.1)) * reciprocal;
                    (quotient.re as f32, quotient.im as f32)
                }
            })
            .collect();
        return ComplexTensor::from_f32(data, values.shape().to_vec()).map_err(|error| {
            errors::internal(orientation, format!("{}: {error}", orientation.name()))
        });
    }
    let divisor = scalar_f64
        .map(|values| (values[0].0, values[0].1))
        .or_else(|| scalar_f32.map(|values| (f64::from(values[0].0), f64::from(values[0].1))))
        .ok_or_else(|| {
            errors::internal(
                orientation,
                "complex scalar solve requires floating storage",
            )
        })?;
    let numerators_f64 = values.as_f64_slice();
    let numerators_f32 = values.as_f32_slice();
    let quotient = |index: usize| -> BuiltinResult<(f64, f64)> {
        let numerator = numerators_f64
            .map(|values| (values[index].0, values[index].1))
            .or_else(|| {
                numerators_f32
                    .map(|values| (f64::from(values[index].0), f64::from(values[index].1)))
            })
            .ok_or_else(|| {
                errors::internal(
                    orientation,
                    "complex matrix solve requires floating storage",
                )
            })?;
        if [numerator.0, numerator.1, divisor.0, divisor.1]
            .into_iter()
            .all(f64::is_finite)
            && (divisor.0 != 0.0 || divisor.1 != 0.0)
        {
            Ok(runmat_value::divide_complex_f64(numerator, divisor))
        } else {
            // Retain the established exceptional/zero-divisor behavior.
            let reciprocal = Complex64::new(1.0, 0.0) / Complex64::new(divisor.0, divisor.1);
            let quotient = Complex64::new(numerator.0, numerator.1) * reciprocal;
            Ok((quotient.re, quotient.im))
        }
    };
    let output = if dtype == NumericDType::F32 {
        let data = (0..values.len())
            .map(|index| quotient(index).map(|(real, imag)| (real as f32, imag as f32)))
            .collect::<BuiltinResult<Vec<_>>>()?;
        ComplexTensor::from_f32(data, values.shape().to_vec())
    } else {
        let data = (0..values.len())
            .map(quotient)
            .collect::<BuiltinResult<Vec<_>>>()?;
        ComplexTensor::new(data, values.shape().to_vec())
    };
    output
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
    tensor.len() == 1
}

fn storage_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}
