//! Host inverse kernels shared by inv and integer matrix powers.
use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{ComplexTensor, NumericDType, Tensor};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InverseError {
    #[error("input must be a square matrix.")]
    NonSquare,
    #[error("inputs must be 2-D matrices.")]
    Rank,
    #[error("matrix is singular to working precision.")]
    Singular,
    #[error("{0}")]
    Storage(String),
}

fn dimensions(shape: &[usize]) -> Result<(usize, usize), InverseError> {
    match shape {
        [] => Ok((1, 1)),
        [1] => Ok((1, 1)),
        [_] => Err(InverseError::NonSquare),
        [rows, cols, tail @ ..] => {
            if tail.iter().any(|&n| n != 1) {
                return Err(InverseError::Rank);
            }
            if rows != cols {
                return Err(InverseError::NonSquare);
            }
            Ok((*rows, *cols))
        }
    }
}
fn output_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}
pub(crate) fn real(matrix: &Tensor) -> Result<Tensor, InverseError> {
    let (rows, cols) = dimensions(matrix.shape())?;
    let dtype = output_dtype(matrix.numeric_dtype());
    if rows == 0 {
        return Tensor::new_with_dtype(Vec::new(), matrix.shape().to_vec(), dtype)
            .map_err(InverseError::Storage);
    }
    let values = super::tensor::tensor_values_f64_cow(matrix);
    let inverse = DMatrix::from_column_slice(rows, cols, &values)
        .try_inverse()
        .ok_or(InverseError::Singular)?;
    Tensor::new_with_dtype(inverse.as_slice().to_vec(), matrix.shape().to_vec(), dtype)
        .map_err(InverseError::Storage)
}
pub(crate) fn complex(matrix: &ComplexTensor) -> Result<ComplexTensor, InverseError> {
    let (rows, cols) = dimensions(matrix.shape())?;
    let dtype = output_dtype(matrix.numeric_dtype());
    if rows == 0 {
        return ComplexTensor::from_f64_values_with_dtype(
            Vec::new(),
            matrix.shape().to_vec(),
            dtype,
        )
        .map_err(InverseError::Storage);
    }
    let values = matrix
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect::<Vec<_>>();
    let inverse = DMatrix::from_column_slice(rows, cols, &values)
        .try_inverse()
        .ok_or(InverseError::Singular)?;
    let values = inverse.as_slice().iter().map(|z| (z.re, z.im)).collect();
    ComplexTensor::from_f64_values_with_dtype(values, matrix.shape().to_vec(), dtype)
        .map_err(InverseError::Storage)
}
