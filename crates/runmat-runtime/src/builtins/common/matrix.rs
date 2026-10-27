//! Matrix operations for MATLAB-compatible arithmetic
//!
//! Implements element-wise and matrix operations following MATLAB semantics.

use crate::builtins::common::{linalg, tensor};
use crate::BuiltinResult;
use runmat_value::{NumericDType, NumericScalar, NumericStorage, Tensor};

fn diagnostic_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

/// Matrix addition: C = A + B
pub fn matrix_add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        let (a_rows, a_cols) = diagnostic_dimensions(a.shape());
        let (b_rows, b_cols) = diagnostic_dimensions(b.shape());
        return Err(format!(
            "Matrix dimensions must agree: {}x{} + {}x{}",
            a_rows, a_cols, b_rows, b_cols
        ));
    }

    let a_values = tensor::tensor_values_f64_cow(a);
    let b_values = tensor::tensor_values_f64_cow(b);
    let data: Vec<f64> = a_values
        .iter()
        .zip(b_values.iter())
        .map(|(x, y)| x + y)
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Matrix subtraction: C = A - B
pub fn matrix_sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        let (a_rows, a_cols) = diagnostic_dimensions(a.shape());
        let (b_rows, b_cols) = diagnostic_dimensions(b.shape());
        return Err(format!(
            "Matrix dimensions must agree: {}x{} - {}x{}",
            a_rows, a_cols, b_rows, b_cols
        ));
    }

    let a_values = tensor::tensor_values_f64_cow(a);
    let b_values = tensor::tensor_values_f64_cow(b);
    let data: Vec<f64> = a_values
        .iter()
        .zip(b_values.iter())
        .map(|(x, y)| x - y)
        .collect();

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Matrix multiplication: C = A * B
pub fn matrix_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    linalg::matmul_real(a, b)
}

/// GPU-aware matmul entry: if both inputs are GpuTensor handles, call provider; otherwise fall back to CPU.
pub async fn value_matmul(
    a: &runmat_value::Value,
    b: &runmat_value::Value,
) -> BuiltinResult<runmat_value::Value> {
    crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes::mtimes_eval(a, b).await
}

pub async fn value_matmul_in_context(
    context: &crate::context::RuntimeContext,
    a: &runmat_value::Value,
    b: &runmat_value::Value,
) -> BuiltinResult<runmat_value::Value> {
    crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes::mtimes_eval_in_context(
        context, a, b,
    )
    .await
}

fn complex_matrix_mul(
    a: &runmat_value::ComplexTensor,
    b: &runmat_value::ComplexTensor,
) -> Result<runmat_value::ComplexTensor, String> {
    linalg::matmul_complex(a, b)
}

/// Scalar multiplication: C = A * s
pub fn matrix_scalar_mul(a: &Tensor, scalar: f64) -> Tensor {
    linalg::scalar_mul_real(a, scalar)
}

/// Integer matrix powers use binary exponentiation; floating negative powers
/// first use the same inverse kernel as inv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixPowerError {
    NonSquare { rows: usize, columns: usize },
    NegativeExponent,
    Inverse(linalg::InverseError),
    IdentitySizeOverflow,
    StorageClassChanged { left: String, right: String },
    Storage(String),
}

impl std::fmt::Display for MatrixPowerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonSquare { rows, columns } => {
                write!(
                    formatter,
                    "Matrix must be square for matrix power: {rows}x{columns}"
                )
            }
            Self::NegativeExponent => {
                formatter.write_str("Negative matrix powers require floating-point storage")
            }
            Self::Inverse(error) => write!(formatter, "Matrix inverse failed: {error}"),
            Self::IdentitySizeOverflow => {
                formatter.write_str("matrix power identity size overflow")
            }
            Self::StorageClassChanged { left, right } => {
                write!(
                    formatter,
                    "matrix power storage class changed from {left} to {right}"
                )
            }
            Self::Storage(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for MatrixPowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inverse(error) => Some(error),
            _ => None,
        }
    }
}

pub fn matrix_power(a: &Tensor, n: i32) -> Result<Tensor, MatrixPowerError> {
    if a.rows() != a.cols() {
        return Err(MatrixPowerError::NonSquare {
            rows: a.rows(),
            columns: a.cols(),
        });
    }

    if n < 0 && !matches!(a.numeric_dtype(), NumericDType::F64 | NumericDType::F32) {
        return Err(MatrixPowerError::NegativeExponent);
    }

    if n == 0 {
        let result = numeric_identity(a.numeric_dtype(), a.rows())?;
        return Tensor::from_numeric_storage(result, a.shape().to_vec())
            .map_err(MatrixPowerError::Storage);
    }

    let base = if n < 0 {
        linalg::inverse::real(a).map_err(MatrixPowerError::Inverse)?
    } else {
        a.clone()
    };
    if n == -1 {
        return Ok(base);
    }
    let mut result = numeric_identity(a.numeric_dtype(), a.rows())?;
    let mut base = base
        .into_numeric_storage()
        .map_err(MatrixPowerError::Storage)?;
    let mut exp = n.unsigned_abs();

    while exp > 0 {
        if exp % 2 == 1 {
            result = numeric_square_matmul(result, &base, a.rows())?;
        }
        exp /= 2;
        if exp > 0 {
            base = numeric_square_matmul(base.clone(), &base, a.rows())?;
        }
    }

    Tensor::from_numeric_storage(result, a.shape().to_vec()).map_err(MatrixPowerError::Storage)
}

fn numeric_identity(dtype: NumericDType, size: usize) -> Result<NumericStorage, MatrixPowerError> {
    let len = size
        .checked_mul(size)
        .ok_or(MatrixPowerError::IdentitySizeOverflow)?;
    let mut storage = NumericStorage::zeros(dtype, len);
    let one = match dtype {
        NumericDType::F64 => NumericScalar::F64(1.0),
        NumericDType::F32 => NumericScalar::F32(1.0),
        NumericDType::I8 => NumericScalar::I8(1),
        NumericDType::I16 => NumericScalar::I16(1),
        NumericDType::I32 => NumericScalar::I32(1),
        NumericDType::I64 => NumericScalar::I64(1),
        NumericDType::U8 => NumericScalar::U8(1),
        NumericDType::U16 => NumericScalar::U16(1),
        NumericDType::U32 => NumericScalar::U32(1),
        NumericDType::U64 => NumericScalar::U64(1),
    };
    for index in 0..size {
        storage
            .set_value(index + index * size, one)
            .map_err(MatrixPowerError::Storage)?;
    }
    Ok(storage)
}

fn typed_square_matmul<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    size: usize,
    zero: T,
    multiply_accumulate: impl Fn(T, T, T) -> T,
) -> Vec<T> {
    let mut output = vec![zero; size * size];
    for column in 0..size {
        for row in 0..size {
            let mut accumulator = zero;
            for inner in 0..size {
                accumulator = multiply_accumulate(
                    accumulator,
                    lhs[row + inner * size],
                    rhs[inner + column * size],
                );
            }
            output[row + column * size] = accumulator;
        }
    }
    output
}

fn numeric_square_matmul(
    lhs: NumericStorage,
    rhs: &NumericStorage,
    size: usize,
) -> Result<NumericStorage, MatrixPowerError> {
    macro_rules! floating {
        ($lhs:expr, $rhs:expr, $variant:ident, $zero:expr) => {
            Ok(NumericStorage::$variant(typed_square_matmul(
                $lhs,
                $rhs,
                size,
                $zero,
                |accumulator, left, right| accumulator + left * right,
            )))
        };
    }
    macro_rules! integer {
        ($lhs:expr, $rhs:expr, $variant:ident, $zero:expr) => {
            Ok(NumericStorage::$variant(typed_square_matmul(
                $lhs,
                $rhs,
                size,
                $zero,
                |accumulator, left, right| accumulator.saturating_add(left.saturating_mul(right)),
            )))
        };
    }
    match (lhs, rhs) {
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            floating!(&lhs, rhs, F64, 0.0_f64)
        }
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            floating!(&lhs, rhs, F32, 0.0_f32)
        }
        (NumericStorage::I8(lhs), NumericStorage::I8(rhs)) => integer!(&lhs, rhs, I8, 0_i8),
        (NumericStorage::I16(lhs), NumericStorage::I16(rhs)) => {
            integer!(&lhs, rhs, I16, 0_i16)
        }
        (NumericStorage::I32(lhs), NumericStorage::I32(rhs)) => {
            integer!(&lhs, rhs, I32, 0_i32)
        }
        (NumericStorage::I64(lhs), NumericStorage::I64(rhs)) => {
            integer!(&lhs, rhs, I64, 0_i64)
        }
        (NumericStorage::U8(lhs), NumericStorage::U8(rhs)) => integer!(&lhs, rhs, U8, 0_u8),
        (NumericStorage::U16(lhs), NumericStorage::U16(rhs)) => {
            integer!(&lhs, rhs, U16, 0_u16)
        }
        (NumericStorage::U32(lhs), NumericStorage::U32(rhs)) => {
            integer!(&lhs, rhs, U32, 0_u32)
        }
        (NumericStorage::U64(lhs), NumericStorage::U64(rhs)) => {
            integer!(&lhs, rhs, U64, 0_u64)
        }
        (lhs, rhs) => Err(MatrixPowerError::StorageClassChanged {
            left: lhs.class_name().to_string(),
            right: rhs.class_name().to_string(),
        }),
    }
}

/// Complex integer powers use the shared inverse for negative exponents and
/// binary exponentiation with complex matrix multiplication.
pub fn complex_matrix_power(
    a: &runmat_value::ComplexTensor,
    n: i32,
) -> Result<runmat_value::ComplexTensor, MatrixPowerError> {
    if a.rows() != a.cols() {
        return Err(MatrixPowerError::NonSquare {
            rows: a.rows(),
            columns: a.cols(),
        });
    }
    if n < 0 && !matches!(a.numeric_dtype(), NumericDType::F64 | NumericDType::F32) {
        return Err(MatrixPowerError::NegativeExponent);
    }
    if n == 0 {
        return complex_matrix_identity(a);
    }
    let mut base = if n < 0 {
        linalg::inverse::complex(a).map_err(MatrixPowerError::Inverse)?
    } else {
        a.clone()
    };
    if n == 1 || n == -1 {
        return Ok(base);
    }
    let mut result = complex_matrix_identity(a)?;
    let mut exp = n.unsigned_abs();
    while exp > 0 {
        if exp % 2 == 1 {
            result = complex_matrix_mul(&result, &base).map_err(MatrixPowerError::Storage)?;
        }
        exp /= 2;
        if exp > 0 {
            base = complex_matrix_mul(&base, &base).map_err(MatrixPowerError::Storage)?;
        }
    }
    Ok(result)
}

fn complex_matrix_identity(
    prototype: &runmat_value::ComplexTensor,
) -> Result<runmat_value::ComplexTensor, MatrixPowerError> {
    let n = prototype.rows();
    let count = n
        .checked_mul(n)
        .ok_or(MatrixPowerError::IdentitySizeOverflow)?;
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); count];
    for i in 0..n {
        data[i * n + i] = (1.0, 0.0);
    }
    runmat_value::ComplexTensor::from_f64_values_with_dtype(
        data,
        vec![n, n],
        prototype.numeric_dtype(),
    )
    .map_err(MatrixPowerError::Storage)
}

/// Create identity matrix
pub fn matrix_eye(n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::new_2d(data, n, n).unwrap() // Always valid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn matrix_errors_keep_their_prior_rank_zero_and_rank_one_dimensions() {
        let scalar = Tensor::new(vec![7.0], vec![]).expect("rank-zero tensor");
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("row tensor");
        assert_eq!(
            matrix_add(&scalar, &row).expect_err("different matrix dimensions"),
            "Matrix dimensions must agree: 0x0 + 1x2"
        );

        let two = Tensor::new(vec![1.0, 2.0], vec![2]).expect("rank-one tensor");
        let three = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("rank-one tensor");
        assert_eq!(
            matrix_sub(&two, &three).expect_err("different matrix dimensions"),
            "Matrix dimensions must agree: 1x2 - 1x3"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn matrix_power_preserves_owned_shape_and_complex_empty_projection() {
        let input = Tensor::new(vec![3.0], vec![1, 1, 1]).expect("singleton matrix");
        let identity = matrix_power(&input, 0).expect("zero power");
        assert_eq!(identity.shape(), [1, 1, 1]);
        assert_eq!(identity.materialize_f64(), vec![1.0]);
        let squared = matrix_power(&input, 2).expect("square");
        assert_eq!(squared.shape(), [1, 1, 1]);
        assert_eq!(squared.materialize_f64(), vec![9.0]);

        let complex = runmat_value::ComplexTensor::new(vec![(3.0, 1.0)], vec![])
            .expect("rank-zero complex tensor");
        let identity = complex_matrix_power(&complex, 0).expect("zero power");
        assert_eq!(identity.shape(), [0, 0]);
        assert!(identity.materialize_f64().is_empty());
    }
}
