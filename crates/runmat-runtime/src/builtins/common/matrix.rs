//! Matrix operations for MATLAB-compatible arithmetic
//!
//! Implements element-wise and matrix operations following MATLAB semantics.

use crate::builtins::common::{linalg, tensor};
use crate::BuiltinResult;
use runmat_value::{NumericDType, NumericScalar, NumericStorage, Tensor};

/// Matrix addition: C = A + B
pub fn matrix_add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} + {}x{}",
            a.rows, a.cols, b.rows, b.cols
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
        return Err(format!(
            "Matrix dimensions must agree: {}x{} - {}x{}",
            a.rows, a.cols, b.rows, b.cols
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
    crate::builtins::math::linalg::ops::mtimes::mtimes_eval(a, b).await
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

/// Matrix power: C = A^n (for positive integer n)
/// This computes A * A * ... * A (n times) via repeated multiplication
pub fn matrix_power(a: &Tensor, n: i32) -> Result<Tensor, String> {
    if a.rows() != a.cols() {
        return Err(format!(
            "Matrix must be square for matrix power: {}x{}",
            a.rows(),
            a.cols()
        ));
    }

    if n < 0 {
        return Err("Negative matrix powers not supported yet".to_string());
    }

    let mut result = numeric_identity(a.numeric_dtype(), a.rows())?;
    if n == 0 {
        return Tensor::from_numeric_storage(result, a.shape.clone());
    }

    let mut base = a.clone().into_numeric_storage()?;
    let mut exp = n as u32;

    while exp > 0 {
        if exp % 2 == 1 {
            result = numeric_square_matmul(result, &base, a.rows())?;
        }
        exp /= 2;
        if exp > 0 {
            base = numeric_square_matmul(base.clone(), &base, a.rows())?;
        }
    }

    Tensor::from_numeric_storage(result, a.shape.clone())
}

fn numeric_identity(dtype: NumericDType, size: usize) -> Result<NumericStorage, String> {
    let len = size
        .checked_mul(size)
        .ok_or_else(|| "matrix power identity size overflow".to_string())?;
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
        storage.set_value(index + index * size, one)?;
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
) -> Result<NumericStorage, String> {
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
        (lhs, rhs) => Err(format!(
            "matrix power storage class changed from {} to {}",
            lhs.class_name(),
            rhs.class_name()
        )),
    }
}

/// Complex matrix power: C = A^n (for positive integer n)
/// Uses binary exponentiation with complex matrix multiply
pub fn complex_matrix_power(
    a: &runmat_value::ComplexTensor,
    n: i32,
) -> Result<runmat_value::ComplexTensor, String> {
    if a.rows != a.cols {
        return Err(format!(
            "Matrix must be square for matrix power: {}x{}",
            a.rows, a.cols
        ));
    }
    if n < 0 {
        return Err("Negative matrix powers not supported yet".to_string());
    }
    if n == 0 {
        return Ok(complex_matrix_eye(a.rows));
    }
    if n == 1 {
        return Ok(a.clone());
    }
    let mut result = complex_matrix_eye(a.rows);
    let mut base = a.clone();
    let mut exp = n as u32;
    while exp > 0 {
        if exp % 2 == 1 {
            result = complex_matrix_mul(&result, &base)?;
        }
        base = complex_matrix_mul(&base, &base)?;
        exp /= 2;
    }
    Ok(result)
}

fn complex_matrix_eye(n: usize) -> runmat_value::ComplexTensor {
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); n * n];
    for i in 0..n {
        data[i * n + i] = (1.0, 0.0);
    }
    runmat_value::ComplexTensor::new_2d(data, n, n).unwrap()
}

/// Create identity matrix
pub fn matrix_eye(n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::new_2d(data, n, n).unwrap() // Always valid
}
