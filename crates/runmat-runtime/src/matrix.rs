//! Matrix operations for MATLAB-compatible arithmetic
//!
//! Implements element-wise and matrix operations following MATLAB semantics.

use crate::builtins::common::linalg;
use crate::BuiltinResult;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

/// Matrix addition: C = A + B
pub fn matrix_add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} + {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
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

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
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
    a: &runmat_builtins::Value,
    b: &runmat_builtins::Value,
) -> BuiltinResult<runmat_builtins::Value> {
    crate::builtins::math::linalg::ops::mtimes::mtimes_eval(a, b).await
}

fn complex_matrix_mul(
    a: &runmat_builtins::ComplexTensor,
    b: &runmat_builtins::ComplexTensor,
) -> Result<runmat_builtins::ComplexTensor, String> {
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

    if n == 0 {
        // A^0 = I (identity matrix)
        return Ok(matrix_eye(a.rows));
    }

    if n == 1 {
        // A^1 = A
        return Ok(a.clone());
    }

    // Compute A^n via repeated multiplication
    // Use binary exponentiation for efficiency
    let mut result = matrix_eye(a.rows());
    let mut base = a.clone();
    let mut exp = n as u32;

    while exp > 0 {
        if exp % 2 == 1 {
            result = matrix_mul(&result, &base)?;
        }
        base = matrix_mul(&base, &base)?;
        exp /= 2;
    }

    Ok(result)
}

/// Complex matrix power: C = A^n (for positive integer n)
/// Uses binary exponentiation with complex matrix multiply
pub fn complex_matrix_power(
    a: &runmat_builtins::ComplexTensor,
    n: i32,
) -> Result<runmat_builtins::ComplexTensor, String> {
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

fn complex_matrix_eye(n: usize) -> runmat_builtins::ComplexTensor {
    let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); n * n];
    for i in 0..n {
        data[i * n + i] = (1.0, 0.0);
    }
    runmat_builtins::ComplexTensor::new_2d(data, n, n).unwrap()
}

/// Create identity matrix
pub fn matrix_eye(n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::new_2d(data, n, n).unwrap() // Always valid
}

const MATRIX_TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];

const MATRIX_ROWS_COLS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "rows",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row count.",
    },
    BuiltinParamDescriptor {
        name: "cols",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column count.",
    },
];
const MATRIX_N_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square matrix size.",
}];
const MATRIX_A_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];
const MATRIX_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output matrix.",
}];

const MATRIX_ZEROS_SIGS: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = matrix_zeros(rows, cols)",
    inputs: &MATRIX_ROWS_COLS_INPUTS,
    outputs: &MATRIX_OUT,
}];
const MATRIX_ONES_SIGS: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = matrix_ones(rows, cols)",
    inputs: &MATRIX_ROWS_COLS_INPUTS,
    outputs: &MATRIX_OUT,
}];
const MATRIX_EYE_SIGS: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = matrix_eye(n)",
    inputs: &MATRIX_N_INPUT,
    outputs: &MATRIX_OUT,
}];
const MATRIX_TRANSPOSE_SIGS: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = matrix_transpose(A)",
    inputs: &MATRIX_A_INPUT,
    outputs: &MATRIX_OUT,
}];

const MATRIX_ZEROS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATRIX_ZEROS_SIGS,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &MATRIX_TEST_ERRORS,
};
const MATRIX_ONES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATRIX_ONES_SIGS,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &MATRIX_TEST_ERRORS,
};
const MATRIX_EYE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATRIX_EYE_SIGS,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &MATRIX_TEST_ERRORS,
};
const MATRIX_TRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATRIX_TRANSPOSE_SIGS,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &MATRIX_TEST_ERRORS,
};

// Simple built-in function for testing matrix operations
#[runtime_builtin(
    name = "matrix_zeros",
    descriptor(crate::matrix::MATRIX_ZEROS_DESCRIPTOR),
    builtin_path = "crate::matrix"
)]
async fn matrix_zeros_builtin(rows: i32, cols: i32) -> crate::BuiltinResult<Tensor> {
    if rows < 0 || cols < 0 {
        return Err(("Matrix dimensions must be non-negative".to_string()).into());
    }
    Ok(Tensor::zeros(vec![rows as usize, cols as usize]))
}

#[runtime_builtin(
    name = "matrix_ones",
    descriptor(crate::matrix::MATRIX_ONES_DESCRIPTOR),
    builtin_path = "crate::matrix"
)]
async fn matrix_ones_builtin(rows: i32, cols: i32) -> crate::BuiltinResult<Tensor> {
    if rows < 0 || cols < 0 {
        return Err(("Matrix dimensions must be non-negative".to_string()).into());
    }
    Ok(Tensor::ones(vec![rows as usize, cols as usize]))
}

#[runtime_builtin(
    name = "matrix_eye",
    descriptor(crate::matrix::MATRIX_EYE_DESCRIPTOR),
    builtin_path = "crate::matrix"
)]
async fn matrix_eye_builtin(n: i32) -> crate::BuiltinResult<Tensor> {
    if n < 0 {
        return Err(("Matrix size must be non-negative".to_string()).into());
    }
    Ok(matrix_eye(n as usize))
}

#[runtime_builtin(
    name = "matrix_transpose",
    descriptor(crate::matrix::MATRIX_TRANSPOSE_DESCRIPTOR),
    builtin_path = "crate::matrix"
)]
async fn matrix_transpose_builtin(a: Tensor) -> crate::BuiltinResult<Tensor> {
    let args = [Value::Tensor(a)];
    let result = crate::call_builtin_async("transpose", &args).await?;
    match result {
        Value::Tensor(tensor) => Ok(tensor),
        other => Err((format!("matrix_transpose: expected tensor, got {other:?}")).into()),
    }
}
