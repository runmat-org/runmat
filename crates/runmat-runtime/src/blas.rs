//! BLAS-accelerated matrix operations
//!
//! High-performance linear algebra using BLAS (Basic Linear Algebra Subprograms).

use runmat_builtins::{Tensor as Matrix, Value};
use runmat_macros::runtime_builtin;

/// Helper function to transpose a matrix from row-major to column-major
fn transpose_to_column_major(matrix: &Matrix) -> Vec<f64> {
    let mut result = vec![0.0; matrix.data.len()];
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            result[j * matrix.rows() + i] = matrix.data[i * matrix.cols() + j];
        }
    }
    result
}

/// Helper function to transpose a matrix from column-major to row-major  
// Removed: not needed with column-major everywhere

/// BLAS-accelerated matrix multiplication: C = A * B
/// Uses DGEMM (double precision general matrix multiply)
pub fn blas_matrix_mul(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.cols() != b.rows() {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows(), a.cols(), b.rows(), b.cols()
        ));
    }

    let m = a.rows() as i32;
    let n = b.cols() as i32;
    let k = a.cols() as i32;

    // Convert to column-major storage for BLAS
    let a_col_major = transpose_to_column_major(a);
    let b_col_major = transpose_to_column_major(b);
    let mut c_col_major = vec![0.0; (m * n) as usize];

    unsafe {
        blas::dgemm(
            b'N',             // trans_a: no transpose (already in column-major)
            b'N',             // trans_b: no transpose (already in column-major)
            m,                // m: number of rows of A and C
            n,                // n: number of columns of B and C
            k,                // k: number of columns of A and rows of B
            1.0,              // alpha: scalar multiplier for A*B
            &a_col_major,     // a: matrix A in column-major
            m,                // lda: leading dimension of A (rows)
            &b_col_major,     // b: matrix B in column-major
            k,                // ldb: leading dimension of B (rows of B)
            0.0,              // beta: scalar multiplier for C
            &mut c_col_major, // c: result matrix C in column-major
            m,                // ldc: leading dimension of C (rows of C)
        );
    }

    // Our Tensor uses column-major layout; DGEMM already produced C in column-major
    Matrix::new_2d(c_col_major, a.rows(), b.cols())
}

/// BLAS-accelerated matrix-vector multiplication: y = A * x
/// Uses DGEMV (double precision general matrix-vector multiply)
pub fn blas_matrix_vector_mul(matrix: &Matrix, vector: &[f64]) -> Result<Vec<f64>, String> {
    if matrix.cols() != vector.len() {
        return Err(format!(
            "Matrix columns {} must match vector length {}",
            matrix.cols(),
            vector.len()
        ));
    }

    let m = matrix.rows() as i32;
    let n = matrix.cols() as i32;
    let mut result = vec![0.0; matrix.rows()];

    // Convert matrix to column-major storage for BLAS
    let matrix_col_major = transpose_to_column_major(matrix);

    unsafe {
        blas::dgemv(
            b'N',              // trans: no transpose (already in column-major)
            m,                 // m: number of rows
            n,                 // n: number of columns
            1.0,               // alpha: scalar multiplier
            &matrix_col_major, // a: matrix A in column-major
            m,                 // lda: leading dimension (rows in column-major)
            vector,            // x: input vector
            1,                 // incx: increment for x
            0.0,               // beta: scalar multiplier for y
            &mut result,       // y: output vector
            1,                 // incy: increment for y
        );
    }

    Ok(result)
}

/// BLAS-accelerated dot product
/// Uses DDOT (double precision dot product)
pub fn blas_dot_product(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Vector lengths must match: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    let n = a.len() as i32;
    unsafe { Ok(blas::ddot(n, a, 1, b, 1)) }
}

/// BLAS-accelerated vector norm (Euclidean norm)
/// Uses DNRM2 (double precision norm)
pub fn blas_vector_norm(vector: &[f64]) -> f64 {
    let n = vector.len() as i32;
    unsafe { blas::dnrm2(n, vector, 1) }
}

/// BLAS-accelerated scalar-vector multiplication: y = alpha * x
/// Uses DSCAL (double precision scalar multiplication)
pub fn blas_scale_vector(vector: &mut [f64], alpha: f64) {
    let n = vector.len() as i32;
    unsafe {
        blas::dscal(n, alpha, vector, 1);
    }
}

/// BLAS-accelerated vector addition: y = alpha * x + y
/// Uses DAXPY (double precision alpha x plus y)
pub fn blas_vector_add(alpha: f64, x: &[f64], y: &mut [f64]) -> Result<(), String> {
    if x.len() != y.len() {
        return Err(format!(
            "Vector lengths must match: {} vs {}",
            x.len(),
            y.len()
        ));
    }

    let n = x.len() as i32;
    unsafe {
        blas::daxpy(n, alpha, x, 1, y, 1);
    }
    Ok(())
}

// Helper function to convert Vec<Value> to Vec<f64>
fn value_vector_to_f64(values: &[Value]) -> Result<Vec<f64>, String> {
    let mut out: Vec<f64> = Vec::new();
    for v in values {
        match v {
            Value::Num(n) => out.push(*n),
            Value::Int(i) => out.push(*i as f64),
            Value::Cell(c) => {
                for elem in &c.data {
                    match elem {
                        Value::Num(n) => out.push(*n),
                        Value::Int(i) => out.push(*i as f64),
                        _ => return Err(format!("Cannot convert {elem:?} to f64")),
                    }
                }
            }
            _ => return Err(format!("Cannot convert {v:?} to f64")),
        }
    }
    Ok(out)
}

// Builtin functions for BLAS operations
#[runtime_builtin(name = "blas_matmul")]
fn blas_matmul_builtin(a: Matrix, b: Matrix) -> Result<Matrix, String> {
    blas_matrix_mul(&a, &b)
}

#[runtime_builtin(name = "dot")]
fn dot_builtin(a: Vec<Value>, b: Vec<Value>) -> Result<f64, String> {
    let a_f64 = value_vector_to_f64(&a)?;
    let b_f64 = value_vector_to_f64(&b)?;
    blas_dot_product(&a_f64, &b_f64)
}

#[runtime_builtin(name = "norm")]
fn norm_builtin(vector: Vec<Value>) -> Result<f64, String> {
    let vector_f64 = value_vector_to_f64(&vector)?;
    Ok(blas_vector_norm(&vector_f64))
}
