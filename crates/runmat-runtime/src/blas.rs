//! BLAS-accelerated matrix operations
//!
//! High-performance linear algebra using BLAS (Basic Linear Algebra Subprograms).
use runmat_value::Tensor as Matrix;

use crate::builtins::common::tensor;

/// Validate the matrix geometry before passing native dimensions/pointers.
pub(crate) fn matrix_dimensions(matrix: &Matrix) -> Result<(i32, i32), String> {
    if matrix.shape().iter().skip(2).any(|&extent| extent != 1)
        || matrix.rows().checked_mul(matrix.cols()) != Some(matrix.len())
    {
        return Err("BLAS/LAPACK input must be a two-dimensional matrix".to_string());
    }
    let rows = i32::try_from(matrix.rows())
        .map_err(|_| "Matrix rows exceed native BLAS/LAPACK limits".to_string())?;
    let cols = i32::try_from(matrix.cols())
        .map_err(|_| "Matrix columns exceed native BLAS/LAPACK limits".to_string())?;
    Ok((rows, cols))
}

/// BLAS-accelerated matrix multiplication: C = A * B
/// Uses DGEMM (double precision general matrix multiply)
pub fn blas_matrix_mul(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.cols() != b.rows() {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }

    let (m, k) = matrix_dimensions(a)?;
    let (_, n) = matrix_dimensions(b)?;
    let count = a
        .rows()
        .checked_mul(b.cols())
        .ok_or("Matrix output size overflow")?;
    let mut c_col_major = vec![0.0; count];
    if m == 0 || n == 0 || k == 0 {
        return Matrix::new_2d(c_col_major, a.rows(), b.cols());
    }

    // Tensor is already column-major. Borrow native F64 storage; other
    // numeric classes retain the existing explicit F64 computation boundary.
    let a_col_major = tensor::tensor_values_f64_cow(a);
    let b_col_major = tensor::tensor_values_f64_cow(b);

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

    let (m, n) = matrix_dimensions(matrix)?;
    let mut result = vec![0.0; matrix.rows()];
    if m == 0 || n == 0 {
        return Ok(result);
    }

    // Borrow the original column-major payload (or convert its numeric class).
    let matrix_col_major = tensor::tensor_values_f64_cow(matrix);

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

    if let Ok(n) = i32::try_from(a.len()) {
        return unsafe { Ok(blas::ddot(n, a, 1, b, 1)) };
    }
    Ok(a.chunks(i32::MAX as usize)
        .zip(b.chunks(i32::MAX as usize))
        .map(|(a, b)| {
            let n = i32::try_from(a.len()).expect("BLAS chunk is bounded by i32::MAX");
            unsafe { blas::ddot(n, a, 1, b, 1) }
        })
        .sum())
}

/// BLAS-accelerated vector norm (Euclidean norm)
/// Uses DNRM2 (double precision norm)
pub fn blas_vector_norm(vector: &[f64]) -> f64 {
    if let Ok(n) = i32::try_from(vector.len()) {
        return unsafe { blas::dnrm2(n, vector, 1) };
    }
    vector
        .chunks(i32::MAX as usize)
        .fold(0.0_f64, |norm, chunk| {
            let n = i32::try_from(chunk.len()).expect("BLAS chunk is bounded by i32::MAX");
            norm.hypot(unsafe { blas::dnrm2(n, chunk, 1) })
        })
}

/// BLAS-accelerated scalar-vector multiplication: y = alpha * x
/// Uses DSCAL (double precision scalar multiplication)
pub fn blas_scale_vector(vector: &mut [f64], alpha: f64) {
    for chunk in vector.chunks_mut(i32::MAX as usize) {
        let n = i32::try_from(chunk.len()).expect("BLAS chunk is bounded by i32::MAX");
        unsafe {
            blas::dscal(n, alpha, chunk, 1);
        }
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

    for (x, y) in x
        .chunks(i32::MAX as usize)
        .zip(y.chunks_mut(i32::MAX as usize))
    {
        let n = i32::try_from(x.len()).expect("BLAS chunk is bounded by i32::MAX");
        unsafe {
            blas::daxpy(n, alpha, x, 1, y, 1);
        }
    }
    Ok(())
}
