//! LAPACK linear algebra routines
//!
//! Advanced linear algebra operations including decompositions and linear system solvers.

use runmat_builtins::{Tensor as Matrix, Value};

/// LU decomposition result
pub struct LuDecomposition {
    pub lu_matrix: Matrix,
    pub pivots: Vec<i32>,
}

/// QR decomposition result  
pub struct QrDecomposition {
    pub q: Matrix,
    pub r: Matrix,
}

/// SVD decomposition result
pub struct SvdDecomposition {
    pub u: Matrix,
    pub s: Vec<f64>,
    pub vt: Matrix,
}

/// Eigenvalue decomposition result
pub struct EigenDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Option<Matrix>,
}

/// Solve linear system Ax = b using LU decomposition with partial pivoting
/// Uses DGESV (double precision general solve)
pub fn lapack_solve_linear_system(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, String> {
    if !is_square(a) {
        return Err("Matrix must be square for linear system solving".to_string());
    }

    if a.rows() != b.len() {
        return Err(format!(
            "Matrix rows {} must match RHS vector length {}",
            a.rows(),
            b.len()
        ));
    }

    let n = a.rows() as i32;
    let nrhs = 1i32;

    // Copy data since LAPACK modifies input
    // Transpose A since LAPACK expects column-major but we store row-major
    let mut a_copy = vec![0.0; a.data.len()];
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            a_copy[j * a.rows() + i] = a.data[i * a.cols() + j]; // transpose
        }
    }
    let mut b_copy = b.to_vec();
    let mut ipiv = vec![0i32; n as usize];
    let mut info = 0i32;

    unsafe {
        lapack::dgesv(
            n,           // n: order of matrix A
            nrhs,        // nrhs: number of right hand sides
            &mut a_copy, // a: coefficient matrix (modified)
            n,           // lda: leading dimension of A
            &mut ipiv,   // ipiv: pivot indices
            &mut b_copy, // b: right hand side (replaced with solution)
            n,           // ldb: leading dimension of B
            &mut info,   // info: success/error code
        );
    }

    if info != 0 {
        return Err(format!("LAPACK DGESV failed with info = {info}"));
    }

    Ok(b_copy)
}

/// LU decomposition with partial pivoting
/// Uses DGETRF (double precision general factorization)
pub fn lapack_lu_decomposition(matrix: &Matrix) -> Result<LuDecomposition, String> {
    if !is_square(matrix) {
        return Err("Matrix must be square for LU decomposition".to_string());
    }

    let n = matrix.rows() as i32;
    // Transpose matrix since LAPACK expects column-major but we store row-major
    let mut a_copy = vec![0.0; matrix.data.len()];
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            a_copy[j * matrix.rows() + i] = matrix.data[i * matrix.cols() + j]; // transpose
        }
    }
    let mut ipiv = vec![0i32; n as usize];
    let mut info = 0i32;

    unsafe {
        lapack::dgetrf(
            n,           // m: number of rows
            n,           // n: number of columns
            &mut a_copy, // a: matrix to factorize
            n,           // lda: leading dimension
            &mut ipiv,   // ipiv: pivot indices
            &mut info,   // info: success/error code
        );
    }

    if info != 0 {
        return Err(format!("LAPACK DGETRF failed with info = {info}"));
    }

    // Transpose result back to row-major
    let mut lu_data = vec![0.0; a_copy.len()];
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            lu_data[i * matrix.cols() + j] = a_copy[j * matrix.rows() + i]; // transpose back
        }
    }
    let lu_matrix = Matrix::new_2d(lu_data, matrix.rows(), matrix.cols())?;

    Ok(LuDecomposition {
        lu_matrix,
        pivots: ipiv,
    })
}

/// QR decomposition
/// Uses DGEQRF (double precision general QR factorization)
pub fn lapack_qr_decomposition(matrix: &Matrix) -> Result<QrDecomposition, String> {
    let m = matrix.rows() as i32;
    let n = matrix.cols() as i32;
    // LAPACK expects column-major; our Tensor stores in column-major already
    let mut a_copy = matrix.data.clone();
    let mut tau = vec![0.0; std::cmp::min(m, n) as usize];
    let mut work = vec![0.0; 1];
    let mut lwork = -1i32;
    let mut info = 0i32;

    // Query optimal work array size
    unsafe {
        lapack::dgeqrf(m, n, &mut a_copy, m, &mut tau, &mut work, lwork, &mut info);
    }

    if info != 0 {
        return Err(format!(
            "LAPACK DGEQRF workspace query failed with info = {info}"
        ));
    }

    lwork = work[0] as i32;
    work.resize(lwork as usize, 0.0);

    // Perform QR factorization
    unsafe {
        lapack::dgeqrf(m, n, &mut a_copy, m, &mut tau, &mut work, lwork, &mut info);
    }

    if info != 0 {
        return Err(format!("LAPACK DGEQRF failed with info = {info}"));
    }

    // Extract R matrix (upper triangular part)
    // Extract R from the upper-triangular part of A (column-major)
    let mut r_data = vec![0.0; (n * n) as usize];
    for col in 0..n as usize {
        for row in 0..=col {
            r_data[row + col * n as usize] = a_copy[row + col * m as usize];
        }
    }
    let r = Matrix::new_2d(r_data, n as usize, n as usize)?;

    // Generate Q matrix using DORGQR
    let mut q_data = a_copy;
    work.resize(lwork as usize, 0.0);

    unsafe {
        lapack::dorgqr(m, n, n, &mut q_data, m, &tau, &mut work, lwork, &mut info);
    }

    if info != 0 {
        return Err(format!("LAPACK DORGQR failed with info = {info}"));
    }

    // Q is m x n in column-major already
    let q = Matrix::new_2d(q_data, matrix.rows(), matrix.cols())?;

    Ok(QrDecomposition { q, r })
}

/// Compute eigenvalues and optionally eigenvectors
/// Uses DSYEV (double precision symmetric eigenvalue problem)
pub fn lapack_eigenvalues(
    matrix: &Matrix,
    compute_vectors: bool,
) -> Result<EigenDecomposition, String> {
    if !is_square(matrix) {
        return Err("Matrix must be square for eigenvalue decomposition".to_string());
    }

    let n = matrix.rows() as i32;
    let jobz = if compute_vectors { b'V' } else { b'N' };
    let uplo = b'U'; // Upper triangular

    let mut a_copy = matrix.data.clone();
    let mut w = vec![0.0; n as usize]; // eigenvalues
    let mut work = vec![0.0; 1];
    let mut lwork = -1i32;
    let mut info = 0i32;

    // Query optimal work array size
    unsafe {
        lapack::dsyev(
            jobz,
            uplo,
            n,
            &mut a_copy,
            n,
            &mut w,
            &mut work,
            lwork,
            &mut info,
        );
    }

    if info != 0 {
        return Err(format!(
            "LAPACK DSYEV workspace query failed with info = {info}"
        ));
    }

    lwork = work[0] as i32;
    work.resize(lwork as usize, 0.0);

    // Compute eigenvalues (and eigenvectors if requested)
    unsafe {
        lapack::dsyev(
            jobz,
            uplo,
            n,
            &mut a_copy,
            n,
            &mut w,
            &mut work,
            lwork,
            &mut info,
        );
    }

    if info != 0 {
        return Err(format!("LAPACK DSYEV failed with info = {info}"));
    }

    let eigenvectors = if compute_vectors {
        Some(Matrix::new_2d(a_copy, matrix.rows(), matrix.cols())?)
    } else {
        None
    };

    Ok(EigenDecomposition {
        eigenvalues: w,
        eigenvectors,
    })
}

/// Matrix determinant using LU decomposition
pub fn lapack_determinant(matrix: &Matrix) -> Result<f64, String> {
    let lu = lapack_lu_decomposition(matrix)?;

    // Determinant is product of diagonal elements times sign of permutation
    let mut det = 1.0;
    let n = matrix.rows();

    // Product of diagonal elements
    for i in 0..n {
        det *= lu.lu_matrix.data[i * n + i];
    }

    // Count number of row swaps to determine sign
    let mut swaps = 0;
    for i in 0..n {
        if lu.pivots[i] != (i + 1) as i32 {
            // LAPACK uses 1-based indexing
            swaps += 1;
        }
    }

    if swaps % 2 == 1 {
        det = -det;
    }

    Ok(det)
}

/// Matrix inverse using LU decomposition
pub fn lapack_matrix_inverse(matrix: &Matrix) -> Result<Matrix, String> {
    if !is_square(matrix) {
        return Err("Matrix must be square for inversion".to_string());
    }

    let n = matrix.rows() as i32;
    // Our storage is column-major; copy as-is
    let mut a_copy = matrix.data.clone();
    let mut ipiv = vec![0i32; n as usize];
    let mut work = vec![0.0; 1];
    let mut lwork = -1i32;
    let mut info = 0i32;

    // LU factorization
    unsafe {
        lapack::dgetrf(n, n, &mut a_copy, n, &mut ipiv, &mut info);
    }

    if info != 0 {
        return Err(format!("LAPACK DGETRF failed with info = {info}"));
    }

    // Query optimal work array size for inversion
    unsafe {
        lapack::dgetri(n, &mut a_copy, n, &ipiv, &mut work, lwork, &mut info);
    }

    if info != 0 {
        return Err(format!(
            "LAPACK DGETRI workspace query failed with info = {info}"
        ));
    }

    lwork = work[0] as i32;
    work.resize(lwork as usize, 0.0);

    // Compute inverse
    unsafe {
        lapack::dgetri(n, &mut a_copy, n, &ipiv, &mut work, lwork, &mut info);
    }

    if info != 0 {
        return Err(format!("LAPACK DGETRI failed with info = {info}"));
    }

    // Already in column-major
    Matrix::new_2d(a_copy, matrix.rows(), matrix.cols())
}

// Helper function to check if a matrix is square
fn is_square(matrix: &Matrix) -> bool {
    matrix.rows() == matrix.cols()
}

// Helper function to convert Vec<Value> to Vec<f64>
#[allow(dead_code)]
pub(crate) fn value_vector_to_f64(values: &[Value]) -> Result<Vec<f64>, String> {
    let mut out: Vec<f64> = Vec::new();
    for v in values {
        match v {
            Value::Num(n) => out.push(*n),
            Value::Int(i) => out.push(i.to_f64()),
            Value::Cell(c) => {
                for elem in &c.data {
                    match &**elem {
                        Value::Num(n) => out.push(*n),
                        Value::Int(i) => out.push(i.to_f64()),
                        _ => return Err(format!("Cannot convert {elem:?} to f64")),
                    }
                }
            }
            _ => return Err(format!("Cannot convert {v:?} to f64")),
        }
    }
    Ok(out)
}

// Helper function to convert Vec<f64> to Vec<Value>
#[allow(dead_code)]
fn f64_vector_to_value(values: Vec<f64>) -> Vec<Value> {
    values.into_iter().map(Value::Num).collect()
}

// Builtin functions for LAPACK operations
#[cfg(test)]
#[allow(dead_code)]
fn solve_builtin(a: Matrix, b: Vec<Value>) -> Result<Value, String> {
    let b_f64 = value_vector_to_f64(&b)?;
    let x = lapack_solve_linear_system(&a, &b_f64)?;
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(x, vec![b_f64.len(), 1]).map_err(|e| format!("solve: {e}"))?,
    ))
}

#[cfg(test)]
#[allow(dead_code)]
fn det_builtin(matrix: Matrix) -> Result<f64, String> {
    lapack_determinant(&matrix)
}

#[cfg(test)]
#[allow(dead_code)]
fn inv_builtin(matrix: Matrix) -> Result<Matrix, String> {
    lapack_matrix_inverse(&matrix)
}

#[cfg(test)]
#[allow(dead_code)]
fn eig_builtin(matrix: Matrix) -> Result<Matrix, String> {
    let decomp = lapack_eigenvalues(&matrix, false)?;
    Ok(decomp.eigenvectors.unwrap())
}
